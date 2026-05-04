from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.sac_smopso.controller_checkpoint import (
    _infer_checkpoint_operator_count,
    _migrate_action_dim_in_place,
    _migrate_operator_count_in_place,
)
from uav_benchmark.algorithms.sac_smopso.controller_networks import _Actor, _Critic
from uav_benchmark.algorithms.sac_smopso.controller_replay import ReplayBuffer, _stack_serialized_states
from uav_benchmark.algorithms.sac_smopso.controller_types import (
    ControllerAction,
    ControllerConfig,
    TemporalRelationalState,
    TemporalRelationalStateSpec,
    _masked_mean_np,
)
from uav_benchmark.algorithms.sac_smopso.torch_support import _TORCH_AVAILABLE, F, nn, torch

__all__ = [
    "ControllerAction",
    "ControllerConfig",
    "HybridSACController",
    "TemporalRelationalState",
    "TemporalRelationalStateSpec",
    "_TORCH_AVAILABLE",
    "torch",
]


class HybridSACController:
    def __init__(
        self,
        state_spec: TemporalRelationalStateSpec,
        action_dim: int,
        operator_names: tuple[str, ...],
        device_tag: str = "cpu",
        config: ControllerConfig | None = None,
        policy_mode: str = "online",
        encoder_mode: str = "learned",
    ) -> None:
        self.state_spec = state_spec
        self.action_dim = int(action_dim)
        self.operator_names = tuple(operator_names)
        self.operator_count = len(self.operator_names)
        self.config = config or ControllerConfig()
        self.policy_mode = str(policy_mode).strip().lower() or "online"
        self.encoder_mode = str(encoder_mode).strip().lower() or "learned"
        self.use_operator_head = bool(self.config.use_operator_head)
        self.total_steps = 0
        self.training_steps = 0
        self.loss_ema = 0.0
        self.last_operator_probs = np.full(self.operator_count, 1.0 / max(1, self.operator_count), dtype=float)
        self.last_source = "heuristic"
        self.checkpoint_loaded = False
        self.checkpoint_path = ""
        self.enabled = bool(_TORCH_AVAILABLE and torch is not None)
        self.device_tag = "heuristic:numpy"
        self.device = None
        self.target_entropy = -float(self.action_dim)
        if self.use_operator_head:
            self.target_entropy -= float(np.log(max(2, self.operator_count)))
        self.replay = ReplayBuffer(self.config.replay_capacity)

        self.actor = None
        self.q1 = None
        self.q2 = None
        self.target_q1 = None
        self.target_q2 = None
        self.actor_optim = None
        self.q1_optim = None
        self.q2_optim = None
        self.log_alpha = None
        self.alpha_optim = None

        if not self.enabled:
            return

        assert torch is not None and nn is not None
        try:
            self.device = torch.device(device_tag)
            self.device_tag = f"torch:{self.device.type}"
        except (RuntimeError, TypeError, ValueError):
            self.device = torch.device("cpu")
            self.device_tag = "torch:cpu"

        hidden_dim = int(self.config.hidden_dim)
        self.actor = _Actor(
            self.state_spec,
            self.action_dim,
            self.operator_count,
            hidden_dim,
            self.encoder_mode,
        ).to(self.device)
        self.q1 = _Critic(
            self.state_spec,
            self.action_dim,
            self.operator_count,
            hidden_dim,
            self.encoder_mode,
        ).to(self.device)
        self.q2 = _Critic(
            self.state_spec,
            self.action_dim,
            self.operator_count,
            hidden_dim,
            self.encoder_mode,
        ).to(self.device)
        self.target_q1 = _Critic(
            self.state_spec,
            self.action_dim,
            self.operator_count,
            hidden_dim,
            self.encoder_mode,
        ).to(self.device)
        self.target_q2 = _Critic(
            self.state_spec,
            self.action_dim,
            self.operator_count,
            hidden_dim,
            self.encoder_mode,
        ).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(self.config.lr))
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=float(self.config.lr))
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=float(self.config.lr))
        initial_alpha = max(1e-4, float(self.config.alpha_init))
        self.log_alpha = torch.tensor(
            np.log(initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=float(self.config.lr))

    @property
    def alpha(self) -> float:
        if not self.enabled or self.log_alpha is None or torch is None:
            return float(self.config.alpha_init)
        return float(torch.exp(self.log_alpha.detach()).item())

    def training_enabled(self) -> bool:
        return self.policy_mode != "frozen"

    def can_use_policy(self) -> bool:
        if not self.enabled or self.actor is None or torch is None or self.device is None:
            return False
        if self.checkpoint_loaded:
            return True
        return self.total_steps >= int(self.config.warmup_steps) and len(self.replay) >= max(
            8,
            min(int(self.config.batch_size), int(self.config.warmup_steps)),
        )

    def set_policy_mode(self, policy_mode: str) -> None:
        normalized = str(policy_mode).strip().lower() or "online"
        if normalized not in {"online", "finetune", "frozen"}:
            raise ValueError(f"Unsupported SAC policy mode: {policy_mode}")
        self.policy_mode = normalized

    def _policy_mix_factor(self) -> float:
        if self.policy_mode == "frozen":
            return 1.0
        anneal_steps = max(1, int(self.config.policy_mix_anneal_steps))
        progress = float(np.clip(self.training_steps / float(anneal_steps), 0.0, 1.0))
        if self.checkpoint_loaded:
            start = float(self.config.loaded_policy_mix_start)
            end = float(self.config.loaded_policy_mix_end)
        else:
            start = float(self.config.scratch_policy_mix_start)
            end = float(self.config.scratch_policy_mix_end)
        return float(np.clip(start + (end - start) * progress, min(start, end), max(start, end)))

    def act(self, state: TemporalRelationalState, deterministic: bool = False) -> ControllerAction:
        heuristic = self._heuristic_action(state, deterministic=deterministic)
        if not self.enabled or self.actor is None or torch is None or self.device is None:
            self.last_source = heuristic.source
            self.last_operator_probs = heuristic.operator_probs.copy()
            return heuristic

        use_policy = self.can_use_policy()
        if not use_policy:
            self.last_source = heuristic.source
            self.last_operator_probs = heuristic.operator_probs.copy()
            return heuristic

        batch_state = _stack_serialized_states([state.serialize()], device=self.device)
        with torch.no_grad():
            continuous_t, _operator_one_hot, _operator_id_t, log_prob_t, operator_probs_t = self.actor.sample(
                batch_state,
                deterministic=deterministic,
                use_operator_head=self.use_operator_head,
            )
        actor_continuous = continuous_t.squeeze(0).cpu().numpy()
        if self.use_operator_head:
            actor_probs = operator_probs_t.squeeze(0).cpu().numpy()
            actor_probs = np.clip(np.asarray(actor_probs, dtype=float), 1e-8, None)
            actor_probs /= np.sum(actor_probs)
        else:
            actor_probs = heuristic.operator_probs.copy()

        mix = self._policy_mix_factor()
        continuous = np.clip(mix * actor_continuous + (1.0 - mix) * heuristic.continuous, -1.0, 1.0)
        if self.use_operator_head:
            operator_probs = np.clip(mix * actor_probs + (1.0 - mix) * heuristic.operator_probs, 1e-8, None)
            operator_probs /= np.sum(operator_probs)
            if deterministic:
                operator_id = int(np.argmax(operator_probs))
            else:
                operator_id = int(np.random.choice(self.operator_count, p=operator_probs))
        else:
            operator_probs = heuristic.operator_probs.copy()
            operator_id = int(heuristic.operator_id)
        action = ControllerAction(
            operator_id=operator_id,
            continuous=continuous.astype(np.float32, copy=False),
            operator_probs=operator_probs.astype(np.float32, copy=False),
            log_prob=float(log_prob_t.item()),
            source="sac-mixed",
        )
        self.last_source = action.source
        self.last_operator_probs = action.operator_probs.copy()
        return action

    def observe(
        self,
        state: TemporalRelationalState,
        action: ControllerAction,
        reward: float,
        next_state: TemporalRelationalState,
        done: bool,
    ) -> None:
        if not self.training_enabled():
            return
        self.replay.add(
            state=state,
            continuous_action=np.asarray(action.continuous, dtype=np.float32).reshape(-1),
            operator_id=int(action.operator_id),
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
        )
        self.total_steps += 1

        if (
            not self.enabled
            or torch is None
            or self.device is None
            or self.actor is None
            or self.q1 is None
            or self.q2 is None
            or self.target_q1 is None
            or self.target_q2 is None
            or self.actor_optim is None
            or self.q1_optim is None
            or self.q2_optim is None
            or self.log_alpha is None
            or self.alpha_optim is None
        ):
            return

        if self.total_steps < int(self.config.warmup_steps):
            return
        if len(self.replay) < max(8, int(self.config.batch_size)):
            return

        for _ in range(max(1, int(self.config.updates_per_step))):
            critic_loss, actor_loss = self._update_networks()
            combined_loss = float(critic_loss + actor_loss)
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * combined_loss if self.training_steps > 0 else combined_loss
            self.training_steps += 1

    def gpu_peak_bytes(self) -> float:
        if not self.enabled or torch is None or self.device is None:
            return 0.0
        if self.device.type != "cuda":
            return 0.0
        try:
            return float(torch.cuda.max_memory_allocated(self.device))
        except (RuntimeError, TypeError, ValueError):
            return 0.0

    def metadata(self) -> dict[str, Any]:
        return {
            "controllerActionDim": float(self.action_dim),
            "controllerOperatorCount": float(self.operator_count),
            "controllerWarmupSteps": float(self.config.warmup_steps),
            "controllerBatchSize": float(self.config.batch_size),
            "controllerReplayCapacity": float(self.config.replay_capacity),
            "controllerReplaySize": float(len(self.replay)),
            "controllerUpdates": float(self.training_steps),
            "controllerAlpha": float(self.alpha),
            "controllerPolicyMix": float(self._policy_mix_factor()),
            "controllerOperatorHeadEnabled": 1.0 if self.use_operator_head else 0.0,
            "policyMode": self.policy_mode,
            "encoderMode": self.encoder_mode,
            "checkpointLoaded": 1.0 if self.checkpoint_loaded else 0.0,
            "stateGlobalDim": float(self.state_spec.global_dim),
            "statePopulationDim": float(self.state_spec.population_dim),
            "stateArchiveDim": float(self.state_spec.archive_dim),
            "stateTopologyDim": float(self.state_spec.topology_dim),
            "stateInteractionDim": float(self.state_spec.interaction_dim),
            "stateEnvironmentDim": float(self.state_spec.environment_dim),
            "stateTemporalDim": float(self.state_spec.temporal_dim),
        }

    def save_checkpoint(self, path: str | Path, extra_metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled or torch is None or self.device is None:
            raise RuntimeError("Torch-enabled SAC controller is required to save a checkpoint.")
        if (
            self.actor is None
            or self.q1 is None
            or self.q2 is None
            or self.target_q1 is None
            or self.target_q2 is None
            or self.actor_optim is None
            or self.q1_optim is None
            or self.q2_optim is None
            or self.log_alpha is None
            or self.alpha_optim is None
        ):
            raise RuntimeError("SAC controller networks are not initialized.")
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "stateSpec": {
                "globalDim": int(self.state_spec.global_dim),
                "populationDim": int(self.state_spec.population_dim),
                "archiveDim": int(self.state_spec.archive_dim),
                "topologyDim": int(self.state_spec.topology_dim),
                "interactionDim": int(self.state_spec.interaction_dim),
                "environmentDim": int(self.state_spec.environment_dim),
                "temporalDim": int(self.state_spec.temporal_dim),
            },
            "actionDim": int(self.action_dim),
            "operatorNames": list(self.operator_names),
            "operatorCount": int(self.operator_count),
            "encoderMode": str(self.encoder_mode),
            "config": {
                "hiddenDim": int(self.config.hidden_dim),
                "lr": float(self.config.lr),
                "gamma": float(self.config.gamma),
                "tau": float(self.config.tau),
                "replayCapacity": int(self.config.replay_capacity),
                "batchSize": int(self.config.batch_size),
                "warmupSteps": int(self.config.warmup_steps),
                "updatesPerStep": int(self.config.updates_per_step),
                "alphaInit": float(self.config.alpha_init),
                "scratchPolicyMixStart": float(self.config.scratch_policy_mix_start),
                "scratchPolicyMixEnd": float(self.config.scratch_policy_mix_end),
                "loadedPolicyMixStart": float(self.config.loaded_policy_mix_start),
                "loadedPolicyMixEnd": float(self.config.loaded_policy_mix_end),
                "policyMixAnnealSteps": int(self.config.policy_mix_anneal_steps),
                "useOperatorHead": 1 if self.use_operator_head else 0,
            },
            "policyMode": str(self.policy_mode),
            "trainingSteps": int(self.training_steps),
            "totalSteps": int(self.total_steps),
            "lossEma": float(self.loss_ema),
            "logAlpha": float(self.log_alpha.detach().cpu().item()),
            "replaySize": int(len(self.replay)),
            "replayData": self.replay.serialize(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "targetQ1": self.target_q1.state_dict(),
            "targetQ2": self.target_q2.state_dict(),
            "actorOptim": self.actor_optim.state_dict(),
            "q1Optim": self.q1_optim.state_dict(),
            "q2Optim": self.q2_optim.state_dict(),
            "alphaOptim": self.alpha_optim.state_dict(),
            "extraMetadata": dict(extra_metadata or {}),
        }
        torch.save(payload, checkpoint_path)
        self.checkpoint_path = str(checkpoint_path)

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = True) -> dict[str, Any]:
        if not self.enabled or torch is None or self.device is None:
            raise RuntimeError("Torch-enabled SAC controller is required to load a checkpoint.")
        if (
            self.actor is None
            or self.q1 is None
            or self.q2 is None
            or self.target_q1 is None
            or self.target_q2 is None
            or self.actor_optim is None
            or self.q1_optim is None
            or self.q2_optim is None
            or self.log_alpha is None
            or self.alpha_optim is None
        ):
            raise RuntimeError("SAC controller networks are not initialized.")
        checkpoint_path = Path(path)
        try:
            payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch versions do not support weights_only
            payload = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_operator_count = _infer_checkpoint_operator_count(payload)
        ckpt_action_dim = int(payload.get("actionDim", -1))
        migrated_action_dim = False
        if ckpt_action_dim != self.action_dim:
            if ckpt_action_dim > 0 and ckpt_action_dim < self.action_dim:
                # Old checkpoint saved with fewer action dims (e.g. 12 -> 13
                # after adding `repair_intensity`). Migrate by zero-padding the
                # actor output heads and critic input layers so that the old
                # behaviour is preserved on the first ``ckpt_action_dim`` dims
                # and the new dims start at their neutral mean (tanh(0) = 0).
                _migrate_action_dim_in_place(
                    payload=payload,
                    old_dim=ckpt_action_dim,
                    new_dim=self.action_dim,
                    operator_count=checkpoint_operator_count,
                )
                migrated_action_dim = True
            else:
                raise ValueError(
                    f"Checkpoint action dim ({ckpt_action_dim}) is larger than "
                    f"controller action dim ({self.action_dim}); cannot shrink "
                    "(would drop trained weights). Retrain the policy with the "
                    "current 12-d lean action space."
                )
        checkpoint_ops = tuple(str(name) for name in payload.get("operatorNames", []))
        migrated_operator_count = False
        if checkpoint_operator_count != self.operator_count:
            if self.use_operator_head:
                raise ValueError("Checkpoint operator set does not match controller operator set.")
            _migrate_operator_count_in_place(
                payload=payload,
                old_count=checkpoint_operator_count,
                new_count=self.operator_count,
                action_dim=int(payload.get("actionDim", self.action_dim)),
            )
            migrated_operator_count = True
        if checkpoint_ops != self.operator_names and self.use_operator_head:
            raise ValueError("Checkpoint operator set does not match controller operator set.")
        if str(payload.get("encoderMode", "")).strip().lower() != self.encoder_mode:
            raise ValueError("Checkpoint encoder mode does not match controller encoder mode.")
        spec = payload.get("stateSpec", {})
        expected = {
            "globalDim": int(self.state_spec.global_dim),
            "populationDim": int(self.state_spec.population_dim),
            "archiveDim": int(self.state_spec.archive_dim),
            "topologyDim": int(self.state_spec.topology_dim),
            "interactionDim": int(self.state_spec.interaction_dim),
            "environmentDim": int(self.state_spec.environment_dim),
            "temporalDim": int(self.state_spec.temporal_dim),
        }
        for key, value in expected.items():
            if int(spec.get(key, -1)) != int(value):
                raise ValueError(f"Checkpoint state spec mismatch for {key}.")

        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.target_q1.load_state_dict(payload["targetQ1"])
        self.target_q2.load_state_dict(payload["targetQ2"])
        with torch.no_grad():
            self.log_alpha.copy_(
                torch.tensor(
                    float(payload.get("logAlpha", np.log(self.config.alpha_init))),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        if load_optimizers and not migrated_action_dim and not migrated_operator_count:
            self.actor_optim.load_state_dict(payload["actorOptim"])
            self.q1_optim.load_state_dict(payload["q1Optim"])
            self.q2_optim.load_state_dict(payload["q2Optim"])
            self.alpha_optim.load_state_dict(payload["alphaOptim"])
            self.replay.load_serialized(payload.get("replayData"))
        elif migrated_action_dim or migrated_operator_count:
            if load_optimizers and not migrated_action_dim:
                self.replay.load_serialized(payload.get("replayData"))
            # Optimizer states and replay buffer reference the old action
            # shape; keep the freshly-initialised optimizers and start with
            # an empty replay. Operator-count migrations keep the replay data
            # because the disabled operator head ignores cached operator ids.
            warnings.warn(
                (
                    f"SAC checkpoint migrated from action_dim={ckpt_action_dim} to action_dim={self.action_dim}; "
                    "optimizer state and replay buffer were reset. Run a short "
                    "retraining pass to refit the policy to the widened action "
                    "space."
                )
                if migrated_action_dim
                else (
                    f"SAC checkpoint operator branch migrated from {checkpoint_operator_count} to {self.operator_count} entries "
                    "while the operator head is disabled; optimizer state was reset "
                    "but replay data was preserved."
                ),
                stacklevel=2,
            )
        self.training_steps = int(payload.get("trainingSteps", 0))
        self.total_steps = int(payload.get("totalSteps", 0))
        self.loss_ema = float(payload.get("lossEma", 0.0))
        self.checkpoint_loaded = True
        self.checkpoint_path = str(checkpoint_path)
        return dict(payload.get("extraMetadata", {}))

    def _heuristic_action(
        self,
        state: TemporalRelationalState,
        deterministic: bool,
    ) -> ControllerAction:
        global_state = np.asarray(state.global_features, dtype=np.float32).reshape(-1)
        topology_mean = _masked_mean_np(state.topology_tokens, state.topology_mask)
        interaction_mean = _masked_mean_np(state.interaction_tokens, state.interaction_mask)

        progress = float(global_state[0]) if global_state.size > 0 else 0.0
        feasible = float(global_state[1]) if global_state.size > 1 else 1.0
        conflict = float(global_state[2]) if global_state.size > 2 else 0.0
        hv_trend = float(global_state[3]) if global_state.size > 3 else 0.5
        diversity = float(global_state[4]) if global_state.size > 4 else 0.5
        stagnation = float(global_state[5]) if global_state.size > 5 else 0.0
        archive_fill = float(global_state[6]) if global_state.size > 6 else 0.5
        archive_occupancy = float(global_state[7]) if global_state.size > 7 else 0.5
        spatial_occupancy = float(global_state[8]) if global_state.size > 8 else 0.5
        mean_violation = float(global_state[10]) if global_state.size > 10 else max(0.0, 1.0 - feasible)
        clearance_pressure = (
            float(global_state[20])
            if global_state.size > 20
            else float(topology_mean[5])
            if topology_mean.size > 5
            else 0.0
        )
        overlap_pressure = (
            float(global_state[21])
            if global_state.size > 21
            else float(topology_mean[6])
            if topology_mean.size > 6
            else 0.0
        )
        interaction_pressure = (
            float(global_state[22])
            if global_state.size > 22
            else float(interaction_mean[2])
            if interaction_mean.size > 2
            else 0.0
        )
        turn_saturation = (
            float(global_state[23])
            if global_state.size > 23
            else float(topology_mean[7])
            if topology_mean.size > 7
            else 0.0
        )

        # Build the live 12-d SAC-SMOPSO heuristic action in canonical order:
        #   [inertia, c1, c2, velocity_scale, kappa_scale, delta_scale,
        #    leader_bias, mutation_prob, repulsion_weight, archive_focus,
        #    repair_intensity, sbx_weight]
        full = np.array(
            [
                0.45 - 0.90 * progress + 0.15 * (1.0 - hv_trend),
                -0.05 + 0.70 * (1.0 - diversity) + 0.20 * overlap_pressure,
                -0.15 + 0.95 * progress + 0.20 * archive_occupancy,
                0.10 * (1.0 - progress) + 0.45 * (1.0 - feasible) + 0.15 * interaction_pressure,
                -0.10 + 0.65 * archive_occupancy + 0.20 * spatial_occupancy,
                -0.15 + 0.85 * stagnation + 0.20 * clearance_pressure,
                -0.35 + 1.10 * progress + 0.20 * turn_saturation,
                -0.55 + 1.50 * max(conflict, mean_violation, interaction_pressure),
                -0.75 + 1.65 * max(conflict, interaction_pressure),
                -0.35 + 1.40 * max(spatial_occupancy, overlap_pressure),
                -0.15 + 0.65 * archive_fill + 0.15 * clearance_pressure,
                # repair_intensity: scale with conflict / violation / early
                # generation. Stays mildly positive so the repair step runs
                # even when the archive looks healthy, but drops to ~0.2 in
                # the final 25% of generations when refinement dominates.
                -0.35
                + 1.60 * max(conflict, mean_violation, interaction_pressure, overlap_pressure)
                + 0.25 * (1.0 - feasible)
                - 0.30 * max(0.0, progress - 0.75),
                # sbx_weight: prefer the reservoir-SBX step when progress
                # stalls, diversity collapses, or path overlap remains high.
                -0.55
                + 1.45 * max(stagnation, 1.0 - hv_trend, overlap_pressure, 1.0 - diversity)
                + 0.20 * max(0.0, 0.4 - feasible)
                - 0.20 * max(0.0, progress - 0.85),
            ],
            dtype=np.float32,
        )
        if self.action_dim < full.size:
            continuous = full[: self.action_dim].astype(np.float32)
        else:
            pad = np.full(self.action_dim - full.size, 0.1, dtype=np.float32)
            continuous = np.concatenate([full, pad])
        continuous = np.clip(continuous, -1.0, 1.0)
        if not deterministic:
            continuous = np.clip(
                continuous + np.random.normal(0.0, 0.08, size=continuous.shape),
                -1.0,
                1.0,
            )

        operator_probs = np.full(self.operator_count, 0.05, dtype=np.float32)
        operator_id = 0
        # Operator layout: 0=base, 1=sbx, 2=de, 3=elite, 4=spread.
        # Rule of thumb:
        #   - Severe constraint violations / conflicts -> `elite` (local refinement
        #     around the best feasible candidates) backed by `de` (which injects
        #     pbest-directed moves that repel particles through the DE mixing).
        #   - Long stagnation and low diversity -> `de` with a `spread` mix.
        #   - Late stage with turn saturation -> `elite`.
        #   - Low archive quality -> `sbx`.
        #   - Otherwise -> `base` PSO move.
        if conflict > 0.30 or mean_violation > 0.25 or interaction_pressure > 0.35:
            operator_id = min(self.operator_count - 1, 3)  # elite
            operator_probs[operator_id] = 0.55
            if self.operator_count > 2:
                operator_probs[2] = 0.25  # de as strong secondary
            if self.operator_count > 1:
                operator_probs[1] = 0.10  # sbx as tertiary for exploration
        elif stagnation > 0.55 and (diversity < 0.45 or overlap_pressure > 0.30):
            operator_id = min(self.operator_count - 1, 2)
            operator_probs[operator_id] = 0.60
            if self.operator_count > 4:
                operator_probs[4] = 0.20
            if self.operator_count > 1:
                operator_probs[1] = 0.10
        elif progress > 0.70 or turn_saturation > 0.70:
            operator_id = min(self.operator_count - 1, 3)
            operator_probs[operator_id] = 0.64
            if self.operator_count > 1:
                operator_probs[1] = 0.16
        elif hv_trend < 0.48 or archive_occupancy < 0.30 or clearance_pressure > 0.35:
            operator_id = min(self.operator_count - 1, 1)
            operator_probs[operator_id] = 0.60
            if self.operator_count > 2:
                operator_probs[2] = 0.18
        else:
            operator_probs[0] = 0.65
            if self.operator_count > 1:
                operator_probs[1] = 0.12
            if self.operator_count > 2:
                operator_probs[2] = 0.12
        operator_probs = np.clip(operator_probs, 1e-6, None)
        operator_probs /= np.sum(operator_probs)
        chosen = operator_id if deterministic else int(np.random.choice(self.operator_count, p=operator_probs))
        return ControllerAction(
            operator_id=int(chosen),
            continuous=continuous.astype(np.float32, copy=False),
            operator_probs=operator_probs.astype(np.float32, copy=False),
            log_prob=0.0,
            source="heuristic",
        )

    def _update_networks(self) -> tuple[float, float]:
        assert (
            self.enabled
            and torch is not None
            and self.device is not None
            and self.actor is not None
            and self.q1 is not None
            and self.q2 is not None
            and self.target_q1 is not None
            and self.target_q2 is not None
            and self.actor_optim is not None
            and self.q1_optim is not None
            and self.q2_optim is not None
            and self.log_alpha is not None
            and self.alpha_optim is not None
        )
        states, cont_actions, operator_one_hot, rewards, next_states, dones = self.replay.sample(
            batch_size=int(self.config.batch_size),
            operator_count=self.operator_count,
            device=self.device,
            use_operator_head=self.use_operator_head,
        )

        with torch.no_grad():
            next_cont, next_operator_one_hot, _next_op, next_log_prob, _next_probs = self.actor.sample(
                next_states,
                use_operator_head=self.use_operator_head,
            )
            target_q = torch.min(
                self.target_q1(next_states, next_cont, next_operator_one_hot),
                self.target_q2(next_states, next_cont, next_operator_one_hot),
            )
            alpha = torch.exp(self.log_alpha.detach())
            q_target = rewards + (1.0 - dones) * float(self.config.gamma) * (target_q - alpha * next_log_prob)

        q1_pred = self.q1(states, cont_actions, operator_one_hot)
        q2_pred = self.q2(states, cont_actions, operator_one_hot)
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.q1_optim.zero_grad(set_to_none=True)
        self.q2_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=5.0)
        self.q1_optim.step()
        self.q2_optim.step()

        policy_cont, policy_operator_one_hot, _policy_op, log_prob, _policy_probs = self.actor.sample(
            states,
            use_operator_head=self.use_operator_head,
        )
        alpha = torch.exp(self.log_alpha)
        q_policy = torch.min(
            self.q1(states, policy_cont, policy_operator_one_hot),
            self.q2(states, policy_cont, policy_operator_one_hot),
        )
        actor_loss = (alpha.detach() * log_prob - q_policy).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        tau = float(self.config.tau)
        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters(), strict=False):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters(), strict=False):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)

        return float(critic_loss.item()), float(actor_loss.item())
