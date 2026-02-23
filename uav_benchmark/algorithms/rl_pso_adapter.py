"""RL↔PSO adaptive controller adapter (simplified).

Bridges the unified RL controller and the NMOPSO engine.
One adapter instance per run.

Call sequence per generation:
  1. action = adapter.observe_and_act(generation)
  2. result = engine.step(action.w, action.c1, action.c2, ...)
  3. adapter.execute_operator(action)
  4. adapter.post_step(hv_before, hv_after, ...)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import uav_benchmark.algorithms.rl_defaults as RLD
from uav_benchmark.algorithms.rl_controller import ContinuousAction, N_FEATURES
from uav_benchmark.algorithms.nmopso_engine import NMOPSOEngine


# ── Decoded PSO action ──────────────────────────────────────────────

@dataclass(slots=True)
class PSOAction:
    """Decoded hyperparameters for one PSO generation."""
    inertia: float
    c1: float
    c2: float
    mutation_prob: float = 0.15
    leader_bias: float = 0.5
    attention_weights: np.ndarray | None = None
    repulsion_weight: float = 0.0
    selected_operator: int = 0   # 0=noop, 1=SBX, 2=DE, 3=elite-refine
    action_idx: int = -1


# ── Adapter ─────────────────────────────────────────────────────────

class RLPSOAdapter:
    """Bridges the RL controller and the PSO engine.

    One adapter instance per run.
    """

    def __init__(
        self,
        controller: Any,
        engine: NMOPSOEngine,
        total_generations: int,
        # Reward
        hv_scale: float = RLD.DEFAULT_REWARD_HV_SCALE,
        div_scale: float = 1.0,
        reward_hv_w: float = RLD.DEFAULT_REWARD_HV_WEIGHT,
        reward_feas_w: float = RLD.DEFAULT_REWARD_FEASIBLE_WEIGHT,
        reward_div_w: float = RLD.DEFAULT_REWARD_DIVERSITY_WEIGHT,
        reward_aux_cost_w: float = RLD.REWARD_AUX_COST_WEIGHT,
        # Operator budget
        aux_eval_budget_factor: float = RLD.AUX_EVAL_BUDGET_FACTOR,
        aux_eval_budget_start_factor: float | None = None,
        aux_eval_budget_end_factor: float | None = None,
        operator_trigger_prob_start: float = 1.0,
        operator_trigger_prob_end: float = 1.0,
        operator_stagnation_boost: float = 0.0,
        operator_stagnation_threshold: int = 5,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
        # Attention
        attention_enabled: bool = True,
        attention_temperature: float = 0.35,
        seed: int = 0,
    ) -> None:
        self.controller = controller
        self.engine = engine
        self.total_generations = int(max(1, total_generations))

        # Reward params
        self.hv_scale = float(max(1e-6, hv_scale))
        self.div_scale = float(max(1e-9, div_scale))
        self.reward_hv_w = float(reward_hv_w)
        self.reward_feas_w = float(reward_feas_w)
        self.reward_div_w = float(reward_div_w)
        self.reward_aux_cost_w = float(max(0.0, reward_aux_cost_w))

        # Budget
        self.aux_eval_budget_per_gen = int(max(0, round(engine.pop_size * aux_eval_budget_factor)))
        budget_start = aux_eval_budget_start_factor if aux_eval_budget_start_factor is not None else aux_eval_budget_factor
        budget_end = aux_eval_budget_end_factor if aux_eval_budget_end_factor is not None else aux_eval_budget_factor
        self.aux_eval_budget_start_factor = float(max(0.0, budget_start))
        self.aux_eval_budget_end_factor = float(max(0.0, budget_end))
        self.operator_trigger_prob_start = float(np.clip(operator_trigger_prob_start, 0.0, 1.0))
        self.operator_trigger_prob_end = float(np.clip(operator_trigger_prob_end, 0.0, 1.0))
        self.operator_stagnation_boost = float(max(0.0, operator_stagnation_boost))
        self.operator_stagnation_threshold = int(max(1, operator_stagnation_threshold))
        self.surrogate_prefilter_enabled = bool(surrogate_prefilter_enabled)
        self.surrogate_prefilter_ratio = float(np.clip(surrogate_prefilter_ratio, 0.0, 1.0))
        self.surrogate_prefilter_min_candidates = int(max(1, surrogate_prefilter_min_candidates))
        self.surrogate_prefilter_k = int(max(1, surrogate_prefilter_k))
        self.attention_enabled = bool(attention_enabled)
        self.attention_temperature = float(max(1e-6, attention_temperature))
        self._rng = np.random.default_rng(int(seed))

        # Logging
        self.rl_actions: list[float] = []
        self.rl_rewards: list[float] = []
        self.rl_features: list[np.ndarray] = []
        self.rl_operators: list[int] = []
        self.rl_effective_operators: list[int] = []
        self.rl_aux_budget_allowed: list[float] = []
        self.rl_aux_eval_used: list[float] = []
        self.rl_aux_eval_usage_ratio: list[float] = []
        self.rl_operator_gate_skips: list[float] = []
        self.rl_attention_entropy: list[float] = []
        self.rl_attention_top1_mass: list[float] = []
        self.rl_attention_invalid_rows: list[float] = []

        # State
        self.last_hv: float = 0.0
        self.stagnation: int = 0
        self._last_features: np.ndarray | None = None
        self._last_action: ContinuousAction | None = None
        self._last_particle_features: np.ndarray | None = None
        self._last_archive_features: np.ndarray | None = None
        self._last_aux_eval_used: float = 0.0
        self._last_aux_budget_allowed: float = 0.0
        self._last_operator_gate_skip: float = 0.0
        self._last_effective_operator: int = 0

        # Timing
        self.rl_controller_time_sec: float = 0.0

        # Counters
        self.rl_sbx_injected: int = 0
        self.rl_de_injected: int = 0
        self.rl_elite_trials: int = 0
        self.rl_operator_gate_skip_count: int = 0
        self.rl_aux_eval_used_total: float = 0.0
        self.rl_aux_budget_total: float = 0.0
        self.rl_prefilter_filtered_total: float = 0.0

    def _progress(self, generation: int) -> float:
        if self.total_generations <= 1:
            return 1.0
        return float(np.clip((int(generation) - 1) / float(self.total_generations - 1), 0.0, 1.0))

    def _schedule_scalar(self, start: float, end: float, generation: int) -> float:
        p = self._progress(generation)
        return float((1.0 - p) * start + p * end)

    def _current_aux_budget(self, generation: int) -> int:
        factor = self._schedule_scalar(
            start=self.aux_eval_budget_start_factor,
            end=self.aux_eval_budget_end_factor,
            generation=generation,
        )
        budget = int(max(0, round(self.engine.pop_size * max(0.0, factor))))
        if self.stagnation <= 1:
            budget = int(max(0, round(0.5 * budget)))
        elif self.stagnation >= self.operator_stagnation_threshold:
            boost_steps = min(4, self.stagnation - self.operator_stagnation_threshold + 1)
            budget = int(max(0, round(budget * (1.0 + 0.20 * boost_steps))))
        budget = int(min(max(0, budget), self.engine.pop_size))
        return budget

    def _current_trigger_prob(self, generation: int) -> float:
        prob = self._schedule_scalar(
            start=self.operator_trigger_prob_start,
            end=self.operator_trigger_prob_end,
            generation=generation,
        )
        if prob < 0.999:
            if self.stagnation <= 0:
                prob *= 0.6
            elif self.stagnation == 1:
                prob *= 0.8
        if self.stagnation >= self.operator_stagnation_threshold:
            prob += self.operator_stagnation_boost
        return float(np.clip(prob, 0.0, 1.0))

    def observe_and_act(
        self,
        generation: int,
        inertia: float,
        inertia_min: float,
        diversity_ref: float,
    ) -> PSOAction:
        """Extract state, query controller, decode action."""
        # State features
        features = self.engine.state_features(
            generation=generation,
            total_generations=self.total_generations,
            last_hv=self.last_hv,
            stagnation=self.stagnation,
            diversity_ref=diversity_ref,
        )

        # Query controller
        t0 = time.perf_counter()
        action = self.controller.select_action(features)
        self.rl_controller_time_sec += float(time.perf_counter() - t0)

        attention_weights: np.ndarray | None = None
        attention_entropy = 0.0
        attention_top1_mass = 0.0
        attention_invalid_rows = 0.0
        if self.attention_enabled and hasattr(self.controller, "compute_attention_weights"):
            particle_features = self.engine.get_particle_features()
            archive_features = self.engine.get_archive_features()
            t1 = time.perf_counter()
            raw_attention = self.controller.compute_attention_weights(
                particle_features=particle_features,
                archive_features=archive_features,
                temperature=self.attention_temperature,
            )
            self.rl_controller_time_sec += float(time.perf_counter() - t1)
            att = np.asarray(raw_attention, dtype=float)
            if att.ndim == 2 and att.shape[0] == self.engine.pop_size and att.shape[1] == len(self.engine.archive):
                if att.shape[1] > 0:
                    att = np.where(np.isfinite(att), att, 0.0)
                    att = np.clip(att, 0.0, None)
                    row_sum = np.sum(att, axis=1, keepdims=True)
                    invalid = np.logical_or(~np.isfinite(row_sum[:, 0]), row_sum[:, 0] <= 1e-12)
                    attention_invalid_rows = float(np.sum(invalid))
                    if np.any(invalid):
                        att[invalid] = 1.0 / float(att.shape[1])
                        row_sum = np.sum(att, axis=1, keepdims=True)
                    att = att / np.maximum(row_sum, 1e-12)
                    top1 = np.max(att, axis=1)
                    attention_top1_mass = float(np.mean(top1))
                    entropy = -np.sum(att * np.log(np.clip(att, 1e-12, 1.0)), axis=1)
                    if att.shape[1] > 1:
                        entropy = entropy / np.log(float(att.shape[1]))
                    else:
                        entropy = np.zeros_like(entropy)
                    attention_entropy = float(np.mean(entropy))
            self._last_particle_features = particle_features
            self._last_archive_features = archive_features
            
            # Temporary bypass: Do not use random, untrained attention. Let the engine use Grid Leaders.
            attention_weights = None
        else:
            self._last_particle_features = None
            self._last_archive_features = None

        # Clip inertia based on controller output
        new_inertia = float(np.clip(action.w, inertia_min, 1.10))

        # Logging
        self.rl_actions.append(float(action.operator))
        self.rl_features.append(features.copy())
        self.rl_operators.append(action.operator)
        self.rl_attention_entropy.append(attention_entropy)
        self.rl_attention_top1_mass.append(attention_top1_mass)
        self.rl_attention_invalid_rows.append(attention_invalid_rows)

        # Store for post_step
        self._last_features = features
        self._last_action = action

        return PSOAction(
            inertia=new_inertia,
            c1=action.c1,
            c2=action.c2,
            mutation_prob=0.15,
            leader_bias=RLD.DEFAULT_LEADER_BIAS,
            attention_weights=attention_weights,
            selected_operator=action.operator,
            action_idx=action.operator,
        )

    def execute_operator(self, action: PSOAction, generation: int) -> None:
        """Execute the selected operator on the engine.

        Call after ``engine.step()`` but before ``post_step()``.
        """
        budget = int(self._current_aux_budget(generation))
        trigger_prob = float(self._current_trigger_prob(generation))
        used_eval = 0
        effective_operator = int(action.selected_operator)
        gate_skip = 0.0

        # Stochastic gate for expensive operators.
        if effective_operator > 0 and (budget <= 0 or float(self._rng.random()) > trigger_prob):
            effective_operator = 0
            gate_skip = 1.0
            self.rl_operator_gate_skip_count += 1

        # Arm 0: No-op
        if effective_operator == 0:
            pass

        # Arm 1: SBX injection
        elif effective_operator == 1:
            if budget > 0:
                try:
                    n = self.engine.inject_sbx(
                        ratio=RLD.SBX_INJECT_RATIO,
                        replace_ratio=RLD.SBX_REPLACE_RATIO,
                        max_evals=budget,
                        surrogate_prefilter_enabled=self.surrogate_prefilter_enabled,
                        surrogate_prefilter_ratio=self.surrogate_prefilter_ratio,
                        surrogate_prefilter_min_candidates=self.surrogate_prefilter_min_candidates,
                        surrogate_prefilter_k=self.surrogate_prefilter_k,
                    )
                except TypeError:
                    n = self.engine.inject_sbx(
                        ratio=RLD.SBX_INJECT_RATIO,
                        replace_ratio=RLD.SBX_REPLACE_RATIO,
                        max_evals=budget,
                    )
                used_eval += int(self.engine.last_operator_evals.get("sbx", 0))
                self.rl_sbx_injected += n

        # Arm 2: DE injection
        elif effective_operator == 2:
            if budget > 0:
                try:
                    n = self.engine.inject_de(
                        f_scale=RLD.DE_F_SCALE,
                        cr_rate=RLD.DE_CR_RATE,
                        ratio=RLD.DE_INJECT_RATIO,
                        replace_ratio=RLD.DE_REPLACE_RATIO,
                        pbest_ratio=RLD.DE_PBEST_RATIO,
                        max_evals=budget,
                        surrogate_prefilter_enabled=self.surrogate_prefilter_enabled,
                        surrogate_prefilter_ratio=self.surrogate_prefilter_ratio,
                        surrogate_prefilter_min_candidates=self.surrogate_prefilter_min_candidates,
                        surrogate_prefilter_k=self.surrogate_prefilter_k,
                    )
                except TypeError:
                    n = self.engine.inject_de(
                        f_scale=RLD.DE_F_SCALE,
                        cr_rate=RLD.DE_CR_RATE,
                        ratio=RLD.DE_INJECT_RATIO,
                        replace_ratio=RLD.DE_REPLACE_RATIO,
                        pbest_ratio=RLD.DE_PBEST_RATIO,
                        max_evals=budget,
                    )
                used_eval += int(self.engine.last_operator_evals.get("de", 0))
                self.rl_de_injected += n

        # Arm 3: Elite refine
        elif effective_operator == 3:
            if budget > 0:
                try:
                    n = self.engine.elite_refine(
                        sigma=RLD.ELITE_SIGMA,
                        top_k=RLD.ELITE_REFINE_TOP_K,
                        iters=RLD.ELITE_REFINE_ITERS,
                        max_evals=budget,
                        surrogate_prefilter_enabled=self.surrogate_prefilter_enabled,
                        surrogate_prefilter_ratio=self.surrogate_prefilter_ratio,
                        surrogate_prefilter_min_candidates=self.surrogate_prefilter_min_candidates,
                        surrogate_prefilter_k=self.surrogate_prefilter_k,
                    )
                except TypeError:
                    n = self.engine.elite_refine(
                        sigma=RLD.ELITE_SIGMA,
                        top_k=RLD.ELITE_REFINE_TOP_K,
                        iters=RLD.ELITE_REFINE_ITERS,
                        max_evals=budget,
                    )
                used_eval += int(self.engine.last_operator_evals.get("elite", 0))
                self.rl_elite_trials += n

        self._last_aux_eval_used = float(max(0, used_eval))
        self._last_aux_budget_allowed = float(max(0, budget))
        self._last_operator_gate_skip = float(gate_skip)
        self._last_effective_operator = int(effective_operator)
        self.rl_aux_eval_used_total += self._last_aux_eval_used
        self.rl_aux_budget_total += self._last_aux_budget_allowed
        self.rl_effective_operators.append(float(self._last_effective_operator))
        self.rl_aux_budget_allowed.append(self._last_aux_budget_allowed)
        self.rl_aux_eval_used.append(self._last_aux_eval_used)
        usage_ratio = self._last_aux_eval_used / max(1.0, self._last_aux_budget_allowed)
        self.rl_aux_eval_usage_ratio.append(float(np.clip(usage_ratio, 0.0, 1.0)))
        self.rl_operator_gate_skips.append(self._last_operator_gate_skip)

        # Keep controller feedback aligned to what was actually executed.
        if self._last_action is not None and self._last_action.operator != self._last_effective_operator:
            self._last_action = ContinuousAction(
                w=float(self._last_action.w),
                c1=float(self._last_action.c1),
                c2=float(self._last_action.c2),
                repulsion_weight=float(self._last_action.repulsion_weight),
                operator=int(self._last_effective_operator),
                action_idx=int(self._last_action.action_idx),
            )

        filtered = 0.0
        if hasattr(self.engine, "last_operator_filtered"):
            filtered_map = getattr(self.engine, "last_operator_filtered", {})
            if self._last_effective_operator == 1:
                filtered = float(filtered_map.get("sbx", 0))
            elif self._last_effective_operator == 2:
                filtered = float(filtered_map.get("de", 0))
            elif self._last_effective_operator == 3:
                filtered = float(filtered_map.get("elite", 0))
        self.rl_prefilter_filtered_total += max(0.0, filtered)

    def post_step(
        self,
        hv_before: float,
        hv_after: float,
        feasible_before: float,
        feasible_after: float,
        diversity_before: float,
        diversity_after: float,
    ) -> None:
        """Compute reward and update controller."""
        features = self._last_features
        action = self._last_action

        # Stagnation tracking
        hv_slope = hv_after - self.last_hv
        if hv_slope <= 1e-8:
            self.stagnation += 1
        else:
            self.stagnation = 0
        self.last_hv = hv_after

        if features is None or action is None:
            return

        # Simplified reward: 3 terms, no cost/conflict penalties
        delta_hv = float(np.tanh((hv_after - hv_before) / self.hv_scale))
        delta_feas = float(np.clip(feasible_after - feasible_before, -1.0, 1.0))
        delta_div = float(np.tanh((diversity_after - diversity_before) / self.div_scale))

        aux_usage_ratio = self._last_aux_eval_used / max(1.0, self._last_aux_budget_allowed)
        aux_cost_penalty = float(self.reward_aux_cost_w * np.clip(aux_usage_ratio, 0.0, 1.0))
        
        # We only penalize multi-UAV conflicts here.
        # Terrain/static obstacle collisions ('collisionViolation') are already handled by 'delta_feas'.
        # Double-penalizing terrain crashes destroys UAV1 performance.
        conflict_after = float(np.mean([
            float(getattr(c, "details", {}).get("conflictRate", 0.0))
            for c in self.engine.candidates
        ])) if getattr(self.engine, "candidates", None) else 0.0
        
        # Heavy penalty for unresolved multi-UAV collisions
        conflict_penalty = float(np.clip(conflict_after * 5.0, 0.0, 1.5))

        reward = float(np.clip(
            self.reward_hv_w * delta_hv
            + self.reward_feas_w * delta_feas
            + self.reward_div_w * delta_div
            - aux_cost_penalty
            - conflict_penalty,
            -1.0, 1.0,
        ))

        self.rl_rewards.append(reward)

        # Update controller (1-step, no n-step returns)
        t0 = time.perf_counter()
        self.controller.update(features, reward, action)
        self.rl_controller_time_sec += float(time.perf_counter() - t0)
        if (
            hasattr(self.controller, "update_attention")
            and self._last_particle_features is not None
            and self._last_archive_features is not None
        ):
            t1 = time.perf_counter()
            self.controller.update_attention(
                particle_features=self._last_particle_features,
                archive_features=self._last_archive_features,
                reward=reward,
                temperature=self.attention_temperature,
            )
            self.rl_controller_time_sec += float(time.perf_counter() - t1)

    def flush_pending(self) -> None:
        """No-op — kept for API compatibility (no n-step buffering)."""
        pass

    def rl_trace(self) -> dict[str, np.ndarray]:
        """Return all RL trace arrays for artifact saving."""
        return {
            "action": np.array(self.rl_actions, dtype=float),
            "reward": np.array(self.rl_rewards, dtype=float),
            "feature": np.array(self.rl_features, dtype=float).reshape(-1, N_FEATURES) if self.rl_features else np.zeros((0, N_FEATURES), dtype=float),
            "operator": np.array(self.rl_operators, dtype=float),
            "effective_operator": np.array(self.rl_effective_operators, dtype=float),
            "aux_budget_allowed": np.array(self.rl_aux_budget_allowed, dtype=float),
            "aux_eval_used": np.array(self.rl_aux_eval_used, dtype=float),
            "aux_eval_usage_ratio": np.array(self.rl_aux_eval_usage_ratio, dtype=float),
            "operator_gate_skip": np.array(self.rl_operator_gate_skips, dtype=float),
            "attention_entropy": np.array(self.rl_attention_entropy, dtype=float),
            "attention_top1_mass": np.array(self.rl_attention_top1_mass, dtype=float),
            "attention_invalid_rows": np.array(self.rl_attention_invalid_rows, dtype=float),
        }

    def rl_metadata(self) -> dict[str, Any]:
        """Return all RL metadata for artifact saving."""
        return {
            "rlSbxInjected": float(self.rl_sbx_injected),
            "rlDeInjected": float(self.rl_de_injected),
            "rlEliteTrials": float(self.rl_elite_trials),
            "rlOperatorGateSkips": float(self.rl_operator_gate_skip_count),
            "rlAuxEvalUsedTotal": float(self.rl_aux_eval_used_total),
            "rlAuxBudgetTotal": float(self.rl_aux_budget_total),
            "rlAuxEvalUsageMean": float(np.mean(np.asarray(self.rl_aux_eval_usage_ratio, dtype=float))) if self.rl_aux_eval_usage_ratio else 0.0,
            "rlAuxBudgetStartFactor": float(self.aux_eval_budget_start_factor),
            "rlAuxBudgetEndFactor": float(self.aux_eval_budget_end_factor),
            "rlOperatorTriggerProbStart": float(self.operator_trigger_prob_start),
            "rlOperatorTriggerProbEnd": float(self.operator_trigger_prob_end),
            "rlOperatorStagnationBoost": float(self.operator_stagnation_boost),
            "rlOperatorStagnationThreshold": float(self.operator_stagnation_threshold),
            "rlRewardAuxCostWeight": float(self.reward_aux_cost_w),
            "rlSurrogatePrefilterEnabled": float(1.0 if self.surrogate_prefilter_enabled else 0.0),
            "rlSurrogatePrefilterRatio": float(self.surrogate_prefilter_ratio),
            "rlSurrogatePrefilterMinCandidates": float(self.surrogate_prefilter_min_candidates),
            "rlSurrogatePrefilterK": float(self.surrogate_prefilter_k),
            "rlPrefilterFilteredTotal": float(self.rl_prefilter_filtered_total),
            "rlControllerTimeSec": float(self.rl_controller_time_sec),
            "rlControllerSummary": self.controller.summary() if hasattr(self.controller, "summary") else {},
        }
