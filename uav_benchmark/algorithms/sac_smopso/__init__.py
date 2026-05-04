from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.sac_smopso.actions import _map_continuous_action
from uav_benchmark.algorithms.sac_smopso.constants import (
    _ACTION_KEYS,
    _CANDIDATE_TOKEN_DIM,
    _ENVIRONMENT_TOKEN_DIM,
    _GLOBAL_STATE_DIM,
    _INTERACTION_TOKEN_DIM,
    _OBJECTIVE_COUNT,
    _OPERATOR_NAMES,
    _TEMPORAL_WINDOW,
    _TOPOLOGY_TOKEN_DIM,
)
from uav_benchmark.algorithms.sac_smopso.controller import (
    ControllerConfig,
    HybridSACController,
    TemporalRelationalStateSpec,
)
from uav_benchmark.algorithms.sac_smopso.geometry import _targeted_geometry_repair
from uav_benchmark.algorithms.sac_smopso.initialization import (
    _build_navigation_bounds,
    _model_constraint_values,
    _report_ready_candidate,
    _search_ready_candidate,
    _structured_initial_population,
)
from uav_benchmark.algorithms.sac_smopso.reservoir import (
    _refresh_unconstrained_population,
    _reservoir_sbx_injection,
)
from uav_benchmark.algorithms.sac_smopso.scoring import _reservoir_score
from uav_benchmark.algorithms.sac_smopso.state import (
    _archive_snapshot,
    _build_temporal_relational_state,
    _problem_descriptors,
    _resolve_policy_mode,
    _resolve_state_representation,
)
from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _ensure_fleet_endpoints,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
)
from uav_benchmark.algorithms.shared.nmopso_engine import NMOPSOEngine, _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.utils.gpu import resolve_gpu
from uav_benchmark.utils.random import seed_everything

__all__ = [
    "_archive_snapshot",
    "_build_navigation_bounds",
    "_build_temporal_relational_state",
    "_compute_reward",
    "_map_continuous_action",
    "_model_constraint_values",
    "_problem_descriptors",
    "_refresh_unconstrained_population",
    "_reservoir_sbx_injection",
    "_reservoir_score",
    "_resolve_policy_mode",
    "_resolve_state_representation",
    "_search_ready_candidate",
    "_structured_initial_population",
    "_targeted_geometry_repair",
    "run_sac_smopso",
]


def _compute_reward(
    before: dict[str, float],
    after: dict[str, float],
    operator_stats: dict[str, float],
    population: int,
) -> float:
    hv_delta = np.tanh(8.0 * (float(after["hv"]) - float(before["hv"])) / max(1e-6, abs(float(before["hv"])) + 1e-3))
    feasible_delta = float(after["feasible_ratio"]) - float(before["feasible_ratio"])
    diversity_delta = np.tanh(
        (float(after["diversity"]) - float(before["diversity"])) / max(1e-6, float(before["diversity"]) + 1.0)
    )
    conflict_delta = float(before["conflict_rate"]) - float(after["conflict_rate"])
    violation_delta = float(before["mean_violation"]) - float(after["mean_violation"])
    geometry_delta = np.tanh(
        (float(before.get("best_geometry", 100.0)) - float(after.get("best_geometry", 100.0)))
        / max(1.0, abs(float(before.get("best_geometry", 100.0))))
    )
    occupancy_delta = 0.5 * (
        float(after["objective_occupancy"] - before["objective_occupancy"])
        + float(after["spatial_occupancy"] - before["spatial_occupancy"])
    )
    effect_ratio = float(operator_stats.get("effectCount", 0.0)) / max(1.0, float(population))
    eval_penalty = max(0.0, float(operator_stats["evalCount"]) - float(population)) / max(1.0, float(population))
    reward = (
        2.6 * hv_delta
        + 1.4 * feasible_delta
        + 0.7 * diversity_delta
        + 0.8 * conflict_delta
        + 0.9 * violation_delta
        + 1.0 * geometry_delta
        + 0.4 * occupancy_delta
        + 0.2 * effect_ratio * max(0.0, geometry_delta + violation_delta)
        - 0.2 * eval_penalty
    )
    return float(np.clip(reward, -5.0, 5.0))


def _run_fleet_sac_smopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 42
    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + 3 * requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)

    extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    representation = str(extra.get("sacRepresentation", extra.get("representation", "cart"))).strip().lower() or "cart"
    if representation not in {"cart", "sc"}:
        representation = "cart"
    if representation == "cart":
        lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    else:
        max_angle_rad = float(np.deg2rad(params.max_turn_deg))
        lower, upper = _build_navigation_bounds(
            model=model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            max_angle_rad=max_angle_rad,
        )
    archive_size = int(extra.get("nRep", max(params.population * 4, 24)))
    metric_interval = int(extra.get("metricInterval", 20))
    separation_min = float(model.get("separationMin", params.separation_min))
    drone_size = float(model.get("droneSize", params.drone_size))
    state_representation, encoder_mode = _resolve_state_representation(extra)
    policy_mode = _resolve_policy_mode(extra)
    checkpoint_path_raw = extra.get("sacCheckpointPath", extra.get("sac_checkpoint_path"))
    checkpoint_path = Path(checkpoint_path_raw).expanduser() if checkpoint_path_raw else None
    save_checkpoint = bool(extra.get("sacSaveCheckpoint", checkpoint_path is not None))
    deterministic_policy = bool(extra.get("sacDeterministicPolicy", policy_mode == "frozen"))
    controller_config = ControllerConfig(
        hidden_dim=int(extra.get("sacHiddenDim", 128)),
        lr=float(extra.get("sacLr", 3e-4)),
        gamma=float(extra.get("sacGamma", 0.98)),
        tau=float(extra.get("sacTau", 0.01)),
        replay_capacity=int(extra.get("sacReplayCapacity", max(512, params.generations * params.population))),
        batch_size=int(extra.get("sacBatchSize", min(32, max(8, params.population)))),
        warmup_steps=int(extra.get("sacWarmupSteps", max(6, params.population))),
        updates_per_step=int(extra.get("sacUpdatesPerStep", 1)),
        alpha_init=float(extra.get("sacAlpha", 0.08)),
        scratch_policy_mix_start=float(extra.get("sacScratchPolicyMixStart", 0.35)),
        scratch_policy_mix_end=float(extra.get("sacScratchPolicyMixEnd", 0.90)),
        loaded_policy_mix_start=float(extra.get("sacLoadedPolicyMixStart", 0.60)),
        loaded_policy_mix_end=float(extra.get("sacLoadedPolicyMixEnd", 1.00)),
        policy_mix_anneal_steps=int(extra.get("sacPolicyMixAnnealSteps", 200)),
        use_operator_head=False,
    )

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    def _evaluate(vectors: np.ndarray) -> list[Candidate]:
        raw_candidates = _evaluate_population(
            vectors,
            model=model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            representation=representation,
        )
        return [
            _search_ready_candidate(
                candidate,
                separation_min=separation_min,
                drone_size=drone_size,
                max_turn_deg=float(params.max_turn_deg),
            )
            for candidate in raw_candidates
        ]

    gpu_info = resolve_gpu(params.gpu_mode)
    policy_device = "cpu"
    if gpu_info.enabled and gpu_info.backend == "torch":
        policy_device = "cuda:0" if "cuda" in gpu_info.device else ("mps" if "mps" in gpu_info.device else "cpu")
    problem_descriptor = _problem_descriptors(model, fleet_size=fleet_size, max_turn_deg=float(params.max_turn_deg))

    run_indices = _resolve_run_indices(params)
    resume_existing_runs = bool(extra.get("resumeExistingRuns", True))
    for run_idx in run_indices:
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=_OBJECTIVE_COUNT,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue

        run_start = time.perf_counter()
        # Deterministic seeding for reproducible NMOPSO/SAC-SMOPSO runs.
        # Previously np.random/torch state was implicit — the same
        # params.seed gave different outputs depending on caller state.
        seed_everything(seed_value + run_idx)
        initial_population = _structured_initial_population(
            model=model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            pop_size=params.population,
            lower=lower,
            upper=upper,
            separation_min=separation_min,
            representation=representation,
        )
        engine = NMOPSOEngine(
            model=model,
            pop_size=params.population,
            lower=lower,
            upper=upper,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            representation=representation,
            objective_count=_OBJECTIVE_COUNT,
            archive_size=archive_size,
            # Use the strict Pareto archive (as NMOPSO does) on fleet=1 so
            # infeasible/dominated candidates cannot creep into the final
            # solution set. R2 indicator archive stays on for multi-UAV
            # where it genuinely helps diversity under constraint pressure.
            use_r2_archive=(fleet_size > 1),
            paper_nmopso=True,
            is_nmopso_family=True,
            grid_cells=int(extra.get("nGrid", 10)),
            grid_kappa=float(extra.get("kappa", 1.0)),
            use_grid_leader=True,
            velocity_clamp_ratio=float(extra.get("velocityClampRatio", 0.5)),
            gpu_mode=params.gpu_mode,
            feature_mode=str(extra.get("sacFeatureMode", "path")),
            evaluate_fn=_evaluate,
            initial_population=initial_population,
        )
        engine.reset()

        state_spec = TemporalRelationalStateSpec(
            global_dim=_GLOBAL_STATE_DIM,
            population_dim=_CANDIDATE_TOKEN_DIM,
            archive_dim=_CANDIDATE_TOKEN_DIM,
            topology_dim=_TOPOLOGY_TOKEN_DIM,
            interaction_dim=_INTERACTION_TOKEN_DIM,
            environment_dim=_ENVIRONMENT_TOKEN_DIM,
            temporal_dim=_GLOBAL_STATE_DIM,
        )
        controller = HybridSACController(
            state_spec=state_spec,
            action_dim=len(_ACTION_KEYS),
            operator_names=_OPERATOR_NAMES,
            device_tag=policy_device,
            config=controller_config,
            policy_mode=policy_mode,
            encoder_mode=encoder_mode,
        )
        if not controller.enabled and (policy_mode != "online" or checkpoint_path is not None or save_checkpoint):
            raise RuntimeError(
                "SAC-SMOPSO checkpointed policy modes require PyTorch. Install torch to use pretraining, finetuning, or frozen-policy evaluation."
            )
        checkpoint_metadata: dict[str, Any] = {}
        if checkpoint_path is not None and checkpoint_path.exists() and controller.enabled:
            checkpoint_metadata = controller.load_checkpoint(
                checkpoint_path,
                load_optimizers=policy_mode != "frozen",
            )
        controller_time_sec = 0.0
        global_history: deque[np.ndarray] = deque(maxlen=_TEMPORAL_WINDOW - 1)
        trace_state: list[np.ndarray] = []
        trace_state_global: list[np.ndarray] = []
        trace_action: list[np.ndarray] = []
        trace_reward: list[float] = []
        trace_hv: list[float] = []
        trace_feasible: list[float] = []
        trace_conflict: list[float] = []
        trace_source: list[float] = []
        trace_archive_fill: list[float] = []
        trace_population_summary: list[np.ndarray] = []
        trace_archive_summary: list[np.ndarray] = []
        trace_topology_summary: list[np.ndarray] = []
        trace_interaction_summary: list[np.ndarray] = []
        trace_environment_summary: list[np.ndarray] = []
        trace_sbx_weight: list[float] = []

        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )
        snapshot = _archive_snapshot(engine)
        last_hv = float(snapshot["hv"])
        diversity_ref = max(1.0, float(snapshot["diversity"]))
        stagnation = 0
        aux_noise = np.random.normal(0.0, 0.04, size=initial_population.shape) * np.maximum(upper - lower, 1e-9)
        aux_seed_vectors = np.clip(initial_population + aux_noise, lower, upper)
        aux_seed_candidates = _evaluate(aux_seed_vectors)
        auxiliary_population: dict[str, Any] = {
            "vectors": np.asarray(aux_seed_vectors, dtype=float).copy(),
            "candidates": list(aux_seed_candidates),
        }
        _refresh_unconstrained_population(
            auxiliary_population,
            fresh_vectors=engine.population,
            fresh_candidates=list(engine.candidates),
            capacity=int(params.population),
            model=model,
        )

        for generation in range(1, params.generations + 1):
            before = _archive_snapshot(engine)
            feasible_elites = [
                candidate
                for candidate in list(engine.archive_candidates)
                if float(getattr(candidate, "details", {}).get("feasible", 0.0)) > 0.5
            ]
            state = _build_temporal_relational_state(
                engine=engine,
                model=model,
                snapshot=before,
                generation=generation - 1,
                total_generations=params.generations,
                last_hv=last_hv,
                stagnation=stagnation,
                diversity_ref=diversity_ref,
                problem_descriptor=problem_descriptor,
                global_history=global_history,
            )

            policy_start = time.perf_counter()
            action = controller.act(state, deterministic=deterministic_policy)
            controller_time_sec += float(time.perf_counter() - policy_start)
            action_params = _map_continuous_action(action.continuous)

            # Learned hybrid schedule (novel).
            # ``sbx_weight`` now controls *how much* CMOSMA-style reservoir
            # recombination is injected on top of the base NMOPSO step, rather
            # than hard-switching the whole generation at a threshold. This
            # keeps PSO exploitation alive every generation while still letting
            # the controller scale up SBX pressure smoothly on the hard cases.
            #
            # ``SAC_SMOPSO_FORCE_SBX`` (optional): pin the SBX weight to a
            # constant in [0,1] for ablation / debugging.
            # ``SAC_SMOPSO_FORCE_REPAIR_INTENSITY`` similarly pins both the
            # in-SBX conflict-repair pass and the targeted geometry repair.
            sbx_weight_value = float(action_params.get("sbx_weight", 0.0))
            _force_sbx = os.environ.get("SAC_SMOPSO_FORCE_SBX")
            if _force_sbx is not None:
                try:
                    sbx_weight_value = float(np.clip(float(_force_sbx), 0.0, 1.0))
                    action_params["sbx_weight"] = sbx_weight_value
                except ValueError:
                    pass
            repair_intensity_value = float(action_params.get("repair_intensity", 0.0))
            _force_repair = os.environ.get("SAC_SMOPSO_FORCE_REPAIR_INTENSITY")
            if _force_repair is not None:
                try:
                    repair_intensity_value = float(np.clip(float(_force_repair), 0.0, 1.0))
                    action_params["repair_intensity"] = repair_intensity_value
                except ValueError:
                    pass

            operator_stats = {"effectCount": 0.0, "evalCount": float(params.population)}
            engine.step(
                inertia=action_params["inertia"],
                c1=action_params["c1"],
                c2=action_params["c2"],
                velocity_scale=action_params["velocity_scale"],
                kappa_scale=action_params["kappa_scale"],
                delta_scale=action_params["delta_scale"],
                region_scale=max(0.25, float(action_params.get("archive_focus", 0.0))),
                leader_bias=action_params["leader_bias"],
                mutation_prob=action_params["mutation_prob"],
                repulsion_weight=action_params["repulsion_weight"],
            )
            _refresh_unconstrained_population(
                auxiliary_population,
                fresh_vectors=engine.population,
                fresh_candidates=list(engine.candidates),
                capacity=int(params.population),
                model=model,
            )

            if sbx_weight_value > 1e-3:
                # The controller schedules crossover pressure between the main
                # constrained swarm and the unconstrained companion population.
                operator_stats = _reservoir_sbx_injection(
                    engine=engine,
                    aux_state=auxiliary_population,
                    sbx_weight=sbx_weight_value,
                    repair_intensity=repair_intensity_value,
                    fleet_size=fleet_size,
                    n_waypoints=n_waypoints,
                    lower=lower,
                    upper=upper,
                    aux_capacity=int(params.population),
                )
                operator_stats["evalCount"] = float(operator_stats.get("evalCount", 0.0)) + float(params.population)
                # Damp stale PSO momentum in proportion to the SBX pressure so
                # strong recombination phases do not get undone immediately.
                engine.velocity *= max(0.10, 1.0 - 0.85 * sbx_weight_value)
            repair_stats = _targeted_geometry_repair(
                engine,
                model=model,
                representation=representation,
                fleet_size=fleet_size,
                n_waypoints=n_waypoints,
                lower=lower,
                upper=upper,
                repair_intensity=repair_intensity_value,
                aux_candidates=list(auxiliary_population.get("candidates", [])),
            )
            operator_stats["effectCount"] = float(operator_stats.get("effectCount", 0.0)) + float(
                repair_stats.get("effectCount", 0.0)
            )
            operator_stats["evalCount"] = float(operator_stats.get("evalCount", 0.0)) + float(
                repair_stats.get("evalCount", 0.0)
            )

            # Preserve any feasible elites discovered so far so that aggressive
            # operators cannot evict them from the archive. The engine's own
            # R2 indicator archive (used for fleet > 1) handles the rest.
            if feasible_elites:
                engine.update_archive(feasible_elites)

            after = _archive_snapshot(engine)
            reward = _compute_reward(
                before=before,
                after=after,
                operator_stats=operator_stats,
                population=params.population,
            )

            improved = (
                after["hv"] > before["hv"] + 1e-6
                or after["mean_violation"] < before["mean_violation"] - 1e-6
                or after["feasible_ratio"] > before["feasible_ratio"] + 1e-6
            )
            stagnation = 0 if improved else stagnation + 1
            diversity_ref = max(diversity_ref, float(after["diversity"]), 1.0)

            next_state = _build_temporal_relational_state(
                engine=engine,
                model=model,
                snapshot=after,
                generation=generation,
                total_generations=params.generations,
                last_hv=float(after["hv"]),
                stagnation=stagnation,
                diversity_ref=diversity_ref,
                problem_descriptor=problem_descriptor,
                global_history=global_history,
            )
            policy_start = time.perf_counter()
            if controller.training_enabled():
                controller.observe(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=generation == params.generations,
                )
                controller_time_sec += float(time.perf_counter() - policy_start)
            last_hv = float(after["hv"])
            global_history.append(np.asarray(next_state.global_features, dtype=float).copy())

            trace_state.append(state.summary_vector())
            trace_state_global.append(np.asarray(state.global_features, dtype=float).copy())
            trace_action.append(np.asarray(action.continuous, dtype=float).copy())
            trace_reward.append(float(reward))
            trace_hv.append(float(after["hv"]))
            trace_feasible.append(float(after["feasible_ratio"]))
            trace_conflict.append(float(after["conflict_rate"]))
            trace_source.append(1.0 if action.source == "sac-mixed" else 0.0)
            trace_archive_fill.append(float(after["archive_fill"]))
            trace_sbx_weight.append(float(sbx_weight_value))
            trace_population_summary.append(
                np.mean(state.population_tokens[state.population_mask > 0.5], axis=0)
                if np.any(state.population_mask > 0.5)
                else np.zeros(_CANDIDATE_TOKEN_DIM, dtype=float)
            )
            trace_archive_summary.append(
                np.mean(state.archive_tokens[state.archive_mask > 0.5], axis=0)
                if np.any(state.archive_mask > 0.5)
                else np.zeros(_CANDIDATE_TOKEN_DIM, dtype=float)
            )
            trace_topology_summary.append(
                np.mean(state.topology_tokens[state.topology_mask > 0.5], axis=0)
                if np.any(state.topology_mask > 0.5)
                else np.zeros(_TOPOLOGY_TOKEN_DIM, dtype=float)
            )
            trace_interaction_summary.append(
                np.mean(state.interaction_tokens[state.interaction_mask > 0.5], axis=0)
                if np.any(state.interaction_mask > 0.5)
                else np.zeros(_INTERACTION_TOKEN_DIM, dtype=float)
            )
            trace_environment_summary.append(
                np.mean(state.environment_tokens[state.environment_mask > 0.5], axis=0)
                if np.any(state.environment_mask > 0.5)
                else np.zeros(_ENVIRONMENT_TOKEN_DIM, dtype=float)
            )

            if params.compute_metrics:
                report_candidates = [
                    _report_ready_candidate(candidate)
                    for candidate in (
                        list(engine.archive_candidates) if engine.archive_candidates else list(engine.candidates)
                    )
                ]
                report_matrix = _candidate_matrix(report_candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    if report_matrix.size > 0:
                        hv_history[generation - 1, 0] = cal_metric(
                            1, report_matrix, params.problem_index, _OBJECTIVE_COUNT
                        )
                        hv_history[generation - 1, 1] = cal_metric(
                            2, report_matrix, params.problem_index, _OBJECTIVE_COUNT
                        )
                    else:
                        hv_history[generation - 1] = 0.0
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})

        final_candidates = [
            _report_ready_candidate(candidate)
            for candidate in (list(engine.archive_candidates) if engine.archive_candidates else list(engine.candidates))
        ]
        rl_trace = {
            "state": np.asarray(trace_state, dtype=float),
            "state_global": np.asarray(trace_state_global, dtype=float),
            "population_summary": np.asarray(trace_population_summary, dtype=float),
            "archive_summary": np.asarray(trace_archive_summary, dtype=float),
            "topology_summary": np.asarray(trace_topology_summary, dtype=float),
            "interaction_summary": np.asarray(trace_interaction_summary, dtype=float),
            "environment_summary": np.asarray(trace_environment_summary, dtype=float),
            "action": np.asarray(trace_action, dtype=float),
            "sbx_weight": np.asarray(trace_sbx_weight, dtype=float),
            "reward": np.asarray(trace_reward, dtype=float),
            "hv": np.asarray(trace_hv, dtype=float),
            "feasible": np.asarray(trace_feasible, dtype=float),
            "conflict": np.asarray(trace_conflict, dtype=float),
            "policy_source": np.asarray(trace_source, dtype=float),
            "archive_fill": np.asarray(trace_archive_fill, dtype=float),
        }
        rl_metadata = controller.metadata()
        rl_metadata["torchEnabled"] = 1.0 if controller.enabled else 0.0
        rl_metadata["stateRepresentation"] = state_representation
        rl_metadata["stateEncoderMode"] = encoder_mode
        rl_metadata["stateHasRelationalTokens"] = 0.0 if encoder_mode == "flat" else 1.0
        rl_metadata["policyDeterministic"] = 1.0 if deterministic_policy else 0.0
        rl_metadata["checkpointPath"] = str(checkpoint_path) if checkpoint_path is not None else ""
        for key, value in checkpoint_metadata.items():
            rl_metadata[f"checkpoint_{key}"] = value

        if checkpoint_path is not None and save_checkpoint and controller.enabled and controller.training_enabled():
            controller.save_checkpoint(
                checkpoint_path,
                extra_metadata={
                    "stateRepresentation": state_representation,
                    "stateEncoderMode": encoder_mode,
                    "policyMode": policy_mode,
                    "problemName": str(params.problem_name),
                    "runIndex": int(run_idx),
                    "fleetSize": int(fleet_size),
                },
            )

        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=_OBJECTIVE_COUNT,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend=engine.gpu_backend,
            gpu_peak_bytes=float(engine.gpu_peak_bytes),
            rl_trace=rl_trace,
            gpu_update_time_sec=float(engine.gpu_update_time_sec),
            rl_controller_time_sec=float(controller_time_sec),
            rl_policy_backend=controller.device_tag,
            rl_policy_gpu_peak_bytes=float(controller.gpu_peak_bytes()),
            rl_policy_loss_ema=float(controller.loss_ema),
            rl_metadata=rl_metadata,
            run_metadata={
                "algorithmName": "SAC-SMOPSO",
                "representation": representation,
                "archiveMode": "r2",
                "stateRepresentation": state_representation,
                "stateEncoderMode": encoder_mode,
                "policyMode": policy_mode,
                "policyDeterministic": int(deterministic_policy),
                "controllerWarmupSteps": int(controller_config.warmup_steps),
            },
        )

        if params.compute_metrics:
            final_matrix = _candidate_matrix(final_candidates)
            if final_matrix.size > 0:
                run_scores[run_idx - 1] = np.array(
                    [
                        cal_metric(1, final_matrix, params.problem_index, _OBJECTIVE_COUNT),
                        cal_metric(2, final_matrix, params.problem_index, _OBJECTIVE_COUNT),
                    ],
                    dtype=float,
                )
            else:
                run_scores[run_idx - 1] = 0.0

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_sac_smopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_sac_smopso(model=model, params=params)
