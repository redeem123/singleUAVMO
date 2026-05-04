from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.adaptive_control import (
    AdaptiveControllerConfig,
    AdaptiveSACController,
    adaptive_state_spec,
    build_adaptive_state,
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
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.smpso import _clone_candidate, _smpso_operator, _update_gbest, _update_pbest
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import tournament_selection
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.utils.gpu import resolve_gpu

_OBJECTIVE_COUNT = 4
_ACTION_KEYS = (
    "inertia_like",
    "leader_pressure",
    "mutation_prob",
    "distribution_index",
    "jitter_sigma",
)
_ACTION_LOWER = np.array([0.1, 0.0, 0.02, 8.0, 0.0], dtype=float)
_ACTION_UPPER = np.array([0.9, 1.0, 0.35, 40.0, 0.18], dtype=float)
_OPERATOR_NAMES = ("base", "sbx", "de", "elite", "spread")


def _map_action(raw: np.ndarray) -> dict[str, float]:
    scaled = 0.5 * (np.clip(np.asarray(raw, dtype=float).reshape(-1)[: len(_ACTION_KEYS)], -1.0, 1.0) + 1.0)
    values = _ACTION_LOWER + scaled * (_ACTION_UPPER - _ACTION_LOWER)
    return {key: float(values[index]) for index, key in enumerate(_ACTION_KEYS)}


def _snapshot(
    candidates: list[Any], archive: list[Any], archive_size: int, last_hv: float, stagnation: int
) -> dict[str, float]:
    matrix = _candidate_matrix(candidates)
    finite = matrix[np.all(np.isfinite(matrix), axis=1)] if matrix.size > 0 else matrix
    hv = cal_metric(1, finite, 0, _OBJECTIVE_COUNT) if finite.size > 0 else 0.0
    diversity = float(np.mean(np.std(finite, axis=0))) if finite.size > 0 else 0.0
    feasible_ratio = 0.0
    conflict_rate = 0.0
    mean_violation = 0.0
    if candidates:
        feasible_ratio = float(np.mean([float(getattr(c, "details", {}).get("feasible", 0.0)) for c in candidates]))
        conflict_rate = float(np.mean([float(getattr(c, "details", {}).get("conflictRate", 0.0)) for c in candidates]))
        mean_violation = float(
            np.mean(
                [
                    float(getattr(c, "details", {}).get("turnViolation", 0.0))
                    + float(getattr(c, "details", {}).get("separationViolation", 0.0))
                    + float(getattr(c, "details", {}).get("collisionViolation", 0.0))
                    for c in candidates
                ]
            )
        )
    return {
        "hv": float(hv),
        "hv_trend": float(np.clip(0.5 + 0.5 * np.tanh((hv - last_hv) / max(1e-6, abs(last_hv) + 1e-3)), 0.0, 1.0)),
        "diversity": float(np.clip(diversity, 0.0, 1.0)),
        "archive_fill": float(np.clip(len(archive) / max(1, archive_size), 0.0, 1.0)),
        "objective_occupancy": float(np.clip(len(archive) / max(1, archive_size), 0.0, 1.0)),
        "spatial_occupancy": float(np.clip(len(archive) / max(1, archive_size), 0.0, 1.0)),
        "feasible_archive": feasible_ratio,
        "mean_violation": float(np.clip(mean_violation, 0.0, 1.0)),
        "feasible_ratio": float(np.clip(feasible_ratio, 0.0, 1.0)),
        "conflict_rate": float(np.clip(conflict_rate, 0.0, 1.0)),
        "quality_signal": float(np.clip(hv, 0.0, 1.0)),
        "stagnation": float(np.clip(stagnation / 10.0, 0.0, 1.0)),
    }


def _algorithm_features(crowd_dis: np.ndarray, archive: list[Any], params: dict[str, float]) -> np.ndarray:
    crowd = np.asarray(crowd_dis, dtype=float).reshape(-1)
    return np.asarray(
        [
            float(np.clip(np.mean(crowd) if crowd.size > 0 else 0.0, 0.0, 1.0)),
            float(np.clip(np.std(crowd) if crowd.size > 0 else 0.0, 0.0, 1.0)),
            float(np.clip(len(archive) / 50.0, 0.0, 1.0)),
            float(np.clip(params["mutation_prob"], 0.0, 1.0)),
            float(np.clip(params["leader_pressure"], 0.0, 1.0)),
            float(np.clip(params["jitter_sigma"] / 0.2, 0.0, 1.0)),
        ],
        dtype=float,
    )


def run_ra_smpso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 42
    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dimensions = int(lower.size)
    archive_size = int(params.extra.get("nRep", params.population))
    metric_interval = int(params.extra.get("metricInterval", 20))
    extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    controller_config = AdaptiveControllerConfig(
        hidden_dim=int(extra.get("sacHiddenDim", 96)),
        lr=float(extra.get("sacLr", 3e-4)),
        gamma=float(extra.get("sacGamma", 0.98)),
        tau=float(extra.get("sacTau", 0.01)),
        replay_capacity=int(extra.get("sacReplayCapacity", max(256, params.generations * params.population))),
        batch_size=int(extra.get("sacBatchSize", min(16, max(4, params.population)))),
        warmup_steps=int(extra.get("sacWarmupSteps", max(4, params.population))),
        updates_per_step=int(extra.get("sacUpdatesPerStep", 1)),
        alpha_init=float(extra.get("sacAlpha", 0.08)),
    )
    state_representation = str(extra.get("stateRepresentation", "TRFTS"))
    encoder_mode = "handcrafted" if state_representation == "TRFTS-HAND" else "learned"
    policy_mode = str(extra.get("sacPolicyMode", "online"))
    checkpoint_path_raw = extra.get("sacCheckpointPath")
    checkpoint_path = Path(checkpoint_path_raw).expanduser() if checkpoint_path_raw else None
    save_checkpoint = bool(extra.get("sacSaveCheckpoint", checkpoint_path is not None))
    deterministic_policy = bool(extra.get("sacDeterministicPolicy", policy_mode == "frozen"))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    def _evaluate(vectors: np.ndarray) -> list[Any]:
        return _evaluate_population(
            vectors, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints, representation="cart"
        )

    gpu_info = resolve_gpu(params.gpu_mode)
    policy_device = "cpu"
    if gpu_info.enabled and gpu_info.backend == "torch":
        policy_device = "cuda:0" if "cuda" in gpu_info.device else ("mps" if "mps" in gpu_info.device else "cpu")

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
        population = np.random.uniform(lower, upper, size=(params.population, dimensions))
        velocity = np.zeros_like(population)
        candidates = _evaluate(population)
        pbest_vectors = population.copy()
        pbest_candidates = [_clone_candidate(candidates[idx], vector=population[idx]) for idx in range(len(candidates))]
        archive, crowd_dis = _update_gbest(pbest_candidates, archive_size)
        if not archive:
            archive = [
                _clone_candidate(candidates[idx], vector=population[idx])
                for idx in range(min(len(candidates), archive_size))
            ]
            crowd_dis = np.ones(len(archive), dtype=float)
        controller = AdaptiveSACController(
            state_spec=adaptive_state_spec(),
            action_dim=len(_ACTION_KEYS),
            operator_names=_OPERATOR_NAMES,
            device_tag=policy_device,
            config=controller_config,
            policy_mode=policy_mode,
            encoder_mode=encoder_mode,
        )
        if not controller.enabled and (policy_mode != "online" or checkpoint_path is not None or save_checkpoint):
            raise RuntimeError("RA-SMPSO checkpointed policy modes require PyTorch.")
        checkpoint_metadata: dict[str, Any] = {}
        if checkpoint_path is not None and checkpoint_path.exists() and controller.enabled:
            checkpoint_metadata = controller.load_checkpoint(checkpoint_path, load_optimizers=policy_mode != "frozen")
        history: deque[np.ndarray] = deque(maxlen=5)
        last_hv = 0.0
        stagnation = 0
        current_params = {
            "mutation_prob": 0.15,
            "distribution_index": 20.0,
            "leader_pressure": 0.5,
            "jitter_sigma": 0.0,
        }
        controller_time_sec = 0.0
        trace_state: list[np.ndarray] = []
        trace_action: list[np.ndarray] = []
        trace_operator: list[float] = []
        trace_reward: list[float] = []
        trace_hv: list[float] = []
        trace_feasible: list[float] = []
        trace_conflict: list[float] = []
        trace_source: list[float] = []
        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )
        for generation in range(1, params.generations + 1):
            before = _snapshot(candidates, archive, archive_size, last_hv, stagnation)
            state = build_adaptive_state(
                candidates=candidates,
                archive_candidates=archive,
                model=model,
                generation=generation - 1,
                total_generations=params.generations,
                last_metrics=before,
                algorithm_features=_algorithm_features(crowd_dis, archive, current_params),
                history=history,
                state_representation=state_representation,
            )
            policy_start = time.perf_counter()
            action = controller.act(state, deterministic=deterministic_policy)
            controller_time_sec += float(time.perf_counter() - policy_start)
            current_params = _map_action(action.continuous)
            if archive:
                archive_vectors = np.stack([candidate.vector for candidate in archive], axis=0)
                if archive_vectors.shape[0] == 1:
                    leader_indices = np.zeros(params.population, dtype=int)
                else:
                    if current_params["leader_pressure"] >= 0.5:
                        leader_indices = tournament_selection(2, params.population, -np.asarray(crowd_dis, dtype=float))
                    else:
                        leader_indices = np.random.randint(0, archive_vectors.shape[0], size=params.population)
                leaders = archive_vectors[leader_indices]
            else:
                leaders = pbest_vectors.copy()
            population, velocity = _smpso_operator(
                population=population,
                velocity=velocity * current_params["inertia_like"],
                pbest=pbest_vectors,
                leaders=leaders,
                lower=lower,
                upper=upper,
                mutation_prob=current_params["mutation_prob"],
                dis_m=current_params["distribution_index"],
            )
            operator_name = _OPERATOR_NAMES[int(np.clip(action.operator_id, 0, len(_OPERATOR_NAMES) - 1))]
            if operator_name == "de":
                population = np.clip(
                    population + np.random.normal(0.0, current_params["jitter_sigma"], size=population.shape),
                    lower,
                    upper,
                )
            elif operator_name == "elite" and archive:
                elite_vectors = np.stack([candidate.vector for candidate in archive[: min(3, len(archive))]], axis=0)
                elite_pick = elite_vectors[np.random.randint(0, elite_vectors.shape[0], size=population.shape[0])]
                population = np.clip(0.7 * population + 0.3 * elite_pick, lower, upper)
            elif operator_name == "spread":
                population = np.clip(
                    population + np.random.normal(0.0, current_params["jitter_sigma"] + 0.05, size=population.shape),
                    lower,
                    upper,
                )
            candidates = _evaluate(population)
            archive, crowd_dis = _update_gbest(archive + candidates, archive_size)
            if not archive:
                archive = [
                    _clone_candidate(candidates[idx], vector=population[idx])
                    for idx in range(min(len(candidates), archive_size))
                ]
                crowd_dis = np.ones(len(archive), dtype=float)
            pbest_vectors, pbest_candidates = _update_pbest(
                pbest_vectors=pbest_vectors,
                pbest_candidates=pbest_candidates,
                population=population,
                candidates=candidates,
            )
            after = _snapshot(candidates, archive, archive_size, before["hv"], stagnation)
            reward = float(
                2.0 * (after["hv"] - before["hv"])
                + 1.2 * (after["feasible_ratio"] - before["feasible_ratio"])
                + 0.8 * (before["conflict_rate"] - after["conflict_rate"])
                + 0.8 * (before["mean_violation"] - after["mean_violation"])
            )
            improved = after["hv"] > before["hv"] + 1e-6 or after["feasible_ratio"] > before["feasible_ratio"] + 1e-6
            stagnation = 0 if improved else stagnation + 1
            last_hv = after["hv"]
            next_state = build_adaptive_state(
                candidates=candidates,
                archive_candidates=archive,
                model=model,
                generation=generation,
                total_generations=params.generations,
                last_metrics=after,
                algorithm_features=_algorithm_features(crowd_dis, archive, current_params),
                history=history,
                state_representation=state_representation,
            )
            if controller.training_enabled():
                policy_start = time.perf_counter()
                controller.observe(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=generation == params.generations,
                )
                controller_time_sec += float(time.perf_counter() - policy_start)
            history.append(np.asarray(next_state.global_features, dtype=float).copy())
            trace_state.append(state.summary_vector())
            trace_action.append(np.asarray(action.continuous, dtype=float).copy())
            trace_operator.append(float(action.operator_id))
            trace_reward.append(float(reward))
            trace_hv.append(float(after["hv"]))
            trace_feasible.append(float(after["feasible_ratio"]))
            trace_conflict.append(float(after["conflict_rate"]))
            trace_source.append(1.0 if action.source == "sac-mixed" else 0.0)
            if params.compute_metrics:
                report_matrix = _candidate_matrix(archive if archive else candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, report_matrix, params.problem_index, _OBJECTIVE_COUNT)
                    hv_history[generation - 1, 1] = cal_metric(2, report_matrix, params.problem_index, _OBJECTIVE_COUNT)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]
        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        if checkpoint_path is not None and save_checkpoint and controller.enabled and controller.training_enabled():
            controller.save_checkpoint(
                checkpoint_path,
                extra_metadata={"algorithmName": "RA-SMPSO", "stateRepresentation": state_representation},
            )
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=archive if archive else candidates,
            problem_index=params.problem_index,
            objective_count=_OBJECTIVE_COUNT,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            rl_trace={
                "state": np.asarray(trace_state, dtype=float),
                "action": np.asarray(trace_action, dtype=float),
                "operator": np.asarray(trace_operator, dtype=float),
                "reward": np.asarray(trace_reward, dtype=float),
                "hv": np.asarray(trace_hv, dtype=float),
                "feasible": np.asarray(trace_feasible, dtype=float),
                "conflict": np.asarray(trace_conflict, dtype=float),
                "policy_source": np.asarray(trace_source, dtype=float),
            },
            rl_controller_time_sec=float(controller_time_sec),
            rl_policy_backend=controller.device_tag,
            rl_policy_gpu_peak_bytes=float(controller.gpu_peak_bytes()),
            rl_policy_loss_ema=float(controller.loss_ema),
            rl_metadata={
                **controller.metadata(),
                "stateRepresentation": state_representation,
                "stateHasRelationalTokens": 0.0 if state_representation == "flat" else 1.0,
                **{f"checkpoint_{key}": value for key, value in checkpoint_metadata.items()},
            },
            run_metadata={
                "algorithmName": "RA-SMPSO",
                "stateRepresentation": state_representation,
                "stateEncoderMode": encoder_mode,
                "policyMode": policy_mode,
            },
        )
        if params.compute_metrics:
            final_matrix = _candidate_matrix(archive if archive else candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_matrix, params.problem_index, _OBJECTIVE_COUNT),
                    cal_metric(2, final_matrix, params.problem_index, _OBJECTIVE_COUNT),
                ],
                dtype=float,
            )
    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores
