from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir

LOGGER = logging.getLogger(__name__)


def _save_fleet_artifacts(
    run_dir: Path,
    final_candidates: list[Candidate],
    problem_index: int,
    objective_count: int,
    runtime_sec: float,
    gpu_backend: str,
    gpu_peak_bytes: float,
    rl_trace: dict[str, np.ndarray] | None = None,
    gpu_update_time_sec: float = 0.0,
    rl_controller_time_sec: float = 0.0,
    rl_policy_backend: str = "none",
    rl_policy_gpu_peak_bytes: float = 0.0,
    rl_policy_loss_ema: float = 0.0,
    rl_metadata: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
    from uav_benchmark.io.results import save_run_summary_json

    ensure_dir(run_dir)

    # Final objective matrix
    final_matrix = _candidate_matrix(final_candidates)
    save_run_popobj(run_dir / "final_popobj.mat", final_matrix, problem_index, objective_count)

    # Run stats
    run_stats: dict[str, Any] = {
        "runtimeSec": float(runtime_sec),
        "gpuBackend": str(gpu_backend),
        "gpuPeakBytes": float(gpu_peak_bytes),
        "gpuUpdateTimeSec": float(gpu_update_time_sec),
        "rlControllerTimeSec": float(rl_controller_time_sec),
        "rlPolicyBackend": str(rl_policy_backend),
        "rlPolicyGpuPeakBytes": float(rl_policy_gpu_peak_bytes),
        "rlPolicyLossEma": float(rl_policy_loss_ema),
    }
    if run_metadata:
        for key, value in run_metadata.items():
            run_stats[str(key)] = value
    if run_dir.parent.name:
        run_stats.setdefault("problemName", run_dir.parent.name)
    algorithm_name = run_dir.parent.parent.name if run_dir.parent.parent != run_dir.parent else ""
    if algorithm_name:
        run_stats.setdefault("algorithm", str(run_stats.get("algorithmName", algorithm_name)))
    save_mat(run_dir / "run_stats.mat", run_stats)

    # RL trace
    if rl_trace is not None:
        for key, arr in rl_trace.items():
            save_mat(run_dir / f"rl_{key}.mat", {f"rl_{key}": np.asarray(arr, dtype=float)})
    if rl_metadata is not None:
        save_mat(run_dir / "rl_metadata.mat", rl_metadata)

    # Mission stats (fleet paths, conflict logs)
    fleet_paths: list[list[np.ndarray]] = []
    conflict_values: list[float] = []
    feasible_values: list[float] = []
    separation_values: list[float] = []
    makespan_values: list[float] = []
    energy_values: list[float] = []
    risk_values: list[float] = []
    max_turn_values: list[float] = []
    turn_violation_values: list[float] = []
    separation_violation_values: list[float] = []
    collision_violation_values: list[float] = []
    min_clearance_values: list[float] = []
    conflict_log_rows: list[np.ndarray] = []
    conflict_log_candidate_index: list[np.ndarray] = []
    for candidate_index, c in enumerate(final_candidates, start=1):
        paths = c.details.get("paths", [])
        fleet_paths.append(paths)
        details = c.details if isinstance(c.details, dict) else {}
        conflict_values.append(float(details.get("conflictRate", np.nan)))
        feasible_values.append(float(details.get("feasible", np.nan)))
        separation_values.append(float(details.get("minSeparation", np.nan)))
        makespan_values.append(float(details.get("makespan", np.nan)))
        energy_values.append(float(details.get("energy", np.nan)))
        risk_values.append(float(details.get("risk", np.nan)))
        max_turn_values.append(float(details.get("maxTurnDeg", np.nan)))
        turn_violation_values.append(float(details.get("turnViolation", np.nan)))
        separation_violation_values.append(float(details.get("separationViolation", np.nan)))
        collision_violation_values.append(float(details.get("collisionViolation", np.nan)))
        min_clearance_values.append(float(details.get("minClearance", np.nan)))
        conflict_log = np.asarray(details.get("conflictLog", np.zeros((0, 5), dtype=float)), dtype=float)
        if conflict_log.ndim == 1 and conflict_log.size == 5:
            conflict_log = conflict_log.reshape(1, 5)
        if conflict_log.ndim == 2 and conflict_log.shape[1] >= 5 and conflict_log.shape[0] > 0:
            trimmed = np.asarray(conflict_log[:, :5], dtype=float)
            conflict_log_rows.append(trimmed)
            conflict_log_candidate_index.append(np.full(trimmed.shape[0], float(candidate_index), dtype=float))

    # Base-fleet-compatible path artifacts: persist primary UAV path as bp_*.mat.
    for idx, paths in enumerate(fleet_paths, start=1):
        if not paths:
            continue
        primary = np.asarray(paths[0], dtype=float)
        if primary.ndim != 2 or primary.shape[1] != 3 or primary.shape[0] < 2:
            continue
        if final_matrix.ndim == 2 and (idx - 1) < final_matrix.shape[0]:
            obj = final_matrix[idx - 1]
        else:
            candidate_obj = np.asarray(final_candidates[idx - 1].objective, dtype=float)
            obj = candidate_obj.reshape(-1) if candidate_obj.ndim != 1 else candidate_obj
        try:
            save_bp(run_dir / f"bp_{idx}.mat", primary, np.asarray(obj, dtype=float))
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping invalid bp artifact for %s candidate %d: %s", run_dir, idx, exc)
            continue

    try:
        save_mat(
            run_dir / "mission_stats.mat",
            {
                "conflictMean": float(np.nanmean(conflict_values)) if conflict_values else 0.0,
                "conflictStd": float(np.nanstd(conflict_values)) if conflict_values else 0.0,
                "nSolutions": float(len(final_candidates)),
                "feasible": np.asarray(feasible_values, dtype=float),
                "conflictRate": np.asarray(conflict_values, dtype=float),
                "minSeparation": np.asarray(separation_values, dtype=float),
                "makespan": np.asarray(makespan_values, dtype=float),
                "energy": np.asarray(energy_values, dtype=float),
                "risk": np.asarray(risk_values, dtype=float),
                "maxTurnDeg": np.asarray(max_turn_values, dtype=float),
                "turnViolation": np.asarray(turn_violation_values, dtype=float),
                "separationViolation": np.asarray(separation_violation_values, dtype=float),
                "collisionViolation": np.asarray(collision_violation_values, dtype=float),
                "minClearance": np.asarray(min_clearance_values, dtype=float),
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("Skipping mission_stats artifact for %s: %s", run_dir, exc)

    # Best paths (for visualization)
    if fleet_paths and final_matrix.size > 0:
        try:
            best_idx = int(np.argmin(np.sum(np.where(np.isfinite(final_matrix), final_matrix, 1e9), axis=1)))
            best_paths = fleet_paths[best_idx]
            if best_paths:
                save_mat(
                    run_dir / "fleet_paths.mat",
                    {f"uav{i + 1}": np.asarray(p, dtype=float) for i, p in enumerate(best_paths)},
                )
        except (OSError, TypeError, ValueError, IndexError) as exc:
            LOGGER.warning("Skipping fleet_paths artifact for %s: %s", run_dir, exc)

    # Conflict log
    detailed_conflicts = np.vstack(conflict_log_rows) if conflict_log_rows else np.zeros((0, 5), dtype=float)
    conflict_candidate_index = (
        np.concatenate(conflict_log_candidate_index) if conflict_log_candidate_index else np.zeros(0, dtype=float)
    )
    save_mat(
        run_dir / "conflict_log.mat",
        {
            "conflicts": detailed_conflicts,
            "conflictLog": detailed_conflicts,
            "candidateIndex": conflict_candidate_index,
            "conflictRates": np.asarray(conflict_values, dtype=float),
        },
    )

    feasible_array = np.asarray(feasible_values, dtype=float)
    run_stats["solutionCount"] = int(final_matrix.shape[0]) if final_matrix.ndim == 2 else int(len(final_candidates))
    run_stats["feasibleCount"] = int(np.sum(feasible_array > 0.5)) if feasible_array.size > 0 else 0
    inferred_fleet_size = 0
    for paths in fleet_paths:
        if paths:
            inferred_fleet_size = len(paths)
            break
    if inferred_fleet_size > 0:
        run_stats.setdefault("fleetSize", inferred_fleet_size)
    save_mat(run_dir / "run_stats.mat", run_stats)

    # Save a normalized summary after the full run stats are available.
    class _DummyParams:
        algorithm = run_stats.get("algorithm", "fleet_run")
        problem_name = run_stats.get("problemName", "unknown")
        fleet_size = int(run_stats.get("fleetSize", 0))
        generations = int(run_stats.get("generations", 0))
        population = int(run_stats.get("population", len(final_candidates)))
        seed = run_stats.get("seed", 0)

    save_run_summary_json(run_dir / "run_summary.json", _DummyParams(), run_stats)
