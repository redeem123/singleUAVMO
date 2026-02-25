from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.core.evaluate_path import evaluate_path_details


def _path_length(path_xyz: np.ndarray) -> float:
    if path_xyz.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path_xyz, axis=0), axis=1)))


def build_single_uav_mission_stats(
    paths_xyz: list[np.ndarray],
    model: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Build per-solution mission stats for single-UAV runs.

    Feasibility is strict and consistent with multi-UAV reporting:
    finite objectives + no collision violation.
    Turn violation remains a reported metric and objective penalty, but is
    not treated as a hard feasibility failure.
    """
    solution_count = len(paths_xyz)
    feasible_values = np.zeros(solution_count, dtype=float)
    conflict_values = np.zeros(solution_count, dtype=float)
    min_sep_values = np.full(solution_count, np.nan, dtype=float)
    makespan_values = np.zeros(solution_count, dtype=float)
    energy_values = np.zeros(solution_count, dtype=float)
    risk_values = np.full(solution_count, np.inf, dtype=float)
    max_turn_values = np.full(solution_count, np.inf, dtype=float)
    turn_violation_values = np.zeros(solution_count, dtype=float)
    separation_violation_values = np.zeros(solution_count, dtype=float)
    collision_violation_values = np.ones(solution_count, dtype=float)
    min_clearance_values = np.full(solution_count, np.nan, dtype=float)

    turn_limit_deg = float(model.get("maxTurnDeg", model.get("maxTurnAngleDeg", 75.0)))

    for index, path in enumerate(paths_xyz):
        path_xyz = np.asarray(path, dtype=float)
        objective, details = evaluate_path_details(path_xyz, model)

        finite = bool(np.all(np.isfinite(objective)))
        collision_violation = float(details.get("collisionViolation", 1.0)) > 0.5
        max_turn_deg = float(details.get("maxTurnDeg", np.inf))
        turn_violation = bool(np.isfinite(max_turn_deg) and (max_turn_deg > turn_limit_deg + 1e-9))
        feasible = finite and (not collision_violation)

        feasible_values[index] = float(feasible)
        collision_violation_values[index] = float(collision_violation)
        max_turn_values[index] = max_turn_deg
        turn_violation_values[index] = float(turn_violation)
        min_clearance_values[index] = float(details.get("minClearance", np.nan))

        path_len = _path_length(path_xyz)
        makespan_values[index] = path_len
        energy_values[index] = path_len
        if objective.size >= 2:
            risk_values[index] = float(objective[1])

    mission_stats: dict[str, Any] = {
        "conflictMean": 0.0,
        "conflictStd": 0.0,
        "nSolutions": float(solution_count),
        "feasible": feasible_values,
        "conflictRate": conflict_values,
        "minSeparation": min_sep_values,
        "makespan": makespan_values,
        "energy": energy_values,
        "risk": risk_values,
        "maxTurnDeg": max_turn_values,
        "turnViolation": turn_violation_values,
        "separationViolation": separation_violation_values,
        "collisionViolation": collision_violation_values,
        "minClearance": min_clearance_values,
    }
    feasible_mask = feasible_values > 0.5
    return mission_stats, feasible_mask
