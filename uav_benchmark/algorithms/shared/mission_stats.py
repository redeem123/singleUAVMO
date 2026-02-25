from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.core.evaluate_mission import evaluate_mission_details


def build_fleet_mission_stats(
    solutions_paths: list[list[np.ndarray]],
    model: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Build mission stats for a list of candidate fleet solutions.

    Each entry in ``solutions_paths`` is one candidate solution and contains
    one path per UAV. This is the canonical stats path for both single and
    fleet reporting.
    """
    solution_count = len(solutions_paths)
    feasible_values = np.zeros(solution_count, dtype=float)
    conflict_values = np.zeros(solution_count, dtype=float)
    min_sep_values = np.full(solution_count, np.nan, dtype=float)
    makespan_values = np.full(solution_count, np.nan, dtype=float)
    energy_values = np.full(solution_count, np.nan, dtype=float)
    risk_values = np.full(solution_count, np.nan, dtype=float)
    max_turn_values = np.full(solution_count, np.nan, dtype=float)
    turn_violation_values = np.zeros(solution_count, dtype=float)
    separation_violation_values = np.zeros(solution_count, dtype=float)
    collision_violation_values = np.zeros(solution_count, dtype=float)
    min_clearance_values = np.full(solution_count, np.nan, dtype=float)

    for index, fleet_paths in enumerate(solutions_paths):
        normalized_paths = [np.asarray(path, dtype=float) for path in fleet_paths]
        objective, details = evaluate_mission_details(normalized_paths, model)
        finite = bool(np.all(np.isfinite(objective)))
        collision_violation = float(details.get("collisionViolation", 0.0)) > 0.5
        separation_violation = float(details.get("separationViolation", 0.0)) > 0.5
        feasible = finite and (not collision_violation) and (not separation_violation)

        feasible_values[index] = float(feasible)
        conflict_values[index] = float(details.get("conflictRate", np.nan))
        min_sep_values[index] = float(details.get("minSeparation", np.nan))
        makespan_values[index] = float(details.get("makespan", np.nan))
        energy_values[index] = float(details.get("energy", np.nan))
        risk_values[index] = float(details.get("risk", np.nan))
        max_turn_values[index] = float(details.get("maxTurnDeg", np.nan))
        turn_violation_values[index] = float(details.get("turnViolation", 0.0))
        separation_violation_values[index] = float(separation_violation)
        collision_violation_values[index] = float(collision_violation)
        min_clearance_values[index] = float(details.get("minClearance", np.nan))

    mission_stats: dict[str, Any] = {
        "conflictMean": float(np.nanmean(conflict_values)) if conflict_values.size > 0 else 0.0,
        "conflictStd": float(np.nanstd(conflict_values)) if conflict_values.size > 0 else 0.0,
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


def build_mission_stats(
    paths_xyz: list[np.ndarray],
    model: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Backward-compatible legacy-path stats builder over shared mission stats."""
    solutions = [[np.asarray(path, dtype=float)] for path in paths_xyz]
    return build_fleet_mission_stats(solutions_paths=solutions, model=model)
