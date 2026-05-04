from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.sac_smopso.initialization import _detail_value
from uav_benchmark.algorithms.shared.pso_types import Candidate


def _reservoir_score(
    candidate: Candidate,
    *,
    separation_min: float = 10.0,
    drone_size: float = 1.0,
    max_turn_deg: float = 75.0,
) -> float:
    """Pre-infeasibility ranking score for the unconstrained reservoir.

    ``candidate.objective`` is set to ``inf`` on any constraint
    violation, so sorting on it collapses every infeasible candidate
    to the same rank. The individual objective terms stored in
    ``details`` are also clipped to [0, 1] before the inf mask, which
    is too coarse to separate a 50° hairpin from a 180° one.

    Instead we combine raw geometry quantities:
      * ``maxTurnDeg`` relative to the active turn limit.
      * Separation shortfall below ``separationMin`` — pairwise
        conflicts; zero when feasible.
      * Clearance shortfall below ``droneSize`` — obstacle collisions.
      * Search-objective magnitude as a tiebreaker between
        geometrically similar candidates.

    Lower is better. This lets SBX steadily pull the population
    toward feasible geometry even while everybody is still nominally
    infeasible (the CMOSMA AP/unconstrained-selection mechanism).
    """
    details = getattr(candidate, "details", {}) or {}
    try:
        turn_limit = max(1e-6, float(max_turn_deg))
        live_turn = _detail_value(details, "maxTurnDeg", float("nan"))
        if np.isfinite(live_turn):
            turn_pressure = float(np.clip(live_turn / turn_limit, 0.0, 2.5))
            turn_excess = max(0.0, (live_turn - turn_limit) / turn_limit)
        else:
            turn_flag = max(0.0, _detail_value(details, "turnViolation", 1.0))
            turn_pressure = 1.0 + turn_flag
            turn_excess = turn_flag

        live_separation = _detail_value(details, "minSeparation", float("nan"))
        if np.isfinite(live_separation):
            sep_term = max(0.0, (float(separation_min) - live_separation) / max(float(separation_min), 1e-6))
        else:
            sep_term = max(0.0, _detail_value(details, "separationViolation", 1.0))

        live_clearance = _detail_value(details, "minClearance", float("nan"))
        if np.isfinite(live_clearance):
            clr_term = max(0.0, (float(drone_size) - live_clearance) / max(float(drone_size), 1e-6))
        else:
            clr_term = max(0.0, _detail_value(details, "collisionViolation", 1.0))

        obj_term = _reservoir_objective_term(candidate, details)
        conflict_term = max(0.0, _detail_value(details, "conflictRate", 0.0))
        infeasible_term = 1.0 - np.clip(_detail_value(details, "feasible", 0.0), 0.0, 1.0)
        return (
            4.0 * turn_excess
            + 1.25 * turn_pressure
            + 3.0 * sep_term
            + 3.0 * clr_term
            + 0.8 * conflict_term
            + 0.5 * infeasible_term
            + 0.5 * obj_term
        )
    except (TypeError, ValueError):
        return 100.0


def _reservoir_objective_term(candidate: Candidate, details: dict[str, Any]) -> float:
    objective_vector = np.asarray(details.get("objective_search", candidate.objective), dtype=float).reshape(-1)
    finite_objective = objective_vector[np.isfinite(objective_vector)]
    if finite_objective.size == 0:
        finite_objective = np.asarray(
            [
                _detail_value(details, "makespan", 1.0),
                _detail_value(details, "energy", 1.0),
                _detail_value(details, "risk", 1.0),
                _detail_value(details, "turnPenalty", 1.0),
            ],
            dtype=float,
        )
        finite_objective = finite_objective[np.isfinite(finite_objective)]
    if finite_objective.size == 0:
        return 1.0
    return float(np.mean(np.clip(finite_objective, 0.0, 3.0))) / 3.0
