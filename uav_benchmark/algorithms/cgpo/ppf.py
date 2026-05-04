from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_benchmark.algorithms.cgpo.cig import ConstraintInteractionGraph
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort


@dataclass(frozen=True, slots=True)
class ParetoPressureField:
    parent_probability: np.ndarray
    feasibility_pressure: np.ndarray
    exploration_scale: np.ndarray
    objective_weights: np.ndarray
    boundary_mass: float
    feasible_mass: float
    pressure_entropy: float
    stratum_counts: dict[str, int]


def _safe_softmax(score: np.ndarray) -> np.ndarray:
    if score.size == 0:
        return np.zeros(0, dtype=float)
    finite = np.where(np.isfinite(score), score, -1e9)
    shifted = finite - float(np.max(finite))
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    total = float(np.sum(exp))
    if total <= 1e-12:
        return np.full(score.shape, 1.0 / max(1, score.size), dtype=float)
    return exp / total


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def compute_pareto_pressure_field(
    objective: np.ndarray,
    violations: np.ndarray,
    feasible: np.ndarray,
    graphs: list[ConstraintInteractionGraph],
    enabled: bool = True,
    boundary_epsilon: float = 0.05,
    pressure_temperature: float = 1.0,
    diversity_weight: float = 0.25,
    boundary_weight: float = 0.25,
    rank_weight: float = 0.75,
    violation_weight: float = 1.10,
) -> ParetoPressureField:
    objective = np.asarray(objective, dtype=float)
    if objective.ndim != 2:
        objective = objective.reshape(0, 0)
    n = objective.shape[0]
    violations = np.asarray(violations, dtype=float).reshape(-1)
    feasible = np.asarray(feasible, dtype=bool).reshape(-1)
    if violations.size != n:
        raise ValueError(f"violations length {violations.size} does not match objective rows {n}")
    if feasible.size != n:
        raise ValueError(f"feasible length {feasible.size} does not match objective rows {n}")

    if n == 0:
        return ParetoPressureField(
            parent_probability=np.zeros(0, dtype=float),
            feasibility_pressure=np.zeros(0, dtype=float),
            exploration_scale=np.zeros(0, dtype=float),
            objective_weights=np.ones(4, dtype=float) / 4.0,
            boundary_mass=0.0,
            feasible_mass=0.0,
            pressure_entropy=0.0,
            stratum_counts={"feasible": 0, "boundary": 0, "diverse": 0, "suppressed": 0},
        )

    safe_objective = np.where(
        np.isfinite(objective), objective, np.nanmax(np.where(np.isfinite(objective), objective, 0.0)) + 1.0
    )
    try:
        ranks, _ = n_d_sort(safe_objective, None, n)
        crowd = crowding_distance(safe_objective, ranks)
    except (ArithmeticError, IndexError, TypeError, ValueError):
        ranks = np.ones(n, dtype=float)
        crowd = np.zeros(n, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    diversity = _normalize(np.asarray(crowd, dtype=float))
    tension = np.asarray([graph.mean_tension for graph in graphs[:n]], dtype=float)
    if tension.size != n:
        raise ValueError(f"graphs length {len(graphs)} does not match objective rows {n}")
    tension_norm = _normalize(tension)
    cv_norm = _normalize(np.log1p(np.maximum(0.0, violations)))
    boundary_usefulness = (
        np.exp(-np.maximum(0.0, violations) / max(float(boundary_epsilon), 1e-9))
        * (0.25 + tension_norm)
        * (0.25 + diversity)
    )
    high_tension = tension_norm >= (float(np.quantile(tension_norm, 0.65)) if tension_norm.size > 1 else 0.0)
    useful_feasible_boundary = feasible & high_tension & (diversity >= 0.20)
    near_boundary = (
        (violations > 1e-12) & (violations <= max(float(boundary_epsilon), 1e-9) * 10.0)
    ) | useful_feasible_boundary

    if enabled:
        diversity_weight = max(0.0, float(diversity_weight))
        boundary_weight = max(0.0, float(boundary_weight))
        rank_weight = max(0.0, float(rank_weight))
        violation_weight = max(0.0, float(violation_weight))
        score = (
            -rank_weight * _normalize(ranks)
            - violation_weight * cv_norm
            + diversity_weight * diversity
            + boundary_weight * _normalize(boundary_usefulness)
        ) / max(float(pressure_temperature), 1e-9)
        parent_probability = _safe_softmax(score)
        feasibility_pressure = np.clip(0.25 + 0.70 * cv_norm + 0.25 * near_boundary.astype(float), 0.10, 1.0)
        exploration_scale = np.clip(0.025 + 0.13 * (1.0 - feasibility_pressure) + 0.06 * diversity, 0.015, 0.18)
        objective_spread = np.std(safe_objective, axis=0) if safe_objective.size else np.ones(4, dtype=float)
        objective_weights = _safe_softmax(_normalize(objective_spread))
    else:
        parent_probability = np.full(n, 1.0 / n, dtype=float)
        feasibility_pressure = np.full(n, 0.55, dtype=float)
        exploration_scale = np.full(n, 0.075, dtype=float)
        objective_weights = np.ones(
            safe_objective.shape[1] if safe_objective.ndim == 2 and safe_objective.shape[1] else 4, dtype=float
        )
        objective_weights /= max(float(np.sum(objective_weights)), 1e-9)

    pressure_entropy = -float(np.sum(parent_probability * np.log(np.maximum(parent_probability, 1e-12))))
    diverse = (~feasible) & (~near_boundary) & (diversity >= np.quantile(diversity, 0.65))
    suppressed = ~(feasible | near_boundary | diverse)
    return ParetoPressureField(
        parent_probability=parent_probability,
        feasibility_pressure=feasibility_pressure,
        exploration_scale=exploration_scale,
        objective_weights=objective_weights,
        boundary_mass=float(np.sum(parent_probability[near_boundary])) if parent_probability.size else 0.0,
        feasible_mass=float(np.sum(parent_probability[feasible])) if parent_probability.size else 0.0,
        pressure_entropy=pressure_entropy,
        stratum_counts={
            "feasible": int(np.sum(feasible)),
            "boundary": int(np.sum(near_boundary)),
            "diverse": int(np.sum(diverse)),
            "suppressed": int(np.sum(suppressed)),
        },
    )
