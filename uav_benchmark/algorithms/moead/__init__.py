from __future__ import annotations

"""MOEA/D runner adapted for this benchmark.

Core decomposition flow is inherited from the PlatEMO MOEAD implementation
(Q. Zhang and H. Li, 2007; PlatEMO educational codebase), then adapted to
this repository's fleet constrained evaluation and artifact pipeline.
"""

import time
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _constraint_violation,
    _constraint_violation_vector,
    _ensure_fleet_endpoints,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _sbx_mutation,
    _should_write_final_hv,
)
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga3_ops import uniform_point
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


from uav_benchmark.algorithms.shared.pareto_utils import _clone_candidate


def _update_ideal_point(ideal: np.ndarray, objective: np.ndarray) -> np.ndarray:
    objective = np.asarray(objective, dtype=float).reshape(-1)
    if np.all(np.isfinite(objective)):
        return np.minimum(ideal, objective)
    return ideal


def _finite_ideal(candidates: list[Candidate], objective_count: int) -> np.ndarray:
    matrix = _candidate_matrix(candidates)
    if matrix.size == 0:
        return np.zeros(objective_count, dtype=float)
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return np.zeros(objective_count, dtype=float)
    return np.min(finite, axis=0)


def _finite_max(candidates: list[Candidate], objective_count: int, fallback: np.ndarray) -> np.ndarray:
    matrix = _candidate_matrix(candidates)
    if matrix.size == 0:
        return fallback.copy()
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return fallback.copy()
    return np.max(finite, axis=0)


def _decomposition_values(
    objectives: np.ndarray,
    weights: np.ndarray,
    ideal: np.ndarray,
    approach_type: int,
    theta: float,
    zmax: np.ndarray,
) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ideal = np.asarray(ideal, dtype=float).reshape(1, -1)
    if objectives.ndim != 2:
        objectives = objectives.reshape(1, -1)
    if weights.ndim != 2:
        weights = weights.reshape(1, -1)
    if objectives.shape[0] == 1 and weights.shape[0] > 1:
        objectives = np.repeat(objectives, weights.shape[0], axis=0)

    if approach_type == 1:
        norm_w = np.linalg.norm(weights, axis=1) + 1e-12
        delta = objectives - ideal
        norm_delta = np.linalg.norm(delta, axis=1) + 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.sum(delta * weights, axis=1) / (norm_w * norm_delta)
            cosine = np.clip(cosine, -1.0, 1.0)
            values = norm_delta * cosine + float(theta) * norm_delta * np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
        values = np.where(np.isfinite(values), values, np.inf)
        return values

    if approach_type == 2:
        values = np.max(np.abs(objectives - ideal) * weights, axis=1)
        return np.where(np.isfinite(values), values, np.inf)

    if approach_type == 3:
        denom = np.maximum(np.asarray(zmax, dtype=float).reshape(1, -1) - ideal, 1e-12)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.max(np.abs(objectives - ideal) / denom * weights, axis=1)
        return np.where(np.isfinite(values), values, np.inf)

    # Modified Tchebycheff.
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.max(np.abs(objectives - ideal) / np.maximum(weights, 1e-12), axis=1)
    return np.where(np.isfinite(values), values, np.inf)


def _offspring_from_two_parents(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    parents = np.vstack([np.asarray(parent_a, dtype=float), np.asarray(parent_b, dtype=float)])
    offspring = _sbx_mutation(parents, lower, upper)
    if offspring.shape[0] == 0:
        return parents[0].copy()
    return np.asarray(offspring[0], dtype=float).copy()


def _run_fleet_moead(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    # NOTE: This algorithm structure follows PlatEMO's MOEAD template:
    # 1) generate weights, 2) build neighborhoods, 3) iterate subproblems,
    # 4) create offspring from neighborhood parents, 5) update ideal point,
    # 6) replace neighbors by decomposition value. The implementation below
    # adapts those steps to constrained fleet mission objectives.
    objective_count = 4
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
    metric_interval = int(params.extra.get("metricInterval", 20))
    method = str(params.extra.get("moeadWeightMethod", "NBI")).strip() or "NBI"

    weights, adjusted_population = uniform_point(int(params.population), objective_count, method)
    pop_size = int(adjusted_population)
    if pop_size != int(params.population):
        import logging
        logging.getLogger(__name__).warning(
            "MOEAD: NBI weight generation adjusted population size %d → %d "
            "(4-objective NBI simplex constraint). Pass moeadWeightMethod='LATIN' "
            "in extra for exact population control.",
            int(params.population), pop_size,
        )
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-6)
    if weights.shape[0] != pop_size:
        weights = weights[:pop_size]

    default_neighbors = int(np.ceil(pop_size / 10.0))
    neighbor_size = int(params.extra.get("moeadNeighbors", default_neighbors))
    neighbor_size = max(2, min(pop_size, neighbor_size))
    approach_type = int(params.extra.get("moeadType", 1))
    approach_type = 1 if approach_type < 1 or approach_type > 4 else approach_type
    theta = float(params.extra.get("moeadTheta", 5.0))

    dist = np.linalg.norm(weights[:, np.newaxis, :] - weights[np.newaxis, :, :], axis=2)
    neighbors = np.argsort(dist, axis=1)[:, :neighbor_size]

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    run_indices = _resolve_run_indices(params)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in run_indices:
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue
        run_start = time.perf_counter()

        population = np.random.uniform(lower, upper, size=(pop_size, dimensions))
        candidates = _evaluate_population(population, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        ideal = _finite_ideal(candidates, objective_count)
        constraint_vector = _constraint_violation_vector(candidates, model)
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            zmax = _finite_max(candidates, objective_count, fallback=ideal + 1.0)
            for index in range(pop_size):
                subset = neighbors[index]
                parent_order = subset[np.random.permutation(subset.shape[0])]
                pa = int(parent_order[0])
                pb = int(parent_order[1]) if parent_order.shape[0] > 1 else int(parent_order[0])
                child_vec = _offspring_from_two_parents(population[pa], population[pb], lower, upper)
                child_candidate = _evaluate_population(
                    child_vec.reshape(1, -1), model, fleet_size=fleet_size, n_waypoints=n_waypoints
                )[0]
                child_con = _constraint_violation(child_candidate, model)
                ideal = _update_ideal_point(ideal, child_candidate.objective)

                old_obj = np.asarray([candidates[int(j)].objective for j in subset], dtype=float)
                old_con = constraint_vector[subset]
                new_obj = np.repeat(np.asarray(child_candidate.objective, dtype=float).reshape(1, -1), subset.shape[0], axis=0)
                old_g = _decomposition_values(old_obj, weights[subset], ideal, approach_type, theta, zmax)
                new_g = _decomposition_values(new_obj, weights[subset], ideal, approach_type, theta, zmax)

                replace_mask = (child_con < old_con) | ((np.abs(child_con - old_con) <= 1e-12) & (new_g <= old_g))
                if np.any(replace_mask):
                    for local_idx, global_idx in enumerate(subset):
                        if not bool(replace_mask[local_idx]):
                            continue
                        gi = int(global_idx)
                        population[gi] = child_vec
                        candidates[gi] = _clone_candidate(child_candidate, vector=child_vec)
                        constraint_vector[gi] = child_con

            if params.compute_metrics:
                final_obj = _candidate_matrix(candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, final_obj, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, final_obj, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_obj, params.problem_index, objective_count),
                    cal_metric(2, final_obj, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_moead(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_moead(model, params)
