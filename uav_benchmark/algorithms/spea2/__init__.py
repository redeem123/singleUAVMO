from __future__ import annotations

"""SPEA2 runner adapted for this benchmark.

Core evolutionary flow is inherited from the PlatEMO SPEA2 implementation
(E. Zitzler, M. Laumanns, and L. Thiele, 2001; PlatEMO educational codebase),
then adapted to this repository's fleet evaluation and artifact pipeline.
"""

import time
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
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
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


from uav_benchmark.algorithms.shared.pareto_utils import _sanitize_objectives, _pairwise_distance


def _cal_fitness(pop_obj: np.ndarray) -> np.ndarray:
    """Compute SPEA2 fitness from objective matrix."""
    objective = _sanitize_objectives(pop_obj)
    n_points = int(objective.shape[0])
    if n_points == 0:
        return np.zeros(0, dtype=float)

    less = objective[:, np.newaxis, :] < objective[np.newaxis, :, :]
    greater = objective[:, np.newaxis, :] > objective[np.newaxis, :, :]
    dominate = np.any(less, axis=2) & ~np.any(greater, axis=2)
    np.fill_diagonal(dominate, False)

    strength = np.sum(dominate, axis=1, dtype=float)
    raw_fitness = dominate.T.astype(float) @ strength

    distance = _pairwise_distance(objective)
    sorted_distance = np.sort(distance, axis=1)
    k_index = max(0, min(n_points - 1, int(np.floor(np.sqrt(float(n_points)))) - 1))
    density = 1.0 / (sorted_distance[:, k_index] + 2.0)
    return raw_fitness + density


def _truncation(pop_obj: np.ndarray, n_remove: int) -> np.ndarray:
    """Select solutions to remove following PlatEMO SPEA2 truncation."""
    objective = _sanitize_objectives(pop_obj)
    n_points = int(objective.shape[0])
    if n_points == 0 or n_remove <= 0:
        return np.zeros(n_points, dtype=bool)

    distance = _pairwise_distance(objective)
    removed = np.zeros(n_points, dtype=bool)
    target = min(n_points, int(n_remove))
    while int(np.sum(removed)) < target:
        remain = np.flatnonzero(~removed)
        if remain.size == 0:
            break
        temp = np.sort(distance[np.ix_(remain, remain)], axis=1)
        keys = tuple(temp[:, col] for col in range(temp.shape[1] - 1, -1, -1))
        rank = np.lexsort(keys)
        removed[remain[int(rank[0])]] = True
    return removed


def _environmental_selection(
    vectors: np.ndarray,
    candidates: list[Candidate],
    n_keep: int,
) -> tuple[np.ndarray, list[Candidate], np.ndarray]:
    total = len(candidates)
    if total == 0 or n_keep <= 0:
        return np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=float), [], np.zeros(0, dtype=float)
    if total <= n_keep:
        fitness_all = _cal_fitness(_candidate_matrix(candidates))
        return vectors.copy(), list(candidates), fitness_all

    objective = _candidate_matrix(candidates)
    fitness = _cal_fitness(objective)
    next_mask = fitness < 1.0
    selected_count = int(np.sum(next_mask))

    if selected_count < n_keep:
        rank = np.argsort(fitness, kind="mergesort")
        next_mask[:] = False
        next_mask[rank[:n_keep]] = True
    elif selected_count > n_keep:
        delete_mask = _truncation(objective[next_mask], selected_count - n_keep)
        tmp = np.flatnonzero(next_mask)
        next_mask[tmp[delete_mask]] = False

    selected = np.flatnonzero(next_mask)
    if selected.size < n_keep:
        remain = np.setdiff1d(np.arange(total, dtype=int), selected, assume_unique=False)
        if remain.size > 0:
            fill_rank = remain[np.argsort(fitness[remain], kind="mergesort")]
            fill = fill_rank[: n_keep - selected.size]
            selected = np.hstack([selected, fill])
    elif selected.size > n_keep:
        selected = selected[:n_keep]

    selected = selected.astype(int, copy=False)
    return vectors[selected], [candidates[int(idx)] for idx in selected], fitness[selected]


def _tournament_selection(k_tournament: int, n_select: int, fitness: np.ndarray) -> np.ndarray:
    fit = np.asarray(fitness, dtype=float).reshape(-1)
    if fit.size == 0 or n_select <= 0:
        return np.zeros(0, dtype=int)
    k = max(1, int(k_tournament))
    pool = np.random.randint(0, fit.size, size=(k, n_select))
    best_rows = np.argmin(fit[pool], axis=0)
    return pool[best_rows, np.arange(n_select)]


def _run_fleet_spea2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    # NOTE: This structure follows PlatEMO's SPEA2 template:
    # 1) initialize population and SPEA2 fitness,
    # 2) tournament-select by fitness,
    # 3) generate offspring via GA operator,
    # 4) environmental selection with SPEA2 truncation.
    # The adaptation below keeps those steps while using this benchmark's
    # fleet objective evaluation and output artifacts.
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
    pop_size = max(2, int(params.population))
    metric_interval = int(params.extra.get("metricInterval", 20))

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
        fitness = _cal_fitness(_candidate_matrix(candidates))
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            mating_pool = _tournament_selection(2, pop_size, fitness)
            offspring = _sbx_mutation(population[mating_pool], lower, upper)
            offspring_candidates = _evaluate_population(offspring, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

            merged_vectors = np.vstack([population, offspring])
            merged_candidates = candidates + offspring_candidates
            population, candidates, fitness = _environmental_selection(merged_vectors, merged_candidates, pop_size)

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


def run_spea2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_spea2(model, params)
