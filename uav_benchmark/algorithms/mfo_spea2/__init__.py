from __future__ import annotations

"""MFO-SPEA2 runner adapted for this benchmark.

Core workflow is adapted from PlatEMO's MFO-SPEA2 implementation:
- dual populations (target/source),
- epsilon-boundary reduction for source task constraints,
- shared offspring generation,
- SPEA2-style environmental selection for both tasks.
"""

import copy
import time
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
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
from uav_benchmark.core.nsga2_ops import tournament_selection
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


def _clone_candidate(candidate: Candidate, vector: np.ndarray | None = None) -> Candidate:
    cloned_details = copy.deepcopy(candidate.details) if isinstance(candidate.details, dict) else {}
    return Candidate(
        vector=np.asarray(vector if vector is not None else candidate.vector, dtype=float).copy(),
        objective=np.asarray(candidate.objective, dtype=float).copy(),
        details=cloned_details,
    )


def _sanitize_objectives(pop_obj: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pop_obj, dtype=float)
    if matrix.size == 0:
        return matrix.reshape(0, 0)
    finite_mask = np.isfinite(matrix)
    if np.all(finite_mask):
        return matrix
    col_max = np.zeros(matrix.shape[1], dtype=float)
    for col in range(matrix.shape[1]):
        col_values = matrix[finite_mask[:, col], col]
        if col_values.size > 0:
            col_max[col] = float(np.max(col_values))
    penalties = np.sum(~finite_mask, axis=1, keepdims=True).astype(float)
    replacement = col_max.reshape(1, -1) + 1e6 + penalties
    return np.where(finite_mask, matrix, replacement)


def _pairwise_distance(pop_obj: np.ndarray) -> np.ndarray:
    if pop_obj.size == 0:
        return np.zeros((0, 0), dtype=float)
    diff = pop_obj[:, np.newaxis, :] - pop_obj[np.newaxis, :, :]
    distance = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distance, np.inf)
    return distance


def _cal_fitness(pop_obj: np.ndarray, pop_con: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute constrained SPEA2 fitness and per-individual rank positions."""
    objective = _sanitize_objectives(pop_obj)
    n_points = int(objective.shape[0])
    if n_points == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    if pop_con is None:
        cv = np.zeros(n_points, dtype=float)
    else:
        con = np.asarray(pop_con, dtype=float)
        if con.ndim == 1:
            con = con.reshape(-1, 1)
        cv = np.sum(np.maximum(0.0, con), axis=1)

    dominate = np.zeros((n_points, n_points), dtype=bool)
    for i in range(n_points - 1):
        for j in range(i + 1, n_points):
            if cv[i] < cv[j]:
                dominate[i, j] = True
                continue
            if cv[i] > cv[j]:
                dominate[j, i] = True
                continue
            any_better = bool(np.any(objective[i] < objective[j]))
            any_worse = bool(np.any(objective[i] > objective[j]))
            if any_better and not any_worse:
                dominate[i, j] = True
            elif any_worse and not any_better:
                dominate[j, i] = True

    strength = np.sum(dominate, axis=1, dtype=float)
    raw = np.zeros(n_points, dtype=float)
    for idx in range(n_points):
        raw[idx] = float(np.sum(strength[dominate[:, idx]]))

    distance = _pairwise_distance(objective)
    sorted_distance = np.sort(distance, axis=1)
    k_index = max(0, min(n_points - 1, int(np.floor(np.sqrt(float(n_points)))) - 1))
    density = 1.0 / (sorted_distance[:, k_index] + 2.0)
    fitness = raw + density

    order = np.argsort(fitness, kind="mergesort")
    rank = np.empty(n_points, dtype=float)
    rank[order] = np.arange(1, n_points + 1, dtype=float)
    return fitness, rank


def _truncation(pop_obj: np.ndarray, n_remove: int) -> np.ndarray:
    """Select solutions to remove following SPEA2 truncation."""
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
    pop_con: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Candidate], np.ndarray]:
    """MFO-SPEA2 environmental selection."""
    total = len(candidates)
    if total == 0 or n_keep <= 0:
        empty = np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=float)
        return empty, [], np.zeros(0, dtype=float)

    objective = _candidate_matrix(candidates)
    fitness, rank = _cal_fitness(objective, pop_con)

    next_mask = fitness < 1.0
    selected_count = int(np.sum(next_mask))

    if selected_count < n_keep:
        order = np.argsort(fitness, kind="mergesort")
        next_mask[:] = False
        next_mask[order[:n_keep]] = True
    elif selected_count > n_keep:
        delete_mask = _truncation(objective[next_mask], selected_count - n_keep)
        idx = np.flatnonzero(next_mask)
        next_mask[idx[delete_mask]] = False

    selected = np.flatnonzero(next_mask)
    if selected.size < n_keep:
        remain = np.setdiff1d(np.arange(total, dtype=int), selected, assume_unique=False)
        if remain.size > 0:
            fill_order = remain[np.argsort(fitness[remain], kind="mergesort")]
            selected = np.hstack([selected, fill_order[: n_keep - selected.size]])
    elif selected.size > n_keep:
        selected = selected[:n_keep]

    selected = selected.astype(int, copy=False)
    return vectors[selected], [candidates[int(idx)] for idx in selected], rank[selected]


def _reduce_boundary(e_f: np.ndarray, k: int, max_k: int) -> np.ndarray:
    """Reduce epsilon boundary for source-task constraints."""
    z = 1e-8
    near_zero = 1e-15
    e_f = np.maximum(0.0, np.asarray(e_f, dtype=float))
    if e_f.ndim == 0:
        e_f = e_f.reshape(1)

    log_term = np.log((e_f + z) / z)
    log_term = np.maximum(log_term, near_zero)
    b = float(max(1, int(max_k))) / np.power(log_term, 1.0 / 10.0)
    b = np.where(b == 0.0, near_zero, b)
    f = e_f * np.exp(-np.power(float(max(1, int(k))) / b, 10.0))
    close_to_z = np.abs(f - z) < near_zero
    f[close_to_z] = z
    eps_n = f - z
    eps_n[eps_n <= 0.0] = 0.0
    return eps_n


def _run_fleet_mfo_spea2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 0
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

        # Target task population
        target_pop = np.random.uniform(lower, upper, size=(pop_size, dimensions))
        target_candidates = _evaluate_population(target_pop, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

        # Source task starts from the target population
        source_pop = target_pop.copy()
        source_candidates = [
            _clone_candidate(candidate, vector=source_pop[idx]) for idx, candidate in enumerate(target_candidates)
        ]

        target_cv = _constraint_violation_vector(target_candidates, model).reshape(-1, 1)
        source_cv = target_cv.copy()
        initial_e = np.maximum(1.0, np.max(np.maximum(0.0, target_cv), axis=0))

        _, rank_tp = _cal_fitness(_candidate_matrix(target_candidates), target_cv)
        _, rank_sp = _cal_fitness(_candidate_matrix(source_candidates), source_cv - initial_e.reshape(1, -1))

        max_k = max(1, int(params.generations) - 1)
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            eps_n = _reduce_boundary(initial_e, k=generation, max_k=max_k)

            # Tournament over a mixed parent pool [TargetPop, SourcePop]
            tournament_fit = np.hstack([rank_tp, rank_sp])
            mating_pool = tournament_selection(2, pop_size, tournament_fit)
            merged_parents = np.vstack([target_pop, source_pop])
            parent_vectors = merged_parents[mating_pool]
            offspring = _sbx_mutation(parent_vectors, lower, upper)
            if offspring.shape[0] > pop_size:
                offspring = offspring[:pop_size]
            elif offspring.shape[0] < pop_size and offspring.shape[0] > 0:
                short = pop_size - offspring.shape[0]
                fill = offspring[np.random.randint(0, offspring.shape[0], size=short)]
                offspring = np.vstack([offspring, fill])
            elif offspring.shape[0] == 0:
                offspring = np.random.uniform(lower, upper, size=(pop_size, dimensions))

            offspring_candidates = _evaluate_population(offspring, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            offspring_cv = _constraint_violation_vector(offspring_candidates, model).reshape(-1, 1)

            # Target task environmental selection
            target_pop, target_candidates, rank_tp = _environmental_selection(
                np.vstack([target_pop, offspring]),
                target_candidates + offspring_candidates,
                pop_size,
                np.vstack([target_cv, offspring_cv]),
            )

            # Source task environmental selection with epsilon-relaxed constraints
            source_pop, source_candidates, rank_sp = _environmental_selection(
                np.vstack([source_pop, offspring]),
                source_candidates + offspring_candidates,
                pop_size,
                np.vstack([source_cv - eps_n.reshape(1, -1), offspring_cv - eps_n.reshape(1, -1)]),
            )

            target_cv = _constraint_violation_vector(target_candidates, model).reshape(-1, 1)
            source_cv = _constraint_violation_vector(source_candidates, model).reshape(-1, 1)

            if params.compute_metrics:
                target_obj = _candidate_matrix(target_candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, target_obj, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, target_obj, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=target_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(target_candidates)
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


def run_mfo_spea2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    use_legacy_runner = bool(params.extra.get("legacyPathRunner", False))
    if (not use_legacy_runner) or int(params.fleet_size) > 1:
        return _run_fleet_mfo_spea2(model, params)
    # Legacy-path fallback keeps benchmark compatibility for MFO-SPEA2 names;
    # a dedicated path-native version can be added later.
    from uav_benchmark.algorithms.nsga2 import run_nsga2

    return run_nsga2(model, params)
