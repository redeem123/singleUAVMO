"""CMOSMA runner adapted for this benchmark.

Core workflow is inherited from the PlatEMO CMOSMA implementation
(C. He et al., 2022; PlatEMO educational codebase), then adapted to this
repository's constrained fleet mission evaluator and artifact pipeline.
"""

from __future__ import annotations

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
from uav_benchmark.algorithms.shared.pareto_utils import _pairwise_distance, _sanitize_objectives
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


def _cal_fitness(pop_obj: np.ndarray, pop_con: np.ndarray | None = None) -> np.ndarray:
    """Compute CMOSMA/SPEA2-style fitness with optional constraint-domination."""
    objective = _sanitize_objectives(pop_obj)
    n_points = int(objective.shape[0])
    if n_points == 0:
        return np.zeros(0, dtype=float)

    if pop_con is None:
        cv = np.zeros(n_points, dtype=float)
    else:
        cv = np.sum(np.maximum(0.0, np.asarray(pop_con, dtype=float)), axis=1)

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
    return raw + density


def _truncation(pop_obj: np.ndarray, n_remove: int) -> np.ndarray:
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
    use_constraints: bool,
    model: dict[str, Any],
) -> tuple[np.ndarray, list[Candidate], np.ndarray]:
    total = len(candidates)
    if total == 0 or n_keep <= 0:
        return np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=float), [], np.zeros(0, dtype=float)

    objective = _candidate_matrix(candidates)
    constraints = _constraint_violation_vector(candidates, model).reshape(-1, 1) if use_constraints else None
    fitness = _cal_fitness(objective, constraints)
    if total <= n_keep:
        order = np.argsort(fitness, kind="mergesort")
        return vectors[order], [candidates[int(i)] for i in order], fitness[order]

    next_mask = fitness < 1.0
    selected_count = int(np.sum(next_mask))
    if selected_count < n_keep:
        rank = np.argsort(fitness, kind="mergesort")
        next_mask[:] = False
        next_mask[rank[:n_keep]] = True
    elif selected_count > n_keep:
        delete_mask = _truncation(objective[next_mask], selected_count - n_keep)
        idx = np.flatnonzero(next_mask)
        next_mask[idx[delete_mask]] = False

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
    order = np.argsort(fitness[selected], kind="mergesort")
    selected = selected[order]
    return vectors[selected], [candidates[int(i)] for i in selected], fitness[selected]


def _initialize_som(shape: tuple[int, ...], h_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    axes = [np.arange(1, int(size) + 1, dtype=float) for size in shape]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(shape))
    n_neurons = int(grid.shape[0])
    if n_neurons <= 1:
        return np.zeros((n_neurons, n_neurons), dtype=float), np.zeros((n_neurons, 0), dtype=int)

    diff = grid[:, np.newaxis, :] - grid[np.newaxis, :, :]
    latent_distance = np.linalg.norm(diff, axis=2)
    order = np.argsort(latent_distance, axis=1)
    neighbor_count = max(1, min(int(h_neighbors), n_neurons - 1))
    neighbors = order[:, 1 : 1 + neighbor_count]
    return latent_distance, neighbors.astype(int)


def _update_som(
    training_set: np.ndarray,
    weights: np.ndarray,
    fe: int,
    max_fe: int,
    latent_distance: np.ndarray,
    sigma0: float,
    tau0: float,
) -> np.ndarray:
    if training_set.size == 0:
        return weights
    updated = np.asarray(weights, dtype=float).copy()
    max_fe_safe = max(1.0, float(max_fe))
    for sample_idx in range(training_set.shape[0]):
        decay = max(0.0, 1.0 - float(fe + sample_idx + 1) / max_fe_safe)
        sigma = float(sigma0) * decay
        tau = float(tau0) * decay
        if sigma <= 0.0 or tau <= 0.0:
            continue
        sample = training_set[sample_idx]
        nearest = int(np.argmin(np.linalg.norm(updated - sample.reshape(1, -1), axis=1)))
        neighborhood = latent_distance[nearest] < sigma
        if not np.any(neighborhood):
            continue
        influence = np.exp(-latent_distance[nearest, neighborhood]).reshape(-1, 1)
        updated[neighborhood] = updated[neighborhood] + tau * influence * (
            sample.reshape(1, -1) - updated[neighborhood]
        )
    return updated


def _associate(population_vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_points = int(population_vectors.shape[0])
    available_solutions = list(range(n_points))
    available_neurons = list(range(n_points))
    assignment = np.zeros(n_points, dtype=int)

    for _ in range(n_points):
        pick = int(np.random.randint(0, len(available_solutions)))
        solution_idx = available_solutions.pop(pick)
        neuron_idx_array = np.asarray(available_neurons, dtype=int)
        diff = weights[neuron_idx_array] - population_vectors[solution_idx].reshape(1, -1)
        nearest_local = int(np.argmin(np.linalg.norm(diff, axis=1)))
        neuron_idx = available_neurons.pop(nearest_local)
        assignment[neuron_idx] = solution_idx
    return assignment


def _mating_pool(xu: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    n_points = int(xu.size)
    pool = np.zeros(n_points, dtype=int)
    all_indices = np.arange(n_points, dtype=int)
    for neuron_idx in range(n_points):
        q = xu[neighbors[neuron_idx]] if neighbors.size > 0 and np.random.rand() < 0.9 else all_indices
        if q.size == 0:
            pool[neuron_idx] = int(np.random.randint(0, n_points))
        else:
            pool[neuron_idx] = int(q[np.random.randint(0, q.size)])
    return pool


def _new_rows(current: np.ndarray, previous: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    if current.size == 0:
        return current.reshape(0, previous.shape[1] if previous.ndim == 2 else 0)
    if previous.size == 0:
        return current.copy()
    keep = np.ones(current.shape[0], dtype=bool)
    for idx in range(current.shape[0]):
        row = current[idx]
        if np.any(np.all(np.isclose(previous, row.reshape(1, -1), rtol=0.0, atol=atol), axis=1)):
            keep[idx] = False
    return current[keep]


def _som_shape(population: int, latent_dim: int, extra: dict[str, Any]) -> tuple[int, ...]:
    raw = extra.get("cmosmaD")
    if isinstance(raw, int):
        value = max(1, int(raw))
        return tuple(value for _ in range(latent_dim))
    if isinstance(raw, str) and raw.strip():
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if len(tokens) == latent_dim:
            parsed = [max(1, int(float(token))) for token in tokens]
            return tuple(parsed)
    if isinstance(raw, (list, tuple)) and len(raw) == latent_dim:
        try:
            parsed = [max(1, int(float(item))) for item in raw]
            return tuple(parsed)
        except (TypeError, ValueError):
            pass
    base = max(1, int(np.ceil(float(max(1, population)) ** (1.0 / float(max(1, latent_dim))))))
    return tuple(base for _ in range(latent_dim))


def _run_fleet_cmosma(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    # NOTE: This follows PlatEMO CMOSMA flow:
    # 1) two populations (FP/AP), 2) two SOM updates, 3) association and
    # neighborhood mating, 4) GA evolution, 5) constrained/unconstrained
    # environmental selection. Adapted for fleet mission objectives.
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
    latent_dim = max(1, objective_count - 1)
    som_shape = _som_shape(int(params.population), latent_dim, params.extra if isinstance(params.extra, dict) else {})
    pop_size = int(np.prod(np.asarray(som_shape, dtype=int)))
    pop_size = max(2, pop_size)
    tau0 = float(params.extra.get("cmosmaTau0", 0.7))
    h_neighbors = int(params.extra.get("cmosmaH", 5))
    metric_interval = int(params.extra.get("metricInterval", 20))
    sigma0 = float(np.sqrt(np.sum(np.square(np.asarray(som_shape, dtype=float))) / float(latent_dim)) / 2.0)

    latent_distance, neighbors = _initialize_som(som_shape, h_neighbors)
    latent_distance2, neighbors2 = _initialize_som(som_shape, h_neighbors)

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
        fp_vectors = np.random.uniform(lower, upper, size=(pop_size, dimensions))
        ap_vectors = np.random.uniform(lower, upper, size=(pop_size, dimensions))
        fp_candidates = _evaluate_population(fp_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        ap_candidates = _evaluate_population(ap_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

        s_train = fp_vectors.copy()
        s_train2 = ap_vectors.copy()
        w = s_train.copy()
        w2 = s_train2.copy()

        max_fe = max(1, int(params.generations) * pop_size)
        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )

        for generation in range(1, params.generations + 1):
            fe = int((generation - 1) * pop_size)
            w = _update_som(s_train, w, fe, max_fe, latent_distance, sigma0, tau0)
            w2 = _update_som(s_train2, w2, fe, max_fe, latent_distance2, sigma0, tau0)

            xu = _associate(fp_vectors, w)
            xu2 = _associate(ap_vectors, w2)
            mating1 = _mating_pool(xu, neighbors)
            mating2 = _mating_pool(xu2, neighbors2)

            prev_fp = fp_vectors.copy()
            prev_ap = ap_vectors.copy()
            parent1 = np.vstack([fp_vectors[xu], fp_vectors[mating1]])
            parent2 = np.vstack([ap_vectors[xu2], ap_vectors[mating2]])
            offspring1 = _sbx_mutation(parent1, lower, upper)
            offspring2 = _sbx_mutation(parent2, lower, upper)
            offspring1_candidates = _evaluate_population(
                offspring1, model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
            offspring2_candidates = _evaluate_population(
                offspring2, model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )

            merged_fp_vectors = np.vstack([fp_vectors, offspring1, offspring2])
            merged_fp_candidates = fp_candidates + offspring1_candidates + offspring2_candidates
            fp_vectors, fp_candidates, _ = _environmental_selection(
                merged_fp_vectors,
                merged_fp_candidates,
                pop_size,
                use_constraints=True,
                model=model,
            )

            merged_ap_vectors = np.vstack([ap_vectors, offspring1, offspring2])
            merged_ap_candidates = ap_candidates + offspring1_candidates + offspring2_candidates
            ap_vectors, ap_candidates, _ = _environmental_selection(
                merged_ap_vectors,
                merged_ap_candidates,
                pop_size,
                use_constraints=False,
                model=model,
            )

            s_train = _new_rows(fp_vectors, prev_fp)
            s_train2 = _new_rows(ap_vectors, prev_ap)

            if params.compute_metrics:
                final_obj = _candidate_matrix(fp_candidates)
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
            final_candidates=fp_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            run_metadata={
                "somPopulation": int(pop_size),
                "somGridDims": np.asarray(som_shape, dtype=float),
                "somTau0": float(tau0),
                "somNeighbors": int(max(0, neighbors.shape[1] if neighbors.ndim == 2 else 0)),
            },
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(fp_candidates)
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


def run_cmosma(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_cmosma(model, params)
