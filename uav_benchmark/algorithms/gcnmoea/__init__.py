"""GCNMOEA runner adapted for this benchmark.

Core workflow is adapted from PlatEMO GCNMOEA implementation:
- reference-vector-guided environmental selection,
- graph construction from decision correlations,
- graph-neighborhood variation with self-attention/DE operators,
- fallback GA branch with an alternate environmental selection path.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from uav_benchmark.algorithms.gcnmoea.selection import (
    _cosine_similarity,
    _density_estimate,
    _environmental_selection,
    _environmental_selection1,
)
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
from uav_benchmark.algorithms.shared.pareto_utils import _clone_candidate
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort, tournament_selection
from uav_benchmark.core.nsga3_ops import uniform_point
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir

__all__ = ["_cosine_similarity", "run_gcnmoea"]


def _overall_cv(cv: np.ndarray) -> np.ndarray:
    values = np.asarray(cv, dtype=float).reshape(-1)
    values = np.where(values > 0.0, values, 0.0)
    return np.abs(values)


def _build_adjacency(decisions: np.ndarray, max_connections: int, prefer_descending: bool) -> np.ndarray:
    n_nodes = decisions.shape[0]
    if n_nodes <= 1:
        return np.zeros((n_nodes, n_nodes), dtype=int)
    corr = np.corrcoef(decisions)
    if corr.ndim != 2 or corr.shape != (n_nodes, n_nodes):
        corr = np.eye(n_nodes, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
    order = np.argsort(corr, axis=None)
    if prefer_descending:
        order = order[::-1]

    degree = np.zeros(n_nodes, dtype=int)
    for flat in order.tolist():
        i, j = np.unravel_index(flat, corr.shape)
        if i == j:
            continue
        if degree[i] >= max_connections or degree[j] >= max_connections:
            continue
        if adjacency[i, j] == 1:
            continue
        adjacency[i, j] = 1
        adjacency[j, i] = 1
        degree[i] += 1
        degree[j] += 1
    return adjacency


def _neighbors(adjacency: np.ndarray, node: int) -> np.ndarray:
    if adjacency.size == 0:
        return np.zeros(0, dtype=int)
    return np.where(adjacency[int(node)] > 0)[0].astype(int)


def _sample_nodes(nodes: np.ndarray, max_count: int) -> np.ndarray:
    arr = np.asarray(nodes, dtype=int).reshape(-1)
    if arr.size <= max_count:
        return arr
    pick = np.random.choice(arr, size=max_count, replace=False)
    return np.asarray(pick, dtype=int)


def _adj_list_from_matrix(adjacency: np.ndarray) -> list[set[int]]:
    n_nodes = adjacency.shape[0]
    out: list[set[int]] = []
    for idx in range(n_nodes):
        out.append(set(np.where(adjacency[idx] > 0)[0].astype(int).tolist()))
    return out


def _bron_kerbosch_max(adjacency: np.ndarray) -> np.ndarray:
    adj_list = _adj_list_from_matrix(adjacency)
    n_nodes = len(adj_list)
    if n_nodes == 0:
        return np.zeros(0, dtype=int)
    best: set[int] = set()

    def _recurse(r: set[int], p: set[int], x: set[int]) -> None:
        nonlocal best
        if not p and not x:
            if len(r) > len(best):
                best = set(r)
            return
        if len(r) + len(p) <= len(best):
            return
        union_px = p | x
        if union_px:
            pivot = max(union_px, key=lambda node: len(p & adj_list[node]))
            iterate = list(p - adj_list[pivot])
        else:
            iterate = list(p)
        iterate.sort(key=lambda node: len(p & adj_list[node]), reverse=True)
        for node in iterate:
            _recurse(r | {node}, p & adj_list[node], x & adj_list[node])
            p.discard(node)
            x.add(node)

    _recurse(set(), set(range(n_nodes)), set())
    return np.asarray(sorted(best), dtype=int)


def _operator_ga_vectors(parents: np.ndarray, lower: np.ndarray, upper: np.ndarray, out_size: int) -> np.ndarray:
    parent_matrix = np.asarray(parents, dtype=float)
    if parent_matrix.ndim != 2 or parent_matrix.shape[0] == 0:
        return np.zeros((0, lower.size), dtype=float)
    offspring = _sbx_mutation(parent_matrix, lower, upper)
    if offspring.shape[0] >= out_size:
        return offspring[:out_size]
    short = out_size - offspring.shape[0]
    fill = (
        offspring[np.random.randint(0, offspring.shape[0], size=short)]
        if offspring.shape[0] > 0
        else np.random.uniform(lower, upper, size=(short, lower.size))
    )
    return np.vstack([offspring, fill])


def _operator_de_vector(
    base: np.ndarray, p2: np.ndarray, p3: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    f_scale = 0.5
    cr_rate = 0.9
    trial = np.asarray(base, dtype=float).copy()
    mutant = np.asarray(base, dtype=float) + f_scale * (np.asarray(p2, dtype=float) - np.asarray(p3, dtype=float))
    mask = np.random.rand(base.size) < cr_rate
    j_rand = int(np.random.randint(0, base.size))
    mask[j_rand] = True
    trial[mask] = mutant[mask]
    return np.clip(trial, lower, upper)


def _self_attention_vectors(
    parents: np.ndarray,
    adjacency: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    w_matrix: np.ndarray,
    a_vector: np.ndarray,
) -> np.ndarray:
    x = np.asarray(parents, dtype=float)
    n_nodes, dim = x.shape
    if n_nodes == 0:
        return np.zeros((0, dim), dtype=float)

    w = np.asarray(w_matrix, dtype=float)
    if w.shape != (dim, dim):
        w = np.random.rand(dim, dim)
    a = np.asarray(a_vector, dtype=float).reshape(-1)
    if a.size != 2 * dim:
        a = np.random.rand(2 * dim)

    xw = x @ w
    e = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j or adjacency[i, j] <= 0:
                continue
            pair = np.concatenate([xw[i], xw[j]])
            e[i, j] = np.exp(np.tanh(float(pair @ a)))

    alpha = np.zeros_like(e)
    for i in range(n_nodes):
        row_sum = float(np.sum(e[i]))
        if row_sum <= 1e-12:
            neigh = np.where(adjacency[i] > 0)[0]
            if neigh.size > 0:
                alpha[i, neigh] = 1.0 / neigh.size
            else:
                alpha[i, i] = 1.0
        else:
            alpha[i] = e[i] / row_sum
    h = alpha @ xw
    return np.clip(h, lower, upper)


def _pagerank(adjacency: np.ndarray, damping: float = 0.85, iterations: int = 40) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return np.zeros(0, dtype=float)
    adj = np.asarray(adjacency, dtype=float)
    out_degree = np.sum(adj, axis=1)
    transition = np.zeros_like(adj)
    for idx in range(n_nodes):
        if out_degree[idx] > 0:
            transition[idx] = adj[idx] / out_degree[idx]
        else:
            transition[idx] = 1.0 / n_nodes
    rank = np.full(n_nodes, 1.0 / n_nodes, dtype=float)
    base = (1.0 - damping) / n_nodes
    for _ in range(max(1, int(iterations))):
        rank = base + damping * (transition.T @ rank)
    return rank


def _betweenness_sampled(adjacency: np.ndarray, max_sources: int = 16) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    if n_nodes <= 2:
        return np.zeros(n_nodes, dtype=float)

    neigh = [np.where(adjacency[i] > 0)[0].astype(int).tolist() for i in range(n_nodes)]
    sources = np.arange(n_nodes, dtype=int)
    if sources.size > max_sources:
        sources = np.random.choice(sources, size=max_sources, replace=False).astype(int)

    bc = np.zeros(n_nodes, dtype=float)
    for s in sources.tolist():
        stack: list[int] = []
        pred: list[list[int]] = [[] for _ in range(n_nodes)]
        sigma = np.zeros(n_nodes, dtype=float)
        sigma[s] = 1.0
        dist = -np.ones(n_nodes, dtype=int)
        dist[s] = 0
        queue: deque[int] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in neigh[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = np.zeros(n_nodes, dtype=float)
        while stack:
            w = stack.pop()
            if sigma[w] <= 0:
                continue
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    if sources.size > 0:
        bc *= n_nodes / max(1, sources.size)
    max_val = float(np.max(bc))
    if max_val > 0:
        bc = bc / max_val
    return bc


def _finite_min_objective(candidates: list[Candidate], objective_count: int) -> np.ndarray:
    matrix = _candidate_matrix(candidates)
    if matrix.size == 0:
        return np.zeros(objective_count, dtype=float)
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return np.zeros(objective_count, dtype=float)
    return np.min(finite, axis=0)


def _tchebycheff_values(obj: np.ndarray, z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    objective = np.asarray(obj, dtype=float)
    ideal = np.asarray(z, dtype=float).reshape(1, -1)
    w = np.asarray(weights, dtype=float)
    if objective.ndim != 2:
        objective = objective.reshape(1, -1)
    if w.ndim != 2:
        w = w.reshape(1, -1)
    if objective.shape[0] == 1 and w.shape[0] > 1:
        objective = np.repeat(objective, w.shape[0], axis=0)
    return np.max(np.abs(objective - ideal) * np.maximum(w, 1e-12), axis=1)


def _replacement_mask(
    g_old: np.ndarray,
    g_new: np.ndarray,
    cv_old: np.ndarray,
    cv_new: np.ndarray,
    epsilon_k: float,
) -> np.ndarray:
    return ((g_old >= g_new) & (((cv_old <= epsilon_k) & (cv_new <= epsilon_k)) | np.isclose(cv_old, cv_new))) | (
        cv_new < cv_old
    )


def _run_fleet_gcnmoea(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
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

    weights, adjusted_population = uniform_point(int(params.population), objective_count, "NBI")
    pop_size = int(adjusted_population)
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-6)
    if weights.shape[0] != pop_size:
        weights = weights[:pop_size]

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dimensions = int(lower.size)
    metric_interval = int(params.extra.get("metricInterval", 20))

    max_connections = int(params.extra.get("gcnMaxConnections", 15))
    max_green = int(params.extra.get("gcnMaxGreenNeighbors", 8))
    max_black = int(params.extra.get("gcnMaxBlackNeighbors", 10))
    graph_rounds = int(params.extra.get("gcnNumSeeds", min(20, pop_size)))
    attention_reinit_prob = float(params.extra.get("gcnAttentionReinitProb", 0.2))
    epsilon_k = float(params.extra.get("gcnEpsilon", 0.0))

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
        z = _finite_min_objective(candidates, objective_count)
        population, candidates, front_no, d2 = _environmental_selection(population, candidates, weights, pop_size)

        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )

        for generation in range(1, params.generations + 1):
            decisions = population.copy()
            use_descending = bool(np.random.rand() < 0.3)
            adjacency = _build_adjacency(decisions, max_connections=max_connections, prefer_descending=use_descending)

            use_graph_phase = bool(np.random.rand() < 0.4) or (generation <= int(np.ceil(0.5 * params.generations)))
            if use_graph_phase:
                red_population: list[int] = []
                green_population: list[np.ndarray] = []
                att_w = np.random.rand(dimensions, dimensions)
                att_a = np.random.rand(2 * dimensions)

                for _ in range(max(1, min(graph_rounds, pop_size))):
                    red = int(np.random.randint(0, pop_size))
                    red_population.append(red)

                    green = _sample_nodes(_neighbors(adjacency, red), max_green)
                    green_population.append(green)
                    if green.size == 0:
                        continue

                    if np.random.rand() < attention_reinit_prob:
                        att_w = np.random.rand(dimensions, dimensions)
                        att_a = np.random.rand(2 * dimensions)

                    for green_node in green.tolist():
                        black = _neighbors(adjacency, green_node)
                        black = black[black != red]
                        black = _sample_nodes(black, max_black)

                        if black.size > 1:
                            sub_adj = adjacency[np.ix_(black, black)]
                            clique_local = _bron_kerbosch_max(sub_adj)
                            black_focus = black[clique_local] if clique_local.size > 0 else black
                            if black_focus.size < 2:
                                pool = [
                                    int(node) for node in black.tolist() if int(node) not in set(black_focus.tolist())
                                ]
                                np.random.shuffle(pool)
                                need = max(0, 2 - int(black_focus.size))
                                black_focus = np.concatenate([black_focus, np.asarray(pool[:need], dtype=int)])
                            if black_focus.size == 0:
                                continue

                            parent_idx = np.concatenate([np.asarray([green_node], dtype=int), black_focus]).astype(int)
                            parent_idx = np.asarray(list(dict.fromkeys(parent_idx.tolist())), dtype=int)
                            parent_vectors = population[parent_idx]

                            if parent_idx.size == 2:
                                offspring_vectors = _operator_ga_vectors(parent_vectors, lower, upper, out_size=2)
                            else:
                                local_adj = adjacency[np.ix_(parent_idx, parent_idx)]
                                offspring_vectors = _self_attention_vectors(
                                    parent_vectors,
                                    local_adj,
                                    lower,
                                    upper,
                                    att_w,
                                    att_a,
                                )
                            offspring_candidates = _evaluate_population(
                                offspring_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints
                            )
                            off_obj = _candidate_matrix(offspring_candidates)
                            finite_off = off_obj[np.all(np.isfinite(off_obj), axis=1)]
                            if finite_off.size > 0:
                                z = np.minimum(z, np.min(finite_off, axis=0))

                            old_candidates = [candidates[int(idx)] for idx in parent_idx.tolist()]
                            old_obj = _candidate_matrix(old_candidates)
                            old_cv = _overall_cv(_constraint_violation_vector(old_candidates, model))
                            new_cv = _overall_cv(_constraint_violation_vector(offspring_candidates, model))
                            g_old = _tchebycheff_values(old_obj, z, weights[parent_idx])
                            g_new = _tchebycheff_values(_candidate_matrix(offspring_candidates), z, weights[parent_idx])

                            replace = _replacement_mask(g_old, g_new, old_cv, new_cv, epsilon_k)
                            for local_idx in np.where(replace)[0].tolist():
                                global_idx = int(parent_idx[local_idx])
                                population[global_idx] = offspring_vectors[local_idx]
                                candidates[global_idx] = _clone_candidate(
                                    offspring_candidates[local_idx], vector=offspring_vectors[local_idx]
                                )
                        elif black.size == 1:
                            parent_idx = np.asarray([green_node, int(black[0])], dtype=int)
                            parent_vectors = population[parent_idx]
                            offspring_vectors = _operator_ga_vectors(parent_vectors, lower, upper, out_size=2)
                            offspring_candidates = _evaluate_population(
                                offspring_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints
                            )
                            merged_vectors = np.vstack([population, offspring_vectors])
                            merged_candidates = candidates + offspring_candidates
                            population, candidates, front_no, d2 = _environmental_selection(
                                merged_vectors, merged_candidates, weights, pop_size
                            )

                        population, candidates, front_no, d2 = _environmental_selection(
                            population, candidates, weights, pop_size
                        )

                pr = _pagerank(adjacency)
                degree = np.sum(adjacency, axis=1).astype(float)
                _ = degree  # kept for parity with the original centrality fusion
                bn = _betweenness_sampled(adjacency, max_sources=min(16, pop_size))
                node_weight = 0.5 * pr + 0.5 * bn

                for idx in range(len(red_population)):
                    red = int(red_population[idx])
                    green = green_population[idx]
                    if green.size == 0:
                        continue
                    if green.size <= 2:
                        best_green = int(green[np.argmax(node_weight[green])])
                        parent_idx = np.asarray([red, best_green], dtype=int)
                        offspring_vectors = _operator_ga_vectors(population[parent_idx], lower, upper, out_size=2)
                        offspring_candidates = _evaluate_population(
                            offspring_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints
                        )
                        population[parent_idx[0]] = offspring_vectors[0]
                        candidates[parent_idx[0]] = _clone_candidate(
                            offspring_candidates[0], vector=offspring_vectors[0]
                        )
                        second = 1 if offspring_vectors.shape[0] > 1 else 0
                        population[parent_idx[1]] = offspring_vectors[second]
                        candidates[parent_idx[1]] = _clone_candidate(
                            offspring_candidates[second], vector=offspring_vectors[second]
                        )
                    else:
                        order = green[np.argsort(-node_weight[green])]
                        p2 = int(order[0])
                        p3 = int(order[1])
                        child_vec = _operator_de_vector(population[red], population[p2], population[p3], lower, upper)
                        child_candidate = _evaluate_population(
                            child_vec.reshape(1, -1), model, fleet_size=fleet_size, n_waypoints=n_waypoints
                        )[0]

                        old_obj = _candidate_matrix([candidates[red]])
                        new_obj = _candidate_matrix([child_candidate])
                        old_cv = _overall_cv(_constraint_violation_vector([candidates[red]], model))
                        new_cv = _overall_cv(_constraint_violation_vector([child_candidate], model))
                        g_old = _tchebycheff_values(old_obj, z, weights[red : red + 1])
                        g_new = _tchebycheff_values(new_obj, z, weights[red : red + 1])
                        if bool(_replacement_mask(g_old, g_new, old_cv, new_cv, epsilon_k)[0]):
                            population[red] = child_vec
                            candidates[red] = _clone_candidate(child_candidate, vector=child_vec)

                    population, candidates, front_no, d2 = _environmental_selection(
                        population, candidates, weights, pop_size
                    )
            else:
                mating_pool = tournament_selection(
                    2, pop_size, np.asarray(front_no, dtype=float), np.asarray(d2, dtype=float)
                )
                offspring_vectors = _operator_ga_vectors(population[mating_pool], lower, upper, out_size=pop_size)
                offspring_candidates = _evaluate_population(
                    offspring_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints
                )
                merged_vectors = np.vstack([population, offspring_vectors])
                merged_candidates = candidates + offspring_candidates
                population, candidates = _environmental_selection1(
                    merged_vectors, merged_candidates, weights, pop_size, objective_count
                )
                front_no, _ = n_d_sort(_candidate_matrix(candidates).copy(), None, len(candidates))
                d2 = _density_estimate(candidates, weights)

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


def run_gcnmoea(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_gcnmoea(model, params)
