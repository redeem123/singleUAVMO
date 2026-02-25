from __future__ import annotations

"""GCNMOEA runner adapted for this benchmark.

Core workflow is adapted from PlatEMO GCNMOEA implementation:
- reference-vector-guided environmental selection,
- graph construction from decision correlations,
- graph-neighborhood variation with self-attention/DE operators,
- fallback GA branch with an alternate environmental selection path.
"""

import copy
import time
from collections import deque
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
from uav_benchmark.core.nsga2_ops import n_d_sort, tournament_selection
from uav_benchmark.core.nsga3_ops import uniform_point
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


def _clone_candidate(candidate: Candidate, vector: np.ndarray | None = None) -> Candidate:
    cloned_details = copy.deepcopy(candidate.details) if isinstance(candidate.details, dict) else {}
    return Candidate(
        vector=np.asarray(vector if vector is not None else candidate.vector, dtype=float).copy(),
        objective=np.asarray(candidate.objective, dtype=float).copy(),
        details=cloned_details,
    )


def _safe_normalize(matrix: np.ndarray) -> np.ndarray:
    data = np.asarray(matrix, dtype=float)
    if data.size == 0:
        return data.reshape(0, 0)
    finite_mask = np.isfinite(data)
    if not np.all(finite_mask):
        col_max = np.zeros(data.shape[1], dtype=float)
        for col in range(data.shape[1]):
            values = data[finite_mask[:, col], col]
            if values.size > 0:
                col_max[col] = float(np.max(values))
        penalties = np.sum(~finite_mask, axis=1, keepdims=True).astype(float)
        replacement = col_max.reshape(1, -1) + 1e6 + penalties
        data = np.where(finite_mask, data, replacement)
    lo = np.min(data, axis=0)
    hi = np.max(data, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (data - lo) / span


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.ndim != 2:
        a = a.reshape(1, -1)
    if b.ndim != 2:
        b = b.reshape(1, -1)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True).T
    denom = np.maximum(a_norm * b_norm, 1e-12)
    return (a @ b.T) / denom


def _density_estimate(candidates: list[Candidate], weights: np.ndarray) -> np.ndarray:
    pop_obj = _candidate_matrix(candidates)
    if pop_obj.size == 0:
        return np.zeros(0, dtype=float)
    norm_obj = _safe_normalize(pop_obj)
    sim = _cosine_similarity(norm_obj, np.asarray(weights, dtype=float))
    region = np.argmax(sim, axis=1)
    counts = np.bincount(region, minlength=weights.shape[0]).astype(float)
    return counts[region]


def _spd_sort(pop_obj: np.ndarray, d1: np.ndarray, d2: np.ndarray, region: np.ndarray, n_sort: int) -> tuple[np.ndarray, int]:
    n_points, n_obj = pop_obj.shape
    front_no = np.full(n_points, np.inf, dtype=float)
    max_front = 0
    target = min(int(n_sort), n_points)

    while int(np.sum(np.isfinite(front_no))) < target:
        max_front += 1
        dominated = np.isfinite(front_no).copy()
        for i in range(n_points):
            if dominated[i]:
                continue
            for j in range(i + 1, n_points):
                if dominated[j]:
                    continue
                domi = 0
                for m in range(n_obj):
                    if pop_obj[i, m] < pop_obj[j, m]:
                        if domi == -1:
                            domi = 0
                            break
                        domi = 1
                    elif pop_obj[i, m] > pop_obj[j, m]:
                        if domi == 1:
                            domi = 0
                            break
                        domi = -1
                if domi == 0 and region[i] == region[j]:
                    lhs = d1[i] + 5.0 * d2[i]
                    rhs = d1[j] + 5.0 * d2[j]
                    if lhs < rhs:
                        domi = 1
                    elif lhs > rhs:
                        domi = -1
                if domi == 1:
                    dominated[j] = True
                elif domi == -1:
                    dominated[i] = True
                    break
            if not dominated[i]:
                front_no[i] = float(max_front)
    return front_no, max_front


def _environmental_selection(
    vectors: np.ndarray,
    candidates: list[Candidate],
    weights: np.ndarray,
    n_keep: int,
) -> tuple[np.ndarray, list[Candidate], np.ndarray, np.ndarray]:
    total = len(candidates)
    if total == 0 or n_keep <= 0:
        empty = np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=float)
        return empty, [], np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    if total <= n_keep:
        front_no, _ = n_d_sort(_candidate_matrix(candidates).copy(), None, total)
        d2 = _density_estimate(candidates, weights)
        return vectors.copy(), list(candidates), np.asarray(front_no, dtype=float), d2

    pop_obj = _candidate_matrix(candidates)
    norm_obj = _safe_normalize(pop_obj)
    sim = _cosine_similarity(norm_obj, weights)
    sim = np.clip(sim, -1.0, 1.0)

    norm_p = np.linalg.norm(norm_obj, axis=1)
    d1_mat = norm_p[:, None] * sim
    d2_mat = norm_p[:, None] * np.sqrt(np.maximum(0.0, 1.0 - sim**2))
    region = np.argmin(d2_mat, axis=1)
    d2 = np.min(d2_mat, axis=1)
    d1 = d1_mat[np.arange(total), region]

    nd_mask = n_d_sort(norm_obj.copy(), None, 1)[0] == 1
    nd_idx = np.where(nd_mask)[0]
    if nd_idx.size > 0:
        extreme_local = np.argmax(norm_obj[nd_idx], axis=0)
        extreme_idx = nd_idx[np.unique(extreme_local)]
        d1[extreme_idx] = 0.0
        d2[extreme_idx] = 0.0

    front_no, max_front = _spd_sort(norm_obj, d1, d2, region + 1, n_keep)
    next_mask = front_no < max_front

    last = np.where(front_no == max_front)[0]
    if last.size > 0 and int(np.sum(next_mask)) < n_keep:
        order = last[np.argsort(d2[last])]
        need = n_keep - int(np.sum(next_mask))
        next_mask[order[:need]] = True

    selected = np.where(next_mask)[0]
    if selected.size < n_keep:
        remain = np.setdiff1d(np.arange(total, dtype=int), selected, assume_unique=False)
        if remain.size > 0:
            fill = remain[np.argsort(d2[remain])]
            selected = np.hstack([selected, fill[: n_keep - selected.size]])
    elif selected.size > n_keep:
        selected = selected[:n_keep]

    selected = selected.astype(int, copy=False)
    return (
        vectors[selected],
        [candidates[int(idx)] for idx in selected],
        front_no[selected],
        d2[selected],
    )


def _level_sort(candidates: list[Candidate], n_levels: int) -> np.ndarray:
    pop_obj = _candidate_matrix(candidates)
    if pop_obj.size == 0:
        return np.zeros(0, dtype=int)
    zmax = np.max(pop_obj, axis=0)
    zmin = np.min(pop_obj, axis=0)
    interval = (zmax - zmin) / max(1, int(n_levels))
    levels = np.zeros(pop_obj.shape[0], dtype=int)

    for idx in range(pop_obj.shape[0]):
        t = 0
        while True:
            t += 1
            leveled = True
            for m in range(pop_obj.shape[1]):
                bound = zmin[m] + (t + 1) * interval[m]
                if pop_obj[idx, m] > bound:
                    leveled = False
                    break
            if leveled or t >= max(1, int(n_levels)):
                levels[idx] = t
                break
    return levels


def _environmental_selection1(
    vectors: np.ndarray,
    candidates: list[Candidate],
    weights: np.ndarray,
    n_keep: int,
    objective_count: int,
) -> tuple[np.ndarray, list[Candidate]]:
    total = len(candidates)
    if total == 0 or n_keep <= 0:
        empty = np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=float)
        return empty, []
    if total <= n_keep:
        return vectors.copy(), list(candidates)

    pop_obj = _candidate_matrix(candidates)
    front_no, max_front = n_d_sort(pop_obj.copy(), None, total)
    nd_idx: list[int] = []
    for front in range(1, int(max_front) + 1):
        members = np.where(front_no == front)[0].tolist()
        nd_idx.extend(members)
        if len(nd_idx) >= n_keep:
            break
    if not nd_idx:
        order = np.argsort(np.sum(pop_obj, axis=1))
        selected = order[:n_keep]
        return vectors[selected], [candidates[int(i)] for i in selected]

    nd_idx_arr = np.asarray(nd_idx, dtype=int)
    nd_candidates = [candidates[int(i)] for i in nd_idx_arr]
    levels = _level_sort(nd_candidates, n_levels=2 * max(1, int(objective_count)))
    lvl_idx: list[int] = []
    for level in range(1, 2 * max(1, int(objective_count)) + 1):
        members = np.where(levels == level)[0].tolist()
        lvl_idx.extend(members)
        if len(lvl_idx) >= n_keep:
            break
    if not lvl_idx:
        lvl_idx = list(range(min(len(nd_candidates), n_keep)))

    pool_global = nd_idx_arr[np.asarray(lvl_idx, dtype=int)].tolist()
    selected_global: list[int] = []
    for wi in range(min(weights.shape[0], n_keep)):
        if not pool_global:
            break
        pool_obj = _safe_normalize(pop_obj[np.asarray(pool_global, dtype=int)])
        sim = _cosine_similarity(pool_obj, np.asarray(weights[wi], dtype=float).reshape(1, -1)).reshape(-1)
        local = int(np.argmax(sim))
        selected_global.append(int(pool_global.pop(local)))

    if len(selected_global) < n_keep:
        fill_pool = [idx for idx in nd_idx_arr.tolist() if idx not in selected_global]
        if len(fill_pool) < (n_keep - len(selected_global)):
            all_pool = [idx for idx in range(total) if idx not in selected_global]
            fill_pool.extend(all_pool)
        selected_global.extend(fill_pool[: n_keep - len(selected_global)])

    selected = np.asarray(selected_global[:n_keep], dtype=int)
    return vectors[selected], [candidates[int(i)] for i in selected]


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
    fill = offspring[np.random.randint(0, offspring.shape[0], size=short)] if offspring.shape[0] > 0 else np.random.uniform(lower, upper, size=(short, lower.size))
    return np.vstack([offspring, fill])


def _operator_de_vector(base: np.ndarray, p2: np.ndarray, p3: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
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
    return ((g_old >= g_new) & (((cv_old <= epsilon_k) & (cv_new <= epsilon_k)) | np.isclose(cv_old, cv_new))) | (cv_new < cv_old)


def _run_fleet_gcnmoea(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
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

        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

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
                            if clique_local.size > 0:
                                black_focus = black[clique_local]
                            else:
                                black_focus = black
                            if black_focus.size < 2:
                                pool = [int(node) for node in black.tolist() if int(node) not in set(black_focus.tolist())]
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
                        candidates[parent_idx[0]] = _clone_candidate(offspring_candidates[0], vector=offspring_vectors[0])
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
                mating_pool = tournament_selection(2, pop_size, np.asarray(front_no, dtype=float), np.asarray(d2, dtype=float))
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
    use_legacy_runner = bool(params.extra.get("legacyPathRunner", False))
    if (not use_legacy_runner) or int(params.fleet_size) > 1:
        return _run_fleet_gcnmoea(model, params)
    # Legacy-path fallback keeps benchmark compatibility for GCNMOEA names;
    # a dedicated path-native implementation can be added later.
    from uav_benchmark.algorithms.nsga2 import run_nsga2

    return run_nsga2(model, params)
