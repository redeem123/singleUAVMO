from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.metrics import HV_REF_MARGIN
from uav_benchmark.core.r2_archive import r2_archive_update


def _candidate_matrix(candidates: list[Candidate]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 1), dtype=float)
    return np.array([c.objective for c in candidates], dtype=float)


def _stack_or_empty(vectors: list[np.ndarray], width: int) -> np.ndarray:
    if not vectors:
        return np.zeros((0, width), dtype=float)
    return np.stack(vectors, axis=0)


def _grid_ids_from_bounds(matrix: np.ndarray, n_grid: int, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    span = maxs - mins
    span[span == 0] = 1.0
    normalized = (matrix - mins) / span
    grid_indices = np.clip((normalized * n_grid).astype(int), 0, n_grid - 1)
    multiplier = np.power(n_grid, np.arange(matrix.shape[1]))
    return np.sum(grid_indices * multiplier, axis=1)


def _finite_min(matrix: np.ndarray) -> np.ndarray:
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return np.zeros(matrix.shape[1])
    return np.min(finite, axis=0)


def _candidate_is_feasible(candidate: Candidate, index: int, objective: np.ndarray | None) -> bool:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    if "feasible" in details:
        return float(details.get("feasible", 0.0)) > 0.5
    if objective is not None and objective.ndim == 2 and index < objective.shape[0]:
        return bool(np.all(np.isfinite(objective[index])))
    return bool(np.all(np.isfinite(np.asarray(candidate.objective, dtype=float))))


def _candidate_feasible_flags(candidates: list[Candidate], objective: np.ndarray | None = None) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=float)
    return np.asarray(
        [_candidate_is_feasible(candidate, index, objective) for index, candidate in enumerate(candidates)],
        dtype=float,
    )


def _archive_front(candidates: list[Candidate], max_size: int) -> list[Candidate]:
    """Pareto archive pruning with crowding-based trimming."""
    from uav_benchmark.core.dominance import dominates

    if not candidates:
        return []
    objectives = _candidate_matrix(candidates)
    n = len(candidates)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(i + 1, n):
            if dominated[j]:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated[j] = True
            elif dominates(objectives[j], objectives[i]):
                dominated[i] = True
                break
    front = [candidates[i] for i in range(n) if not dominated[i]]
    if len(front) <= max_size:
        return front
    # Crowding distance trimming
    from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort

    obj = _candidate_matrix(front)
    front_no, _ = n_d_sort(obj.copy(), None, len(front))
    cd = crowding_distance(obj, front_no)
    keep_idx = np.argsort(-cd)[:max_size]
    return [front[i] for i in keep_idx]


def _r2_archive_from_candidates(
    archive_candidates: list[Candidate],
    new_candidates: list[Candidate],
    max_size: int,
    weights: np.ndarray,
    z_ideal: np.ndarray,
    eps_rel: float = 1e-8,
) -> tuple[list[Candidate], np.ndarray]:
    """R2-contribution archive pruning, preserving Candidate objects."""
    all_cands = list(archive_candidates) + list(new_candidates)
    if not all_cands:
        return [], z_ideal.copy()

    all_obj = np.stack([c.objective for c in all_cands], axis=0)
    all_vec = np.stack([c.vector for c in all_cands], axis=0)
    n_arch = len(archive_candidates)

    new_obj, new_vec, z_ideal_out = r2_archive_update(
        archive_obj=all_obj[:n_arch] if archive_candidates else np.zeros((0, all_obj.shape[1])),
        archive_vectors=all_vec[:n_arch] if archive_candidates else np.zeros((0, all_vec.shape[1])),
        candidate_obj=all_obj[n_arch:] if new_candidates else np.zeros((0, all_obj.shape[1])),
        candidate_vectors=all_vec[n_arch:] if new_candidates else np.zeros((0, all_vec.shape[1])),
        max_size=max_size,
        weights=weights,
        z_ideal=z_ideal,
        eps_rel=eps_rel,
    )

    # Rebuild Candidate list by matching returned vectors to originals
    kept: list[Candidate] = []
    used: set[int] = set()
    for i in range(new_vec.shape[0]):
        for j, c in enumerate(all_cands):
            if j not in used and np.array_equal(c.vector, new_vec[i]):
                kept.append(c)
                used.add(j)
                break
        else:
            dists = np.linalg.norm(np.stack([c.vector for c in all_cands], axis=0) - new_vec[i], axis=1)
            for j_used in used:
                dists[j_used] = np.inf
            best = int(np.argmin(dists))
            kept.append(all_cands[best])
            used.add(best)
    return kept, z_ideal_out


def _grid_cell_id(matrix: np.ndarray, n_grid: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=int)
    return _grid_ids_from_bounds(matrix, n_grid, np.min(matrix, axis=0), np.max(matrix, axis=0))


def _hypergrid_cell_id(matrix: np.ndarray, n_grid: int) -> np.ndarray:
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return np.zeros(matrix.shape[0], dtype=int)
    return _grid_ids_from_bounds(matrix, n_grid, np.min(finite, axis=0), np.max(finite, axis=0))


def _hypergrid_occupied_count(matrix: np.ndarray, n_grid: int) -> int:
    ids = _hypergrid_cell_id(matrix, n_grid)
    return int(len(np.unique(ids)))


def _sample_hypergrid_leaders(
    matrix: np.ndarray,
    n_pick: int,
    n_grid: int,
    kappa: float,
) -> tuple[np.ndarray, int]:
    cell_ids = _hypergrid_cell_id(matrix, n_grid)
    unique_cells, _, inverse = np.unique(cell_ids, return_index=True, return_inverse=True)
    n_unique = len(unique_cells)
    cell_counts = np.bincount(inverse, minlength=n_unique).astype(float)
    raw = np.exp(-kappa * cell_counts)
    prob = raw / np.sum(raw)
    cell_draws = np.random.choice(n_unique, size=n_pick, replace=True, p=prob)
    picks = np.zeros(n_pick, dtype=int)
    for i, cell in enumerate(cell_draws):
        members = np.where(inverse == cell)[0]
        picks[i] = members[np.random.randint(len(members))]
    return picks, n_unique


def _leader_index(
    archive: list[Candidate],
    leader_bias: float,
    use_grid: bool = False,
    n_grid: int = 8,
) -> int:
    if not archive:
        return 0
    if use_grid:
        matrix = _candidate_matrix(archive)
        finite_mask = np.all(np.isfinite(matrix), axis=1)
        if not np.any(finite_mask):
            return int(np.random.randint(len(archive)))
        finite_idx = np.where(finite_mask)[0]
        cell_ids = _grid_cell_id(matrix[finite_mask], n_grid)
        unique_cells = np.unique(cell_ids)
        cell_counts = {c: np.sum(cell_ids == c) for c in unique_cells}
        min_count = min(cell_counts.values())
        sparse_cells = [c for c, cnt in cell_counts.items() if cnt == min_count]
        cell = sparse_cells[np.random.randint(len(sparse_cells))]
        members = finite_idx[cell_ids == cell]
        return int(members[np.random.randint(len(members))])
    n = len(archive)
    if n == 1:
        return 0
    weights = np.exp(-leader_bias * np.arange(n, dtype=float) / max(1, n - 1))
    weights /= weights.sum()
    return int(np.random.choice(n, p=weights))


def _fixed_hv_reference(matrix: np.ndarray) -> np.ndarray | None:
    if matrix.size == 0:
        return None
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return None
    reference = np.max(finite, axis=0) * HV_REF_MARGIN
    reference = np.asarray(reference, dtype=float)
    reference[reference <= 0] = 1.0
    return reference


def _finite_mean(values: list[float], default: float = 0.0) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else default


def _normalize_feature_mode(raw: Any) -> str:
    mode = str(raw).strip().lower()
    if mode in {"path", "full", "pathwise"}:
        return "path"
    return "lite"


def _objective_score(matrix: np.ndarray) -> np.ndarray:
    """Weighted sum score for ranking, lower=better."""
    if matrix.size == 0:
        return np.zeros(0)
    finite = matrix.copy()
    finite[~np.isfinite(finite)] = 1e18
    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    normalized = (finite - mins) / span
    return np.sum(normalized, axis=1)


def _gpu_velocity_update(
    population: np.ndarray,
    velocity: np.ndarray,
    pbest: np.ndarray,
    leaders: np.ndarray,
    inertia: float,
    c1: float,
    c2: float,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_limit: np.ndarray | None,
    gpu_mode: str,
    fleet_size: int = 1,
    n_waypoints: int = 1,
    repulsion_weight: float = 0.0,
    safe_distance: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """PSO velocity + position update (CPU-only legacy path)."""
    n, d = population.shape
    r1 = np.random.rand(n, d)
    r2 = np.random.rand(n, d)

    backend = "numpy:cpu"

    new_vel = inertia * velocity + c1 * r1 * (pbest - population) + c2 * r2 * (leaders - population)

    if fleet_size > 1 and repulsion_weight > 0.0:
        uav_d = d // fleet_size
        pop_multi = population.reshape(n, fleet_size, uav_d)
        pop_sum = np.sum(pop_multi, axis=1, keepdims=True)
        centroid_others = (pop_sum - pop_multi) / (fleet_size - 1)
        R = pop_multi - centroid_others
        R = R.reshape(n, d)
        r3 = np.random.rand(n, d)
        new_vel = new_vel + repulsion_weight * r3 * R

    if velocity_limit is not None:
        new_vel = np.clip(new_vel, -velocity_limit, velocity_limit)
    new_pop = np.clip(population + new_vel, lower, upper)
    return new_pop, new_vel, backend


def _de_current_to_pbest(
    population: np.ndarray,
    objective: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    de_count: int,
    f_scale: float,
    cr_rate: float,
    pbest_ratio: float,
) -> np.ndarray:
    """DE/current-to-pbest/1 mutation + binomial crossover."""
    n, d = population.shape
    if n < 4 or de_count <= 0:
        return np.zeros((0, d))
    scores = _objective_score(objective)
    pbest_size = max(1, int(n * pbest_ratio))
    pbest_indices = np.argsort(scores)[:pbest_size]
    base_idx = np.random.randint(0, n, size=de_count)
    pbest_idx = pbest_indices[np.random.randint(0, pbest_size, size=de_count)]
    r1 = np.random.randint(0, n, size=de_count)
    r2 = np.random.randint(0, n, size=de_count)
    # Ensure r1 != r2 != base
    for i in range(de_count):
        attempts = 0
        while (r1[i] == base_idx[i] or r1[i] == r2[i]) and attempts < 20:
            r1[i] = np.random.randint(0, n)
            attempts += 1
        attempts = 0
        while (r2[i] == base_idx[i] or r2[i] == r1[i]) and attempts < 20:
            r2[i] = np.random.randint(0, n)
            attempts += 1
    mutant = (
        population[base_idx]
        + f_scale * (population[pbest_idx] - population[base_idx])
        + f_scale * (population[r1] - population[r2])
    )
    # Binomial crossover
    trial = population[base_idx].copy()
    cr_mask = np.random.rand(de_count, d) < cr_rate
    j_rand = np.random.randint(0, d, size=de_count)
    for i in range(de_count):
        cr_mask[i, j_rand[i]] = True
    trial[cr_mask] = mutant[cr_mask]
    return np.clip(trial, lower, upper)


def _elite_refine_vectors(
    archive: list[Candidate],
    lower: np.ndarray,
    upper: np.ndarray,
    span: np.ndarray,
    sigma: float,
    top_k: int,
    iters: int,
    max_trials: int | None = None,
) -> np.ndarray:
    """Generate Gaussian perturbation trials around top-K archive members."""
    if not archive or top_k <= 0 or iters <= 0:
        return np.zeros((0, lower.size), dtype=float)
    obj = _candidate_matrix(archive)
    scores = _objective_score(obj)
    top_indices = np.argsort(scores)[: min(top_k, len(archive))]
    vectors: list[np.ndarray] = []
    trial_count = 0
    for idx in top_indices:
        base = archive[int(idx)].vector.copy()
        for _ in range(iters):
            if max_trials is not None and trial_count >= max(0, int(max_trials)):
                return _stack_or_empty(vectors, base.size)
            noise = np.random.normal(0, sigma, size=base.shape) * span
            vectors.append(np.clip(base + noise, lower, upper))
            trial_count += 1
    return _stack_or_empty(vectors, lower.size)


def _objective_spread_vectors(
    archive: list[Candidate],
    lower: np.ndarray,
    upper: np.ndarray,
    span: np.ndarray,
    top_k_per_obj: int,
    trials_per_pick: int,
    sigma: float,
    max_trials: int | None = None,
) -> np.ndarray:
    """Generate perturbation trials around top-K members per objective."""
    if not archive or top_k_per_obj <= 0 or trials_per_pick <= 0:
        return np.zeros((0, lower.size), dtype=float)
    obj = _candidate_matrix(archive)
    n_obj = obj.shape[1]
    vectors: list[np.ndarray] = []
    trial_count = 0
    for dim in range(n_obj):
        col = obj[:, dim].copy()
        col[~np.isfinite(col)] = 1e18
        top_idx = np.argsort(col)[: min(top_k_per_obj, len(archive))]
        for idx in top_idx:
            base = archive[int(idx)].vector.copy()
            for _ in range(trials_per_pick):
                if max_trials is not None and trial_count >= max(0, int(max_trials)):
                    return _stack_or_empty(vectors, base.size)
                noise = np.random.normal(0, sigma, size=base.shape) * span
                vectors.append(np.clip(base + noise, lower, upper))
                trial_count += 1
    return _stack_or_empty(vectors, lower.size)


def _sbx_mutation(parents: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """SBX crossover + polynomial mutation for NSGA-II-style injection."""
    sbx_crossover_probability = 1.0
    sbx_distribution_index = 20.0
    pm_mutation_probability = 1.0 / max(1, parents.shape[1])
    pm_distribution_index = 20.0
    n, d = parents.shape
    if n < 2:
        return parents.copy()
    pairs = n // 2 * 2
    p1 = parents[:pairs:2]
    p2 = parents[1:pairs:2]
    # SBX
    u = np.random.rand(p1.shape[0], d)
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (sbx_distribution_index + 1)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (sbx_distribution_index + 1)),
    )
    cx_mask = np.random.rand(p1.shape[0], d) < sbx_crossover_probability
    c1 = np.where(cx_mask, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), p1)
    c2 = np.where(cx_mask, 0.5 * ((1 - beta) * p1 + (1 + beta) * p2), p2)
    offspring = np.vstack([c1, c2])
    # Polynomial mutation
    pm_mask = np.random.rand(*offspring.shape) < pm_mutation_probability
    if np.any(pm_mask):
        u_mut = np.random.rand(*offspring.shape)
        delta = np.where(
            u_mut < 0.5,
            (2.0 * u_mut) ** (1.0 / (pm_distribution_index + 1)) - 1.0,
            1.0 - (2.0 * (1.0 - u_mut)) ** (1.0 / (pm_distribution_index + 1)),
        )
        span_arr = upper - lower
        span_arr[span_arr == 0] = 1.0
        offspring[pm_mask] += (delta * span_arr)[pm_mask]
    return np.clip(offspring, lower, upper)
