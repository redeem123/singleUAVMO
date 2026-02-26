"""Self-contained NMOPSO engine for fleet path planning."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import uav_benchmark.algorithms.shared.pso_defaults as PSD
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.metrics import HV_REF_MARGIN, cal_metric
from uav_benchmark.core.r2_archive import (
    r2_archive_update,
    r2_indicator,
    uniform_weight_vectors,
)


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Return value of :meth:`NMOPSOEngine.step`."""
    hv: float = 0.0
    diversity: float = 0.0
    feasible_ratio: float = 0.0
    conflict_rate: float = 0.0
    gpu_backend: str = "numpy:cpu"
    gpu_time_sec: float = 0.0


# ── Helpers (moved from fleet runner module) ───────────────────────

def _candidate_matrix(candidates: list[Candidate]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 1), dtype=float)
    return np.array([c.objective for c in candidates], dtype=float)


def _finite_min(matrix: np.ndarray) -> np.ndarray:
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return np.zeros(matrix.shape[1])
    return np.min(finite, axis=0)


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
            dists = np.linalg.norm(
                np.stack([c.vector for c in all_cands], axis=0) - new_vec[i], axis=1
            )
            for j_used in used:
                dists[j_used] = np.inf
            best = int(np.argmin(dists))
            kept.append(all_cands[best])
            used.add(best)
    return kept, z_ideal_out


def _grid_cell_id(matrix: np.ndarray, n_grid: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=int)
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    normalized = (matrix - mins) / span
    grid_indices = np.clip((normalized * n_grid).astype(int), 0, n_grid - 1)
    multiplier = np.power(n_grid, np.arange(matrix.shape[1]))
    return np.sum(grid_indices * multiplier, axis=1)


def _hypergrid_cell_id(matrix: np.ndarray, n_grid: int) -> np.ndarray:
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return np.zeros(matrix.shape[0], dtype=int)
    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    normalized = (matrix - mins) / span
    grid_indices = np.clip((normalized * n_grid).astype(int), 0, n_grid - 1)
    multiplier = np.power(n_grid, np.arange(matrix.shape[1]))
    return np.sum(grid_indices * multiplier, axis=1)


def _hypergrid_occupied_count(matrix: np.ndarray, n_grid: int) -> int:
    ids = _hypergrid_cell_id(matrix, n_grid)
    return int(len(np.unique(ids)))


def _sample_hypergrid_leaders(
    matrix: np.ndarray, n_pick: int, n_grid: int, kappa: float,
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
    archive: list[Candidate], leader_bias: float,
    use_grid: bool = False, n_grid: int = 8,
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
    mutant = population[base_idx] + f_scale * (population[pbest_idx] - population[base_idx]) + f_scale * (population[r1] - population[r2])
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
    top_indices = np.argsort(scores)[:min(top_k, len(archive))]
    vectors: list[np.ndarray] = []
    trial_count = 0
    for idx in top_indices:
        base = archive[int(idx)].vector.copy()
        for _ in range(iters):
            if max_trials is not None and trial_count >= max(0, int(max_trials)):
                if vectors:
                    return np.stack(vectors, axis=0)
                return np.zeros((0, base.size), dtype=float)
            noise = np.random.normal(0, sigma, size=base.shape) * span
            vectors.append(np.clip(base + noise, lower, upper))
            trial_count += 1
    if not vectors:
        return np.zeros((0, lower.size), dtype=float)
    return np.stack(vectors, axis=0)


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
        top_idx = np.argsort(col)[:min(top_k_per_obj, len(archive))]
        for idx in top_idx:
            base = archive[int(idx)].vector.copy()
            for _ in range(trials_per_pick):
                if max_trials is not None and trial_count >= max(0, int(max_trials)):
                    if vectors:
                        return np.stack(vectors, axis=0)
                    return np.zeros((0, base.size), dtype=float)
                noise = np.random.normal(0, sigma, size=base.shape) * span
                vectors.append(np.clip(base + noise, lower, upper))
                trial_count += 1
    if not vectors:
        return np.zeros((0, lower.size), dtype=float)
    return np.stack(vectors, axis=0)


def _sbx_mutation(parents: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """SBX crossover + polynomial mutation for NSGA-II-style injection."""
    from uav_benchmark.core.nsga2_ops import (
        SBX_CROSSOVER_PROBABILITY, SBX_DISTRIBUTION_INDEX,
        PM_MUTATION_PROBABILITY, PM_DISTRIBUTION_INDEX,
    )
    n, d = parents.shape
    if n < 2:
        return parents.copy()
    pairs = n // 2 * 2
    p1 = parents[:pairs:2]
    p2 = parents[1:pairs:2]
    # SBX
    u = np.random.rand(p1.shape[0], d)
    beta = np.where(u <= 0.5,
                    (2.0 * u) ** (1.0 / (SBX_DISTRIBUTION_INDEX + 1)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (SBX_DISTRIBUTION_INDEX + 1)))
    cx_mask = np.random.rand(p1.shape[0], d) < SBX_CROSSOVER_PROBABILITY
    c1 = np.where(cx_mask, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), p1)
    c2 = np.where(cx_mask, 0.5 * ((1 - beta) * p1 + (1 + beta) * p2), p2)
    offspring = np.vstack([c1, c2])
    # Polynomial mutation
    pm_mask = np.random.rand(*offspring.shape) < PM_MUTATION_PROBABILITY
    if np.any(pm_mask):
        u_mut = np.random.rand(*offspring.shape)
        delta = np.where(
            u_mut < 0.5,
            (2.0 * u_mut) ** (1.0 / (PM_DISTRIBUTION_INDEX + 1)) - 1.0,
            1.0 - (2.0 * (1.0 - u_mut)) ** (1.0 / (PM_DISTRIBUTION_INDEX + 1)),
        )
        span_arr = upper - lower
        span_arr[span_arr == 0] = 1.0
        offspring[pm_mask] += (delta * span_arr)[pm_mask]
    return np.clip(offspring, lower, upper)


# ── Engine ──────────────────────────────────────────────────────────

class NMOPSOEngine:
    """Stateful NMOPSO engine for fleet path planning.

    Owns: population, velocity, pbest, archive, evaluation.
    Exposes ``step()`` + operator arms for adaptive search policies.
    """

    def __init__(
        self,
        model: dict[str, Any],
        pop_size: int,
        lower: np.ndarray,
        upper: np.ndarray,
        fleet_size: int,
        n_waypoints: int,
        representation: str,
        objective_count: int = 4,
        archive_size: int = 100,
        use_r2_archive: bool = True,
        paper_nmopso: bool = True,
        is_nmopso_family: bool = True,
        grid_cells: int = 10,
        grid_kappa: float = 1.0,
        use_grid_leader: bool = True,
        velocity_clamp_ratio: float = 0.5,
        gpu_mode: str = "off",
        feature_mode: str = "lite",
        evaluate_fn: Callable[[np.ndarray], list[Candidate]] | None = None,
    ):
        self.model = model
        self.pop_size = pop_size
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.span = np.maximum(self.upper - self.lower, 1e-9)
        self.dimensions = int(self.lower.size)
        self.fleet_size = fleet_size
        self.n_waypoints = n_waypoints
        self.representation = representation
        self.objective_count = objective_count
        self.archive_size = archive_size
        self.use_r2_archive = use_r2_archive
        self.paper_nmopso = paper_nmopso
        self.is_nmopso_family = is_nmopso_family
        self.grid_cells = grid_cells
        self.grid_kappa = grid_kappa
        self.use_grid_leader = use_grid_leader
        self.velocity_limit_base = velocity_clamp_ratio * self.span
        self.gpu_mode = gpu_mode
        self.feature_mode = _normalize_feature_mode(feature_mode)
        self._evaluate_population = evaluate_fn

        # R2 archive setup
        self.r2_weights = uniform_weight_vectors(n_obj=objective_count, n_divisions=15) if use_r2_archive else np.zeros((0, objective_count))
        self.r2_z_ideal = np.full(objective_count, np.inf)

        # State (initialized in reset())
        self.population: np.ndarray = np.zeros((0, 0))
        self.velocity: np.ndarray = np.zeros((0, 0))
        self.pbest: np.ndarray = np.zeros((0, 0))
        self.pbest_obj: np.ndarray = np.zeros((0, 0))
        self.candidates: list[Candidate] = []
        self.archive: list[Candidate] = []
        self.current_obj: np.ndarray = np.zeros((0, 0))
        self.hv_ref_point: np.ndarray | None = None
        self.metric_rng: np.random.Generator = np.random.default_rng(0)
        self.generation: int = 0

        # Timing
        self.gpu_backend: str = "numpy:cpu"
        self.gpu_peak_bytes: float = 0.0
        self.gpu_update_time_sec: float = 0.0
        self.last_operator_evals: dict[str, int] = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}
        self.last_operator_proposed: dict[str, int] = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}
        self.last_operator_filtered: dict[str, int] = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}

    def reset(self) -> None:
        """Initialize/reset all PSO state for a new run."""
        self.population = np.random.uniform(self.lower, self.upper, size=(self.pop_size, self.dimensions))
        self.metric_rng = np.random.default_rng(0)
        self.velocity = np.zeros_like(self.population)
        self.candidates = self._evaluate(self.population)
        self.pbest = self.population.copy()
        self.pbest_obj = _candidate_matrix(self.candidates).copy()
        self.current_obj = self.pbest_obj.copy()
        self.generation = 0
        self.hv_ref_point = None
        self.gpu_backend = "numpy:cpu"
        self.gpu_peak_bytes = 0.0
        self.gpu_update_time_sec = 0.0
        self.last_operator_evals = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}
        self.last_operator_proposed = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}
        self.last_operator_filtered = {"sbx": 0, "de": 0, "elite": 0, "spread": 0}

        # Initial archive
        if self.use_r2_archive:
            self.r2_z_ideal = np.full(self.objective_count, np.inf)
            self.archive, self.r2_z_ideal = _r2_archive_from_candidates(
                [], self.candidates, max_size=self.archive_size,
                weights=self.r2_weights, z_ideal=self.r2_z_ideal,
            )
        else:
            self.archive = _archive_front(self.candidates, max_size=self.archive_size)

    def _evaluate(self, vectors: np.ndarray) -> list[Candidate]:
        """Evaluate a population matrix, returning Candidates."""
        if self._evaluate_population is None:
            raise RuntimeError("NMOPSOEngine requires an evaluation callback.")
        raw = self._evaluate_population(vectors)
        return [
            c if isinstance(c, Candidate) else Candidate(vector=c.vector, objective=c.objective, details=c.details)
            for c in raw
        ]

    def _candidate_centroid(self, candidate: Candidate) -> np.ndarray:
        """Estimate a 3D spatial centroid from candidate telemetry."""
        if self.feature_mode != "path":
            vec = np.asarray(candidate.vector, dtype=float).reshape(-1)
            if vec.size >= 3:
                usable = vec[: (vec.size // 3) * 3]
                if usable.size >= 3:
                    reshaped = usable.reshape(-1, 3)
                    centroid = np.mean(reshaped, axis=0)
                    if np.all(np.isfinite(centroid)):
                        return centroid
            return np.zeros(3, dtype=float)

        details = candidate.details if isinstance(candidate.details, dict) else {}
        paths = details.get("paths", [])
        points: list[np.ndarray] = []
        for path in paths:
            arr = np.asarray(path, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            xyz = arr[:, :3]
            finite_mask = np.all(np.isfinite(xyz), axis=1)
            if np.any(finite_mask):
                points.append(xyz[finite_mask])
        if points:
            merged = np.vstack(points)
            return np.mean(merged, axis=0)

        vec = np.asarray(candidate.vector, dtype=float).reshape(-1)
        if vec.size >= 3:
            usable = vec[: (vec.size // 3) * 3]
            if usable.size >= 3:
                reshaped = usable.reshape(-1, 3)
                centroid = np.mean(reshaped, axis=0)
                if np.all(np.isfinite(centroid)):
                    return centroid
        return np.zeros(3, dtype=float)

    def _centroids_from_vectors(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        n_rows = matrix.shape[0]
        c0 = np.mean(matrix[:, 0::3], axis=1) if matrix.shape[1] > 0 else np.zeros(n_rows, dtype=float)
        c1 = np.mean(matrix[:, 1::3], axis=1) if matrix.shape[1] > 1 else np.zeros(n_rows, dtype=float)
        c2 = np.mean(matrix[:, 2::3], axis=1) if matrix.shape[1] > 2 else np.zeros(n_rows, dtype=float)
        out = np.stack([c0, c1, c2], axis=1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_objectives(self, obj: np.ndarray) -> np.ndarray:
        if obj.size == 0:
            return np.zeros((0, self.objective_count), dtype=float)
        finite = np.where(np.isfinite(obj), obj, np.nan)
        col_min = np.nanmin(finite, axis=0)
        col_max = np.nanmax(finite, axis=0)
        col_min = np.where(np.isfinite(col_min), col_min, 0.0)
        col_max = np.where(np.isfinite(col_max), col_max, col_min + 1.0)
        span = np.maximum(col_max - col_min, 1e-9)
        safe_obj = np.where(np.isfinite(obj), obj, col_max + span)
        norm = (safe_obj - col_min) / span
        return np.clip(norm, 0.0, 1.0)

    def _normalize_centroids(self, centroids: np.ndarray) -> np.ndarray:
        if centroids.size == 0:
            return np.zeros((0, 3), dtype=float)
        lower = np.array(
            [
                float(self.model.get("xmin", np.nan)),
                float(self.model.get("ymin", np.nan)),
                float(self.model.get("zmin", np.nan)),
            ],
            dtype=float,
        )
        upper = np.array(
            [
                float(self.model.get("xmax", np.nan)),
                float(self.model.get("ymax", np.nan)),
                float(self.model.get("zmax", np.nan)),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            finite = np.where(np.isfinite(centroids), centroids, np.nan)
            lower = np.nanmin(finite, axis=0)
            upper = np.nanmax(finite, axis=0)
            lower = np.where(np.isfinite(lower), lower, 0.0)
            upper = np.where(np.isfinite(upper), upper, lower + 1.0)
        span = np.maximum(upper - lower, 1e-9)
        safe = np.where(np.isfinite(centroids), centroids, lower)
        norm = (safe - lower) / span
        return np.clip(norm, 0.0, 1.0)

    def get_particle_features(self) -> np.ndarray:
        """Return per-particle features with shape (N_pop, M+4)."""
        if self.pop_size <= 0:
            return np.zeros((0, self.objective_count + 4), dtype=float)

        obj = np.asarray(self.current_obj, dtype=float)
        if obj.shape != (self.pop_size, self.objective_count):
            obj = _candidate_matrix(self.candidates)
            if obj.shape != (self.pop_size, self.objective_count):
                pad = np.zeros((self.pop_size, self.objective_count), dtype=float)
                rows = min(self.pop_size, obj.shape[0])
                cols = min(self.objective_count, obj.shape[1]) if obj.ndim == 2 else 0
                if rows > 0 and cols > 0:
                    pad[:rows, :cols] = obj[:rows, :cols]
                obj = pad

        norm_obj = self._normalize_objectives(obj)
        if self.feature_mode == "path":
            centroids = np.zeros((self.pop_size, 3), dtype=float)
            for idx in range(min(self.pop_size, len(self.candidates))):
                centroids[idx] = self._candidate_centroid(self.candidates[idx])
        else:
            if self.population.size and self.population.shape[0] == self.pop_size:
                centroids = self._centroids_from_vectors(self.population)
            else:
                candidate_vectors = np.zeros((self.pop_size, self.dimensions), dtype=float)
                for idx in range(min(self.pop_size, len(self.candidates))):
                    candidate_vectors[idx] = np.asarray(self.candidates[idx].vector, dtype=float)
                centroids = self._centroids_from_vectors(candidate_vectors)
        norm_centroids = self._normalize_centroids(centroids)

        vel_mag = np.linalg.norm(np.asarray(self.velocity, dtype=float), axis=1) if self.velocity.size else np.zeros(self.pop_size, dtype=float)
        denom = float(np.linalg.norm(self.velocity_limit_base))
        if not np.isfinite(denom) or denom <= 1e-9:
            denom = max(float(np.nanmax(vel_mag)) if vel_mag.size else 1.0, 1e-9)
        vel_norm = np.clip(vel_mag / denom, 0.0, 1.0).reshape(-1, 1)

        feat = np.concatenate([norm_obj, norm_centroids, vel_norm], axis=1)
        return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)

    def get_archive_features(self) -> np.ndarray:
        """Return per-archive features with shape (N_arch, M+4)."""
        n_arch = len(self.archive)
        if n_arch == 0:
            return np.zeros((0, self.objective_count + 4), dtype=float)

        obj = _candidate_matrix(self.archive)
        if obj.shape[1] != self.objective_count:
            pad = np.zeros((n_arch, self.objective_count), dtype=float)
            cols = min(self.objective_count, obj.shape[1]) if obj.ndim == 2 else 0
            if cols > 0:
                pad[:, :cols] = obj[:, :cols]
            obj = pad
        norm_obj = self._normalize_objectives(obj)

        if self.feature_mode == "path":
            centroids = np.zeros((n_arch, 3), dtype=float)
            for idx, candidate in enumerate(self.archive):
                centroids[idx] = self._candidate_centroid(candidate)
        else:
            archive_vectors = np.stack([np.asarray(c.vector, dtype=float) for c in self.archive], axis=0)
            centroids = self._centroids_from_vectors(archive_vectors)
        norm_centroids = self._normalize_centroids(centroids)
        surrogate_speed = np.zeros((n_arch, 1), dtype=float)
        feat = np.concatenate([norm_obj, norm_centroids, surrogate_speed], axis=1)
        return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)

    def attention_leader_select(self, attention_weights: np.ndarray | None) -> np.ndarray | None:
        """Compute leader vectors from attention weights.

        Returns ``None`` when weights are invalid so caller can fall back to
        the original sampling path.
        """
        if attention_weights is None:
            return None
        if not self.archive:
            return self.pbest.copy()

        weights = np.asarray(attention_weights, dtype=float)
        n_arch = len(self.archive)
        if weights.shape != (self.pop_size, n_arch):
            return None

        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = np.clip(weights, 0.0, None)
        row_sum = np.sum(weights, axis=1, keepdims=True)
        invalid = row_sum[:, 0] <= 1e-12
        if np.any(invalid):
            weights[invalid] = 1.0 / max(1, n_arch)
            row_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / np.maximum(row_sum, 1e-12)

        archive_vectors = np.stack([c.vector for c in self.archive], axis=0)
        leaders = weights @ archive_vectors
        if leaders.shape != (self.pop_size, self.dimensions):
            return None
        return np.clip(leaders, self.lower, self.upper)

    def step(
        self,
        inertia: float,
        c1: float,
        c2: float,
        velocity_scale: float = 1.0,
        kappa_scale: float = 1.0,
        delta_scale: float = 1.0,
        region_scale: float = 1.0,
        leader_bias: float = 0.5,
        mutation_prob: float = 0.1,
        attention_weights: np.ndarray | None = None,
        repulsion_weight: float = 0.0,
    ) -> StepResult:
        """Execute one PSO generation: leader select → velocity → mutation → evaluate."""
        self.generation += 1
        result = StepResult()

        # --- Leader selection ---
        finite_archive_matrix = self._finite_archive_matrix()
        occupied_cells = 1

        if self.archive:
            leader_vectors = self.attention_leader_select(attention_weights)
            if leader_vectors is None:
                # Fallback to standard grid/random selection if no attention weights
                occupied_cells = max(1, _hypergrid_occupied_count(finite_archive_matrix, self.grid_cells)) if self.paper_nmopso else 1
                if self.paper_nmopso and finite_archive_matrix.size > 0:
                    finite_idx = np.where(np.all(np.isfinite(_candidate_matrix(self.archive)), axis=1))[0]
                    if finite_idx.size > 0:
                        picks, occupied_cells = _sample_hypergrid_leaders(
                            _candidate_matrix(self.archive)[finite_idx],
                            self.pop_size,
                            self.grid_cells,
                            self.grid_kappa * kappa_scale,
                        )
                        leader_vectors = np.stack([self.archive[finite_idx[p]].vector for p in picks], axis=0)
                if leader_vectors is None:
                    picks = [_leader_index(self.archive, leader_bias, self.use_grid_leader, self.grid_cells) for _ in range(self.pop_size)]
                    leader_vectors = np.stack([self.archive[p].vector for p in picks], axis=0)
        else:
            leader_vectors = self.pbest.copy()

        # --- Velocity & Position Update ---
        velocity_limit = self.velocity_limit_base * velocity_scale
        gpu_t0 = time.perf_counter()
        self.population, self.velocity, self.gpu_backend = _gpu_velocity_update(
            population=self.population,
            velocity=self.velocity,
            pbest=self.pbest,
            leaders=leader_vectors,
            inertia=inertia,
            c1=c1, c2=c2,
            lower=self.lower, upper=self.upper,
            velocity_limit=velocity_limit,
            gpu_mode=self.gpu_mode,
            fleet_size=self.fleet_size,
            n_waypoints=self.n_waypoints,
            repulsion_weight=repulsion_weight,
            safe_distance=float(self.model.get("separationMin", self.model.get("safeDist", 10.0))),
        )
        gpu_elapsed = float(time.perf_counter() - gpu_t0)
        self.gpu_update_time_sec += gpu_elapsed
        result.gpu_time_sec = gpu_elapsed
        result.gpu_backend = self.gpu_backend

        # --- Base Mutation ---
        if mutation_prob > 1e-9:
            delta_cells = float(max(1, _hypergrid_occupied_count(finite_archive_matrix, self.grid_cells))) if self.paper_nmopso else 1.0
            if self.paper_nmopso:
                gain = float(np.tanh((delta_cells * delta_scale * region_scale) / max(1.0, float(occupied_cells))))
                mutation_mask = np.random.rand(self.pop_size) < mutation_prob
                if np.any(mutation_mask):
                    noise = np.random.normal(0.0, 1.0, size=(int(np.sum(mutation_mask)), self.dimensions))
                    self.population[mutation_mask] = np.clip(
                        self.population[mutation_mask] + noise * gain * self.pbest[mutation_mask],
                        self.lower,
                        self.upper,
                    )
            else:
                mutation_mask = np.random.rand(self.pop_size) < mutation_prob
                if np.any(mutation_mask):
                    gen_progress = self.generation / max(1, self.generation + 100)  # placeholder; caller should set
                    sigma_scale = (
                        PSD.MUTATION_SIGMA_HIGH * (1.0 - gen_progress) + PSD.MUTATION_SIGMA_LOW
                        if self.is_nmopso_family
                        else PSD.MUTATION_SIGMA_GENERIC
                    )
                    mutation_sigma = sigma_scale * self.span
                    noise = np.random.normal(0.0, 1.0, size=(int(np.sum(mutation_mask)), self.dimensions)) * mutation_sigma
                    self.population[mutation_mask] = np.clip(self.population[mutation_mask] + noise, self.lower, self.upper)

        # --- Targeted Collision Repair (Fleet Heuristic) ---
        if self.fleet_size > 1 and hasattr(self, "candidates") and self.candidates:
            # Only apply proportional repair to highly constrained fleet spaces
            for i, cand in enumerate(self.candidates):
                if float(cand.details.get("conflictRate", 0.0)) > 0.0:
                    c_log = np.asarray(cand.details.get("conflictLog", []))
                    if c_log.size > 0 and c_log.ndim == 2:
                        bad_drones = np.unique(c_log[:, 1:3].astype(int))
                        for drone_idx in bad_drones:
                            if 0 <= drone_idx < self.fleet_size:
                                start_dim = drone_idx * self.n_waypoints * (self.dimensions // (self.fleet_size * self.n_waypoints))
                                end_dim = (drone_idx + 1) * self.n_waypoints * (self.dimensions // (self.fleet_size * self.n_waypoints))
                                
                                global_mask = np.zeros(self.dimensions, dtype=bool)
                                global_mask[start_dim:end_dim] = True
                                
                                # If Spherical, protect the radii `r` to preserve overall length efficiency
                                if self.paper_nmopso and (end_dim - start_dim) % 3 == 0:
                                    r_indices = np.arange(start_dim, end_dim, 3)
                                    global_mask[r_indices] = False
                                
                                if np.any(global_mask):
                                    # Use a gentler aggressive sigma (2%) to break symmetry loops
                                    # without violently steering the drones 45 degrees into skyscrapers.
                                    agg_sigma = 0.02 * self.span[global_mask]
                                    noise = np.random.normal(0.0, 1.0, size=np.sum(global_mask)) * agg_sigma
                                    self.population[i, global_mask] = np.clip(
                                        self.population[i, global_mask] + noise, 
                                        self.lower[global_mask], 
                                        self.upper[global_mask]
                                    )

        # --- Evaluate ---
        self.candidates = self._evaluate(self.population)
        self.current_obj = _candidate_matrix(self.candidates)

        # --- Update pbest ---
        pbest_matrix = np.asarray(self.pbest_obj, dtype=float)
        better = np.logical_and(
            np.all(self.current_obj <= pbest_matrix, axis=1),
            np.any(self.current_obj < pbest_matrix, axis=1),
        )
        ties = np.logical_and(
            np.all(self.current_obj == pbest_matrix, axis=1),
            np.random.rand(self.pop_size) < 0.5,
        )
        replace = np.logical_or(better, ties)
        if np.any(replace):
            self.pbest[replace] = self.population[replace]
            self.pbest_obj[replace] = self.current_obj[replace]

        # --- Archive update ---
        self.update_archive(self.candidates)

        # --- Compute metrics ---
        finite_archive = self._finite_archive_matrix()
        if self.hv_ref_point is None and finite_archive.size > 0:
            self.hv_ref_point = _fixed_hv_reference(finite_archive)
        if finite_archive.size > 0:
            result.hv = cal_metric(1, finite_archive, 0, self.objective_count,
                                   ref_point=self.hv_ref_point)
            result.diversity = float(np.mean(np.std(finite_archive, axis=0)))
        result.feasible_ratio = float(np.mean(np.all(np.isfinite(self.current_obj), axis=1)))
        result.conflict_rate = _finite_mean(
            [float(c.details.get("conflictRate", np.nan)) for c in self.candidates],
            default=0.0,
        )
        return result

    def update_archive(self, new_candidates: list[Candidate]) -> None:
        """Add candidates to the archive with Pareto/R2 pruning."""
        if self.use_r2_archive:
            self.archive, self.r2_z_ideal = _r2_archive_from_candidates(
                self.archive, new_candidates, max_size=self.archive_size,
                weights=self.r2_weights, z_ideal=self.r2_z_ideal,
            )
        else:
            self.archive = _archive_front(self.archive + new_candidates, max_size=self.archive_size)

    def _surrogate_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Build surrogate training tuples (decision vector -> objective score)."""
        train_x: list[np.ndarray] = []
        train_y: list[np.ndarray] = []

        if self.population.size > 0 and self.current_obj.size > 0:
            pop_y = _objective_score(np.asarray(self.current_obj, dtype=float))
            if pop_y.size == self.population.shape[0]:
                train_x.append(np.asarray(self.population, dtype=float))
                train_y.append(pop_y.reshape(-1))

        if self.archive:
            arch_x = np.stack([c.vector for c in self.archive], axis=0)
            arch_obj = _candidate_matrix(self.archive)
            arch_y = _objective_score(arch_obj)
            if arch_y.size == arch_x.shape[0]:
                train_x.append(np.asarray(arch_x, dtype=float))
                train_y.append(arch_y.reshape(-1))

        if not train_x or not train_y:
            return np.zeros((0, self.dimensions), dtype=float), np.zeros(0, dtype=float)

        x = np.vstack(train_x)
        y = np.concatenate(train_y)
        mask = np.logical_and(np.all(np.isfinite(x), axis=1), np.isfinite(y))
        if not np.any(mask):
            return np.zeros((0, self.dimensions), dtype=float), np.zeros(0, dtype=float)
        return x[mask], y[mask]

    def _surrogate_knn_predict(self, candidate_vectors: np.ndarray, train_x: np.ndarray, train_y: np.ndarray, k: int) -> np.ndarray:
        """Predict objective score proxy via inverse-distance weighted KNN."""
        c = np.asarray(candidate_vectors, dtype=float)
        x = np.asarray(train_x, dtype=float)
        y = np.asarray(train_y, dtype=float).reshape(-1)
        if c.size == 0:
            return np.zeros(0, dtype=float)
        if x.size == 0 or y.size == 0:
            return np.full(c.shape[0], np.inf, dtype=float)

        k_eff = int(max(1, min(int(k), x.shape[0])))
        diff = c[:, None, :] - x[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
        nn_dist2 = np.take_along_axis(dist2, idx, axis=1)
        nn_y = y[idx]
        weights = 1.0 / np.sqrt(np.maximum(nn_dist2, 1e-12))
        numer = np.sum(weights * nn_y, axis=1)
        denom = np.sum(weights, axis=1)
        pred = numer / np.maximum(denom, 1e-12)
        return np.asarray(pred, dtype=float)

    def _prefilter_candidate_vectors(
        self,
        vectors: np.ndarray,
        max_evals: int | None,
        prefilter_enabled: bool,
        prefilter_ratio: float,
        prefilter_min_candidates: int,
        prefilter_k: int,
    ) -> tuple[np.ndarray, int, int]:
        """Surrogate prefilter to reduce expensive objective evaluations."""
        cand = np.asarray(vectors, dtype=float)
        if cand.ndim != 2 or cand.shape[0] == 0:
            return np.zeros((0, self.dimensions), dtype=float), 0, 0
        proposed = int(cand.shape[0])

        keep = proposed
        if max_evals is not None:
            keep = min(keep, max(0, int(max_evals)))
        if prefilter_enabled and prefilter_ratio < 1.0:
            ratio_keep = int(np.ceil(max(0.0, prefilter_ratio) * proposed))
            keep = min(keep, max(int(prefilter_min_candidates), ratio_keep))
        keep = int(np.clip(keep, 0, proposed))
        if keep <= 0:
            return np.zeros((0, self.dimensions), dtype=float), proposed, 0
        if keep >= proposed:
            return cand, proposed, proposed

        train_x, train_y = self._surrogate_training_data()
        if train_x.shape[0] >= max(4, int(prefilter_k)):
            pred = self._surrogate_knn_predict(cand, train_x, train_y, k=int(prefilter_k))
            order = np.argsort(pred)
        else:
            # Fallback heuristic when surrogate data is insufficient.
            center = np.mean(self.population, axis=0) if self.population.size > 0 else np.mean(cand, axis=0)
            order = np.argsort(np.linalg.norm(cand - center, axis=1))

        exploit_keep = int(max(1, round(0.8 * keep)))
        explore_keep = keep - exploit_keep
        chosen = list(order[:exploit_keep].tolist())
        if explore_keep > 0:
            tail = order[exploit_keep:]
            if tail.size > 0:
                picks = np.random.choice(tail, size=min(explore_keep, tail.size), replace=False)
                chosen.extend(np.asarray(picks, dtype=int).tolist())
        if not chosen:
            return np.zeros((0, self.dimensions), dtype=float), proposed, 0
        seen: set[int] = set()
        ordered_idx: list[int] = []
        for idx in chosen:
            i = int(idx)
            if i in seen:
                continue
            seen.add(i)
            ordered_idx.append(i)
            if len(ordered_idx) >= keep:
                break
        chosen_idx = np.asarray(ordered_idx, dtype=int)
        return cand[chosen_idx], proposed, int(chosen_idx.size)

    def inject_sbx(
        self,
        ratio: float,
        replace_ratio: float,
        mutation_scale: float = 1.0,
        max_evals: int | None = None,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
    ) -> int:
        """Arm 1: SBX crossover injection. Returns number of replacements."""
        inject_count = int(max(2, round(self.pop_size * ratio)))
        if inject_count % 2 == 1:
            inject_count += 1
        if max_evals is not None:
            inject_count = min(inject_count, max(0, int(max_evals)))
            if inject_count % 2 == 1:
                inject_count -= 1
        if inject_count < 2:
            self.last_operator_evals["sbx"] = 0
            self.last_operator_proposed["sbx"] = 0
            self.last_operator_filtered["sbx"] = 0
            return 0
        parent_idx = np.random.randint(0, self.pop_size, size=inject_count)
        offspring_vectors = _sbx_mutation(self.population[parent_idx], self.lower, self.upper)
        if mutation_scale > 0.0 and mutation_scale != 1.0:
            base = self.population[np.random.randint(0, self.pop_size, size=inject_count)]
            offspring_vectors = np.clip(
                base + (offspring_vectors - base) * mutation_scale,
                self.lower, self.upper,
            )
        filtered_vectors, proposed_count, eval_count = self._prefilter_candidate_vectors(
            vectors=offspring_vectors,
            max_evals=max_evals,
            prefilter_enabled=surrogate_prefilter_enabled,
            prefilter_ratio=surrogate_prefilter_ratio,
            prefilter_min_candidates=surrogate_prefilter_min_candidates,
            prefilter_k=surrogate_prefilter_k,
        )
        self.last_operator_proposed["sbx"] = int(proposed_count)
        self.last_operator_evals["sbx"] = int(eval_count)
        self.last_operator_filtered["sbx"] = int(max(0, proposed_count - eval_count))
        if eval_count <= 0:
            return 0

        off_candidates = self._evaluate(filtered_vectors)
        off_obj = _candidate_matrix(off_candidates)
        self.update_archive(off_candidates)

        replaced = 0
        if replace_ratio > 0.0 and off_obj.size > 0 and self.current_obj.size > 0:
            replace_count = int(min(
                max(1, round(self.pop_size * replace_ratio)),
                self.pop_size, off_obj.shape[0],
            ))
            cur_score = _objective_score(self.current_obj)
            off_score = _objective_score(off_obj)
            worst_idx = np.argsort(cur_score)[-replace_count:]
            best_off = np.argsort(off_score)[:replace_count]
            self.population[worst_idx] = filtered_vectors[best_off]
            self.velocity[worst_idx] = 0.0
            for li, oi in zip(worst_idx.tolist(), best_off.tolist()):
                self.candidates[int(li)] = off_candidates[int(oi)]
            self.current_obj[worst_idx] = off_obj[best_off]
            replaced = int(replace_count)
        return replaced

    def inject_de(
        self,
        f_scale: float,
        cr_rate: float,
        ratio: float,
        replace_ratio: float,
        pbest_ratio: float,
        max_evals: int | None = None,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
    ) -> int:
        """Arm 2: DE/current-to-pbest injection. Returns number of replacements."""
        if self.current_obj.size == 0:
            self.last_operator_evals["de"] = 0
            self.last_operator_proposed["de"] = 0
            self.last_operator_filtered["de"] = 0
            return 0
        de_count = int(min(max(4, round(self.pop_size * ratio)), self.pop_size))
        if max_evals is not None:
            de_count = min(de_count, max(0, int(max_evals)))
        if de_count < 4:
            self.last_operator_evals["de"] = 0
            self.last_operator_proposed["de"] = 0
            self.last_operator_filtered["de"] = 0
            return 0
        de_vectors = _de_current_to_pbest(
            population=self.population,
            objective=self.current_obj,
            lower=self.lower, upper=self.upper,
            de_count=de_count,
            f_scale=f_scale, cr_rate=cr_rate,
            pbest_ratio=pbest_ratio,
        )
        if de_vectors.size == 0:
            self.last_operator_evals["de"] = 0
            self.last_operator_proposed["de"] = 0
            self.last_operator_filtered["de"] = 0
            return 0
        filtered_vectors, proposed_count, eval_count = self._prefilter_candidate_vectors(
            vectors=de_vectors,
            max_evals=max_evals,
            prefilter_enabled=surrogate_prefilter_enabled,
            prefilter_ratio=surrogate_prefilter_ratio,
            prefilter_min_candidates=surrogate_prefilter_min_candidates,
            prefilter_k=surrogate_prefilter_k,
        )
        self.last_operator_proposed["de"] = int(proposed_count)
        self.last_operator_evals["de"] = int(eval_count)
        self.last_operator_filtered["de"] = int(max(0, proposed_count - eval_count))
        if eval_count <= 0:
            return 0

        de_candidates = self._evaluate(filtered_vectors)
        de_obj = _candidate_matrix(de_candidates)
        self.update_archive(de_candidates)

        replaced = 0
        if replace_ratio > 0.0 and de_obj.size > 0 and self.current_obj.size > 0:
            de_replace = int(min(max(1, round(self.pop_size * replace_ratio)), self.pop_size, de_obj.shape[0]))
            cur_score = _objective_score(self.current_obj)
            de_score = _objective_score(de_obj)
            worst_idx = np.argsort(cur_score)[-de_replace:]
            best_de = np.argsort(de_score)[:de_replace]
            self.population[worst_idx] = filtered_vectors[best_de]
            self.velocity[worst_idx] = 0.0
            for li, di in zip(worst_idx.tolist(), best_de.tolist()):
                self.candidates[int(li)] = de_candidates[int(di)]
            self.current_obj[worst_idx] = de_obj[best_de]
            replaced = int(de_replace)
        return replaced

    def elite_refine(
        self,
        sigma: float,
        top_k: int,
        iters: int,
        max_evals: int | None = None,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
    ) -> int:
        """Arm 3a: Elite refinement. Returns number of refined candidates."""
        if not self.archive or top_k <= 0 or iters <= 0:
            self.last_operator_evals["elite"] = 0
            self.last_operator_proposed["elite"] = 0
            self.last_operator_filtered["elite"] = 0
            return 0
        trial_vectors = _elite_refine_vectors(
            archive=self.archive,
            lower=self.lower, upper=self.upper,
            span=self.span,
            sigma=sigma, top_k=top_k, iters=iters,
            max_trials=max_evals,
        )
        filtered_vectors, proposed_count, eval_count = self._prefilter_candidate_vectors(
            vectors=trial_vectors,
            max_evals=max_evals,
            prefilter_enabled=surrogate_prefilter_enabled,
            prefilter_ratio=surrogate_prefilter_ratio,
            prefilter_min_candidates=surrogate_prefilter_min_candidates,
            prefilter_k=surrogate_prefilter_k,
        )
        self.last_operator_proposed["elite"] = int(proposed_count)
        self.last_operator_evals["elite"] = int(eval_count)
        self.last_operator_filtered["elite"] = int(max(0, proposed_count - eval_count))
        if eval_count <= 0:
            return 0

        refined = self._evaluate(filtered_vectors)
        if refined:
            self.update_archive(refined)
        return int(len(refined))

    def objective_spread(
        self,
        top_k_per_obj: int,
        trials_per_pick: int,
        sigma: float,
        replace_ratio: float = 0.0,
        max_evals: int | None = None,
        surrogate_prefilter_enabled: bool = False,
        surrogate_prefilter_ratio: float = 1.0,
        surrogate_prefilter_min_candidates: int = 1,
        surrogate_prefilter_k: int = 8,
    ) -> int:
        """Arm 3b: Objective spread. Returns number injected into population."""
        if not self.archive or top_k_per_obj <= 0 or trials_per_pick <= 0:
            self.last_operator_evals["spread"] = 0
            self.last_operator_proposed["spread"] = 0
            self.last_operator_filtered["spread"] = 0
            return 0
        spread_vectors = _objective_spread_vectors(
            archive=self.archive,
            lower=self.lower, upper=self.upper,
            span=self.span,
            top_k_per_obj=top_k_per_obj,
            trials_per_pick=trials_per_pick,
            sigma=sigma,
            max_trials=max_evals,
        )
        filtered_vectors, proposed_count, eval_count = self._prefilter_candidate_vectors(
            vectors=spread_vectors,
            max_evals=max_evals,
            prefilter_enabled=surrogate_prefilter_enabled,
            prefilter_ratio=surrogate_prefilter_ratio,
            prefilter_min_candidates=surrogate_prefilter_min_candidates,
            prefilter_k=surrogate_prefilter_k,
        )
        self.last_operator_proposed["spread"] = int(proposed_count)
        self.last_operator_evals["spread"] = int(eval_count)
        self.last_operator_filtered["spread"] = int(max(0, proposed_count - eval_count))
        if eval_count <= 0:
            self.last_operator_evals["spread"] = 0
            return 0
        spread = self._evaluate(filtered_vectors)
        if not spread:
            self.last_operator_evals["spread"] = 0
            return 0
        self.update_archive(spread)
        injected = 0
        if replace_ratio > 0.0 and self.current_obj.size > 0:
            spread_obj = _candidate_matrix(spread)
            evaluated_vectors = np.stack([c.vector for c in spread], axis=0)
            replace_count = int(min(max(1, round(self.pop_size * replace_ratio)), self.pop_size, spread_obj.shape[0]))
            cur_score = _objective_score(self.current_obj)
            sp_score = _objective_score(spread_obj)
            worst_idx = np.argsort(cur_score)[-replace_count:]
            best_idx = np.argsort(sp_score)[:replace_count]
            self.population[worst_idx] = evaluated_vectors[best_idx]
            self.velocity[worst_idx] = 0.0
            for li, si in zip(worst_idx.tolist(), best_idx.tolist()):
                self.candidates[int(li)] = spread[int(si)]
            self.current_obj[worst_idx] = spread_obj[best_idx]
            injected = int(replace_count)
        return injected

    def r2_before(self) -> float:
        """Compute current R2 indicator for FRRMAB credit."""
        if not self.use_r2_archive or not self.archive:
            return 0.0
        arch_obj = _candidate_matrix(self.archive)
        feas = arch_obj[np.all(np.isfinite(arch_obj), axis=1)]
        if feas.size == 0:
            return 0.0
        return r2_indicator(feas, self.r2_weights, self.r2_z_ideal)

    def state_features(
        self,
        generation: int,
        total_generations: int,
        last_hv: float,
        stagnation: int,
        diversity_ref: float,
    ) -> np.ndarray:
        """Build the 6-dim optimizer state vector."""
        finite_archive = self._finite_archive_matrix()
        if self.hv_ref_point is None and finite_archive.size > 0:
            self.hv_ref_point = _fixed_hv_reference(finite_archive)
        hv_now = cal_metric(1, finite_archive, 0, self.objective_count,
                            ref_point=self.hv_ref_point, rng=self.metric_rng) if finite_archive.size > 0 else 0.0
        diversity = float(np.mean(np.std(finite_archive, axis=0))) if finite_archive.size > 0 else 0.0
        feasible_ratio = float(np.mean(np.all(np.isfinite(self.current_obj), axis=1)))
        
        # 'conflictRate' tracks UAV-to-UAV collisions
        conflict_rate = _finite_mean(
            [float(getattr(c, "details", {}).get("conflictRate", 0.0)) for c in self.candidates],
            default=0.0,
        )
        hv_slope = hv_now - last_hv
        return np.array([
            generation / max(1, total_generations),
            np.clip(feasible_ratio, 0.0, 1.0),
            np.clip(max(0.0, conflict_rate) / 0.02, 0.0, 1.0),
            0.5 * (np.tanh(hv_slope / 0.01) + 1.0),
            np.clip(np.log1p(max(0.0, diversity)) / np.log1p(3.0 * diversity_ref), 0.0, 1.0),
            min(1.0, stagnation / max(1, total_generations)),
        ], dtype=float)

    def _finite_archive_matrix(self) -> np.ndarray:
        matrix = _candidate_matrix(self.archive)
        if matrix.size == 0:
            return matrix
        return matrix[np.all(np.isfinite(matrix), axis=1)]

    @property
    def archive_candidates(self) -> list[Candidate]:
        return self.archive
