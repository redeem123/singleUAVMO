"""MOQGWO family runner with attention fusion + Atlas archive."""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
    _ensure_fleet_endpoints,
)
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.algorithms.nmopso import (
    build_atlas_config,
    topology_signature,
    topology_bin_from_signature,
    robustness_from_cost,
    AtlasConfig,
)
from uav_benchmark.algorithms.moqgwo.gpu_strict_ops import (
    QGWOGPUStrictEngine,
    evaluate_population_gpu_strict,
    gpu_peak_bytes_for_device,
    require_torch_gpu_for_moqgwo,
)


# ─────────────────────────────────────────────────────────────────────
# GWO Engine
# ─────────────────────────────────────────────────────────────────────

class QGWO_Engine:
    """Grey Wolf Optimizer core with optional attention-guided leader fusion."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        pop_size: int,
        use_attention: bool = True,
    ) -> None:
        self.lower    = lower
        self.upper    = upper
        self.dim      = lower.size
        self.pop_size = pop_size
        self.use_attention = bool(use_attention)
        self.positions = np.random.uniform(lower, upper, size=(pop_size, self.dim))
        self.leaders   = np.zeros((3, self.dim))

    # -- Attention -------------------------------------------------------
    def _attention_weights(self, pos: np.ndarray, leaders: np.ndarray) -> np.ndarray:
        """Return softmax weights over (alpha, beta, delta) for one wolf."""
        pos_norm = pos / (np.linalg.norm(pos) + 1e-12)
        leader_norms = leaders / (np.linalg.norm(leaders, axis=1, keepdims=True) + 1e-12)
        scores = leader_norms @ pos_norm
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        return weights / (weights.sum() + 1e-12)

    # -- One generation step --------------------------------------------
    def step(self, generation: int, max_generations: int) -> np.ndarray:
        """Vectorised MOQGWO update with linear GWO and optional attention fusion."""
        # Paper-standard GWO schedule: linear decay only.
        a = 2.0 - generation * (2.0 / max_generations)

        # Standard GWO estimate from 3 leaders
        X_terms = np.zeros((3, self.pop_size, self.dim), dtype=float)
        for j in range(3):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            A  = 2.0 * a * r1 - a
            C  = 2.0 * r2
            D  = np.abs(C * self.leaders[j] - self.positions)
            X_terms[j] = self.leaders[j] - A * D
        X_GWO = np.mean(X_terms, axis=0)

        if self.use_attention:
            # Attention-guided leader fusion: reweight alpha/beta/delta update
            # terms directly, so attention modulates GWO dynamics rather than
            # adding a separate centroid pull.
            leader_weights = np.stack([
                self._attention_weights(self.positions[i], self.leaders)
                for i in range(self.pop_size)
            ])  # (pop, 3)
            terms_by_wolf = np.transpose(X_terms, (1, 0, 2))  # (pop, 3, dim)
            new_positions = np.sum(leader_weights[:, :, None] * terms_by_wolf, axis=1)
        else:
            new_positions = X_GWO

        # Sanitize and clip
        finite_mask = np.isfinite(new_positions)
        if not np.all(finite_mask):
            center = 0.5 * (self.lower + self.upper)
            new_positions = np.where(finite_mask, new_positions, center)

        self.positions = np.clip(new_positions, self.lower, self.upper)
        return self.positions


# ─────────────────────────────────────────────────────────────────────
# Grid Archive
# ─────────────────────────────────────────────────────────────────────

def _build_grid(
    obj_matrix: np.ndarray,
    divisions: int,
    inflation_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if obj_matrix.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, obj_matrix.shape[1]))
    obj_safe = np.asarray(obj_matrix, dtype=float).copy()
    n_obj = obj_safe.shape[1]
    min_vals = np.zeros(n_obj, dtype=float)
    max_vals = np.zeros(n_obj, dtype=float)
    for j in range(n_obj):
        col = obj_safe[:, j]
        finite = np.isfinite(col)
        if np.any(finite):
            min_vals[j] = float(np.min(col[finite]))
            max_vals[j] = float(np.max(col[finite]))
            col[~finite] = max_vals[j]
            obj_safe[:, j] = col
        else:
            min_vals[j] = 0.0
            max_vals[j] = 1.0
            obj_safe[:, j] = 0.0
    if inflation_alpha > 0.0:
        delta = max_vals - min_vals
        min_vals = min_vals - inflation_alpha * delta
        max_vals = max_vals + inflation_alpha * delta
    with np.errstate(divide="ignore", invalid="ignore"):
        step = (max_vals - min_vals) / divisions
        raw  = np.floor((obj_safe - min_vals) / step)
    raw  = np.nan_to_num(raw, nan=0.0, posinf=divisions - 1, neginf=0.0)
    cell = np.clip(raw.astype(int), 0, divisions - 1)
    basis  = divisions ** np.arange(obj_matrix.shape[1])
    linear = (cell * basis).sum(axis=1)
    _, unique, counts = np.unique(linear, return_inverse=True, return_counts=True)
    return linear, unique, counts


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float).reshape(-1)
    n = logits.size
    if n == 0:
        return logits
    finite_mask = np.isfinite(logits)
    if not np.any(finite_mask):
        return np.ones(n, dtype=float) / float(n)
    work = np.where(finite_mask, logits, -np.inf)
    max_logit = np.max(work)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exps = np.exp(work - max_logit)
    exps[~finite_mask] = 0.0
    total = float(np.sum(exps))
    if not np.isfinite(total) or total <= 0.0:
        return np.ones(n, dtype=float) / float(n)
    return exps / total


def _occupancy_per_solution(indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(0, dtype=float)
    _, inverse, counts = np.unique(indices, return_inverse=True, return_counts=True)
    return counts[inverse].astype(float)


def _weighted_occ_sample(
    grid_indices: np.ndarray,
    atlas_indices: np.ndarray | None,
    objective_weight: float,
    atlas_weight: float,
    scale: float,
    inverse: bool,
) -> int:
    obj_occ = _occupancy_per_solution(grid_indices)
    atlas_occ = (
        _occupancy_per_solution(atlas_indices)
        if (atlas_indices is not None and atlas_indices.size == grid_indices.size)
        else np.ones_like(obj_occ)
    )
    occ = objective_weight * obj_occ + atlas_weight * atlas_occ
    logits = (-scale * occ) if inverse else (scale * occ)
    probs = _stable_softmax(logits)
    return int(np.random.choice(probs.shape[0], p=probs))


def _paper_select_cell_member(grid_indices: np.ndarray) -> int:
    """Paper-style leader selection: roulette over cells with P(cell) ∝ 1/Ni."""
    if grid_indices.size == 0:
        return 0
    cells, inverse, counts = np.unique(grid_indices, return_inverse=True, return_counts=True)
    del cells
    cell_weights = 1.0 / np.maximum(1.0, counts.astype(float))
    cell_probs = cell_weights / np.sum(cell_weights)
    chosen_cell = int(np.random.choice(cell_probs.shape[0], p=cell_probs))
    members = np.where(inverse == chosen_cell)[0]
    return int(members[np.random.randint(0, members.size)])


def _paper_delete_cell_member(grid_indices: np.ndarray) -> int:
    """Paper-style archive delete: roulette over most crowded tendency with P(cell) ∝ Ni."""
    if grid_indices.size == 0:
        return 0
    cells, inverse, counts = np.unique(grid_indices, return_inverse=True, return_counts=True)
    del cells
    cell_weights = counts.astype(float)
    cell_probs = cell_weights / np.sum(cell_weights)
    chosen_cell = int(np.random.choice(cell_probs.shape[0], p=cell_probs))
    members = np.where(inverse == chosen_cell)[0]
    return int(members[np.random.randint(0, members.size)])


def _dominates_objective(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= (b + eps)) and np.any(a < (b - eps)))


def _paper_archive_update(
    archive: list[Candidate],
    new_cands: list[Candidate],
    max_size: int,
    divisions: int,
) -> list[Candidate]:
    """Foundation-paper style archive controller (objective-only)."""
    rep = list(archive)
    if rep:
        rep_obj = np.stack([c.objective for c in rep])
        fronts, _ = n_d_sort(rep_obj.copy(), None, rep_obj.shape[0])
        rep = [rep[i] for i in np.where(fronts == 1)[0]]
    while len(rep) > max_size:
        rep_obj = np.stack([c.objective for c in rep])
        grid, _, _ = _build_grid(rep_obj, divisions, inflation_alpha=0.1)
        kill_idx = _paper_delete_cell_member(grid)
        rep.pop(kill_idx)

    for cand in new_cands:
        if not rep:
            rep.append(cand)
            continue

        dominated_by_rep = False
        dominates_rep: list[int] = []
        for idx, member in enumerate(rep):
            if _dominates_objective(member.objective, cand.objective):
                dominated_by_rep = True
                break
            if _dominates_objective(cand.objective, member.objective):
                dominates_rep.append(idx)
        if dominated_by_rep:
            continue
        if dominates_rep:
            for idx in sorted(dominates_rep, reverse=True):
                rep.pop(idx)

        if len(rep) >= max_size:
            rep_obj = np.stack([c.objective for c in rep])
            grid, _, _ = _build_grid(rep_obj, divisions, inflation_alpha=0.1)
            kill_idx = _paper_delete_cell_member(grid)
            rep.pop(kill_idx)
        rep.append(cand)

    if not rep:
        return rep
    rep_obj = np.stack([c.objective for c in rep])
    fronts, _ = n_d_sort(rep_obj.copy(), None, rep_obj.shape[0])
    rep = [rep[i] for i in np.where(fronts == 1)[0]]
    while len(rep) > max_size:
        rep_obj = np.stack([c.objective for c in rep])
        grid, _, _ = _build_grid(rep_obj, divisions, inflation_alpha=0.1)
        kill_idx = _paper_delete_cell_member(grid)
        rep.pop(kill_idx)
    return rep


# ─────────────────────────────────────────────────────────────────────
# Archive Update
# ─────────────────────────────────────────────────────────────────────

def _update_archive(
    archive: list[Candidate],
    new_cands: list[Candidate],
    atlas_indices: np.ndarray | None,    # pre-computed for all (archive + new)
    max_size: int,
    divisions: int,
    atlas_config: AtlasConfig,
    paper_standard: bool = False,
) -> tuple[list[Candidate], np.ndarray | None]:
    """Merge + prune archive using non-dominated sorting + Atlas truncation.

    Returns (kept_candidates, kept_atlas_indices).
    """
    if paper_standard:
        kept = _paper_archive_update(archive, new_cands, max_size, divisions)
        return kept, None

    all_cands = list(archive) + list(new_cands)
    if not all_cands:
        return [], (np.zeros(0, dtype=int) if atlas_indices is not None else None)

    total      = len(all_cands)
    obj_all    = np.stack([c.objective for c in all_cands])
    atlas_all: np.ndarray | None = None
    if atlas_indices is not None and atlas_indices.size == total:
        atlas_all = atlas_indices

    # ── Phase 1: Non-dominated sorting (progressive fronts) ──────────
    fronts, _ = n_d_sort(obj_all.copy(), None, total)
    selected: list[int] = []
    rank = 1
    while len(selected) < max_size:
        local_front = np.where(fronts == rank)[0]
        if local_front.size == 0:
            break
        global_front = local_front
        remaining = max_size - len(selected)
        if global_front.size <= remaining:
            selected.extend(global_front.tolist())
            rank += 1
            continue

        # ── Phase 2: Atlas truncation on partial front ──────────────
        pool_global = np.concatenate([np.asarray(selected, dtype=int), global_front])
        pool_obj = obj_all[pool_global]
        pool_atl = atlas_all[pool_global] if atlas_all is not None else None
        grid, _, _ = _build_grid(pool_obj, divisions)

        delete_mask = np.zeros(pool_global.size, dtype=bool)
        while int((~delete_mask).sum()) > max_size:
            active = np.where(~delete_mask)[0]
            active_grid = grid[active]
            active_atl = pool_atl[active] if pool_atl is not None else None
            kill_local = _weighted_occ_sample(
                active_grid,
                active_atl if atlas_config.enabled else None,
                atlas_config.objective_weight,
                atlas_config.atlas_weight,
                scale=10.0,
                inverse=False,
            )
            delete_mask[active[kill_local]] = True

        keep_global = pool_global[np.where(~delete_mask)[0]]
        kept_atlas = atlas_all[keep_global] if atlas_all is not None else None
        return [all_cands[i] for i in keep_global], kept_atlas

    if not selected:
        selected = np.arange(min(max_size, total), dtype=int).tolist()
    keep_global = np.asarray(selected[:max_size], dtype=int)
    kept_atlas = atlas_all[keep_global] if atlas_all is not None else None
    return [all_cands[i] for i in keep_global], kept_atlas


# ─────────────────────────────────────────────────────────────────────
# Atlas index computation
# ─────────────────────────────────────────────────────────────────────

def _atlas_for_candidates(
    cands: list[Candidate],
    model: dict,
    atlas_config: AtlasConfig,
) -> np.ndarray:
    """Compute topology-robustness atlas indices for a list of Candidates."""
    indices = np.zeros(len(cands), dtype=int)
    for i, cand in enumerate(cands):
        paths = cand.details.get("paths", []) if isinstance(cand.details, dict) else []
        fleet_sigs = []
        for p in paths:
            sig = topology_signature(p, model, atlas_config.max_obstacles)
            fleet_sigs.append(sig)
        avg_sig = np.mean(fleet_sigs, axis=0) if fleet_sigs else np.zeros(1)
        _, rob_bin = robustness_from_cost(cand.objective, atlas_config.n_robust_bins)
        top_bin    = topology_bin_from_signature(avg_sig, atlas_config)
        indices[i] = rob_bin * 1000 + top_bin
    return indices


# ─────────────────────────────────────────────────────────────────────
# Leader Selection — objective-grid + atlas-aware
# ─────────────────────────────────────────────────────────────────────

def _select_leaders(
    archive: list[Candidate],
    atlas_indices: np.ndarray | None,
    divisions: int,
    atlas_config: AtlasConfig,
    paper_standard: bool = False,
) -> np.ndarray:
    """Select 3 leaders (alpha, beta, delta) from archive grid occupancy."""
    n = len(archive)
    if n == 0:
        return np.zeros((3, archive[0].vector.size if archive else 1))

    pool_idx = np.arange(n)

    pool_obj  = np.stack([archive[i].objective for i in pool_idx])
    pool_atl  = atlas_indices[pool_idx] if atlas_indices is not None else None
    grid, _, _ = _build_grid(pool_obj, divisions, inflation_alpha=(0.1 if paper_standard else 0.0))

    leaders = []
    available_local = np.arange(pool_idx.size)
    for _ in range(3):
        if available_local.size == 0:
            available_local = np.arange(pool_idx.size)
        active_grid = grid[available_local]
        active_atl = pool_atl[available_local] if pool_atl is not None else None
        if paper_standard:
            idx_in_active = _paper_select_cell_member(active_grid)
        else:
            idx_in_active = _weighted_occ_sample(
                active_grid,
                active_atl if atlas_config.enabled else None,
                atlas_config.objective_weight,
                atlas_config.atlas_weight,
                scale=10.0,
                inverse=True,
            )
        chosen_local = available_local[idx_in_active]
        leaders.append(archive[pool_idx[chosen_local]].vector.copy())
        available_local = available_local[available_local != chosen_local]

    return np.stack(leaders)  # shape (3, dim)


# ─────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────

def _resolve_variant(raw: Any) -> str:
    key = str(raw).strip().lower()
    if key in {"", "full", "a2", "a2moqgwo", "a2-moqgwo"}:
        return "full"
    if key in {"no_attention", "no-attention", "noattention"}:
        return "no_attention"
    if key in {"standard_gwo", "standard-gwo", "gwo", "standard"}:
        return "standard_gwo"
    if key in {"gpu_strict", "gpu-strict", "moqgwo-gpu-strict", "moqgwo_gpu_strict"}:
        return "gpu_strict"
    return "full"


def _apply_variant(params: BenchmarkParams, *, variant: str | None = None, use_atlas: bool | None = None) -> BenchmarkParams:
    merged_extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    if variant is not None:
        merged_extra["moqgwoVariant"] = variant
    if use_atlas is not None:
        merged_extra["moqgwoUseAtlas"] = bool(use_atlas)
    return replace(params, extra=merged_extra)


def run_fleet_moqgwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """MOQGWO family runner with attention fusion and Atlas-aware archive."""
    objective_count = 4
    model = dict(model)
    n_waypoints     = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value      = int(params.seed) if params.seed is not None else 0

    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["maxTurnDeg"]              = float(params.max_turn_deg)
    model["is_rl"]                   = False
    model["hardCollisionConstraint"] = True

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    variant = _resolve_variant(params.extra.get("moqgwoVariant", "full"))
    paper_standard = variant == "standard_gwo"
    use_atlas = bool(params.extra.get("moqgwoUseAtlas", True))
    use_gpu_strict = variant == "gpu_strict"
    use_attention = (variant != "no_attention") and (not paper_standard)
    if paper_standard:
        # Keep paper-standard MOGWO untouched.
        use_atlas = False
    torch_module = None
    gpu_device = None
    gpu_backend = "numpy:cpu"
    if use_gpu_strict:
        torch_module, gpu_device, gpu_backend = require_torch_gpu_for_moqgwo(params.gpu_mode)

    archive_size   = int(params.extra.get("nRep", params.population))
    grid_divisions = int(params.extra.get("nGrid", 10))
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = (np.zeros((params.runs, 2), dtype=float)
                  if params.compute_metrics else np.zeros((0, 2), dtype=float))

    atlas_config = build_atlas_config({
        "useTopologyRobustArchive": use_atlas,
        "atlasTopologyBins": int(params.extra.get("atlasTopologyBins", 24)),
        "atlasRobustBins": int(params.extra.get("atlasRobustBins", 4)),
        "atlasMaxObstacles": int(params.extra.get("atlasMaxObstacles", 3)),
        "atlasHashLevels": int(params.extra.get("atlasHashLevels", 6)),
        "atlasObjectiveWeight": float(params.extra.get("atlasObjectiveWeight", 0.5)),
        "atlasTopologyWeight": float(params.extra.get("atlasTopologyWeight", 0.5)),
    })
    if not atlas_config.enabled:
        atlas_config.objective_weight = 1.0
        atlas_config.atlas_weight = 0.0
    run_indices     = _resolve_run_indices(params)
    resume_existing = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in run_indices:
        run_start = time.perf_counter()
        run_dir   = results_path / f"Run_{run_idx}"

        if resume_existing:
            resumed = _resume_run_scores(
                run_dir=run_dir, problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resumed is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resumed
                continue

        np.random.seed(seed_value * 1000 + run_idx)
        if use_gpu_strict:
            torch_module.manual_seed(seed_value * 1000 + run_idx)
            if str(gpu_device).startswith("cuda"):
                torch_module.cuda.reset_peak_memory_stats(gpu_device)

        # ── Initialise ────────────────────────────────────────────────
        if use_gpu_strict:
            engine = QGWOGPUStrictEngine(
                lower,
                upper,
                params.population,
                torch_module=torch_module,
                device=gpu_device,
                use_attention=use_attention,
                use_quantum=False,
            )
        else:
            engine = QGWO_Engine(
                lower,
                upper,
                params.population,
                use_attention=use_attention,
            )
        hv_hist = (np.zeros((params.generations, 2), dtype=float)
                   if params.compute_metrics else np.zeros((0, 2), dtype=float))
        run_gpu_peak_bytes = 0.0

        # Initial evaluation
        if use_gpu_strict:
            init_cands = evaluate_population_gpu_strict(
                engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints,
                torch_module=torch_module, device=gpu_device,
            )
            run_gpu_peak_bytes = max(run_gpu_peak_bytes, gpu_peak_bytes_for_device(torch_module, gpu_device))
        else:
            init_cands = _evaluate_population(
                engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
        init_atlas = _atlas_for_candidates(init_cands, model, atlas_config) if atlas_config.enabled else None

        archive: list[Candidate] = []
        arc_atlas: np.ndarray | None = np.zeros(0, dtype=int) if atlas_config.enabled else None
        active_atlas_config = atlas_config

        # Bootstrap archive
        archive, arc_atlas = _update_archive(
            [], init_cands, init_atlas,
            archive_size, grid_divisions, active_atlas_config,
            paper_standard=paper_standard,
        )

        # Set initial leaders from archive
        if archive:
            selected_leaders = _select_leaders(
                archive, arc_atlas, grid_divisions, active_atlas_config,
                paper_standard=paper_standard,
            )
            if use_gpu_strict:
                engine.set_leaders(selected_leaders)
            else:
                engine.leaders = selected_leaders

        # ── Generation Loop ───────────────────────────────────────────
        for gen in range(1, params.generations + 1):
            # Update positions
            new_positions = engine.step(gen, params.generations)

            # Evaluate new population
            if use_gpu_strict:
                new_cands = evaluate_population_gpu_strict(
                    new_positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints,
                    torch_module=torch_module, device=gpu_device,
                )
                run_gpu_peak_bytes = max(run_gpu_peak_bytes, gpu_peak_bytes_for_device(torch_module, gpu_device))
            else:
                new_cands = _evaluate_population(
                    new_positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
                )
            new_atlas = (
                _atlas_for_candidates(new_cands, model, active_atlas_config)
                if active_atlas_config.enabled
                else None
            )

            # Archive update
            combined_atlas = None
            if arc_atlas is not None and new_atlas is not None:
                combined_atlas = np.concatenate([arc_atlas, new_atlas])
            archive, arc_atlas = _update_archive(
                archive, new_cands, combined_atlas,
                archive_size, grid_divisions, active_atlas_config,
                paper_standard=paper_standard,
            )

            if archive:
                selected_leaders = _select_leaders(
                    archive, arc_atlas, grid_divisions, active_atlas_config,
                    paper_standard=paper_standard,
                )
                if use_gpu_strict:
                    engine.set_leaders(selected_leaders)
                else:
                    engine.leaders = selected_leaders

            # Metrics
            if params.compute_metrics and hv_hist.shape[0] > 0:
                if gen == 1 or gen == params.generations or gen % metric_interval == 0:
                    if archive:
                        arc_obj = np.stack([c.objective for c in archive])
                        hv_hist[gen-1, 0] = cal_metric(1, arc_obj, params.problem_index, objective_count)
                        hv_hist[gen-1, 1] = cal_metric(2, arc_obj, params.problem_index, objective_count)
                elif gen > 1:
                    hv_hist[gen-1] = hv_hist[gen-2]

        # ── Finalize ──────────────────────────────────────────────────
        ensure_dir(run_dir)
        if params.compute_metrics and hv_hist.shape[0] > 0:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_hist})

        if not archive:
            # Pathological fallback
            if use_gpu_strict:
                last_cands = evaluate_population_gpu_strict(
                    engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints,
                    torch_module=torch_module, device=gpu_device,
                )
            else:
                last_cands = _evaluate_population(
                    engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
                )
            if paper_standard:
                last_obj = np.stack([c.objective for c in last_cands]) if last_cands else np.zeros((0, objective_count))
                if last_obj.size > 0:
                    fronts, _ = n_d_sort(last_obj.copy(), None, last_obj.shape[0])
                    front1 = np.where(fronts == 1)[0]
                    archive = [last_cands[i] for i in front1[:archive_size]]
                else:
                    archive = []
            else:
                last_obj = np.stack([c.objective for c in last_cands]) if last_cands else np.zeros((0, objective_count))
                if last_obj.size > 0:
                    fronts, _ = n_d_sort(last_obj.copy(), None, last_obj.shape[0])
                    selected = np.where(fronts == 1)[0]
                    if selected.size == 0:
                        selected = np.arange(min(archive_size, last_obj.shape[0]), dtype=int)
                    archive = [last_cands[i] for i in selected[:archive_size]]
                else:
                    archive = []

        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=archive,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend=gpu_backend if use_gpu_strict else "numpy:cpu",
            gpu_peak_bytes=float(run_gpu_peak_bytes if use_gpu_strict else 0.0),
            run_metadata={
                "algorithmName": "MOQGWO-GPU-STRICT" if use_gpu_strict else "MOQGWO",
                "representation": "cart",
                "moqgwoVariant": str(variant),
                "moqgwoUseAtlas": float(1.0 if atlas_config.enabled else 0.0),
                "requestedPopulation": float(params.population),
                "effectivePopulation": float(params.population),
                "archiveSize": float(archive_size),
            },
        )

        if params.compute_metrics:
            arc_obj = np.stack([c.objective for c in archive])
            run_scores[run_idx - 1] = np.array([
                cal_metric(1, arc_obj, params.problem_index, objective_count),
                cal_metric(2, arc_obj, params.problem_index, objective_count),
            ], dtype=float)

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_fleet_moqgwo_no_attention(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_moqgwo(model, _apply_variant(params, variant="no_attention"))


def run_fleet_moqgwo_no_atlas(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_moqgwo(model, _apply_variant(params, use_atlas=False))


def run_fleet_moqgwo_standard_gwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_moqgwo(model, _apply_variant(params, variant="standard_gwo"))


def run_fleet_moqgwo_gpu_strict(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_moqgwo(model, _apply_variant(params, variant="gpu_strict"))
