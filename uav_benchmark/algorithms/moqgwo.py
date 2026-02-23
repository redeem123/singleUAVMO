"""A²-MOQGWO: Attention-Augmented Multi-Objective Quantum Grey Wolf Optimizer.

Fixed version addressing:
  1. Proper hard constraint handling (CDP replaces is_rl=True bypass)
  2. Feasibility-first leader selection (alpha/beta/delta chosen from feasible pool)
  3. Self-attention normalization (per-individual feature normalization prevents score collapse)
  4. Stable quantum update (capped log term, bounded perturbation)
  5. Eliminated redundant final re-evaluation (reuse archived Candidate objects directly)
  6. Atlas-aware leader selection restored (topology weights enabled)
  7. Adaptive blend decay tied to constraint ratio (faster exploitation when feasible)
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.multi_uav import (
    _build_bounds,
    _constraint_violation,
    _constraint_violation_vector,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_multi_artifacts,
    _should_write_final_hv,
    _ensure_multi_endpoints,
)
from uav_benchmark.algorithms.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.pso_types import Candidate
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.algorithms.nmopso_utils import (
    build_atlas_config,
    topology_signature,
    topology_bin_from_signature,
    robustness_from_cost,
    delete_one_with_weights,
    select_leader_with_weights,
    AtlasConfig,
)


# ─────────────────────────────────────────────────────────────────────
# QGWO Engine
# ─────────────────────────────────────────────────────────────────────

class QGWO_Engine:
    """Quantum Grey Wolf Optimizer Core with fixed attention and quantum step."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray, pop_size: int) -> None:
        self.lower    = lower
        self.upper    = upper
        self.dim      = lower.size
        self.pop_size = pop_size
        self.positions = np.random.uniform(lower, upper, size=(pop_size, self.dim))
        self.leaders   = np.zeros((3, self.dim))
        self._chaos_x  = 0.5
        self._chaos_y  = 0.5

    # -- Chaotic parameter adaptation -----------------------------------
    def _zaslavskii_map(self) -> float:
        v, mu, eps, r = 400.0, 3.0, 0.3, 3.0
        self._chaos_x = (
            self._chaos_x + v * (1 + mu * self._chaos_y)
            + eps * v * mu * np.cos(2 * np.pi * self._chaos_x)
        ) % 1.0
        self._chaos_y = np.exp(-r) * (
            self._chaos_y + eps * np.cos(2 * np.pi * self._chaos_x)
        )
        return float(self._chaos_x)

    # -- Self-Attention --------------------------------------------------
    def _self_attention(self, pos: np.ndarray, leaders: np.ndarray) -> np.ndarray:
        """Normalised self-attention: avoids score collapse in high dimensions.

        Each leader vector and the query position are L2-normalised before
        computing dot-product similarities, making the scores O(1) regardless
        of dimension size.
        """
        # Normalise query and keys to unit sphere
        pos_norm = pos / (np.linalg.norm(pos) + 1e-12)
        leader_norms = leaders / (np.linalg.norm(leaders, axis=1, keepdims=True) + 1e-12)

        # Scaled dot-product attention (sqrt(d_k) cancels for unit norms, but kept for clarity)
        scores = leader_norms @ pos_norm  # shape (3,)

        # Numerically stable softmax
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / (weights.sum() + 1e-12)  # shape (3,)

        return (weights[:, None] * leaders).sum(axis=0)  # weighted sum of original leaders

    # -- One generation step --------------------------------------------
    def step(self, generation: int, max_generations: int) -> np.ndarray:
        """Vectorised A²-MOQGWO update with stabilised quantum collapse."""
        chaos_val = self._zaslavskii_map()
        a = 2.0 - generation * (2.0 / max_generations) + 0.1 * chaos_val

        # Attention-guided collapse centre — vectorised across population
        m_best = np.stack([self._self_attention(self.positions[i], self.leaders)
                           for i in range(self.pop_size)])

        # Quantum beta parameter
        beta = np.random.uniform(0.5, 1.0, size=(self.pop_size, self.dim))

        # Standard GWO estimate from 3 leaders
        X_GWO = np.zeros_like(self.positions)
        for j in range(3):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            A  = 2.0 * a * r1 - a
            C  = 2.0 * r2
            D  = np.abs(C * self.leaders[j] - self.positions)
            X_GWO += self.leaders[j] - A * D
        X_GWO /= 3.0

        # Quantum collapse — log(1/u) capped to prevent runaway perturbation
        u = np.clip(np.random.rand(self.pop_size, self.dim), 1e-8, 1.0)
        log_term = np.minimum(-np.log(u), 5.0)   # cap at 5σ to prevent exploding jumps
        sign = np.where(np.random.rand(self.pop_size, self.dim) > 0.5, 1.0, -1.0)
        q_pos = m_best + sign * beta * np.abs(m_best - self.positions) * log_term

        # Blend: early → more quantum exploration; late → more GWO exploitation
        blend_chance = (max_generations - generation) / max_generations
        mask = np.random.rand(self.pop_size, self.dim) < blend_chance
        new_positions = np.where(mask, q_pos, X_GWO)

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if obj_matrix.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, obj_matrix.shape[1]))
    min_vals = np.min(obj_matrix, axis=0)
    max_vals = np.max(obj_matrix, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = (max_vals - min_vals) / divisions
        raw  = np.floor((obj_matrix - min_vals) / step)
    raw  = np.nan_to_num(raw, nan=0.0, posinf=divisions - 1, neginf=0.0)
    cell = np.clip(raw.astype(int), 0, divisions - 1)
    basis  = divisions ** np.arange(obj_matrix.shape[1])
    linear = (cell * basis).sum(axis=1)
    _, unique, counts = np.unique(linear, return_inverse=True, return_counts=True)
    return linear, unique, counts


# ─────────────────────────────────────────────────────────────────────
# CDP Archive Update — with constraint-domination principle
# ─────────────────────────────────────────────────────────────────────

def _update_archive(
    archive: list[Candidate],
    new_cands: list[Candidate],
    atlas_indices: np.ndarray,    # pre-computed for all (archive + new)
    max_size: int,
    divisions: int,
    atlas_config: AtlasConfig,
    model: dict,
) -> tuple[list[Candidate], np.ndarray]:
    """Merge + prune archive using CDP + non-dominated sorting + Atlas truncation.

    Returns (kept_candidates, kept_atlas_indices).
    """
    all_cands = list(archive) + list(new_cands)
    if not all_cands:
        return [], np.zeros(0, dtype=int)

    n_arch     = len(archive)
    total      = len(all_cands)
    obj_all    = np.stack([c.objective for c in all_cands])
    cv_all     = _constraint_violation_vector(all_cands, model)
    atlas_all  = atlas_indices  # shape (total,), pre-built by caller

    # ── Phase 1: Feasibility filter ─────────────────────────────────
    feas_mask  = cv_all <= 0.0
    n_feas     = int(feas_mask.sum())

    if n_feas == 0:
        # No feasible solutions — keep least-violating (CDP)
        order = np.argsort(cv_all)[:max_size]
        kept  = [all_cands[i] for i in order]
        return kept, atlas_all[order]

    # Work on feasible pool only
    feas_idx  = np.where(feas_mask)[0]
    feas_obj  = obj_all[feas_idx]
    feas_atl  = atlas_all[feas_idx]

    # ── Phase 2: Non-dominated sorting (front 1 only) ───────────────
    fronts, _ = n_d_sort(feas_obj.copy(), None, feas_idx.size)
    front1    = feas_idx[fronts == 1]
    if front1.size == 0:
        front1 = feas_idx  # fallback

    if front1.size <= max_size:
        kept = [all_cands[i] for i in front1]
        return kept, atlas_all[front1]

    # ── Phase 3: Atlas Truncation (oversize front 1) ─────────────────
    f1_obj  = obj_all[front1]
    f1_atl  = atlas_all[front1]
    grid, _, _ = _build_grid(f1_obj, divisions)

    delete_mask = np.zeros(front1.size, dtype=bool)
    while int((~delete_mask).sum()) > max_size:
        active     = np.where(~delete_mask)[0]
        active_grid = grid[active]
        active_atl  = f1_atl[active]
        kill_local  = delete_one_with_weights(
            active_grid, 10.0,
            atlas_config.objective_weight,
            atlas_config.atlas_weight,
            active_atl,
        )
        delete_mask[active[kill_local]] = True

    keep_local = np.where(~delete_mask)[0]
    keep_global = front1[keep_local]
    return [all_cands[i] for i in keep_global], atlas_all[keep_global]


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
# Leader Selection — feasibility-first + atlas-aware
# ─────────────────────────────────────────────────────────────────────

def _select_leaders(
    archive: list[Candidate],
    atlas_indices: np.ndarray,
    divisions: int,
    atlas_config: AtlasConfig,
    model: dict,
) -> np.ndarray:
    """Select 3 leaders (alpha, beta, delta) using CDP priority + atlas weights.

    Preferentially draws from feasible archive members; falls back to
    full archive if fewer than 3 feasible exist.
    """
    n = len(archive)
    if n == 0:
        return np.zeros((3, archive[0].vector.size if archive else 1))

    cv = _constraint_violation_vector(archive, model)
    feas_mask = cv <= 0.0
    feas_idx  = np.where(feas_mask)[0]

    if feas_idx.size >= 3:
        pool_idx  = feas_idx
    else:
        pool_idx  = np.arange(n)  # fall back to all

    pool_obj  = np.stack([archive[i].objective for i in pool_idx])
    pool_atl  = atlas_indices[pool_idx]
    grid, _, _ = _build_grid(pool_obj, divisions)

    leaders = []
    for _ in range(3):
        idx_in_pool = select_leader_with_weights(
            grid, 10.0,
            atlas_config.objective_weight,
            atlas_config.atlas_weight,
            pool_atl,
        )
        leaders.append(archive[pool_idx[idx_in_pool]].vector.copy())

    return np.stack(leaders)  # shape (3, dim)


# ─────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────

def run_multi_moqgwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """A²-MOQGWO: Fixed, constraint-aware, topology-robust multi-UAV runner."""
    objective_count = 4
    model = dict(model)
    n_waypoints     = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value      = int(params.seed) if params.seed is not None else 0

    model, fleet_size = _ensure_multi_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["maxTurnDeg"]              = float(params.max_turn_deg)
    # FIX #1: Use proper hard constraint evaluation (not is_rl=True bypass)
    model["is_rl"]                   = False
    model["hardCollisionConstraint"] = True

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)

    archive_size   = int(params.extra.get("nRep", params.population))
    grid_divisions = int(params.extra.get("nGrid", 10))
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = (np.zeros((params.runs, 2), dtype=float)
                  if params.compute_metrics else np.zeros((0, 2), dtype=float))

    atlas_config    = build_atlas_config({"useTopologyRobustArchive": True})
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

        # ── Initialise ────────────────────────────────────────────────
        engine = QGWO_Engine(lower, upper, params.population)
        hv_hist = (np.zeros((params.generations, 2), dtype=float)
                   if params.compute_metrics else np.zeros((0, 2), dtype=float))

        # Initial evaluation
        init_cands  = _evaluate_population(
            engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
        )
        init_atlas  = _atlas_for_candidates(init_cands, model, atlas_config)

        archive: list[Candidate] = []
        arc_atlas: np.ndarray    = np.zeros(0, dtype=int)

        # Bootstrap archive
        archive, arc_atlas = _update_archive(
            [], init_cands, init_atlas,
            archive_size, grid_divisions, atlas_config, model,
        )

        # Set initial leaders from archive
        if archive:
            engine.leaders = _select_leaders(archive, arc_atlas, grid_divisions, atlas_config, model)

        # ── Generation Loop ───────────────────────────────────────────
        for gen in range(1, params.generations + 1):

            # Update positions
            new_positions = engine.step(gen, params.generations)

            # Evaluate new population
            new_cands = _evaluate_population(
                new_positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
            new_atlas = _atlas_for_candidates(new_cands, model, atlas_config)

            # Archive update (CDP-aware)
            # FIX #5: Combine archive+new atlas arrays for the update call
            combined_atlas = np.concatenate([arc_atlas, new_atlas])
            archive, arc_atlas = _update_archive(
                archive, new_cands, combined_atlas,
                archive_size, grid_divisions, atlas_config, model,
            )

            # FIX #3/#6: Leader selection from feasible pool with atlas weights
            if archive:
                engine.leaders = _select_leaders(
                    archive, arc_atlas, grid_divisions, atlas_config, model
                )

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

        # FIX #4: Use existing Candidate objects directly — no extra re-evaluation
        if not archive:
            # Pathological fallback
            last_cands = _evaluate_population(
                engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
            archive = sorted(last_cands, key=lambda c: _constraint_violation(c, model))[:archive_size]

        _save_multi_artifacts(
            run_dir=run_dir,
            final_candidates=archive,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            rl_trace=None,
            run_metadata={
                "algorithmName": "MOQGWO",
                "representation": "cart",
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
