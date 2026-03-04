"""MOGWO family runner with attention fusion."""
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


_ATTN_TAU_OBJ = 0.20
_ATTN_BLEND_EPS = 0.03
_ATTN_ROW_DEGENERATE_EPS = 1e-6
_ATTN_EPS = 1e-12
_ATTN_FEAS_LAMBDA_MAX = 0.82


def _fit_matrix(values: np.ndarray, rows: int, cols: int, fill: float) -> np.ndarray:
    out = np.full((rows, cols), float(fill), dtype=float)
    raw = np.asarray(values, dtype=float)
    if raw.size == 0:
        return out
    raw = raw.reshape(-1, cols) if raw.ndim != 2 else raw
    if raw.shape[1] != cols:
        return out
    use = min(rows, raw.shape[0])
    out[:use] = raw[:use]
    return out


# ─────────────────────────────────────────────────────────────────────
# GWO Engine
# ─────────────────────────────────────────────────────────────────────

class QGWO_Engine:
    """Grey Wolf Optimizer core with objective-conditioned attention."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        pop_size: int,
        use_attention: bool = True,
        use_feasibility_pressure: bool = True,
    ) -> None:
        self.lower    = lower
        self.upper    = upper
        self.dim      = lower.size
        self.pop_size = pop_size
        self.use_attention = bool(use_attention)
        self.use_feasibility_pressure = bool(use_feasibility_pressure)
        self.positions = np.random.uniform(lower, upper, size=(pop_size, self.dim))
        self.leaders   = np.zeros((3, self.dim))
        self._wolf_objectives = np.zeros((self.pop_size, 4), dtype=float)
        self._leader_objectives = np.zeros((3, 4), dtype=float)
        self._feasibility_pressure = 0.0
        self.last_attention_stats: dict[str, float] = {
            "entropy_mean": 0.0,
            "lambda_feasibility": 0.0,
        }

    def set_attention_context(
        self,
        *,
        wolf_objectives: np.ndarray,
        feasibility_pressure: float,
        leader_objectives: np.ndarray,
        wolf_risk: np.ndarray | None = None,
        leader_risk: np.ndarray | None = None,
    ) -> None:
        if wolf_risk is not None or leader_risk is not None:
            import warnings
            warnings.warn(
                "set_attention_context: wolf_risk and leader_risk are not used by the "
                "attention mechanism and are ignored. Pass None or omit them.",
                stacklevel=2,
            )
        self._wolf_objectives = np.clip(_fit_matrix(wolf_objectives, self.pop_size, 4, fill=1.0), 0.0, 1.0)
        self._leader_objectives = np.clip(_fit_matrix(leader_objectives, 3, 4, fill=1.0), 0.0, 1.0)
        self._feasibility_pressure = (
            float(np.clip(feasibility_pressure, 0.0, 1.0)) if self.use_feasibility_pressure else 0.0
        )

    @staticmethod
    def _normalize_channel_rows(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            return np.zeros((0, 0), dtype=float)
        mean = np.mean(scores, axis=1, keepdims=True)
        std = np.std(scores, axis=1, keepdims=True)
        centered = scores - mean
        good = std > _ATTN_EPS
        out = np.zeros_like(scores)
        if np.any(good):
            out = np.divide(centered, np.where(good, std, 1.0))
        return out

    def _attention_weights(self) -> np.ndarray:
        p = float(np.clip(self._feasibility_pressure, 0.0, 1.0))
        w2 = 0.25 + 0.30 * p
        w_other = (1.0 - w2) / 3.0
        objective_weights = np.asarray([w_other, w2, w_other, w_other], dtype=float).reshape(1, 1, 4)

        diff = np.abs(self._wolf_objectives[:, None, :] - self._leader_objectives[None, :, :])
        score_obj = -np.sum(objective_weights * diff, axis=2) / _ATTN_TAU_OBJ
        score_obj = self._normalize_channel_rows(score_obj)

        if self.use_feasibility_pressure:
            fronts, _ = n_d_sort(self._leader_objectives.copy(), None, self._leader_objectives.shape[0])
            rank_raw = np.asarray(fronts, dtype=float).reshape(-1)
            rank_score = -(rank_raw - 1.0)
            score_rank = np.broadcast_to(rank_score.reshape(1, -1), score_obj.shape)
            score_rank = self._normalize_channel_rows(score_rank)
            lambda_feas = float(np.clip(0.12 + 0.70 * p, 0.0, _ATTN_FEAS_LAMBDA_MAX))
        else:
            score_rank = np.zeros_like(score_obj)
            lambda_feas = 0.0
        lambda_obj = max(0.0, 1.0 - lambda_feas)
        score = lambda_obj * score_obj + lambda_feas * score_rank

        tau_eff = 0.88 - 0.33 * p
        tau_eff = float(np.clip(tau_eff, 0.50, 0.88))
        score = score / tau_eff
        score = score - np.max(score, axis=1, keepdims=True)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            weights = np.exp(score)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights = np.divide(weights, np.where(weights_sum > _ATTN_EPS, weights_sum, 1.0))
        weights = (1.0 - _ATTN_BLEND_EPS) * weights + (_ATTN_BLEND_EPS / 3.0)
        weights = np.divide(weights, np.maximum(np.sum(weights, axis=1, keepdims=True), _ATTN_EPS))
        row_span = np.max(score, axis=1) - np.min(score, axis=1)
        invalid = ~np.isfinite(weights).all(axis=1)
        degenerate = invalid | (row_span <= _ATTN_ROW_DEGENERATE_EPS)
        if np.any(degenerate):
            weights[degenerate] = (1.0 / 3.0)
        weights = np.divide(weights, np.maximum(np.sum(weights, axis=1, keepdims=True), _ATTN_EPS))

        entropy = -np.sum(weights * np.log(np.clip(weights, _ATTN_EPS, 1.0)), axis=1)
        self.last_attention_stats = {
            "entropy_mean": float(np.mean(entropy)) if entropy.size > 0 else 0.0,
            "lambda_feasibility": float(lambda_feas),
            # Backward-compatible key for old analysis scripts.
            "lambda_safe": float(lambda_feas),
        }
        return weights

    # -- One generation step --------------------------------------------
    def step(self, generation: int, max_generations: int) -> np.ndarray:
        """Vectorised MOGWO update with linear GWO and optional attention fusion."""
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
            leader_weights = self._attention_weights()  # (pop, 3)
            terms_by_wolf = np.transpose(X_terms, (1, 0, 2))  # (pop, 3, dim)
            new_positions = np.sum(leader_weights[:, :, None] * terms_by_wolf, axis=1)
        else:
            self.last_attention_stats = {
                "entropy_mean": 0.0,
                "lambda_feasibility": 0.0,
                "lambda_safe": 0.0,
            }
            new_positions = X_GWO

        # Sanitize and clip
        finite_mask = np.isfinite(new_positions)
        if not np.all(finite_mask):
            center = 0.5 * (self.lower + self.upper)
            new_positions = np.where(finite_mask, new_positions, center)

        self.positions = np.clip(new_positions, self.lower, self.upper)
        return self.positions


def _candidate_objective_context(candidate: Candidate) -> np.ndarray:
    obj_raw = np.asarray(candidate.objective, dtype=float).reshape(-1)
    details = candidate.details if isinstance(candidate.details, dict) else {}
    detail_proxy = np.asarray([
        float(details.get("makespan", np.nan)),
        float(details.get("energy", np.nan)),
        float(details.get("risk", np.nan)),
        float(details.get("turnPenalty", np.nan)),
    ], dtype=float)
    out = np.ones(4, dtype=float)
    use = min(4, obj_raw.size)
    if use > 0:
        out[:use] = obj_raw[:use]
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = detail_proxy[bad]
    out[~np.isfinite(out)] = 1.0
    return np.clip(out, 0.0, 1.0)


def _candidate_is_feasible(candidate: Candidate) -> bool:
    obj = np.asarray(candidate.objective, dtype=float).reshape(-1)
    if obj.size == 0 or np.any(~np.isfinite(obj)):
        return False
    details = candidate.details if isinstance(candidate.details, dict) else {}
    if "feasible" in details:
        return float(details.get("feasible", 0.0)) > 0.5
    collision = float(details.get("collisionViolation", 0.0)) > 0.5
    separation = float(details.get("separationViolation", 0.0)) > 0.5
    return not (collision or separation)


def _attention_context_from_candidates(
    candidates: list[Candidate],
) -> tuple[np.ndarray, float]:
    count = len(candidates)
    objectives = np.ones((count, 4), dtype=float)
    feasible_mask = np.zeros(count, dtype=bool)
    for idx, cand in enumerate(candidates):
        objectives[idx] = _candidate_objective_context(cand)
        feasible_mask[idx] = _candidate_is_feasible(cand)
    feasible_ratio = float(np.mean(feasible_mask.astype(float))) if count > 0 else 1.0
    return objectives, feasible_ratio


def _attention_leader_context(
    archive: list[Candidate],
    leader_indices: np.ndarray,
) -> np.ndarray:
    obj = np.ones((3, 4), dtype=float)
    if len(archive) <= 0 or leader_indices.size <= 0:
        return obj
    for slot, raw_idx in enumerate(np.asarray(leader_indices, dtype=int).reshape(-1)[:3]):
        idx = int(np.clip(raw_idx, 0, len(archive) - 1))
        cand = archive[idx]
        obj[slot] = _candidate_objective_context(cand)
    return obj


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
    scale: float,
    inverse: bool,
) -> int:
    obj_occ = _occupancy_per_solution(grid_indices)
    occ = np.asarray(obj_occ, dtype=float)
    if inverse:
        logits = -scale * occ
    else:
        logits = scale * occ
    probs = _stable_softmax(logits)
    return int(np.random.choice(probs.shape[0], p=probs))


# ─────────────────────────────────────────────────────────────────────
# Archive Update
# ─────────────────────────────────────────────────────────────────────

def _update_archive(
    archive: list[Candidate],
    new_cands: list[Candidate],
    max_size: int,
    divisions: int,
) -> list[Candidate]:
    """Merge + prune archive using non-dominated sorting + occupancy truncation."""
    all_cands = list(archive) + list(new_cands)
    if not all_cands:
        return []

    total      = len(all_cands)
    obj_all    = np.stack([c.objective for c in all_cands])

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

        # ── Phase 2: occupancy truncation on partial front ──────────
        pool_global = np.concatenate([np.asarray(selected, dtype=int), global_front])
        pool_obj = obj_all[pool_global]
        grid, _, _ = _build_grid(pool_obj, divisions)

        delete_mask = np.zeros(pool_global.size, dtype=bool)
        while int((~delete_mask).sum()) > max_size:
            active = np.where(~delete_mask)[0]
            active_grid = grid[active]
            kill_local = _weighted_occ_sample(
                active_grid,
                scale=10.0,
                inverse=False,
            )
            delete_mask[active[kill_local]] = True

        keep_global = pool_global[np.where(~delete_mask)[0]]
        return [all_cands[i] for i in keep_global]

    if not selected:
        selected = np.arange(min(max_size, total), dtype=int).tolist()
    keep_global = np.asarray(selected[:max_size], dtype=int)
    return [all_cands[i] for i in keep_global]


# ─────────────────────────────────────────────────────────────────────
# Leader Selection — objective-grid only
# ─────────────────────────────────────────────────────────────────────

def _select_leaders(
    archive: list[Candidate],
    divisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select 3 leaders (alpha, beta, delta) from archive grid occupancy."""
    n = len(archive)
    if n == 0:
        return np.zeros((3, 1), dtype=float), np.zeros(3, dtype=int)

    pool_idx = np.arange(n)

    pool_obj  = np.stack([archive[i].objective for i in pool_idx])
    grid, _, _ = _build_grid(pool_obj, divisions, inflation_alpha=0.0)

    leaders = []
    leader_archive_indices: list[int] = []
    available_local = np.arange(pool_idx.size)
    for _ in range(3):
        if available_local.size == 0:
            available_local = np.arange(pool_idx.size)
        candidate_local = available_local
        active_grid = grid[candidate_local]
        idx_in_active = _weighted_occ_sample(
            active_grid,
            scale=10.0,
            inverse=True,
        )
        chosen_local = candidate_local[idx_in_active]
        chosen_archive_idx = int(pool_idx[chosen_local])
        leaders.append(archive[chosen_archive_idx].vector.copy())
        leader_archive_indices.append(chosen_archive_idx)
        available_local = available_local[available_local != chosen_local]

    return np.stack(leaders), np.asarray(leader_archive_indices, dtype=int)


# ─────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────

def _resolve_variant(raw: Any) -> str:
    key = str(raw).strip().lower()
    if key in {"", "full", "a2", "a2mogwo", "a2-mogwo"}:
        return "full"
    if key in {"no_attention", "no-attention", "noattention"}:
        return "no_attention"
    if key in {"standard_gwo", "standard-gwo", "gwo", "standard"}:
        return "no_attention"
    return "full"


def _apply_variant(params: BenchmarkParams, *, variant: str | None = None) -> BenchmarkParams:
    merged_extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    if variant is not None:
        merged_extra["mogwoVariant"] = variant
    return replace(params, extra=merged_extra)


def run_fleet_mogwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """MOGWO family runner with attention fusion and objective-grid archive."""
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
    variant = _resolve_variant(params.extra.get("mogwoVariant", "full"))
    use_attention = variant != "no_attention"
    gpu_backend = "numpy:cpu"

    archive_size   = int(params.extra.get("nRep", params.population))
    grid_divisions = int(params.extra.get("nGrid", 10))
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = (np.zeros((params.runs, 2), dtype=float)
                  if params.compute_metrics else np.zeros((0, 2), dtype=float))

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

        engine = QGWO_Engine(
            lower,
            upper,
            params.population,
            use_attention=use_attention,
            use_feasibility_pressure=True,
        )
        attention_context_enabled = bool(use_attention and hasattr(engine, "set_attention_context"))
        hv_hist = (np.zeros((params.generations, 2), dtype=float)
                   if params.compute_metrics else np.zeros((0, 2), dtype=float))
        run_gpu_peak_bytes = 0.0
        attention_entropy_sum = 0.0
        attention_lambda_feas_sum = 0.0
        attention_steps = 0

        # Initial evaluation
        init_cands = _evaluate_population(
            engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
        )
        if attention_context_enabled:
            init_obj_ctx, init_feasible_ratio = _attention_context_from_candidates(init_cands)
        else:
            init_obj_ctx = np.ones((params.population, 4), dtype=float)
            init_feasible_ratio = 1.0

        archive: list[Candidate] = []

        # Bootstrap archive
        archive = _update_archive(
            [], init_cands,
            archive_size, grid_divisions,
        )

        # Set initial leaders from archive
        if archive:
            selected_leaders, selected_indices = _select_leaders(
                archive,
                grid_divisions,
            )
            engine.leaders = selected_leaders
            if attention_context_enabled:
                leader_obj_ctx = _attention_leader_context(
                    archive,
                    selected_indices,
                )
                engine.set_attention_context(
                    wolf_objectives=init_obj_ctx,
                    feasibility_pressure=float(np.clip(1.0 - init_feasible_ratio, 0.0, 1.0)),
                    leader_objectives=leader_obj_ctx,
                )

        # ── Generation Loop ───────────────────────────────────────────
        for gen in range(1, params.generations + 1):
            # Update positions
            new_positions = engine.step(gen, params.generations)
            if use_attention:
                stats = getattr(engine, "last_attention_stats", None)
                if isinstance(stats, dict):
                    attention_entropy_sum += float(stats.get("entropy_mean", 0.0))
                    attention_lambda_feas_sum += float(
                        stats.get("lambda_feasibility", stats.get("lambda_safe", 0.0))
                    )
                    attention_steps += 1

            # Evaluate new population
            new_cands = _evaluate_population(
                new_positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
            if attention_context_enabled:
                new_obj_ctx, new_feasible_ratio = _attention_context_from_candidates(new_cands)
            else:
                new_obj_ctx = np.ones((params.population, 4), dtype=float)
                new_feasible_ratio = 1.0

            # Archive update
            archive = _update_archive(
                archive, new_cands,
                archive_size, grid_divisions,
            )

            if archive:
                selected_leaders, selected_indices = _select_leaders(
                    archive,
                    grid_divisions,
                )
                engine.leaders = selected_leaders
                if attention_context_enabled:
                    leader_obj_ctx = _attention_leader_context(
                        archive,
                        selected_indices,
                    )
                    engine.set_attention_context(
                        wolf_objectives=new_obj_ctx,
                        feasibility_pressure=float(np.clip(1.0 - new_feasible_ratio, 0.0, 1.0)),
                        leader_objectives=leader_obj_ctx,
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

        if not archive:
            # Pathological fallback
            last_cands = _evaluate_population(
                engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
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
            gpu_backend=gpu_backend,
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": "MOGWO",
                "representation": "cart",
                "mogwoVariant": str(variant),
                "requestedPopulation": float(params.population),
                "effectivePopulation": float(params.population),
                "archiveSize": float(archive_size),
                "mogwoAttentionEntropyMean": float(attention_entropy_sum / max(1, attention_steps)),
                "mogwoLambdaFeasibilityMean": float(attention_lambda_feas_sum / max(1, attention_steps)),
                # Backward-compatible key for previous analysis tables.
                "mogwoLambdaSafeMean": float(attention_lambda_feas_sum / max(1, attention_steps)),
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


def run_fleet_mogwo_no_attention(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_mogwo(model, _apply_variant(params, variant="no_attention"))


def run_fleet_mogwo_standard_gwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    # Backward-compatible alias. This repository uses the same implementation
    # as MOGWO-NO-ATTENTION.
    return run_fleet_mogwo(model, _apply_variant(params, variant="no_attention"))
