from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.mogwo.constants import _ATTN_EPS, _DIVERSITY_EPS
from uav_benchmark.algorithms.shared.fleet_runner import _constraint_violation
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.nsga2_ops import n_d_sort


def _candidate_objective_context(candidate: Candidate) -> np.ndarray:
    obj_raw = np.asarray(candidate.objective, dtype=float).reshape(-1)
    details = candidate.details if isinstance(candidate.details, dict) else {}
    detail_proxy = np.asarray(
        [
            float(details.get("makespan", np.nan)),
            float(details.get("energy", np.nan)),
            float(details.get("risk", np.nan)),
            float(details.get("turnPenalty", np.nan)),
        ],
        dtype=float,
    )
    out = np.ones(4, dtype=float)
    use = min(4, obj_raw.size)
    if use > 0:
        out[:use] = obj_raw[:use]
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = detail_proxy[bad]
    out[~np.isfinite(out)] = 1.0
    return np.clip(out, 0.0, 1.0)


def _stack_candidate_vectors(candidates: list[Candidate], dim: int) -> np.ndarray:
    if not candidates:
        return np.zeros((0, dim), dtype=float)
    return np.stack([np.asarray(candidate.vector, dtype=float) for candidate in candidates], axis=0)


def _stack_candidate_contexts(candidates: list[Candidate]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 4), dtype=float)
    return np.stack([_candidate_objective_context(candidate) for candidate in candidates], axis=0)


def _objective_rank_matrix(
    objectives: np.ndarray,
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    obj_all = np.asarray(objectives, dtype=float)
    ref_obj = np.abs(np.asarray(reference if reference is not None else obj_all, dtype=float))
    objective_scale = np.ones(obj_all.shape[1], dtype=float)
    for objective_idx in range(obj_all.shape[1]):
        column = ref_obj[:, objective_idx]
        finite_col = column[np.isfinite(column)]
        if finite_col.size > 0:
            objective_scale[objective_idx] = float(max(1.0, np.max(finite_col)))
    return np.where(np.isfinite(obj_all), obj_all, objective_scale[None, :])


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


def _candidate_objective_sum(candidate: Candidate) -> float:
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    if objective.size == 0:
        return float("inf")
    safe = np.where(np.isfinite(objective), objective, 1.0)
    return float(np.sum(safe))


def _repair_candidate_rank(candidate: Candidate, model: dict[str, Any]) -> tuple[int, float, float]:
    cand_cv = float(max(0.0, _constraint_violation(candidate, model)))
    feasible_rank = 0 if cand_cv <= 0.0 else 1
    return (feasible_rank, cand_cv, _candidate_objective_sum(candidate))


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


def _archive_objective_context(archive: list[Candidate]) -> np.ndarray:
    return _stack_candidate_contexts(archive)


def _objective_diversity_level(objectives: np.ndarray) -> float:
    matrix = np.asarray(objectives, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.shape[0] < 2:
        return 0.0
    span = np.max(finite, axis=0) - np.min(finite, axis=0)
    spread = np.std(finite, axis=0)
    normalized = np.divide(spread, np.maximum(span, _DIVERSITY_EPS))
    score = float(np.mean(np.clip(normalized, 0.0, 1.0)))
    return float(np.clip(score, 0.0, 1.0))


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
        raw = np.floor((obj_safe - min_vals) / step)
    raw = np.nan_to_num(raw, nan=0.0, posinf=divisions - 1, neginf=0.0)
    cell = np.clip(raw.astype(int), 0, divisions - 1)
    basis = divisions ** np.arange(obj_matrix.shape[1])
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
    logits = -scale * occ if inverse else scale * occ
    probs = _stable_softmax(logits)
    return int(np.random.choice(probs.shape[0], p=probs))


def _select_archive_members(
    candidates: list[Candidate],
    obj_rank: np.ndarray,
    fronts: np.ndarray,
    max_size: int,
    divisions: int,
) -> list[Candidate]:
    total = len(candidates)
    selected: list[int] = []
    rank = 1
    while len(selected) < max_size:
        local_front = np.where(fronts == rank)[0]
        if local_front.size == 0:
            break
        remaining = max_size - len(selected)
        if local_front.size <= remaining:
            selected.extend(local_front.tolist())
            rank += 1
            continue

        pool_global = np.concatenate([np.asarray(selected, dtype=int), local_front])
        pool_obj = obj_rank[pool_global]
        grid, _, _ = _build_grid(pool_obj, divisions)

        delete_mask = np.zeros(pool_global.size, dtype=bool)
        while int((~delete_mask).sum()) > max_size:
            active = np.where(~delete_mask)[0]
            active_grid = grid[active]
            kill_local = _weighted_occ_sample(active_grid, scale=10.0, inverse=False)
            delete_mask[active[kill_local]] = True

        keep_global = pool_global[np.where(~delete_mask)[0]]
        return [candidates[int(idx)] for idx in keep_global]

    if not selected:
        selected = np.arange(min(max_size, total), dtype=int).tolist()
    keep_global = np.asarray(selected[:max_size], dtype=int)
    return [candidates[int(idx)] for idx in keep_global]


# ─────────────────────────────────────────────────────────────────────
# Archive Update
# ─────────────────────────────────────────────────────────────────────


def _update_archive(
    archive: list[Candidate],
    new_cands: list[Candidate],
    model: dict[str, Any],
    max_size: int,
    divisions: int,
    use_constraints: bool = True,
) -> list[Candidate]:
    """Merge + prune archive using feasibility-first sorting + occupancy truncation."""
    all_cands = list(archive) + list(new_cands)
    if not all_cands:
        return []

    total = len(all_cands)
    obj_all = np.stack([c.objective for c in all_cands])

    if use_constraints:
        cv_all = np.asarray([_constraint_violation(candidate, model) for candidate in all_cands], dtype=float)
        cv_all = np.where(np.isfinite(cv_all), np.maximum(cv_all, 0.0), 1.0)
        feasible_mask = cv_all <= 0.0

        # Use feasible points to set finite fallbacks so constraint sorting can
        # apply strict feasibility-first ranking via pop_con.
        ref_obj = np.abs(obj_all[feasible_mask]) if np.any(feasible_mask) else np.abs(obj_all)
        constraint_matrix: np.ndarray | None = cv_all[:, None]
    else:
        ref_obj = np.abs(obj_all)
        constraint_matrix = None
    obj_rank = _objective_rank_matrix(obj_all, reference=ref_obj)

    # ── Phase 1: Non-dominated sorting (progressive fronts) ──────────
    fronts, _ = n_d_sort(obj_rank.copy(), constraint_matrix, total)
    return _select_archive_members(all_cands, obj_rank, fronts, max_size, divisions)


# ─────────────────────────────────────────────────────────────────────
# Leader Selection — objective-grid only
# ─────────────────────────────────────────────────────────────────────


def _select_leaders(
    archive: list[Candidate],
    divisions: int,
    use_advanced_archive: bool = True,
    use_mean_selection: bool = False,
    model: dict[str, Any] | None = None,
    relaxation_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select 3 leaders (alpha, beta, delta) from archive grid occupancy.

    Feasible archive members are prioritized whenever available.
    """
    n = len(archive)
    if n == 0:
        return np.zeros((3, 1), dtype=float), np.zeros(3, dtype=int), np.ones(3, dtype=float)

    if use_advanced_archive:
        feasible_idx = np.asarray(
            [idx for idx, candidate in enumerate(archive) if _candidate_is_feasible(candidate)],
            dtype=int,
        )
        relaxed_idx = np.zeros(0, dtype=int)
        if model is not None and relaxation_eps > _ATTN_EPS:
            relaxed_idx = np.asarray(
                [
                    idx
                    for idx, candidate in enumerate(archive)
                    if _constraint_violation(candidate, model) <= relaxation_eps + _ATTN_EPS
                ],
                dtype=int,
            )
        if feasible_idx.size >= 3 or relaxed_idx.size <= 0:
            pool_idx = feasible_idx if feasible_idx.size > 0 else np.arange(n)
        else:
            merged = np.concatenate([feasible_idx, relaxed_idx])
            pool_idx = np.unique(merged) if merged.size > 0 else np.arange(n)
    else:
        pool_idx = np.arange(n)

    if use_mean_selection:
        pool_obj = np.stack([archive[i].objective for i in pool_idx]).astype(float, copy=False)
        finite_mask = np.isfinite(pool_obj)
        if not np.all(finite_mask):
            max_finite = np.ones(pool_obj.shape[1], dtype=float)
            for objective_idx in range(pool_obj.shape[1]):
                column = pool_obj[:, objective_idx]
                finite_col = column[np.isfinite(column)]
                if finite_col.size > 0:
                    max_finite[objective_idx] = float(np.max(finite_col))
            pool_obj = np.where(finite_mask, pool_obj, max_finite[None, :] + 1.0)
        # Simple mean-based ranking for "True GWO" baseline
        mn = np.min(pool_obj, axis=0)
        mx = np.max(pool_obj, axis=0)
        span = mx - mn
        norm_obj = (pool_obj - mn) / np.where(span < 1e-9, 1.0, span)
        mean_scores = np.mean(norm_obj, axis=1)
        top_pool_indices = np.argsort(mean_scores)[:3]

        leaders = []
        leader_archive_indices = []
        leader_occupancy = []
        for lp_idx in top_pool_indices:
            archive_idx = int(pool_idx[lp_idx])
            leaders.append(archive[archive_idx].vector.copy())
            leader_archive_indices.append(archive_idx)
            leader_occupancy.append(1.0)

        while len(leaders) < 3:
            leaders.append(leaders[0].copy() if leaders else np.zeros(archive[0].vector.size))
            leader_archive_indices.append(leader_archive_indices[0] if leader_archive_indices else 0)
            leader_occupancy.append(1.0)

        return (
            np.stack(leaders),
            np.asarray(leader_archive_indices, dtype=int),
            np.asarray(leader_occupancy, dtype=float),
        )

    pool_obj = np.stack([archive[i].objective for i in pool_idx])
    grid, _, _ = _build_grid(pool_obj, divisions, inflation_alpha=0.0)
    grid_occ = _occupancy_per_solution(grid)

    leaders = []
    leader_archive_indices: list[int] = []
    leader_occupancy: list[float] = []
    available_local = np.arange(pool_idx.size)
    for _ in range(3):
        if available_local.size == 0:
            available_local = np.arange(pool_idx.size)
        candidate_local = available_local
        if use_advanced_archive:
            active_grid = grid[candidate_local]
            idx_in_active = _weighted_occ_sample(
                active_grid,
                scale=10.0,
                inverse=True,
            )
        else:
            idx_in_active = np.random.choice(candidate_local.size)

        chosen_local = candidate_local[idx_in_active]
        chosen_archive_idx = int(pool_idx[chosen_local])
        leaders.append(archive[chosen_archive_idx].vector.copy())
        leader_archive_indices.append(chosen_archive_idx)
        leader_occupancy.append(float(grid_occ[chosen_local]))
        available_local = available_local[available_local != chosen_local]

    return (
        np.stack(leaders),
        np.asarray(leader_archive_indices, dtype=int),
        np.asarray(leader_occupancy, dtype=float),
    )


# ─────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────
