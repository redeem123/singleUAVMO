from __future__ import annotations

import numpy as np

from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.nsga2_ops import n_d_sort


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
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    a_scale = np.max(np.abs(a), axis=1, keepdims=True)
    b_scale = np.max(np.abs(b), axis=1, keepdims=True)
    a_scale[~np.isfinite(a_scale) | (a_scale <= 0.0)] = 1.0
    b_scale[~np.isfinite(b_scale) | (b_scale <= 0.0)] = 1.0
    a = a / a_scale
    b = b / b_scale
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[~np.isfinite(a_norm) | (a_norm <= 0.0)] = 1.0
    b_norm[~np.isfinite(b_norm) | (b_norm <= 0.0)] = 1.0
    similarity = (a / a_norm) @ (b / b_norm).T
    return np.clip(similarity, -1.0, 1.0)


def _density_estimate(candidates: list[Candidate], weights: np.ndarray) -> np.ndarray:
    pop_obj = _candidate_matrix(candidates)
    if pop_obj.size == 0:
        return np.zeros(0, dtype=float)
    norm_obj = _safe_normalize(pop_obj)
    sim = _cosine_similarity(norm_obj, np.asarray(weights, dtype=float))
    region = np.argmax(sim, axis=1)
    counts = np.bincount(region, minlength=weights.shape[0]).astype(float)
    return counts[region]


def _spd_sort(
    pop_obj: np.ndarray, d1: np.ndarray, d2: np.ndarray, region: np.ndarray, n_sort: int
) -> tuple[np.ndarray, int]:
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
    pop_obj = _safe_normalize(pop_obj)
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
