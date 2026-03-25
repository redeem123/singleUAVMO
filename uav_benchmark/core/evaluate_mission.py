from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import KDTree

from uav_benchmark.core.evaluate_path import evaluate_path_details


def _resample_path(path_xyz: np.ndarray, n_samples: int) -> np.ndarray:
    if path_xyz.shape[0] == 0:
        return np.zeros((n_samples, 3), dtype=float)
    if path_xyz.shape[0] == 1:
        return np.repeat(path_xyz, n_samples, axis=0)
    deltas = np.diff(path_xyz, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cum = np.hstack([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(path_xyz[:1], n_samples, axis=0)
    targets = np.linspace(0.0, total, n_samples)
    out = np.zeros((n_samples, 3), dtype=float)
    seg_idx = 0
    for idx, t in enumerate(targets):
        while seg_idx < len(seg_lengths) - 1 and cum[seg_idx + 1] < t:
            seg_idx += 1
        t0 = cum[seg_idx]
        t1 = cum[seg_idx + 1]
        if t1 <= t0:
            out[idx] = path_xyz[seg_idx]
            continue
        alpha = (t - t0) / (t1 - t0)
        out[idx] = (1.0 - alpha) * path_xyz[seg_idx] + alpha * path_xyz[seg_idx + 1]
    return out


def evaluate_mission_details(
    paths_xyz: list[np.ndarray],
    model: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not paths_xyz:
        inf = np.array([np.inf, np.inf, np.inf, np.inf], dtype=float)
        return inf, {
            "feasible": 0.0,
            "conflictRate": 1.0,
            "minSeparation": 0.0,
            "makespan": np.inf,
            "energy": np.inf,
            "risk": np.inf,
            "maxTurnDeg": np.inf,
            "conflictLog": np.zeros((0, 5), dtype=float),
        }

    separation_min = float(model.get("separationMin", model.get("safeDist", 10.0)))
    max_turn_deg_limit = float(model.get("maxTurnDeg", model.get("maxTurnAngleDeg", 75.0)))
    separation_weight = float(model.get("fleetSeparationWeight", 1.0))
    hard_collision = bool(model.get("hardCollisionConstraint", True))

    path_objs = []
    infeasible = False
    collision_violation = False
    min_clearance = np.inf
    max_turn_observed = 0.0
    path_eval_model = dict(model)
    # Fleet decoding may introduce short auxiliary segments; keep J1
    # finite and let turning/safety terms penalize poor geometry.
    if "rmin" not in path_eval_model:
        path_eval_model["rmin"] = 0.0
    for path in paths_xyz:
        path = np.asarray(path, dtype=float)
        obj, path_details = evaluate_path_details(path, path_eval_model)
        path_objs.append(obj)
        if np.any(~np.isfinite(obj)):
            infeasible = True
        if float(path_details.get("collisionViolation", 0.0)) > 0.5:
            collision_violation = True
        path_clearance = float(path_details.get("minClearance", np.nan))
        if np.isfinite(path_clearance):
            min_clearance = min(min_clearance, path_clearance)
        max_turn = float(path_details.get("maxTurnDeg", 0.0))
        max_turn_observed = max(max_turn_observed, max_turn)
    path_obj_mat = np.asarray(path_objs, dtype=float)

    # Synchronize by normalized progress to evaluate pairwise separation.
    n_samples = int(max(20, max(path.shape[0] for path in paths_xyz) * 4))
    synced = np.stack([_resample_path(np.asarray(path, dtype=float), n_samples) for path in paths_xyz], axis=0)
    
    fleet_size, n_samples, _ = synced.shape
    violation_sum = 0.0
    min_sep = np.inf
    conflict_rows: list[list[float]] = []
    
    for step in range(n_samples):
        points = synced[:, step, :]
        if fleet_size < 2:
            continue
        tree = KDTree(points)
        indices = tree.query_ball_tree(tree, r=separation_min)
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i < j:
                    dist = float(np.linalg.norm(points[i] - points[j]))
                    min_sep = min(min_sep, dist)
                    violation = max(0.0, separation_min - dist)
                    violation_sum += float(violation / max(separation_min, 1e-9))
                    conflict_rows.append([float(step), float(i), float(j), dist, violation])

    pair_count = (fleet_size * (fleet_size - 1)) // 2
    denom = max(1, pair_count * n_samples)
    conflict_rate = float(violation_sum / denom)
    # Unified objective set with legacy-path: aggregate per-path [J1, J2, J3, J4],
    # then inject inter-UAV separation into the shared safety objective J2.
    if path_obj_mat.size > 0:
        aggregated = np.mean(path_obj_mat, axis=0)
        if np.any(~np.isfinite(aggregated)):
            fallback = np.nanmax(path_obj_mat[np.isfinite(path_obj_mat)]) if np.any(np.isfinite(path_obj_mat)) else 1_000.0
            aggregated = np.where(np.isfinite(aggregated), aggregated, float(fallback) * 5.0)
        aggregated = np.asarray(aggregated, dtype=float)
        aggregated[1] = float(aggregated[1] + separation_weight * conflict_rate)
        obj = aggregated
    else:
        obj = np.array([np.inf, np.inf, np.inf, np.inf], dtype=float)

    if np.all(np.isfinite(obj)):
        obj = np.clip(obj, 0.0, 1.0)

    makespan = float(obj[0])
    energy = float(obj[1])
    risk = float(obj[2])
    turn_penalty = float(obj[3])
    separation_violation = bool(np.isfinite(min_sep) and min_sep < separation_min)
    turn_violation = bool(max_turn_observed > max_turn_deg_limit + 1e-9)
    if infeasible or separation_violation or turn_violation or (hard_collision and collision_violation):
        obj[:] = np.inf
    details = {
        "feasible": float(np.all(np.isfinite(obj))),
        "conflictRate": conflict_rate,
        "minSeparation": float(min_sep) if np.isfinite(min_sep) else float("nan"),
        "makespan": makespan,
        "energy": energy,
        "risk": risk,
        "maxTurnDeg": float(max_turn_observed),
        "turnViolation": float(turn_violation),
        "turnPenalty": turn_penalty,
        "separationViolation": float(separation_violation),
        "collisionViolation": float(collision_violation),
        "minClearance": float(min_clearance if np.isfinite(min_clearance) else np.nan),
        "conflictLog": np.asarray(conflict_rows, dtype=float).reshape(-1, 5) if conflict_rows else np.zeros((0, 5), dtype=float),
        "pathObjectives": path_obj_mat,
    }
    return obj, details


def evaluate_mission(paths_xyz: list[np.ndarray], model: dict[str, Any]) -> np.ndarray:
    obj, _ = evaluate_mission_details(paths_xyz, model)
    return obj
