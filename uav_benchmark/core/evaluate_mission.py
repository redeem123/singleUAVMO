from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.core.evaluate_path import evaluate_path


def _path_length(path_xyz: np.ndarray) -> float:
    if path_xyz.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path_xyz, axis=0), axis=1)))


def _climb_cost(path_xyz: np.ndarray) -> float:
    if path_xyz.shape[0] < 2:
        return 0.0
    dz = np.diff(path_xyz[:, 2])
    return float(np.sum(np.maximum(0.0, dz)))


def _max_turn_deg(path_xyz: np.ndarray) -> float:
    if path_xyz.shape[0] < 3:
        return 0.0
    v1 = path_xyz[1:-1] - path_xyz[:-2]
    v2 = path_xyz[2:] - path_xyz[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 0) & (n2 > 0)
    if not np.any(valid):
        return 0.0
    cross_norms = np.linalg.norm(np.cross(v1[valid], v2[valid]), axis=1)
    dots = np.sum(v1[valid] * v2[valid], axis=1)
    angles = np.arctan2(cross_norms, dots)
    return float(np.degrees(np.max(np.abs(angles))))


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
    max_turn_deg = float(model.get("maxTurnDeg", model.get("maxTurnAngleDeg", 75.0)))
    climb_weight = float(model.get("climbWeight", 0.2))

    path_objs = []
    lengths = []
    energies = []
    infeasible = False
    turn_violation = False
    turn_excess_terms: list[float] = []
    max_turn_observed = 0.0
    path_eval_model = dict(model)
    # Multi-UAV decoding may introduce short auxiliary segments; keep J1
    # finite and let turning/safety terms penalize poor geometry.
    if "rmin" not in path_eval_model:
        path_eval_model["rmin"] = 0.0
    for path in paths_xyz:
        path = np.asarray(path, dtype=float)
        obj = evaluate_path(path, path_eval_model)
        path_objs.append(obj)
        lengths.append(_path_length(path))
        energies.append(_path_length(path) + climb_weight * _climb_cost(path))
        if np.any(~np.isfinite(obj)):
            infeasible = True
        max_turn = _max_turn_deg(path)
        max_turn_observed = max(max_turn_observed, max_turn)
        if max_turn > max_turn_deg:
            turn_violation = True
            turn_excess_terms.append((max_turn - max_turn_deg) / max(max_turn_deg, 1e-9))
        else:
            turn_excess_terms.append(0.0)
    path_obj_mat = np.asarray(path_objs, dtype=float)

    # Synchronize by normalized progress to evaluate pairwise separation.
    n_samples = int(max(20, max(path.shape[0] for path in paths_xyz) * 4))
    synced = np.stack([_resample_path(np.asarray(path, dtype=float), n_samples) for path in paths_xyz], axis=0)
    pair_count = 0
    violation_sum = 0.0
    min_sep = np.inf
    conflict_rows: list[list[float]] = []
    fleet_size = synced.shape[0]
    for i in range(fleet_size):
        for j in range(i + 1, fleet_size):
            pair_count += 1
            distances = np.linalg.norm(synced[i] - synced[j], axis=1)
            min_sep = min(min_sep, float(np.min(distances)))
            violations = np.maximum(0.0, separation_min - distances)
            violation_sum += float(np.sum(violations / max(separation_min, 1e-9)))
            bad_steps = np.where(violations > 0.0)[0]
            for step in bad_steps:
                conflict_rows.append([float(step), float(i), float(j), float(distances[step]), float(violations[step])])

    denom = max(1, pair_count * n_samples)
    conflict_rate = float(violation_sum / denom)
    risk_terms = path_obj_mat[:, 1].copy() if path_obj_mat.size else np.zeros(0, dtype=float)
    if risk_terms.size > 0 and np.any(~np.isfinite(risk_terms)):
        fallback = float(np.nanmax(risk_terms[np.isfinite(risk_terms)])) if np.any(np.isfinite(risk_terms)) else 1_000.0
        risk_terms[~np.isfinite(risk_terms)] = fallback * 5.0
    risk = float(np.mean(risk_terms)) if risk_terms.size else np.inf
    makespan = float(np.max(lengths)) if lengths else np.inf
    energy = float(np.sum(energies)) if energies else np.inf
    turn_penalty = float(np.mean(np.asarray(turn_excess_terms, dtype=float))) if turn_excess_terms else 0.0
    obj = np.array([makespan, energy, risk, conflict_rate + 0.35 * turn_penalty], dtype=float)
    separation_violation = bool(np.isfinite(min_sep) and min_sep < separation_min)
    if infeasible or separation_violation:
        obj[:] = np.inf
    details = {
        "feasible": float(np.all(np.isfinite(obj))),
        "conflictRate": conflict_rate,
        "minSeparation": float(min_sep if np.isfinite(min_sep) else 0.0),
        "makespan": makespan,
        "energy": energy,
        "risk": risk,
        "maxTurnDeg": float(max_turn_observed),
        "turnViolation": float(turn_violation),
        "turnPenalty": turn_penalty,
        "separationViolation": float(separation_violation),
        "conflictLog": np.asarray(conflict_rows, dtype=float).reshape(-1, 5) if conflict_rows else np.zeros((0, 5), dtype=float),
        "pathObjectives": path_obj_mat,
    }
    return obj, details


def evaluate_mission(paths_xyz: list[np.ndarray], model: dict[str, Any]) -> np.ndarray:
    obj, _ = evaluate_mission_details(paths_xyz, model)
    return obj
