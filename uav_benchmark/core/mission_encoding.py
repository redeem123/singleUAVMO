from __future__ import annotations

from typing import Any

import numpy as np


def decision_size(fleet_size: int, n_waypoints: int) -> int:
    return int(fleet_size) * int(n_waypoints) * 3


def decode_decision(
    decision: np.ndarray,
    fleet_size: int,
    n_waypoints: int,
) -> np.ndarray:
    vector = np.asarray(decision, dtype=float).reshape(-1)
    expected = decision_size(fleet_size, n_waypoints)
    if vector.size != expected:
        raise ValueError(f"Decision size mismatch: got {vector.size}, expected {expected}")
    return vector.reshape(fleet_size, n_waypoints, 3)


def _model_bounds(model: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        float(model["xmin"]),
        float(model["xmax"]),
        float(model["ymin"]),
        float(model["ymax"]),
        float(model["zmin"]),
        float(model["zmax"]),
    )


def _safe_height(model: dict[str, Any]) -> float | None:
    safe_h = model.get("safeH")
    return None if safe_h is None else float(safe_h)


def _ground_height(height_map: np.ndarray, x: float, y: float, xmax: int, ymax: int) -> float:
    if not np.isfinite(x):
        x = 0.5 * (1.0 + float(xmax))
    if not np.isfinite(y):
        y = 0.5 * (1.0 + float(ymax))
    px = float(np.clip(x, 1.0, float(xmax))) - 1.0
    py = float(np.clip(y, 1.0, float(ymax))) - 1.0
    x0 = int(np.floor(px))
    x1 = min(x0 + 1, xmax - 1)
    y0 = int(np.floor(py))
    y1 = min(y0 + 1, ymax - 1)
    x0 = max(0, min(x0, xmax - 1))
    y0 = max(0, min(y0, ymax - 1))
    tx = px - float(x0)
    ty = py - float(y0)
    v00 = float(height_map[y0, x0])
    v01 = float(height_map[y0, x1])
    v10 = float(height_map[y1, x0])
    v11 = float(height_map[y1, x1])
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v01 + (1.0 - tx) * ty * v10 + tx * ty * v11


def _to_abs_point(model: dict[str, Any], point_xy_relz: np.ndarray, safe_h: float | None) -> np.ndarray:
    x, y, z_rel = float(point_xy_relz[0]), float(point_xy_relz[1]), float(point_xy_relz[2])
    xmin, xmax, ymin, ymax, zmin, zmax = _model_bounds(model)
    if not np.isfinite(x):
        x = 0.5 * (xmin + xmax)
    if not np.isfinite(y):
        y = 0.5 * (ymin + ymax)
    if not np.isfinite(z_rel):
        z_rel = float(safe_h) if safe_h is not None else zmin
    x = float(np.clip(x, xmin, xmax))
    y = float(np.clip(y, ymin, ymax))
    z_rel = float(np.clip(z_rel, zmin, zmax))
    if safe_h is not None:
        z_rel = max(z_rel, safe_h)
    ground = _ground_height(
        np.asarray(model["H"], dtype=float),
        x,
        y,
        int(xmax),
        int(ymax),
    )
    return np.array([x, y, z_rel + ground], dtype=float)


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return vec / norm


def _progress_schedule(progress_raw: np.ndarray, n_waypoints: int) -> np.ndarray:
    if n_waypoints <= 0:
        return np.zeros(0, dtype=float)
    base = np.linspace(1, n_waypoints, n_waypoints, dtype=float) / (n_waypoints + 1.0)
    centered = np.asarray(progress_raw, dtype=float).reshape(-1)
    centered = centered[:n_waypoints]
    if centered.size < n_waypoints:
        centered = np.pad(centered, (0, n_waypoints - centered.size), mode="edge")
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    centered = centered - np.mean(centered)
    max_abs = float(np.max(np.abs(centered))) if centered.size > 0 else 0.0
    if max_abs > 1e-12:
        centered = centered / max_abs
    jitter = centered * (0.18 / (n_waypoints + 1.0))
    schedule = np.sort(base + jitter)
    min_gap = 0.40 / (n_waypoints + 1.0)
    schedule[0] = max(min_gap, min(1.0 - min_gap * n_waypoints, schedule[0]))
    for idx in range(1, n_waypoints):
        schedule[idx] = max(schedule[idx], schedule[idx - 1] + min_gap)
    if schedule[-1] > 1.0 - min_gap:
        start = max(min_gap, 1.0 - min_gap * n_waypoints)
        schedule = start + min_gap * np.arange(1, n_waypoints + 1, dtype=float)
    schedule = np.clip(schedule, min_gap, 1.0 - min_gap)
    return schedule


def _smooth_signal(values: np.ndarray, passes: int = 1) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if out.size <= 2:
        return out
    for _ in range(max(1, int(passes))):
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


def _normalize_signal(values: np.ndarray, lower: float, upper: float, passes: int) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    normalized = (raw - lower) / (upper - lower) if upper > lower else np.zeros_like(raw)
    normalized = np.clip((normalized - 0.5) * 2.0, -1.0, 1.0)
    return _smooth_signal(normalized, passes=passes)


def _fit_internal_waypoints(internal: np.ndarray, fallback: np.ndarray, n_waypoints: int) -> np.ndarray:
    if internal.shape[0] < n_waypoints:
        if internal.shape[0] > 0:
            pad = np.repeat(internal[-1:, :], n_waypoints - internal.shape[0], axis=0)
            return np.vstack([internal, pad])
        return np.repeat(np.asarray(fallback, dtype=float).reshape(1, -1), n_waypoints, axis=0)
    if internal.shape[0] > n_waypoints:
        picks = np.linspace(0, internal.shape[0] - 1, n_waypoints).astype(int)
        return internal[picks]
    return internal


def decision_to_paths(
    decision: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> list[np.ndarray]:
    block = decode_decision(decision, fleet_size=fleet_size, n_waypoints=n_waypoints)
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    if starts.shape[0] < fleet_size or goals.shape[0] < fleet_size:
        raise ValueError("Model does not contain enough starts/goals for requested fleet size")
    safe_h = _safe_height(model)
    xmin, xmax, ymin, ymax, zmin, zmax = _model_bounds(model)
    map_diag = float(np.hypot(xmax - xmin, ymax - ymin))
    lateral_max = 0.03 * map_diag
    separation_min = float(model.get("separationMin", model.get("safeDist", 10.0)))

    paths: list[np.ndarray] = []
    for uav_idx in range(fleet_size):
        start = starts[uav_idx].reshape(-1)[:3]
        goal = goals[uav_idx].reshape(-1)[:3]
        direction_xy = _safe_unit(goal[:2] - start[:2])
        perp_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=float)

        schedule = _progress_schedule(block[uav_idx, :, 0], n_waypoints)
        lateral_norm = _normalize_signal(block[uav_idx, :, 1], ymin, ymax, passes=2)
        altitude_norm = _normalize_signal(block[uav_idx, :, 2], zmin, zmax, passes=1)

        lane_shift_xy = (uav_idx - (fleet_size - 1) / 2.0) * (0.90 * separation_min)
        waypoints_abs: list[np.ndarray] = []
        for idx in range(n_waypoints):
            t = float(schedule[idx])
            base_xy = start[:2] + t * (goal[:2] - start[:2])
            xy = base_xy + perp_xy * (lane_shift_xy + lateral_norm[idx] * lateral_max)
            xy[0] = float(np.clip(xy[0], xmin, xmax))
            xy[1] = float(np.clip(xy[1], ymin, ymax))

            z_base = float(start[2] + t * (goal[2] - start[2]))
            z_rel = z_base + altitude_norm[idx] * (0.12 * (zmax - zmin))
            if safe_h is not None:
                z_rel = max(z_rel, safe_h)
            z_rel = float(np.clip(z_rel, zmin, zmax))
            waypoints_abs.append(_to_abs_point(model, np.array([xy[0], xy[1], z_rel], dtype=float), safe_h))

        start_abs = _to_abs_point(model, start, safe_h)
        goal_abs = _to_abs_point(model, goal, safe_h)
        path = np.vstack([start_abs, *waypoints_abs, goal_abs])
        paths.append(path)
    return paths


def paths_to_decision(
    paths_xyz: list[np.ndarray],
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> np.ndarray:
    if len(paths_xyz) != fleet_size:
        raise ValueError(f"Expected {fleet_size} paths, got {len(paths_xyz)}")
    starts = np.asarray(model["starts"], dtype=float)
    _ = np.asarray(model["goals"], dtype=float)  # keep interface explicit for future constraints
    height_map = np.asarray(model["H"], dtype=float)
    xmin, xmax, ymin, ymax, zmin, zmax = _model_bounds(model)
    xmax_index = int(xmax)
    ymax_index = int(ymax)
    decision = np.zeros((fleet_size, n_waypoints, 3), dtype=float)
    for idx, path in enumerate(paths_xyz):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("Each path must be an N x 3 matrix")
        # Use internal waypoints only; interpolate or trim to n_waypoints.
        internal = path[1:-1] if path.shape[0] > 2 else np.zeros((0, 3), dtype=float)
        internal = _fit_internal_waypoints(internal, starts[idx], n_waypoints)
        for j in range(n_waypoints):
            x, y, z_abs = internal[j]
            x = float(np.clip(x, xmin, xmax))
            y = float(np.clip(y, ymin, ymax))
            ground = _ground_height(height_map, x, y, xmax_index, ymax_index)
            z_rel = float(np.clip(z_abs - ground, zmin, zmax))
            decision[idx, j] = np.array([x, y, z_rel], dtype=float)
    return decision.reshape(-1)
