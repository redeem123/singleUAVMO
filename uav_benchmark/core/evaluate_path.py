from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit


def _value(model: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in model and model[key] is not None:
            return float(np.asarray(model[key]).reshape(-1)[0])
    return float(default)


@njit(fastmath=True)
def _interpolate_path(path_xyz: np.ndarray, step_size: float) -> np.ndarray:
    """Interpolate path segments so no gap exceeds *step_size*."""
    n_pts = path_xyz.shape[0]
    if n_pts < 2:
        return path_xyz.copy()
    
    n_seg = n_pts - 1
    distances = np.empty(n_seg, dtype=np.float64)
    steps_per_seg = np.empty(n_seg, dtype=np.int64)
    total_points = 1
    
    for i in range(n_seg):
        dx = path_xyz[i+1, 0] - path_xyz[i, 0]
        dy = path_xyz[i+1, 1] - path_xyz[i, 1]
        dz = path_xyz[i+1, 2] - path_xyz[i, 2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        distances[i] = d
        
        steps = int(math.ceil(d / step_size))
        if steps < 1:
            steps = 1
        steps_per_seg[i] = steps
        total_points += steps
        
    result = np.empty((total_points, 3), dtype=np.float64)
    result[0] = path_xyz[0]
    cursor = 1
    
    for seg_idx in range(n_seg):
        n_steps = steps_per_seg[seg_idx]
        p0 = path_xyz[seg_idx]
        p1 = path_xyz[seg_idx + 1]
        for step in range(1, n_steps + 1):
            t = float(step) / float(n_steps)
            result[cursor, 0] = (1.0 - t) * p0[0] + t * p1[0]
            result[cursor, 1] = (1.0 - t) * p0[1] + t * p1[1]
            result[cursor, 2] = (1.0 - t) * p0[2] + t * p1[2]
            cursor += 1
            
    return result[:cursor]


@njit(fastmath=True)
def _dist_point_to_segment_2d(point: np.ndarray, start_point: np.ndarray, end_point: np.ndarray) -> float:
    """Distance from a single 2-D point to a line segment."""
    segment = end_point - start_point
    segment_norm = float(np.dot(segment, segment))
    if segment_norm == 0:
        return float(np.linalg.norm(point - start_point))
    factor = float(np.dot(point - start_point, segment) / segment_norm)
    factor = max(0.0, min(1.0, factor))
    projection = start_point + factor * segment
    return float(np.linalg.norm(point - projection))


@njit(fastmath=True)
def _dist_points_to_segments_2d(
    centers: np.ndarray, seg_starts: np.ndarray, seg_ends: np.ndarray
) -> np.ndarray:
    """Vectorised min-distance from multiple obstacle centres to multiple segments.

    Parameters
    ----------
    centers : (M, 2)  obstacle centre coordinates
    seg_starts : (N, 2)  segment start coordinates
    seg_ends : (N, 2)  segment end coordinates

    Returns
    -------
    distances : (N, M)  distance from each segment to each centre
    """
    n_seg = seg_starts.shape[0]
    n_obs = centers.shape[0]
    distances = np.empty((n_seg, n_obs), dtype=np.float64)

    for i in range(n_seg):
        sx0 = seg_starts[i, 0]
        sy0 = seg_starts[i, 1]
        sx1 = seg_ends[i, 0]
        sy1 = seg_ends[i, 1]
        
        dx = sx1 - sx0
        dy = sy1 - sy0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-30:
            seg_len_sq = 1e-30
            
        for j in range(n_obs):
            cx = centers[j, 0]
            cy = centers[j, 1]
            
            p_dx = cx - sx0
            p_dy = cy - sy0
            
            t = (p_dx * dx + p_dy * dy) / seg_len_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
                
            proj_x = sx0 + t * dx
            proj_y = sy0 + t * dy
            
            dist_x = cx - proj_x
            dist_y = cy - proj_y
            distances[i, j] = math.sqrt(dist_x * dist_x + dist_y * dist_y)

    return distances


@njit(fastmath=True)
def _bilinear_interpolate(height_map: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perform bilinear interpolation on the height map for given (x, y) coordinates.
    Expects 0-based coordinates.
    """
    h, w = height_map.shape
    n_pts = x.shape[0]
    out = np.empty(n_pts, dtype=np.float64)
    for i in range(n_pts):
        px = x[i]
        py = y[i]
        
        x0 = int(np.floor(px))
        x1 = x0 + 1
        y0 = int(np.floor(py))
        y1 = y0 + 1
        
        x0_c = min(max(x0, 0), w - 1)
        x1_c = min(max(x1, 0), w - 1)
        y0_c = min(max(y0, 0), h - 1)
        y1_c = min(max(y1, 0), h - 1)
        
        wx1 = min(max(float(x1) - px, 0.0), 1.0)
        wx0 = 1.0 - wx1
        wy1 = min(max(float(y1) - py, 0.0), 1.0)
        wy0 = 1.0 - wy1
        
        v00 = height_map[y0_c, x0_c]
        v10 = height_map[y1_c, x0_c]
        v01 = height_map[y0_c, x1_c]
        v11 = height_map[y1_c, x1_c]
        
        out[i] = v00 * wx1 * wy1 + v10 * wx1 * wy0 + v01 * wx0 * wy1 + v11 * wx0 * wy0
        
    return out


def evaluate_path_details(path_xyz: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate a UAV path and expose additional feasibility diagnostics."""
    infinite_cost = float("inf")
    path_xyz = np.asarray(path_xyz, dtype=float)
    if path_xyz.ndim != 2 or path_xyz.shape[1] != 3 or path_xyz.shape[0] < 2:
        return (
            np.array([infinite_cost, infinite_cost, infinite_cost, infinite_cost], dtype=float),
            {"collisionViolation": 1.0, "minClearance": float("-inf"), "maxTurnDeg": float("inf")},
        )

    x_coord = path_xyz[:, 0]
    y_coord = path_xyz[:, 1]
    z_absolute = path_xyz[:, 2]
    xmin = _value(model, "xmin")
    xmax = _value(model, "xmax")
    ymin = _value(model, "ymin")
    ymax = _value(model, "ymax")
    if np.any(x_coord < xmin) or np.any(x_coord > xmax) or np.any(y_coord < ymin) or np.any(y_coord > ymax):
        return (
            np.array([infinite_cost, infinite_cost, infinite_cost, infinite_cost], dtype=float),
            {"collisionViolation": 1.0, "minClearance": float("-inf"), "maxTurnDeg": float("inf")},
        )

    height_map = np.asarray(model["H"], dtype=float)
    # Use 0-based coordinates for bilinear lookup (MATLAB/Legacy uses 1-based indices)
    ground = _bilinear_interpolate(height_map, x_coord - 1.0, y_coord - 1.0)
    z_relative = z_absolute - ground

    start_point = path_xyz[0]
    end_point = path_xyz[-1]
    min_segment_length = 0.0
    if "rmin" in model and model["rmin"] is not None:
        min_segment_length = float(np.asarray(model["rmin"]).reshape(-1)[0])
    elif "n" in model and float(model["n"]) > 0:
        min_segment_length = float(np.linalg.norm(end_point - start_point) / (3.0 * float(model["n"])))

    segment_vectors = np.diff(path_xyz, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    if np.any(segment_lengths <= min_segment_length):
        first_objective = infinite_cost
    else:
        total_length = float(np.sum(segment_lengths))
        if total_length <= 0:
            first_objective = infinite_cost
        else:
            straight = float(np.linalg.norm(end_point - start_point))
            first_objective = 1.0 - straight / total_length

    # ── Build obstacle matrix ──────────────────────────────────────
    obstacles: list[np.ndarray] = []
    if "threats" in model and model["threats"] is not None:
        threat_array = np.asarray(model["threats"], dtype=float)
        if threat_array.ndim == 2 and threat_array.shape[1] >= 4:
            obstacles.append(threat_array[:, :4])
    if "nofly_c" in model and model["nofly_c"] is not None and "nofly_r" in model and model["nofly_r"] is not None:
        centers = np.asarray(model["nofly_c"], dtype=float)
        if centers.ndim == 1:
            centers = centers.reshape(1, -1)
        if centers.shape[1] >= 2:
            centers = centers[:, :2]
            radii = np.asarray(model["nofly_r"], dtype=float).reshape(-1)
            if radii.size == 1:
                radii = np.repeat(radii, centers.shape[0])
            elif radii.size < centers.shape[0]:
                radii = np.pad(radii, (0, centers.shape[0] - radii.size), mode="edge")
            nofly = np.column_stack([centers[:, 0], centers[:, 1], np.zeros(centers.shape[0]), radii[: centers.shape[0]]])
            obstacles.append(nofly)
    obstacle_matrix = np.vstack(obstacles) if obstacles else np.zeros((0, 4), dtype=float)

    drone_size = _value(model, "droneSize", "drone_size", default=1.0)
    safe_dist = _value(model, "safeDist", "safe_dist", default=10.0)
    step_size = _value(model, "collisionStep", default=1.0)
    if step_size <= 0:
        step_size = 1.0
    interpolated = _interpolate_path(path_xyz, step_size)
    x_interp = interpolated[:, 0]
    y_interp = interpolated[:, 1]
    z_interp_abs = interpolated[:, 2]

    ground_interp = _bilinear_interpolate(height_map, x_interp - 1.0, y_interp - 1.0)
    z_interp_rel = z_interp_abs - ground_interp

    # ── Objective 2: obstacle/terrain clearance (vectorised) ───────
    min_clearance_global = float("inf")
    if interpolated.shape[0] < 2:
        second_objective = 0.0
    else:
        n_seg = interpolated.shape[0] - 1
        # terrain clearance per segment: min of the two endpoints
        terrain_clearance = np.minimum(z_interp_rel[:-1], z_interp_rel[1:])  # (n_seg,)

        if obstacle_matrix.shape[0] > 0:
            seg_starts = np.column_stack([x_interp[:-1], y_interp[:-1]])  # (n_seg, 2)
            seg_ends = np.column_stack([x_interp[1:], y_interp[1:]])  # (n_seg, 2)
            obs_centers = obstacle_matrix[:, :2]  # (M, 2)
            obs_radii = obstacle_matrix[:, 3]  # (M,)
            # dist_matrix: (n_seg, M) — distance from each segment to each obstacle centre
            dist_matrix = _dist_points_to_segments_2d(obs_centers, seg_starts, seg_ends)
            # subtract radii → clearance to obstacle surface
            obs_clearance = dist_matrix - obs_radii[np.newaxis, :]  # (n_seg, M)
            min_obs_clearance = np.min(obs_clearance, axis=1)  # (n_seg,)
            min_clearance = np.minimum(terrain_clearance, min_obs_clearance)
        else:
            min_clearance = terrain_clearance

        # Use a continuous barrier-style penalty instead of an infinite wall.
        # This keeps optimization numerically stable while still strongly
        # discouraging collisions (clearance <= drone_size).
        safe_dist_eff = max(safe_dist, 1e-9)
        collision_scale = max(drone_size, 1e-9)
        segment_penalty = np.where(
            min_clearance >= drone_size + safe_dist_eff,
            0.0,
            np.where(
                min_clearance > drone_size,
                1.0 - (min_clearance - drone_size) / safe_dist_eff,
                1.0 + np.maximum(0.0, (drone_size - min_clearance) / collision_scale),
            ),
        )
        second_objective = float(np.sum(segment_penalty)) / max(1, n_seg)
        if min_clearance.size > 0:
            min_clearance_global = float(np.min(min_clearance))

    collision_margin = _value(model, "collisionHardMargin", default=0.0)
    collision_floor = max(0.0, drone_size + collision_margin)
    collision_violation = float(min_clearance_global <= collision_floor + 1e-9)

    # ── Objective 3: altitude deviation (vectorised) ──────────────
    zmax_val = _value(model, "zmax")
    zmin_val = _value(model, "zmin")
    if zmax_val <= zmin_val:
        third_objective = infinite_cost
    else:
        mean_altitude = (zmax_val + zmin_val) / 2.0
        bounds_tol = 1e-6
        out_of_bounds = (z_relative < zmin_val - bounds_tol) | (z_relative > zmax_val + bounds_tol)
        if np.any(out_of_bounds):
            third_objective = infinite_cost
        else:
            altitude_penalties = 2.0 * np.abs(z_relative - mean_altitude) / (zmax_val - zmin_val)
            third_objective = float(np.mean(altitude_penalties))

    # ── Objective 4: turning angle (vectorised) ──────────────────
    # Soft-penalize sharp spikes so a few near-90° turns cannot hide
    # behind many small turns.
    turn_limit_deg = _value(model, "maxTurnDeg", "maxTurnAngleDeg", default=75.0)
    turn_limit_rad = _value(model, "maxTurnRad", "maxTurnAngleRad", default=math.radians(turn_limit_deg))
    spike_weight = max(0.0, _value(model, "turnSpikePenaltyWeight", "j4SpikePenaltyWeight", default=1.0))
    if path_xyz.shape[0] < 3:
        fourth_objective = 0.0
        max_turn_deg = 0.0
    else:
        v1 = path_xyz[1:-1] - path_xyz[:-2]  # (N-2, 3)
        v2 = path_xyz[2:] - path_xyz[1:-1]   # (N-2, 3)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        valid = (n1 > 0) & (n2 > 0)
        if not np.any(valid):
            fourth_objective = 0.0
            max_turn_deg = 0.0
        else:
            cross_norms = np.linalg.norm(np.cross(v1[valid], v2[valid]), axis=1)
            dots = np.sum(v1[valid] * v2[valid], axis=1)
            angles = np.arctan2(cross_norms, dots)
            # Include zero angles for degenerate segments
            all_angles = np.zeros(v1.shape[0], dtype=float)
            all_angles[valid] = angles
            abs_angles = np.abs(all_angles)
            mean_turn = float(np.mean(abs_angles / math.pi))
            max_turn = float(np.max(abs_angles))
            excess = max(0.0, max_turn - turn_limit_rad)
            spike_penalty = spike_weight * (excess / math.pi)
            fourth_objective = mean_turn + spike_penalty
            max_turn_deg = float(np.degrees(max_turn))

    objective = np.array([first_objective, second_objective, third_objective, fourth_objective], dtype=float)
    if np.all(np.isfinite(objective)):
        objective = np.clip(objective, 0.0, 1.0)
    details = {
        "collisionViolation": collision_violation,
        "minClearance": float(min_clearance_global),
        "maxTurnDeg": max_turn_deg,
    }
    return objective, details


def evaluate_path(path_xyz: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Evaluate a UAV path against the 4-objective cost function.

    Objectives
    ----------
    J1 : path length ratio  (1 − straight/total)
    J2 : mean obstacle/terrain clearance penalty
    J3 : mean altitude deviation penalty
    J4 : mean turning-angle penalty with soft max-turn penalty
    """
    objective, _ = evaluate_path_details(path_xyz, model)
    return objective
