from __future__ import annotations

import math
from typing import Any

import numpy as np


def _value(model: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in model and model[key] is not None:
            return float(np.asarray(model[key]).reshape(-1)[0])
    return float(default)


def _interpolate_path(path_xyz: np.ndarray, step_size: float) -> np.ndarray:
    """Interpolate path segments so no gap exceeds *step_size*."""
    if path_xyz.shape[0] < 2:
        return path_xyz.copy()
    diffs = np.diff(path_xyz, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    steps_per_seg = np.maximum(1, np.ceil(distances / step_size).astype(int))
    total_points = 1 + int(np.sum(steps_per_seg))
    result = np.empty((total_points, 3), dtype=float)
    result[0] = path_xyz[0]
    cursor = 1
    for seg_idx in range(path_xyz.shape[0] - 1):
        n_steps = steps_per_seg[seg_idx]
        t = np.arange(1, n_steps + 1, dtype=float) / n_steps
        result[cursor : cursor + n_steps] = (
            (1.0 - t[:, np.newaxis]) * path_xyz[seg_idx] + t[:, np.newaxis] * path_xyz[seg_idx + 1]
        )
        cursor += n_steps
    return result[:cursor]


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
    # seg_dirs: (N, 2)
    seg_dirs = seg_ends - seg_starts
    seg_len_sq = np.sum(seg_dirs ** 2, axis=1, keepdims=True)  # (N, 1)
    seg_len_sq = np.maximum(seg_len_sq, 1e-30)  # avoid /0

    # diff: (N, M, 2) via broadcasting  — centres is (1, M, 2)
    diff = centers[np.newaxis, :, :] - seg_starts[:, np.newaxis, :]  # (N, M, 2)
    # project: (N, M)
    t = np.sum(diff * seg_dirs[:, np.newaxis, :], axis=2) / seg_len_sq  # (N, M)
    t = np.clip(t, 0.0, 1.0)
    # projection points: (N, M, 2)
    proj = seg_starts[:, np.newaxis, :] + t[:, :, np.newaxis] * seg_dirs[:, np.newaxis, :]
    return np.linalg.norm(centers[np.newaxis, :, :] - proj, axis=2)  # (N, M)


def evaluate_path(path_xyz: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Evaluate a UAV path against the 4-objective cost function.

    Objectives
    ----------
    J1 : path length ratio  (1 − straight/total)
    J2 : mean obstacle/terrain clearance penalty
    J3 : mean altitude deviation penalty
    J4 : mean turning-angle penalty with soft max-turn penalty
    """
    infinite_cost = float("inf")
    path_xyz = np.asarray(path_xyz, dtype=float)
    if path_xyz.ndim != 2 or path_xyz.shape[1] != 3 or path_xyz.shape[0] < 2:
        return np.array([infinite_cost, infinite_cost, infinite_cost, infinite_cost], dtype=float)

    x_coord = path_xyz[:, 0]
    y_coord = path_xyz[:, 1]
    z_absolute = path_xyz[:, 2]
    xmin = _value(model, "xmin")
    xmax = _value(model, "xmax")
    ymin = _value(model, "ymin")
    ymax = _value(model, "ymax")
    if np.any(x_coord < xmin) or np.any(x_coord > xmax) or np.any(y_coord < ymin) or np.any(y_coord > ymax):
        return np.array([infinite_cost, infinite_cost, infinite_cost, infinite_cost], dtype=float)

    height_map = np.asarray(model["H"], dtype=float)
    x_index = np.clip(np.rint(x_coord).astype(int), 1, int(xmax)) - 1
    y_index = np.clip(np.rint(y_coord).astype(int), 1, int(ymax)) - 1
    ground = height_map[y_index, x_index]
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
    x_index_interp = np.clip(np.rint(x_interp).astype(int), 1, int(xmax)) - 1
    y_index_interp = np.clip(np.rint(y_interp).astype(int), 1, int(ymax)) - 1
    ground_interp = height_map[y_index_interp, x_index_interp]
    z_interp_rel = z_interp_abs - ground_interp

    # ── Objective 2: obstacle/terrain clearance (vectorised) ───────
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
    else:
        v1 = path_xyz[1:-1] - path_xyz[:-2]  # (N-2, 3)
        v2 = path_xyz[2:] - path_xyz[1:-1]   # (N-2, 3)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        valid = (n1 > 0) & (n2 > 0)
        if not np.any(valid):
            fourth_objective = 0.0
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

    return np.array([first_objective, second_objective, third_objective, fourth_objective], dtype=float)
