from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import _constraint_violation
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.evaluate_path import (
    _bilinear_interpolate,
    _dist_points_to_segments_2d,
    _interpolate_path,
)
from uav_benchmark.core.mission_encoding import paths_to_decision
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort

_ISSUE_PAIR = 1
_ISSUE_CLEARANCE = 2
_ISSUE_TURN = 3
_ISSUE_TRADE = 4


@dataclass(slots=True)
class ConflictTopology:
    issue_code: int
    key: tuple[int, ...]
    target_uav: int
    other_uav: int
    center_ratio: float
    center_index: int
    span: int
    severity: float
    dominant_objective: int


def _is_feasible(candidate: Candidate) -> bool:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    return bool(float(details.get("feasible", 0.0)) > 0.5 and np.all(np.isfinite(objective)))


def _strict_objective_sum(candidate: Candidate) -> float:
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    if objective.size == 0 or np.any(~np.isfinite(objective)):
        return float("inf")
    return float(np.sum(objective))


def _objective_bins(objective: np.ndarray, bins: int = 5) -> tuple[int, ...]:
    values = np.asarray(objective, dtype=float).reshape(-1)
    if values.size == 0:
        return (0, 0, 0, 0)
    clipped = np.clip(np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    quantized = np.floor(clipped * float(bins)).astype(int)
    quantized = np.clip(quantized, 0, bins - 1)
    if quantized.size < 4:
        quantized = np.pad(quantized, (0, 4 - quantized.size), mode="edge")
    return tuple(int(item) for item in quantized[:4])


def _ratio_to_index(n_points: int, ratio: float) -> int:
    if n_points <= 2:
        return 1 if n_points > 1 else 0
    clipped = float(np.clip(ratio, 0.0, 1.0))
    return int(np.clip(round(clipped * (n_points - 1)), 1, n_points - 2))


def _window_indices(n_points: int, center_index: int, span: int) -> np.ndarray:
    if n_points <= 2:
        return np.zeros(0, dtype=int)
    lo = max(1, int(center_index) - int(span))
    hi = min(n_points - 2, int(center_index) + int(span))
    if lo > hi:
        return np.zeros(0, dtype=int)
    return np.arange(lo, hi + 1, dtype=int)


def _gaussian_weights(indices: np.ndarray, center_index: int, span: int) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(0, dtype=float)
    width = max(1.0, float(span))
    delta = (indices.astype(float) - float(center_index)) / width
    return np.exp(-0.5 * delta * delta)


def _dominant_objective(candidate: Candidate) -> int:
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    if objective.size == 0:
        return 0
    finite = np.where(np.isfinite(objective), objective, -np.inf)
    if not np.any(np.isfinite(finite)):
        return 0
    return int(np.argmax(finite))


def _path_scores(candidate: Candidate) -> np.ndarray:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    matrix = np.asarray(details.get("pathObjectives", np.zeros((0, 4), dtype=float)), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros(0, dtype=float)
    finite = np.where(np.isfinite(matrix), matrix, 1.0)
    return np.sum(finite, axis=1)


def _trade_target_uav(candidate: Candidate, dominant_objective: int) -> int:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    matrix = np.asarray(details.get("pathObjectives", np.zeros((0, 4), dtype=float)), dtype=float)
    if matrix.ndim == 2 and matrix.shape[0] > 0:
        column = matrix[:, min(dominant_objective, matrix.shape[1] - 1)]
        finite = np.where(np.isfinite(column), column, -np.inf)
        if np.any(np.isfinite(finite)):
            return int(np.argmax(finite))
        scores = _path_scores(candidate)
        if scores.size > 0:
            return int(np.argmax(scores))
    return 0


def _build_obstacle_matrix(model: dict[str, Any]) -> np.ndarray:
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
            radii = np.asarray(model["nofly_r"], dtype=float).reshape(-1)
            if radii.size == 1:
                radii = np.repeat(radii, centers.shape[0])
            elif radii.size < centers.shape[0]:
                radii = np.pad(radii, (0, centers.shape[0] - radii.size), mode="edge")
            nofly = np.column_stack(
                [centers[:, 0], centers[:, 1], np.zeros(centers.shape[0]), radii[: centers.shape[0]]]
            )
            obstacles.append(nofly)
    if not obstacles:
        return np.zeros((0, 4), dtype=float)
    return np.vstack(obstacles).astype(float)


def _path_clearance_profile(path_xyz: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = np.asarray(path_xyz, dtype=float)
    if path.shape[0] < 2:
        return path.copy(), np.zeros(0, dtype=float)

    step_size = float(model.get("collisionStep", 1.0))
    if step_size <= 0.0:
        step_size = 1.0
    interpolated = _interpolate_path(path, step_size)
    if interpolated.shape[0] < 2:
        return interpolated, np.zeros(0, dtype=float)

    height_map = np.asarray(model["H"], dtype=float)
    x_interp = np.asarray(interpolated[:, 0], dtype=float)
    y_interp = np.asarray(interpolated[:, 1], dtype=float)
    ground_interp = _bilinear_interpolate(height_map, x_interp - 1.0, y_interp - 1.0)
    z_rel = np.asarray(interpolated[:, 2], dtype=float) - ground_interp
    terrain_clearance = np.minimum(z_rel[:-1], z_rel[1:])

    obstacle_matrix = np.asarray(model.get("_tacticObstacleMatrix", np.zeros((0, 4), dtype=float)), dtype=float)
    if obstacle_matrix.ndim != 2 or obstacle_matrix.shape[0] == 0:
        return interpolated, terrain_clearance

    seg_starts = np.column_stack([x_interp[:-1], y_interp[:-1]])
    seg_ends = np.column_stack([x_interp[1:], y_interp[1:]])
    distance_matrix = _dist_points_to_segments_2d(obstacle_matrix[:, :2], seg_starts, seg_ends)
    obstacle_clearance = distance_matrix - obstacle_matrix[:, 3].reshape(1, -1)
    min_obstacle_clearance = np.min(obstacle_clearance, axis=1)
    return interpolated, np.minimum(terrain_clearance, min_obstacle_clearance)


def _turn_profile_deg(path_xyz: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float)
    if path.shape[0] < 3:
        return np.zeros(0, dtype=float)
    v1 = path[1:-1] - path[:-2]
    v2 = path[2:] - path[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-12) & (n2 > 1e-12)
    angles = np.zeros(v1.shape[0], dtype=float)
    if np.any(valid):
        cross_norm = np.linalg.norm(np.cross(v1[valid], v2[valid]), axis=1)
        dots = np.sum(v1[valid] * v2[valid], axis=1)
        angles[valid] = np.degrees(np.abs(np.arctan2(cross_norm, dots)))
    return angles


def _path_perpendicular(path_xyz: np.ndarray, center_index: int) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float)
    left = max(0, int(center_index) - 1)
    right = min(path.shape[0] - 1, int(center_index) + 1)
    direction = path[right, :2] - path[left, :2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = path[-1, :2] - path[0, :2]
        norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    direction = direction / norm
    return np.array([-direction[1], direction[0]], dtype=float)


def _smooth_window(path_xyz: np.ndarray, start_idx: int, end_idx: int, passes: int = 1) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    if path.shape[0] <= 3:
        return path
    lo = max(1, int(start_idx))
    hi = min(path.shape[0] - 2, int(end_idx))
    if lo > hi:
        return path
    for _ in range(max(1, int(passes))):
        updated = path.copy()
        for idx in range(lo, hi + 1):
            updated[idx] = 0.25 * path[idx - 1] + 0.5 * path[idx] + 0.25 * path[idx + 1]
        path = updated
    return path


def _clip_absolute_path(path_xyz: np.ndarray, model: dict[str, Any], original: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    xmin = float(model["xmin"])
    xmax = float(model["xmax"])
    ymin = float(model["ymin"])
    ymax = float(model["ymax"])
    zmin = float(model["zmin"])
    zmax = float(model["zmax"])
    safe_h = float(model["safeH"]) if "safeH" in model and model["safeH"] is not None else None

    path[:, 0] = np.clip(path[:, 0], xmin, xmax)
    path[:, 1] = np.clip(path[:, 1], ymin, ymax)
    ground = _bilinear_interpolate(np.asarray(model["H"], dtype=float), path[:, 0] - 1.0, path[:, 1] - 1.0)
    z_lo = ground + zmin
    if safe_h is not None:
        z_lo = np.maximum(z_lo, ground + safe_h)
    z_hi = ground + zmax
    path[:, 2] = np.clip(path[:, 2], z_lo, z_hi)
    path[0] = np.asarray(original[0], dtype=float)
    path[-1] = np.asarray(original[-1], dtype=float)
    return path


def _extract_topology(candidate: Candidate, model: dict[str, Any]) -> ConflictTopology:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    cached = details.get("_tacticTopology")
    if isinstance(cached, ConflictTopology):
        return cached

    paths = [np.asarray(path, dtype=float) for path in details.get("paths", [])]
    dominant_objective = _dominant_objective(candidate)
    default_target = _trade_target_uav(candidate, dominant_objective)

    n_samples = int(max(20, max(path.shape[0] for path in paths) * 4)) if paths else 20

    conflict_log = np.asarray(details.get("conflictLog", np.zeros((0, 5), dtype=float)), dtype=float)
    if conflict_log.ndim == 2 and conflict_log.shape[0] > 0:
        worst_idx = int(np.argmax(conflict_log[:, 4]))
        worst = conflict_log[worst_idx]
        left_idx = int(np.clip(round(worst[1]), 0, max(0, len(paths) - 1)))
        right_idx = int(np.clip(round(worst[2]), 0, max(0, len(paths) - 1)))
        path_scores = _path_scores(candidate)
        if path_scores.size > max(left_idx, right_idx):
            target_idx = left_idx if path_scores[left_idx] >= path_scores[right_idx] else right_idx
        else:
            target_idx = left_idx
        ratio = float(worst[0] / max(1.0, float(n_samples - 1)))
        center_index = _ratio_to_index(paths[target_idx].shape[0], ratio) if paths else 0
        separation_min = float(model.get("separationMin", model.get("safeDist", 10.0)))
        severity = max(0.0, float(worst[4])) / max(separation_min, 1e-9)
        severity = float(np.clip(severity, 0.0, 1.5))
        time_bin = int(np.clip(np.floor(ratio * 5.0), 0, 4))
        sev_bin = int(np.clip(np.floor(min(0.999, severity) * 4.0), 0, 3))
        topology = ConflictTopology(
            issue_code=_ISSUE_PAIR,
            key=(_ISSUE_PAIR, min(left_idx, right_idx), max(left_idx, right_idx), time_bin, sev_bin),
            target_uav=target_idx,
            other_uav=right_idx if target_idx == left_idx else left_idx,
            center_ratio=ratio,
            center_index=center_index,
            span=max(1, min(3, 1 + sev_bin)),
            severity=severity,
            dominant_objective=dominant_objective,
        )
        details["_tacticTopology"] = topology
        return topology

    drone_size = float(model.get("droneSize", 1.0))
    safe_dist = float(model.get("safeDist", 10.0))
    clearance_limit = drone_size + 0.25 * safe_dist
    best_clearance = float("inf")
    best_clearance_uav = default_target
    best_clearance_ratio = 0.5
    for uav_idx, path in enumerate(paths):
        _interp, clearance = _path_clearance_profile(path, model)
        if clearance.size == 0:
            continue
        local_idx = int(np.argmin(clearance))
        local_clearance = float(clearance[local_idx])
        if local_clearance < best_clearance:
            best_clearance = local_clearance
            best_clearance_uav = uav_idx
            best_clearance_ratio = float(local_idx / max(1, clearance.size - 1))
    if paths and (float(details.get("collisionViolation", 0.0)) > 0.5 or best_clearance < clearance_limit):
        center_index = _ratio_to_index(paths[best_clearance_uav].shape[0], best_clearance_ratio)
        severity = max(0.0, clearance_limit - best_clearance) / max(clearance_limit, 1e-9)
        severity = float(np.clip(severity, 0.0, 1.5))
        region_bin = int(np.clip(np.floor(best_clearance_ratio * 5.0), 0, 4))
        sev_bin = int(np.clip(np.floor(min(0.999, severity) * 4.0), 0, 3))
        topology = ConflictTopology(
            issue_code=_ISSUE_CLEARANCE,
            key=(_ISSUE_CLEARANCE, best_clearance_uav, region_bin, sev_bin),
            target_uav=best_clearance_uav,
            other_uav=-1,
            center_ratio=best_clearance_ratio,
            center_index=center_index,
            span=max(1, min(3, 1 + sev_bin)),
            severity=severity,
            dominant_objective=dominant_objective,
        )
        details["_tacticTopology"] = topology
        return topology

    turn_limit = float(model.get("maxTurnDeg", model.get("maxTurnAngleDeg", 75.0)))
    best_excess = 0.0
    best_turn_uav = default_target
    best_turn_ratio = 0.5
    for uav_idx, path in enumerate(paths):
        turn_deg = _turn_profile_deg(path)
        if turn_deg.size == 0:
            continue
        local_idx = int(np.argmax(turn_deg))
        excess = max(0.0, float(turn_deg[local_idx]) - turn_limit)
        if excess > best_excess:
            best_excess = excess
            best_turn_uav = uav_idx
            best_turn_ratio = float((local_idx + 1) / max(1, path.shape[0] - 1))
    if paths and (float(details.get("turnViolation", 0.0)) > 0.5 or best_excess > 1e-6):
        center_index = _ratio_to_index(paths[best_turn_uav].shape[0], best_turn_ratio)
        severity = float(np.clip(best_excess / max(turn_limit, 1e-9), 0.0, 1.5))
        region_bin = int(np.clip(np.floor(best_turn_ratio * 5.0), 0, 4))
        sev_bin = int(np.clip(np.floor(min(0.999, severity) * 4.0), 0, 3))
        topology = ConflictTopology(
            issue_code=_ISSUE_TURN,
            key=(_ISSUE_TURN, best_turn_uav, region_bin, sev_bin),
            target_uav=best_turn_uav,
            other_uav=-1,
            center_ratio=best_turn_ratio,
            center_index=center_index,
            span=max(1, min(2, 1 + sev_bin)),
            severity=severity,
            dominant_objective=dominant_objective,
        )
        details["_tacticTopology"] = topology
        return topology

    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    obj_bins = _objective_bins(objective)
    topology = ConflictTopology(
        issue_code=_ISSUE_TRADE,
        key=(_ISSUE_TRADE, dominant_objective, *obj_bins),
        target_uav=default_target,
        other_uav=-1,
        center_ratio=0.5,
        center_index=_ratio_to_index(paths[default_target].shape[0], 0.5) if paths else 0,
        span=2,
        severity=float(np.max(np.where(np.isfinite(objective), objective, 0.0))) if objective.size > 0 else 0.0,
        dominant_objective=dominant_objective,
    )
    details["_tacticTopology"] = topology
    return topology


def _topology_priority(
    candidate: Candidate, crowd_value: float, occupancy: dict[tuple[int, ...], int], model: dict[str, Any]
) -> tuple[float, float, float, float]:
    key = _extract_topology(candidate, model).key
    occ = float(occupancy.get(key, 0))
    if np.isinf(crowd_value):
        crowd_rank = 0.0
        crowd_term = -1e12
    else:
        crowd_rank = 1.0
        crowd_term = -float(crowd_value)
    return (occ, crowd_rank, crowd_term, _strict_objective_sum(candidate))


def _greedy_topology_order(
    candidates: list[Candidate],
    indices: list[int],
    crowding: np.ndarray,
    occupancy: dict[tuple[int, ...], int],
    model: dict[str, Any],
) -> list[int]:
    remaining = [int(idx) for idx in indices]
    ordered: list[int] = []
    while remaining:
        best_pos = min(
            range(len(remaining)),
            key=lambda pos: _topology_priority(
                candidates[remaining[pos]],
                float(crowding[remaining[pos]]) if remaining[pos] < crowding.size else 0.0,
                occupancy,
                model,
            ),
        )
        chosen = remaining.pop(best_pos)
        ordered.append(chosen)
        key = _extract_topology(candidates[chosen], model).key
        occupancy[key] = occupancy.get(key, 0) + 1
    return ordered


def _select_next_population(pool: list[Candidate], n_keep: int, model: dict[str, Any]) -> list[Candidate]:
    if not pool or n_keep <= 0:
        return []

    selected: list[Candidate] = []
    occupancy: dict[tuple[int, ...], int] = {}
    feasible = [candidate for candidate in pool if _is_feasible(candidate)]
    if feasible:
        obj = _candidate_matrix(feasible)
        front_no, _ = n_d_sort(obj.copy(), None, len(feasible))
        crowd = crowding_distance(obj, front_no)
        fronts = sorted(int(item) for item in np.unique(front_no[np.isfinite(front_no)]))
        for front in fronts:
            members = np.where(front_no == float(front))[0].tolist()
            if not members:
                continue
            ordered = _greedy_topology_order(feasible, members, crowd, occupancy, model)
            space = n_keep - len(selected)
            if space <= 0:
                break
            for idx in ordered[:space]:
                selected.append(feasible[idx])
            if len(selected) >= n_keep:
                return selected

    if len(selected) >= n_keep:
        return selected[:n_keep]

    infeasible = [candidate for candidate in pool if not _is_feasible(candidate)]
    remaining = list(infeasible)
    while remaining and len(selected) < n_keep:
        best_pos = min(
            range(len(remaining)),
            key=lambda pos: (
                occupancy.get(_extract_topology(remaining[pos], model).key, 0),
                float(_constraint_violation(remaining[pos], model)),
                _strict_objective_sum(remaining[pos]),
            ),
        )
        chosen = remaining.pop(best_pos)
        selected.append(chosen)
        key = _extract_topology(chosen, model).key
        occupancy[key] = occupancy.get(key, 0) + 1
    return selected[:n_keep]


def _update_topology_archive(
    archive: list[Candidate], new_candidates: list[Candidate], max_size: int, model: dict[str, Any]
) -> list[Candidate]:
    feasible_pool = [candidate for candidate in (list(archive) + list(new_candidates)) if _is_feasible(candidate)]
    if not feasible_pool or max_size <= 0:
        return []
    obj = _candidate_matrix(feasible_pool)
    front_no, _ = n_d_sort(obj.copy(), None, len(feasible_pool))
    first_front_idx = np.where(front_no == 1.0)[0]
    if first_front_idx.size == 0:
        return []
    front_candidates = [feasible_pool[int(idx)] for idx in first_front_idx.tolist()]
    if len(front_candidates) <= max_size:
        return front_candidates
    front_obj = _candidate_matrix(front_candidates)
    crowd = crowding_distance(front_obj, np.ones(front_obj.shape[0], dtype=float))
    ordered = _greedy_topology_order(front_candidates, list(range(len(front_candidates))), crowd, {}, model)
    return [front_candidates[idx] for idx in ordered[:max_size]]


def _choose_guide(parent: Candidate, archive: list[Candidate], model: dict[str, Any]) -> Candidate | None:
    if not archive:
        return None
    parent_topology = _extract_topology(parent, model)
    dominant = parent_topology.dominant_objective
    parent_vector = np.asarray(parent.vector, dtype=float)
    candidates = [
        candidate
        for candidate in archive
        if not np.allclose(np.asarray(candidate.vector, dtype=float), parent_vector, atol=1e-8, rtol=0.0)
    ]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            0 if _extract_topology(candidate, model).key != parent_topology.key else 1,
            float(np.asarray(candidate.objective, dtype=float).reshape(-1)[dominant])
            if np.all(np.isfinite(candidate.objective))
            else float("inf"),
            _strict_objective_sum(candidate),
        ),
    )
    if not ranked:
        return None
    top_k = min(5, len(ranked))
    return ranked[int(np.random.randint(0, top_k))]


def _edit_pair_conflict(paths: list[np.ndarray], topology: ConflictTopology, model: dict[str, Any]) -> list[np.ndarray]:
    target_idx = int(np.clip(topology.target_uav, 0, len(paths) - 1))
    other_idx = int(np.clip(topology.other_uav, 0, len(paths) - 1))
    target = np.asarray(paths[target_idx], dtype=float).copy()
    other = np.asarray(paths[other_idx], dtype=float)
    window = _window_indices(target.shape[0], topology.center_index, topology.span)
    if window.size == 0:
        return paths

    other_center = _ratio_to_index(other.shape[0], topology.center_ratio)
    delta = target[topology.center_index, :2] - other[other_center, :2]
    norm = float(np.linalg.norm(delta))
    direction = _path_perpendicular(target, topology.center_index) if norm <= 1e-12 else delta / norm

    separation_min = float(model.get("separationMin", model.get("safeDist", 10.0)))
    magnitude = max(0.5, topology.severity * separation_min * 0.75)
    weights = _gaussian_weights(window, topology.center_index, topology.span)
    target[window, 0] += direction[0] * magnitude * weights
    target[window, 1] += direction[1] * magnitude * weights
    target[window, 2] += 0.12 * magnitude * weights
    target = _smooth_window(target, int(window[0]), int(window[-1]), passes=1)
    updated = list(paths)
    updated[target_idx] = _clip_absolute_path(target, model, np.asarray(paths[target_idx], dtype=float))
    return updated


def _edit_clearance_issue(
    paths: list[np.ndarray], topology: ConflictTopology, model: dict[str, Any]
) -> list[np.ndarray]:
    target_idx = int(np.clip(topology.target_uav, 0, len(paths) - 1))
    target = np.asarray(paths[target_idx], dtype=float).copy()
    window = _window_indices(target.shape[0], topology.center_index, topology.span)
    if window.size == 0:
        return paths

    drone_size = float(model.get("droneSize", 1.0))
    safe_dist = float(model.get("safeDist", 10.0))
    lift = max(1.0, topology.severity * (drone_size + 0.40 * safe_dist))
    direction = _path_perpendicular(target, topology.center_index)
    weights = _gaussian_weights(window, topology.center_index, topology.span)
    target[window, 2] += lift * weights
    target[window, 0] += direction[0] * (0.15 * safe_dist) * weights
    target[window, 1] += direction[1] * (0.15 * safe_dist) * weights
    target = _smooth_window(target, int(window[0]), int(window[-1]), passes=1)
    updated = list(paths)
    updated[target_idx] = _clip_absolute_path(target, model, np.asarray(paths[target_idx], dtype=float))
    return updated


def _edit_turn_issue(paths: list[np.ndarray], topology: ConflictTopology, model: dict[str, Any]) -> list[np.ndarray]:
    target_idx = int(np.clip(topology.target_uav, 0, len(paths) - 1))
    target = np.asarray(paths[target_idx], dtype=float).copy()
    center = int(np.clip(topology.center_index, 1, max(1, target.shape[0] - 2)))
    if target.shape[0] > 2:
        anchor = 0.5 * (target[center - 1] + target[center + 1])
        target[center] = 0.35 * target[center] + 0.65 * anchor
    window = _window_indices(target.shape[0], center, max(1, topology.span))
    if window.size > 0:
        target = _smooth_window(target, int(window[0]), int(window[-1]), passes=2)
    updated = list(paths)
    updated[target_idx] = _clip_absolute_path(target, model, np.asarray(paths[target_idx], dtype=float))
    return updated


def _straighten_window(path_xyz: np.ndarray, indices: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    if indices.size == 0:
        return path
    left = path[int(indices[0]) - 1]
    right = path[int(indices[-1]) + 1]
    total = float(indices.size + 1)
    for offset, idx in enumerate(indices, start=1):
        alpha = float(offset) / total
        straight = (1.0 - alpha) * left + alpha * right
        path[int(idx)] = 0.70 * path[int(idx)] + 0.30 * straight
    return path


def _edit_tradeoff(
    parent: Candidate,
    archive: list[Candidate],
    topology: ConflictTopology,
    model: dict[str, Any],
    progress: float,
) -> list[np.ndarray]:
    paths = [np.asarray(path, dtype=float).copy() for path in parent.details.get("paths", [])]
    if not paths:
        return paths
    target_idx = int(np.clip(topology.target_uav, 0, len(paths) - 1))
    target = np.asarray(paths[target_idx], dtype=float).copy()
    window = _window_indices(target.shape[0], topology.center_index, topology.span)
    if window.size == 0:
        return paths

    guide = _choose_guide(parent, archive, model)
    if guide is not None:
        guide_paths = [np.asarray(path, dtype=float) for path in guide.details.get("paths", [])]
        if len(guide_paths) > target_idx and guide_paths[target_idx].shape == target.shape:
            alpha = 0.18 + 0.12 * (1.0 - float(np.clip(progress, 0.0, 1.0)))
            target[window] = (1.0 - alpha) * target[window] + alpha * guide_paths[target_idx][window]

    dominant = int(topology.dominant_objective)
    safe_dist = float(model.get("safeDist", 10.0))
    if dominant == 0:
        target = _straighten_window(target, window)
    elif dominant == 1:
        direction = _path_perpendicular(target, topology.center_index)
        weights = _gaussian_weights(window, topology.center_index, topology.span)
        target[window, 0] += direction[0] * (0.12 * safe_dist) * weights
        target[window, 1] += direction[1] * (0.12 * safe_dist) * weights
        target[window, 2] += (0.10 * safe_dist) * weights
    elif dominant == 2:
        ground = _bilinear_interpolate(
            np.asarray(model["H"], dtype=float), target[window, 0] - 1.0, target[window, 1] - 1.0
        )
        target_abs = ground + 0.5 * (float(model["zmin"]) + float(model["zmax"]))
        target[window, 2] = 0.75 * target[window, 2] + 0.25 * target_abs
    else:
        target = _smooth_window(target, int(window[0]), int(window[-1]), passes=2)

    target = _smooth_window(target, int(window[0]), int(window[-1]), passes=1)
    paths[target_idx] = _clip_absolute_path(target, model, np.asarray(parent.details["paths"][target_idx], dtype=float))
    return paths


def _topology_edit_vector(
    parent: Candidate,
    archive: list[Candidate],
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    lower: np.ndarray,
    upper: np.ndarray,
    progress: float,
) -> tuple[np.ndarray, ConflictTopology]:
    paths = [np.asarray(path, dtype=float).copy() for path in parent.details.get("paths", [])]
    topology = _extract_topology(parent, model)

    if topology.issue_code == _ISSUE_PAIR:
        edited_paths = _edit_pair_conflict(paths, topology, model)
    elif topology.issue_code == _ISSUE_CLEARANCE:
        edited_paths = _edit_clearance_issue(paths, topology, model)
    elif topology.issue_code == _ISSUE_TURN:
        edited_paths = _edit_turn_issue(paths, topology, model)
    else:
        edited_paths = _edit_tradeoff(parent, archive, topology, model, progress)

    if not edited_paths:
        edited_vector = np.asarray(parent.vector, dtype=float).copy()
    else:
        edited_vector = paths_to_decision(edited_paths, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        edited_vector = np.asarray(edited_vector, dtype=float).reshape(-1)

    if np.allclose(edited_vector, np.asarray(parent.vector, dtype=float), atol=1e-8, rtol=0.0):
        guide = _choose_guide(parent, archive, model)
        if guide is not None:
            edited_vector = 0.88 * np.asarray(parent.vector, dtype=float) + 0.12 * np.asarray(guide.vector, dtype=float)
        if np.allclose(edited_vector, np.asarray(parent.vector, dtype=float), atol=1e-8, rtol=0.0):
            span = upper - lower
            noise = np.random.normal(0.0, 0.01 + 0.01 * (1.0 - progress), size=span.shape) * span
            edited_vector = np.asarray(parent.vector, dtype=float) + noise
    return np.clip(edited_vector, lower, upper), topology
