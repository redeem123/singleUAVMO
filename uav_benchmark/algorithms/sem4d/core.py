from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.core.evaluate_path import evaluate_path_details
from uav_benchmark.core.mission_encoding import decision_to_paths


@dataclass(slots=True)
class _SEM4DIndividual:
    vector: np.ndarray
    objective: np.ndarray
    task_id: int
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class _ShieldConfig:
    iterations: int
    time_samples: int
    max_insertions: int
    separation_buffer: float
    nofly_buffer: float
    dynamic_buffer: float
    repair_gain: float
    energy_max: float | None
    max_turn_deg: float
    smoothing_passes: int


@dataclass(slots=True)
class _ShieldReport:
    correction_norm: float = 0.0
    terrain_corrections: int = 0
    inter_uav_corrections: int = 0
    dynamic_obstacle_corrections: int = 0
    nofly_corrections: int = 0
    energy_corrections: int = 0
    motion_corrections: int = 0
    energy_violation: float = 0.0
    terrain_risk: float = 0.0
    dynamic_risk: float = 0.0
    nofly_risk: float = 0.0


def _extra_float(params: BenchmarkParams, key: str, default: float) -> float:
    return float(params.extra.get(key, params.extra.get(_camel_to_snake(key), default)))


def _extra_int(params: BenchmarkParams, key: str, default: int) -> int:
    return int(params.extra.get(key, params.extra.get(_camel_to_snake(key), default)))


def _camel_to_snake(text: str) -> str:
    out = []
    for index, char in enumerate(str(text)):
        if char.isupper() and index > 0:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _shield_config(model: dict[str, Any], params: BenchmarkParams) -> _ShieldConfig:
    energy_raw = params.extra.get(
        "sem4dEnergyMax",
        params.extra.get("sem4d_energy_max", model.get("energyMax", model.get("maxEnergy"))),
    )
    energy_max: float | None
    if energy_raw is None:
        energy_max = None
    else:
        energy_value = float(np.asarray(energy_raw, dtype=float).reshape(-1)[0])
        energy_max = energy_value if energy_value > 0.0 else None

    return _ShieldConfig(
        iterations=max(1, _extra_int(params, "sem4dShieldIterations", 3)),
        time_samples=max(8, _extra_int(params, "sem4dTimeSamples", 48)),
        max_insertions=max(1, _extra_int(params, "sem4dMaxShieldInsertions", 8)),
        separation_buffer=max(0.0, _extra_float(params, "sem4dSeparationBuffer", 0.15)),
        nofly_buffer=max(0.0, _extra_float(params, "sem4dNoFlyBuffer", 2.0)),
        dynamic_buffer=max(0.0, _extra_float(params, "sem4dDynamicBuffer", 2.0)),
        repair_gain=float(np.clip(_extra_float(params, "sem4dRepairGain", 0.65), 0.05, 2.0)),
        energy_max=energy_max,
        max_turn_deg=float(params.max_turn_deg),
        smoothing_passes=max(1, _extra_int(params, "sem4dSmoothingPasses", 2)),
    )


def _ground_height(model: dict[str, Any], x: float, y: float) -> float:
    height_map = np.asarray(model["H"], dtype=float)
    xmax = int(float(model["xmax"]))
    ymax = int(float(model["ymax"]))
    px = float(np.clip(x, 1.0, float(xmax))) - 1.0
    py = float(np.clip(y, 1.0, float(ymax))) - 1.0
    x0 = max(0, min(int(np.floor(px)), xmax - 1))
    y0 = max(0, min(int(np.floor(py)), ymax - 1))
    x1 = min(x0 + 1, xmax - 1)
    y1 = min(y0 + 1, ymax - 1)
    tx = px - float(x0)
    ty = py - float(y0)
    v00 = float(height_map[y0, x0])
    v01 = float(height_map[y0, x1])
    v10 = float(height_map[y1, x0])
    v11 = float(height_map[y1, x1])
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v01 + (1.0 - tx) * ty * v10 + tx * ty * v11


def _enforce_path_bounds(path: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    bounded = np.asarray(path, dtype=float).copy()
    if bounded.shape[0] == 0:
        return bounded
    xmin = float(model["xmin"])
    xmax = float(model["xmax"])
    ymin = float(model["ymin"])
    ymax = float(model["ymax"])
    zmin = float(model["zmin"])
    zmax = float(model["zmax"])
    safe_h_raw = model.get("safeH")
    safe_h = None if safe_h_raw is None else float(np.asarray(safe_h_raw).reshape(-1)[0])
    drone_size = float(model.get("droneSize", model.get("drone_size", 1.0)))
    safe_h = max(float(safe_h) if safe_h is not None else zmin, drone_size + 0.25)
    bounded[:, 0] = np.clip(bounded[:, 0], xmin, xmax)
    bounded[:, 1] = np.clip(bounded[:, 1], ymin, ymax)
    for index in range(bounded.shape[0]):
        ground = _ground_height(model, float(bounded[index, 0]), float(bounded[index, 1]))
        rel_z = float(bounded[index, 2] - ground)
        if safe_h is not None:
            rel_z = max(rel_z, safe_h)
        rel_z = float(np.clip(rel_z, zmin, zmax))
        bounded[index, 2] = ground + rel_z
    return bounded


def _resample_with_positions(path: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    path = np.asarray(path, dtype=float)
    if path.shape[0] == 0:
        return np.zeros((n_samples, 3), dtype=float), np.zeros(n_samples, dtype=float)
    if path.shape[0] == 1:
        return np.repeat(path, n_samples, axis=0), np.zeros(n_samples, dtype=float)
    deltas = np.diff(path, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.hstack([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(path[:1], n_samples, axis=0), np.zeros(n_samples, dtype=float)
    targets = np.linspace(0.0, total, n_samples)
    samples = np.zeros((n_samples, 3), dtype=float)
    positions = np.zeros(n_samples, dtype=float)
    seg_idx = 0
    for out_idx, target in enumerate(targets):
        while seg_idx < lengths.size - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        start = cumulative[seg_idx]
        end = cumulative[seg_idx + 1]
        alpha = 0.0 if end <= start else float((target - start) / (end - start))
        samples[out_idx] = (1.0 - alpha) * path[seg_idx] + alpha * path[seg_idx + 1]
        positions[out_idx] = float(seg_idx) + alpha
    return samples, positions


def _internal_index(position: float, path_len: int) -> int:
    if path_len <= 2:
        return 0
    return int(np.clip(int(round(position)), 1, path_len - 2))


def _perpendicular_direction(path: np.ndarray, index: int) -> np.ndarray:
    if path.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=float)
    left = max(0, min(index - 1, path.shape[0] - 1))
    right = max(0, min(index + 1, path.shape[0] - 1))
    heading = path[right, :2] - path[left, :2]
    if float(np.linalg.norm(heading)) <= 1e-12:
        heading = path[-1, :2] - path[0, :2]
    perp = np.array([-heading[1], heading[0]], dtype=float)
    norm = float(np.linalg.norm(perp))
    if norm <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return perp / norm


def _row_matrix(raw: Any, width: int) -> np.ndarray:
    if raw is None:
        return np.zeros((0, width), dtype=float)
    matrix = np.asarray(raw, dtype=float)
    if matrix.size == 0:
        return np.zeros((0, width), dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] < width:
        matrix = np.pad(matrix, ((0, 0), (0, width - matrix.shape[1])), mode="constant")
    return matrix[:, :width]


def _match_radii(raw: Any, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=float)
    radii = np.asarray(raw if raw is not None else np.zeros(0), dtype=float).reshape(-1)
    if radii.size == 0:
        return np.zeros(count, dtype=float)
    if radii.size == 1:
        return np.repeat(radii, count)
    if radii.size < count:
        radii = np.pad(radii, (0, count - radii.size), mode="edge")
    return radii[:count]


def _static_discs(model: dict[str, Any]) -> np.ndarray:
    discs: list[np.ndarray] = []
    nofly_centers = _row_matrix(model.get("nofly_c"), 2)
    if nofly_centers.shape[0] > 0:
        radii = _match_radii(model.get("nofly_r"), nofly_centers.shape[0])
        discs.append(np.column_stack([nofly_centers[:, 0], nofly_centers[:, 1], radii]))
    threats = _row_matrix(model.get("threats"), 4)
    if threats.shape[0] > 0:
        discs.append(np.column_stack([threats[:, 0], threats[:, 1], np.maximum(0.0, threats[:, 3])]))
    if not discs:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(discs)


def _dynamic_obstacles(model: dict[str, Any]) -> np.ndarray:
    raw = model.get("dynamicObstacles", model.get("dynamic_obstacles"))
    if raw is None:
        return np.zeros((0, 9), dtype=float)
    if isinstance(raw, dict):
        centers = _row_matrix(raw.get("centers", raw.get("center")), 3)
        if centers.shape[0] == 0:
            return np.zeros((0, 9), dtype=float)
        radii = _match_radii(raw.get("radii", raw.get("radius")), centers.shape[0])
        velocity = _row_matrix(raw.get("velocities", raw.get("velocity")), 3)
        if velocity.shape[0] == 0:
            velocity = np.zeros_like(centers)
        if velocity.shape[0] < centers.shape[0]:
            velocity = np.vstack([velocity, np.repeat(velocity[-1:], centers.shape[0] - velocity.shape[0], axis=0)])
        t0 = _match_radii(raw.get("tStart", raw.get("t_start")), centers.shape[0])
        t1 = _match_radii(raw.get("tEnd", raw.get("t_end")), centers.shape[0])
        t1 = np.where(t1 <= t0, 1.0, t1)
        return np.column_stack([centers, radii, velocity[: centers.shape[0]], t0, t1])
    matrix = _row_matrix(raw, 9)
    if matrix.shape[0] == 0:
        return matrix
    # Accepted row layout: x, y, z, radius, vx, vy, vz, t_start, t_end.
    matrix[:, 3] = np.maximum(0.0, matrix[:, 3])
    matrix[:, 8] = np.where(matrix[:, 8] <= matrix[:, 7], 1.0, matrix[:, 8])
    return matrix


def _dynamic_obstacle_violation(
    paths: list[np.ndarray],
    model: dict[str, Any],
    config: _ShieldConfig,
) -> tuple[bool, float, float]:
    obstacles = _dynamic_obstacles(model)
    if obstacles.shape[0] == 0:
        return False, 0.0, float("nan")

    times = np.linspace(0.0, 1.0, config.time_samples)
    worst_violation = 0.0
    min_margin = float("inf")
    for path in paths:
        samples, _positions = _resample_with_positions(path, config.time_samples)
        for step, t in enumerate(times):
            point = samples[step]
            for obstacle in obstacles:
                t_start = float(obstacle[7])
                t_end = float(obstacle[8])
                if t < t_start or t > t_end:
                    continue
                center = obstacle[:3] + obstacle[4:7] * t
                radius = float(obstacle[3] + config.dynamic_buffer)
                if radius <= 0.0:
                    continue
                distance = float(np.linalg.norm(point - center))
                margin = distance - radius
                min_margin = min(min_margin, margin)
                if margin < 0.0:
                    worst_violation = max(worst_violation, -margin / max(radius, 1e-9))

    if not np.isfinite(min_margin):
        min_margin = float("nan")
    return bool(worst_violation > 1e-9), float(worst_violation), float(min_margin)


def _repair_static_discs(
    paths: list[np.ndarray], model: dict[str, Any], config: _ShieldConfig, report: _ShieldReport
) -> None:
    discs = _static_discs(model)
    if discs.shape[0] == 0:
        return
    for path_index, path in enumerate(paths):
        if path.shape[0] <= 2:
            continue
        samples, positions = _resample_with_positions(path, config.time_samples)
        for step, sample in enumerate(samples):
            point_xy = sample[:2]
            for cx, cy, radius in discs:
                target_radius = float(radius + config.nofly_buffer)
                if target_radius <= 0.0:
                    continue
                delta = point_xy - np.array([cx, cy], dtype=float)
                distance = float(np.linalg.norm(delta))
                penetration = target_radius - distance
                if penetration <= 0.0:
                    continue
                waypoint_index = _internal_index(float(positions[step]), path.shape[0])
                direction = _perpendicular_direction(path, waypoint_index) if distance <= 1e-12 else delta / distance
                path[waypoint_index, :2] += direction * penetration * config.repair_gain
                report.nofly_corrections += 1
                report.nofly_risk += float(penetration / max(target_radius, 1e-9))
        for waypoint_index in range(1, path.shape[0] - 1):
            point_xy = path[waypoint_index, :2]
            for cx, cy, radius in discs:
                target_radius = float(radius + config.nofly_buffer)
                if target_radius <= 0.0:
                    continue
                delta = point_xy - np.array([cx, cy], dtype=float)
                distance = float(np.linalg.norm(delta))
                penetration = target_radius - distance
                if penetration <= 0.0:
                    continue
                direction = _perpendicular_direction(path, waypoint_index) if distance <= 1e-12 else delta / distance
                shift = direction * penetration * config.repair_gain
                path[waypoint_index, :2] += shift
                point_xy = path[waypoint_index, :2]
                report.nofly_corrections += 1
                report.nofly_risk += float(penetration / max(target_radius, 1e-9))
        paths[path_index] = _enforce_path_bounds(path, model)


def _terrain_clearance_floor(model: dict[str, Any]) -> float:
    drone_size = float(model.get("droneSize", model.get("drone_size", 1.0)))
    safe_h_raw = model.get("safeH")
    if safe_h_raw is None:
        return max(0.0, drone_size + 0.25)
    return max(float(np.asarray(safe_h_raw).reshape(-1)[0]), drone_size + 0.25)


def _repair_terrain_clearance(
    paths: list[np.ndarray], model: dict[str, Any], config: _ShieldConfig, report: _ShieldReport
) -> None:
    clearance_floor = _terrain_clearance_floor(model)
    if clearance_floor <= 0.0:
        return
    for path_index, path in enumerate(paths):
        if path.shape[0] <= 2:
            continue
        samples, positions = _resample_with_positions(path, config.time_samples)
        insertions: list[tuple[float, float, int, np.ndarray]] = []
        for step, sample in enumerate(samples):
            ground = _ground_height(model, float(sample[0]), float(sample[1]))
            clearance = float(sample[2] - ground)
            deficit = clearance_floor - clearance
            if deficit <= 0.0:
                continue
            waypoint_index = _internal_index(float(positions[step]), path.shape[0])
            position = float(positions[step])
            segment_index = int(np.clip(math.floor(position), 0, path.shape[0] - 2))
            inserted = np.asarray(sample, dtype=float).copy()
            inserted[2] = ground + clearance_floor
            insertions.append((float(deficit), position, segment_index, inserted))
            path[waypoint_index, 2] += 0.25 * deficit * config.repair_gain
            report.terrain_corrections += 1
            report.terrain_risk += float(deficit / max(clearance_floor, 1e-9))
        if insertions:
            # Add a small number of execution-time safety waypoints at the
            # worst terrain incursions. This is more faithful to a shield
            # than forcing the sparse evolutionary genotype to encode every
            # local terrain correction.
            insertions.sort(key=lambda item: item[0], reverse=True)
            selected = sorted(insertions[: config.max_insertions], key=lambda item: item[1], reverse=True)
            for _deficit, _position, segment_index, inserted in selected:
                path = np.insert(path, segment_index + 1, inserted, axis=0)
        paths[path_index] = _enforce_path_bounds(path, model)


def _repair_inter_uav_conflicts(
    paths: list[np.ndarray], model: dict[str, Any], config: _ShieldConfig, report: _ShieldReport
) -> None:
    if len(paths) < 2:
        return
    separation = float(model.get("separationMin", model.get("safeDist", 10.0)))
    target = separation * (1.0 + config.separation_buffer)
    samples = []
    positions = []
    for path in paths:
        sample, pos = _resample_with_positions(path, config.time_samples)
        samples.append(sample)
        positions.append(pos)
    for step in range(config.time_samples):
        for left in range(len(paths)):
            for right in range(left + 1, len(paths)):
                p_left = samples[left][step]
                p_right = samples[right][step]
                delta_xy = p_left[:2] - p_right[:2]
                distance = float(np.linalg.norm(p_left - p_right))
                violation = target - distance
                if violation <= 0.0:
                    continue
                direction_norm = float(np.linalg.norm(delta_xy))
                if direction_norm <= 1e-12:
                    left_idx = _internal_index(float(positions[left][step]), paths[left].shape[0])
                    direction = _perpendicular_direction(paths[left], left_idx)
                else:
                    direction = delta_xy / direction_norm
                left_idx = _internal_index(float(positions[left][step]), paths[left].shape[0])
                right_idx = _internal_index(float(positions[right][step]), paths[right].shape[0])
                shift = 0.5 * violation * config.repair_gain
                paths[left][left_idx, :2] += direction * shift
                paths[right][right_idx, :2] -= direction * shift
                if (left + right + step) % 2 == 0:
                    paths[left][left_idx, 2] += 0.25 * shift
                else:
                    paths[right][right_idx, 2] += 0.25 * shift
                report.inter_uav_corrections += 1
    for index, path in enumerate(paths):
        paths[index] = _enforce_path_bounds(path, model)


def _repair_dynamic_obstacles(
    paths: list[np.ndarray], model: dict[str, Any], config: _ShieldConfig, report: _ShieldReport
) -> None:
    obstacles = _dynamic_obstacles(model)
    if obstacles.shape[0] == 0:
        return
    times = np.linspace(0.0, 1.0, config.time_samples)
    for path_index, path in enumerate(paths):
        if path.shape[0] <= 2:
            continue
        samples, positions = _resample_with_positions(path, config.time_samples)
        for step, t in enumerate(times):
            point = samples[step]
            for obstacle in obstacles:
                t_start = float(obstacle[7])
                t_end = float(obstacle[8])
                if t < t_start or t > t_end:
                    continue
                center = obstacle[:3] + obstacle[4:7] * t
                radius = float(obstacle[3] + config.dynamic_buffer)
                if radius <= 0.0:
                    continue
                delta_xy = point[:2] - center[:2]
                distance_xy = float(np.linalg.norm(delta_xy))
                penetration = radius - distance_xy
                if penetration <= 0.0:
                    continue
                waypoint_index = _internal_index(float(positions[step]), path.shape[0])
                direction = (
                    _perpendicular_direction(path, waypoint_index) if distance_xy <= 1e-12 else delta_xy / distance_xy
                )
                shift = penetration * config.repair_gain
                path[waypoint_index, :2] += direction * shift
                path[waypoint_index, 2] += 0.35 * shift
                report.dynamic_obstacle_corrections += 1
                report.dynamic_risk += float(penetration / max(radius, 1e-9))
        paths[path_index] = _enforce_path_bounds(path, model)


def _path_energy(path: np.ndarray) -> float:
    if path.shape[0] < 2:
        return 0.0
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    climb = np.maximum(0.0, segments[:, 2])
    turn_energy = 0.0
    if path.shape[0] >= 3:
        vectors = np.diff(path[:, :2], axis=0)
        left = vectors[:-1]
        right = vectors[1:]
        denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        valid = denom > 1e-12
        if np.any(valid):
            dots = np.sum(left[valid] * right[valid], axis=1) / denom[valid]
            angles = np.arccos(np.clip(dots, -1.0, 1.0))
            turn_energy = float(np.sum(angles / math.pi))
    return float(np.sum(lengths) + 0.35 * np.sum(climb) + 0.10 * turn_energy)


def _fleet_energy(paths: list[np.ndarray]) -> float:
    return float(sum(_path_energy(path) for path in paths))


def _energy_limit_violation(paths: list[np.ndarray], config: _ShieldConfig) -> tuple[bool, float, float]:
    if config.energy_max is None:
        return False, 0.0, _fleet_energy(paths)
    energy = _fleet_energy(paths)
    excess = energy - float(config.energy_max)
    if excess <= 0.0:
        return False, 0.0, energy
    return True, float(excess / max(float(config.energy_max), 1e-9)), energy


def _repair_energy(
    paths: list[np.ndarray], model: dict[str, Any], config: _ShieldConfig, report: _ShieldReport
) -> None:
    if config.energy_max is None:
        return
    energy = _fleet_energy(paths)
    excess = energy - float(config.energy_max)
    if excess <= 0.0:
        return
    report.energy_violation = float(excess / max(float(config.energy_max), 1e-9))
    alpha = float(np.clip(0.15 + report.energy_violation, 0.15, 0.65))
    for index, path in enumerate(paths):
        if path.shape[0] <= 2:
            continue
        straight = np.linspace(path[0], path[-1], path.shape[0])
        path[1:-1] = (1.0 - alpha) * path[1:-1] + alpha * straight[1:-1]
        paths[index] = _enforce_path_bounds(path, model)
    report.energy_corrections += 1


def _smooth_path(path: np.ndarray, passes: int) -> np.ndarray:
    smoothed = np.asarray(path, dtype=float).copy()
    if smoothed.shape[0] <= 3:
        return smoothed
    for _ in range(max(1, int(passes))):
        smoothed[1:-1] = 0.25 * smoothed[:-2] + 0.50 * smoothed[1:-1] + 0.25 * smoothed[2:]
    return smoothed


def _repair_motion(
    paths: list[np.ndarray], model: dict[str, Any], config: _ShieldConfig, report: _ShieldReport
) -> None:
    local_model = dict(model)
    local_model["maxTurnDeg"] = float(config.max_turn_deg)
    for index, path in enumerate(paths):
        _, details = evaluate_path_details(path, local_model)
        max_turn = float(details.get("maxTurnDeg", 0.0))
        if max_turn <= config.max_turn_deg + 1e-9:
            continue
        paths[index] = _enforce_path_bounds(_smooth_path(path, config.smoothing_passes), model)
        report.motion_corrections += 1


def _shield_paths(
    paths: list[np.ndarray], model: dict[str, Any], config: _ShieldConfig
) -> tuple[list[np.ndarray], _ShieldReport]:
    repaired = [_enforce_path_bounds(np.asarray(path, dtype=float), model) for path in paths]
    before = np.concatenate([path.reshape(-1) for path in repaired]) if repaired else np.zeros(0, dtype=float)
    report = _ShieldReport()
    for _ in range(config.iterations):
        _repair_terrain_clearance(repaired, model, config, report)
        _repair_static_discs(repaired, model, config, report)
        _repair_dynamic_obstacles(repaired, model, config, report)
        _repair_inter_uav_conflicts(repaired, model, config, report)
        _repair_energy(repaired, model, config, report)
        _repair_motion(repaired, model, config, report)
        _repair_terrain_clearance(repaired, model, config, report)
        _repair_inter_uav_conflicts(repaired, model, config, report)
        _repair_motion(repaired, model, config, report)
    after = np.concatenate([path.reshape(-1) for path in repaired]) if repaired else np.zeros(0, dtype=float)
    if before.shape == after.shape and before.size > 0:
        report.correction_norm = float(np.linalg.norm(after - before))
    elif before.size > 0 or after.size > 0:
        report.correction_norm = float(abs(after.size - before.size))
    return repaired, report


def _arrival_fairness(paths: list[np.ndarray]) -> float:
    lengths = np.asarray(
        [float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if path.shape[0] >= 2 else 0.0 for path in paths],
        dtype=float,
    )
    if lengths.size <= 1:
        return 0.0
    mean = float(np.mean(lengths))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(lengths) / mean)


def _map_diagonal(model: dict[str, Any]) -> float:
    return float(
        math.sqrt(
            (float(model["xmax"]) - float(model["xmin"])) ** 2
            + (float(model["ymax"]) - float(model["ymin"])) ** 2
            + (float(model["zmax"]) - float(model["zmin"])) ** 2
        )
    )


def _sem4d_objective(
    paths: list[np.ndarray],
    mission_objective: np.ndarray,
    mission_details: dict[str, Any],
    report: _ShieldReport,
    model: dict[str, Any],
    params: BenchmarkParams,
) -> np.ndarray:
    del mission_objective
    if not paths:
        return np.full(4, np.inf, dtype=float)

    diagonal = max(_map_diagonal(model), 1e-9)
    lengths = np.asarray(
        [float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if path.shape[0] >= 2 else 0.0 for path in paths],
        dtype=float,
    )
    travel_time = float(np.clip(np.max(lengths) / (1.75 * diagonal), 0.0, 1.0))

    energy_scale = float(
        params.extra.get("sem4dEnergyScale", params.extra.get("sem4d_energy_scale", len(paths) * diagonal * 1.35))
    )
    energy = float(np.clip(_fleet_energy(paths) / max(energy_scale, 1e-9), 0.0, 1.0))

    conflict = float(np.clip(mission_details.get("conflictRate", 0.0), 0.0, 1.0))
    clearance = float(np.clip(mission_details.get("energy", 0.0), 0.0, 1.0))
    altitude_risk = float(np.clip(mission_details.get("risk", 0.0), 0.0, 1.0))
    terrain_risk = float(np.clip(report.terrain_risk / max(1.0, _config_time_scale(params)), 0.0, 1.0))
    dynamic_risk = float(np.clip(report.dynamic_risk / max(1.0, _config_time_scale(params)), 0.0, 1.0))
    nofly_risk = float(np.clip(report.nofly_risk / max(1.0, len(paths)), 0.0, 1.0))
    risk = float(
        np.clip(max(clearance, altitude_risk, conflict * 4.0, terrain_risk, dynamic_risk, nofly_risk), 0.0, 1.0)
    )

    path_obj = np.asarray(mission_details.get("pathObjectives", np.zeros((0, 4))), dtype=float)
    if path_obj.ndim == 2 and path_obj.shape[1] >= 4 and path_obj.shape[0] > 0:
        turn = float(np.nanmean(np.clip(path_obj[:, 3], 0.0, 1.0)))
    else:
        turn = float(np.clip(mission_details.get("turnPenalty", 0.0), 0.0, 1.0))
    fairness = float(np.clip(_arrival_fairness(paths), 0.0, 1.0))
    smoothness_fairness = float(np.clip(0.70 * turn + 0.30 * fairness, 0.0, 1.0))
    return np.array([travel_time, energy, risk, smoothness_fairness], dtype=float)


def _config_time_scale(params: BenchmarkParams) -> float:
    return float(max(1, _extra_int(params, "sem4dTimeSamples", 48)))


def _candidate_from_paths(
    vector: np.ndarray,
    paths: list[np.ndarray],
    model: dict[str, Any],
    params: BenchmarkParams,
    config: _ShieldConfig,
    task_id: int,
) -> Candidate:
    pre_objective, pre_details = evaluate_mission_details(paths, model)
    if task_id == 0:
        shielded_paths, shield_report = _shield_paths(paths, model, config)
        objective, details = evaluate_mission_details(shielded_paths, model)
        sem_objective = _sem4d_objective(shielded_paths, objective, details, shield_report, model, params)
        dynamic_violation, dynamic_violation_score, dynamic_margin = _dynamic_obstacle_violation(
            shielded_paths,
            model,
            config,
        )
        energy_violation, energy_violation_score, shielded_energy = _energy_limit_violation(shielded_paths, config)
        feasible = float(details.get("feasible", 0.0)) > 0.5 and not dynamic_violation and not energy_violation
        if not feasible:
            sem_objective[:] = np.inf
            details["feasible"] = 0.0
        details["paths"] = shielded_paths
        details["preShieldObjective"] = np.asarray(pre_objective, dtype=float)
        details["preShieldConflictRate"] = float(pre_details.get("conflictRate", np.nan))
        details["shieldCorrectionNorm"] = float(shield_report.correction_norm)
        details["shieldTerrainCorrections"] = float(shield_report.terrain_corrections)
        details["shieldInterUavCorrections"] = float(shield_report.inter_uav_corrections)
        details["shieldDynamicObstacleCorrections"] = float(shield_report.dynamic_obstacle_corrections)
        details["shieldNoFlyCorrections"] = float(shield_report.nofly_corrections)
        details["shieldEnergyCorrections"] = float(shield_report.energy_corrections)
        details["shieldMotionCorrections"] = float(shield_report.motion_corrections)
        details["shieldEnergyViolation"] = float(shield_report.energy_violation)
        details["shieldTerrainRisk"] = float(shield_report.terrain_risk)
        details["shieldDynamicRisk"] = float(shield_report.dynamic_risk)
        details["shieldNoFlyRisk"] = float(shield_report.nofly_risk)
        details["postShieldDynamicObstacleViolation"] = float(dynamic_violation)
        details["postShieldDynamicObstacleViolationScore"] = float(dynamic_violation_score)
        details["postShieldDynamicObstacleMargin"] = float(dynamic_margin)
        details["postShieldEnergyViolation"] = float(energy_violation)
        details["postShieldEnergyViolationScore"] = float(energy_violation_score)
        details["postShieldEnergy"] = float(shielded_energy)
        details["sem4dTravelTime"] = float(sem_objective[0]) if np.isfinite(sem_objective[0]) else np.inf
        details["sem4dEnergy"] = float(sem_objective[1]) if np.isfinite(sem_objective[1]) else np.inf
        details["sem4dRisk"] = float(sem_objective[2]) if np.isfinite(sem_objective[2]) else np.inf
        details["sem4dSmoothnessFairness"] = float(sem_objective[3]) if np.isfinite(sem_objective[3]) else np.inf
        details["fairness"] = float(_arrival_fairness(shielded_paths))
        details["makespan"] = details["sem4dTravelTime"]
        details["energy"] = details["sem4dEnergy"]
        details["risk"] = details["sem4dRisk"]
        return Candidate(vector=np.asarray(vector, dtype=float), objective=sem_objective, details=details)

    details = dict(pre_details)
    path_obj = np.asarray(details.get("pathObjectives", np.zeros((0, 4))), dtype=float)
    uav_index = max(0, min(task_id - 1, path_obj.shape[0] - 1)) if path_obj.ndim == 2 and path_obj.shape[0] > 0 else 0
    if path_obj.ndim == 2 and path_obj.shape[0] > 0 and path_obj.shape[1] >= 4:
        objective = np.asarray(path_obj[uav_index, :4], dtype=float).copy()
        if np.all(np.isfinite(objective)):
            objective[1] = float(np.clip(objective[1] + 0.25 * float(details.get("conflictRate", 0.0)), 0.0, 1.0))
    else:
        objective = np.asarray(pre_objective, dtype=float)
    details["paths"] = paths
    details["sem4dTaskId"] = float(task_id)
    return Candidate(vector=np.asarray(vector, dtype=float), objective=objective, details=details)


def _build_aux_model(model: dict[str, Any], params: BenchmarkParams) -> dict[str, Any]:
    aux = dict(model)
    separation_scale = _extra_float(params, "sem4dAuxSeparationScale", 0.65)
    safe_scale = _extra_float(params, "sem4dAuxSafeDistScale", 0.75)
    nofly_scale = _extra_float(params, "sem4dAuxNoFlyScale", 0.85)
    aux["separationMin"] = max(1.0, float(model.get("separationMin", 10.0)) * separation_scale)
    aux["safeDist"] = max(1.0, float(model.get("safeDist", 10.0)) * safe_scale)
    if "nofly_r" in aux and aux["nofly_r"] is not None:
        aux["nofly_r"] = np.asarray(aux["nofly_r"], dtype=float) * nofly_scale
    return aux


def _evaluate_vectors_for_task(
    vectors: np.ndarray,
    task_id: int,
    model: dict[str, Any],
    aux_model: dict[str, Any],
    params: BenchmarkParams,
    config: _ShieldConfig,
    fleet_size: int,
    n_waypoints: int,
) -> list[Candidate]:
    active_model = model if task_id == 0 else aux_model
    candidates: list[Candidate] = []
    for vector in np.asarray(vectors, dtype=float):
        try:
            paths = decision_to_paths(vector, model=active_model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            candidate = _candidate_from_paths(
                vector=np.asarray(vector, dtype=float).copy(),
                paths=paths,
                model=active_model if task_id != 0 else model,
                params=params,
                config=config,
                task_id=task_id,
            )
        except (KeyError, IndexError, RuntimeError, TypeError, ValueError):
            candidate = Candidate(
                vector=np.asarray(vector, dtype=float).copy(),
                objective=np.full(4, np.inf, dtype=float),
                details={"feasible": 0.0, "paths": []},
            )
        candidates.append(candidate)
    return candidates
