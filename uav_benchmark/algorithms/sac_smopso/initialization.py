from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import _transformation_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.mission_encoding import paths_to_decision


def _build_navigation_bounds(
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    max_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(model.get("starts", model["start"]), dtype=float)
    goals = np.asarray(model.get("goals", model["end"]), dtype=float)
    if starts.ndim == 1:
        starts = np.tile(starts.reshape(1, -1), (fleet_size, 1))
    if goals.ndim == 1:
        goals = np.tile(goals.reshape(1, -1), (fleet_size, 1))

    def _endpoint_abs(point: np.ndarray, enforce_safe_h: bool) -> np.ndarray:
        px = float(point[0])
        py = float(point[1])
        z_rel = float(point[2])
        if enforce_safe_h and "safeH" in model and model["safeH"] is not None:
            z_rel = max(z_rel, float(model["safeH"]))
        return np.array([px, py, z_rel + _ground_height(model, px, py)], dtype=float)

    lower = np.zeros((fleet_size, n_waypoints, 3), dtype=float)
    upper = np.zeros((fleet_size, n_waypoints, 3), dtype=float)
    for uav_idx in range(fleet_size):
        start = _endpoint_abs(starts[uav_idx].reshape(-1)[:3], enforce_safe_h=True)
        goal = _endpoint_abs(goals[uav_idx].reshape(-1)[:3], enforce_safe_h=False)
        path_diag = float(np.linalg.norm(goal - start))
        r_max = max(1e-3, 3.0 * path_diag / max(1, n_waypoints))
        r_min = max(1e-4, r_max / 9.0)
        lower[uav_idx, :, 0] = r_min
        upper[uav_idx, :, 0] = r_max
        lower[uav_idx, :, 1] = -max_angle_rad
        upper[uav_idx, :, 1] = max_angle_rad
        lower[uav_idx, :, 2] = -max_angle_rad
        upper[uav_idx, :, 2] = max_angle_rad
    return lower.reshape(-1), upper.reshape(-1)


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


def _corridor_seed_paths(
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    *,
    separation_min: float,
    altitude_bias: float,
    lane_scale: float = 1.6,
    altitude_stagger: float = 0.15,
) -> list[np.ndarray]:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    safe_h = float(model.get("safeH", model.get("zmin", 0.0)))
    zmin = float(model["zmin"])
    zmax = float(model["zmax"])
    xmin = float(model["xmin"])
    xmax = float(model["xmax"])
    ymin = float(model["ymin"])
    ymax = float(model["ymax"])
    map_diag = float(np.hypot(xmax - xmin, ymax - ymin))
    lateral_scale = 0.02 * map_diag
    schedule = np.linspace(1, n_waypoints, n_waypoints, dtype=float) / (n_waypoints + 1.0)
    # Lane separation wide enough to satisfy the minimum-separation constraint
    # with margin. For fleets > 1 we use at least `lane_scale * separation_min`
    # between adjacent UAVs. The previous implementation used 0.85, which
    # made the initial population violate separation from the first step.
    effective_separation = max(separation_min, 1.0) * max(lane_scale, 1.0)
    altitude_span = max(0.0, zmax - zmin)
    paths: list[np.ndarray] = []
    for uav_idx in range(fleet_size):
        start_rel = starts[uav_idx].reshape(-1)[:3]
        goal_rel = goals[uav_idx].reshape(-1)[:3]
        direction_xy = goal_rel[:2] - start_rel[:2]
        direction_norm = float(np.linalg.norm(direction_xy))
        tangent = np.array([1.0, 0.0], dtype=float) if direction_norm <= 1e-9 else direction_xy / direction_norm
        perp = np.array([-tangent[1], tangent[0]], dtype=float)
        lane_shift = (uav_idx - (fleet_size - 1) / 2.0) * effective_separation
        # Stagger each UAV into its own altitude band so that, even if their
        # xy projections touch, they remain vertically separated.
        if fleet_size > 1 and altitude_span > 1e-6 and altitude_stagger > 0.0:
            altitude_offset = (
                ((uav_idx - (fleet_size - 1) / 2.0) / max(1.0, (fleet_size - 1) / 2.0))
                * altitude_stagger
                * altitude_span
            )
        else:
            altitude_offset = 0.0
        rel_floor = max(
            safe_h,
            float(start_rel[2]),
            float(goal_rel[2]),
            safe_h + altitude_bias * altitude_span,
        )
        start_abs = np.array(
            [start_rel[0], start_rel[1], _ground_height(model, start_rel[0], start_rel[1]) + max(start_rel[2], safe_h)],
            dtype=float,
        )
        goal_abs = np.array(
            [goal_rel[0], goal_rel[1], _ground_height(model, goal_rel[0], goal_rel[1]) + max(goal_rel[2], safe_h)],
            dtype=float,
        )
        waypoints: list[np.ndarray] = []
        for t in schedule:
            base_xy = start_rel[:2] + t * (goal_rel[:2] - start_rel[:2])
            xy = base_xy + perp * lane_shift + perp * (0.35 * lateral_scale * np.sin(np.pi * t))
            x = float(np.clip(xy[0], xmin, xmax))
            y = float(np.clip(xy[1], ymin, ymax))
            rel_alt = max(
                rel_floor + altitude_offset + 0.08 * altitude_span * np.sin(np.pi * t),
                safe_h,
            )
            rel_alt = float(np.clip(rel_alt, zmin, zmax))
            waypoints.append(np.array([x, y, _ground_height(model, x, y) + rel_alt], dtype=float))
        paths.append(np.vstack([start_abs, *waypoints, goal_abs]))
    return paths


def _paths_to_spherical_decision(
    paths_xyz: list[np.ndarray],
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> np.ndarray:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    safe_h = float(model.get("safeH", model.get("zmin", 0.0)))
    decision = np.zeros((fleet_size, n_waypoints, 3), dtype=float)
    for uav_idx in range(fleet_size):
        start = starts[uav_idx].reshape(-1)[:3].copy()
        goal = goals[uav_idx].reshape(-1)[:3].copy()
        start[2] = _ground_height(model, start[0], start[1]) + max(start[2], safe_h)
        goal[2] = _ground_height(model, goal[0], goal[1]) + goal[2]
        direction = np.asarray(goal - start, dtype=float)
        phi_start = float(np.arctan2(direction[1], direction[0]))
        psi_start = float(np.arctan2(direction[2], max(np.linalg.norm(direction[:2]), 1e-9)))
        current = np.array(
            [[1.0, 0.0, 0.0, start[0]], [0.0, 1.0, 0.0, start[1]], [0.0, 0.0, 1.0, start[2]], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        current = current @ _transformation_matrix(0.0, phi_start, psi_start)
        internal = np.asarray(paths_xyz[uav_idx], dtype=float)[1:-1]
        if internal.shape[0] != n_waypoints:
            raise ValueError("Internal path waypoint count does not match decision shape.")
        for waypoint_index, point_abs in enumerate(internal):
            target = np.asarray(point_abs[:3], dtype=float)
            rotation = current[:3, :3]
            current_pos = current[:3, 3]
            local_delta = rotation.T @ (target - current_pos)
            radius = float(np.linalg.norm(local_delta))
            xy_norm = max(float(np.linalg.norm(local_delta[:2])), 1e-9)
            phi = float(np.arctan2(local_delta[1], local_delta[0]))
            psi = float(np.arctan2(local_delta[2], xy_norm))
            decision[uav_idx, waypoint_index] = np.array([radius, phi, psi], dtype=float)
            current = current @ _transformation_matrix(radius, phi, psi)
    return decision.reshape(-1)


def _structured_initial_population(
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    pop_size: int,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    separation_min: float,
    representation: str,
) -> np.ndarray:
    # For fleet_size == 1 we deliberately return pure uniform-random samples
    # in spherical-coordinate space — identical to NMOPSO's initial
    # population — so single-UAV instances no longer pay a cost for
    # multi-UAV structured seeding. Benchmarks showed corridor seeds
    # actively hurt single-UAV c_100 / s_120 by collapsing initial
    # diversity and converging the swarm into narrow lanes before the
    # PSO could explore.
    if fleet_size == 1:
        lower_flat = np.asarray(lower, dtype=float).reshape(-1)
        upper_flat = np.asarray(upper, dtype=float).reshape(-1)
        return np.random.uniform(lower_flat, upper_flat, size=(pop_size, lower_flat.size))

    # For multi-UAV, explore a range of lane widths and altitude staggerings
    # so the initial swarm contains candidates that already satisfy the
    # minimum-separation constraint with different tradeoffs on path length.
    seed_specs = (
        (0.25, 1.7, 0.10),
        (0.35, 2.0, 0.20),
        (0.45, 2.4, 0.25),
        (0.30, 1.8, -0.15),
        (0.55, 1.6, 0.30),
        (0.40, 2.8, 0.00),
        (0.50, 2.2, -0.10),
    )
    base_vectors: list[np.ndarray] = []
    for altitude_bias, lane_scale, altitude_stagger in seed_specs:
        paths = _corridor_seed_paths(
            model=model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            separation_min=separation_min,
            altitude_bias=altitude_bias,
            lane_scale=lane_scale,
            altitude_stagger=altitude_stagger,
        )
        if str(representation).strip().lower() == "cart":
            base_vectors.append(paths_to_decision(paths, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints))
        else:
            base_vectors.append(_paths_to_spherical_decision(paths, model, fleet_size, n_waypoints))

    population = np.random.uniform(lower, upper, size=(pop_size, lower.size))
    # Structured-seed ratio: 60% in multi-UAV, 50% in single-UAV.
    structured_ratio = 0.60 if fleet_size > 1 else 0.50
    structured_count = min(pop_size, max(4, int(round(pop_size * structured_ratio))))
    for index in range(structured_count):
        base = base_vectors[index % len(base_vectors)]
        if index < len(base_vectors):
            population[index] = np.clip(base, lower, upper)
            continue
        # Use small noise for earlier slots so we keep copies close to the
        # pristine seed, then progressively widen noise for later slots to
        # explore the neighbourhood around each seed.
        progress = index / max(1, structured_count - 1)
        noise_scale = 0.01 + 0.06 * progress
        noise = np.random.normal(0.0, noise_scale, size=base.shape) * np.maximum(upper - lower, 1e-9)
        population[index] = np.clip(base + noise, lower, upper)
    return population


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float(default)


def _detail_value(details: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = details.get(key, default)
    flat = np.asarray(value, dtype=float).reshape(-1)
    return float(flat[0]) if flat.size > 0 and np.isfinite(flat[0]) else float(default)


def _model_constraint_values(model: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(model.get("separationMin", model.get("safeDist", 10.0))),
        float(model.get("droneSize", 1.0)),
        float(model.get("maxTurnDeg", 75.0)),
    )


def _search_objective_from_details(
    details: dict[str, Any],
    *,
    separation_min: float,
    drone_size: float,
    max_turn_deg: float,
) -> np.ndarray:
    makespan = np.clip(_detail_value(details, "makespan", 1.0), 0.0, 1.0)
    energy = np.clip(_detail_value(details, "energy", 1.0), 0.0, 1.0)
    risk = np.clip(_detail_value(details, "risk", 1.0), 0.0, 1.0)
    turn_penalty = np.clip(_detail_value(details, "turnPenalty", 1.0), 0.0, 1.5)
    conflict = max(0.0, _detail_value(details, "conflictRate", 0.0))

    min_sep = _detail_value(details, "minSeparation", float("nan"))
    min_clearance = _detail_value(details, "minClearance", float("nan"))
    max_turn_used = _detail_value(details, "maxTurnDeg", float("nan"))

    separation_penalty = (
        max(0.0, (separation_min - min_sep) / max(separation_min, 1e-9)) if np.isfinite(min_sep) else 1.0
    )
    clearance_penalty = (
        max(0.0, (drone_size - min_clearance) / max(drone_size, 1e-9)) if np.isfinite(min_clearance) else 1.0
    )
    turn_penalty_excess = (
        max(0.0, (max_turn_used - max_turn_deg) / max(max_turn_deg, 1e-9))
        if np.isfinite(max_turn_used)
        else _detail_value(details, "turnViolation", 1.0)
    )

    collision_flag = _detail_value(details, "collisionViolation", 0.0)
    separation_flag = _detail_value(details, "separationViolation", 0.0)
    turn_flag = _detail_value(details, "turnViolation", 0.0)

    surrogate = np.asarray(
        [
            1.0 + makespan + 0.35 * (turn_penalty_excess + separation_penalty + clearance_penalty) + 0.10 * conflict,
            1.0 + energy + 0.55 * clearance_penalty + 0.35 * separation_penalty + 0.20 * conflict,
            1.0 + risk + 0.45 * clearance_penalty + 0.20 * separation_penalty + 0.20 * collision_flag,
            1.0 + turn_penalty + 0.90 * turn_penalty_excess + 0.15 * (collision_flag + separation_flag + turn_flag),
        ],
        dtype=float,
    )
    return np.clip(np.nan_to_num(surrogate, nan=2.0, posinf=3.0, neginf=3.0), 1.0, 3.0)


def _search_ready_candidate(
    candidate: Candidate,
    *,
    separation_min: float,
    drone_size: float,
    max_turn_deg: float,
) -> Candidate:
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    details = dict(candidate.details) if isinstance(candidate.details, dict) else {}
    details.setdefault("objective_search", objective.copy())
    if np.all(np.isfinite(objective)):
        return Candidate(
            vector=np.asarray(candidate.vector, dtype=float).copy(), objective=objective.copy(), details=details
        )
    surrogate = _search_objective_from_details(
        details,
        separation_min=separation_min,
        drone_size=drone_size,
        max_turn_deg=max_turn_deg,
    )
    details["objective_search"] = surrogate.copy()
    return Candidate(vector=np.asarray(candidate.vector, dtype=float).copy(), objective=surrogate, details=details)


def _report_ready_candidate(candidate: Candidate) -> Candidate:
    details = dict(candidate.details) if isinstance(candidate.details, dict) else {}
    raw_objective = np.asarray(details.get("objective_raw", candidate.objective), dtype=float).reshape(-1)
    return Candidate(
        vector=np.asarray(candidate.vector, dtype=float).copy(), objective=raw_objective.copy(), details=details
    )
