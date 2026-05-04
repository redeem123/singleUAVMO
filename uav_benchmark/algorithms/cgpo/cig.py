from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.core.evaluate_path import _bilinear_interpolate, _collect_obstacles


@dataclass(frozen=True, slots=True)
class ConstraintInteractionGraph:
    """Dynamic constraint graph plus its induced waypoint tension fields."""

    tension: np.ndarray
    scalar_tension: np.ndarray
    per_uav_tension: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    edge_type: np.ndarray
    mean_tension: float
    max_tension: float
    terrain_edges: int
    obstacle_edges: int
    turn_edges: int
    pairwise_edges: int
    objective_edges: int
    conflict_clusters: int
    active_constraint_types: tuple[str, ...]


_NODE_WAYPOINT = 0.0
_NODE_TERRAIN = 1.0
_NODE_ALTITUDE = 2.0
_NODE_OBSTACLE = 3.0
_NODE_TURN = 4.0
_NODE_PAIRWISE = 5.0
_NODE_SMOOTHING = 6.0

_EDGE_TERRAIN = 1
_EDGE_ALTITUDE = 2
_EDGE_OBSTACLE = 3
_EDGE_TURN = 4
_EDGE_PAIRWISE = 5
_EDGE_SMOOTHING = 6


def _waypoint_from_node(node_idx: int, n_points: int) -> tuple[int, int]:
    return int(node_idx // n_points), int(node_idx % n_points)


def _waypoint_node(uav_idx: int, point_idx: int, n_points: int) -> int:
    return int(uav_idx * n_points + point_idx)


def _ground(model: dict[str, Any], xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    return _bilinear_interpolate(np.asarray(model["H"], dtype=float), points[:, 0] - 1.0, points[:, 1] - 1.0)


def _horizontal_turn(path: np.ndarray, index: int) -> float:
    if index <= 0 or index >= path.shape[0] - 1:
        return 0.0
    a = path[index, :2] - path[index - 1, :2]
    b = path[index + 1, :2] - path[index, :2]
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-9 or bn <= 1e-9:
        return 0.0
    dot = float(np.dot(a, b) / max(an * bn, 1e-9))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _safe_paths(paths: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for path in paths:
        arr = np.asarray(path, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 2:
            out.append(arr)
    return out


_DEFAULT_SMOOTHING_WEIGHT = 0.05


def build_constraint_interaction_graph(
    paths: list[np.ndarray],
    model: dict[str, Any],
    objective_weights: np.ndarray | None = None,  # noqa: ARG001 - retained for ABI compatibility
    use_edge_coupling: bool = True,
    use_terrain_edges: bool = True,
    use_obstacle_edges: bool = True,
    use_turn_edges: bool = True,
    use_smoothing_edges: bool | None = None,
    use_pairwise_edges: bool | None = None,
    temporal_window: int = 1,
    pairwise_edge_radius_scale: float = 1.5,
    terrain_boundary_beta: float = 0.25,
    smoothing_weight: float = _DEFAULT_SMOOTHING_WEIGHT,
) -> ConstraintInteractionGraph:
    """Build the constraint-interaction graph for a fleet of waypoint paths.

    The ``objective_weights`` argument is ignored: prior versions coupled the
    smoothing-edge weight to the PPF objective spread, which made CIG a
    function of PPF and confounded the ablation interpretation.  The graph is
    now independent of the selection mechanism and uses a fixed
    ``smoothing_weight`` instead.
    """
    safe_paths = _safe_paths(paths)
    fleet_size = len(safe_paths)
    n_points = max((path.shape[0] for path in safe_paths), default=0)
    tension = np.zeros((fleet_size, n_points, 3), dtype=float)
    node_features: list[list[float]] = []
    edge_pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    edge_types: list[int] = []
    active: set[str] = set()
    terrain_edges = 0
    obstacle_edges = 0
    turn_edges = 0
    pairwise_edges = 0
    objective_edges = 0
    conflict_clusters = 0

    if fleet_size == 0 or n_points == 0:
        empty = np.zeros((fleet_size, n_points), dtype=float)
        return ConstraintInteractionGraph(
            tension=tension,
            scalar_tension=empty,
            per_uav_tension=np.zeros(fleet_size, dtype=float),
            node_features=np.zeros((0, 8), dtype=float),
            edge_index=np.zeros((2, 0), dtype=int),
            edge_weight=np.zeros(0, dtype=float),
            edge_type=np.zeros(0, dtype=int),
            mean_tension=0.0,
            max_tension=0.0,
            terrain_edges=0,
            obstacle_edges=0,
            turn_edges=0,
            pairwise_edges=0,
            objective_edges=0,
            conflict_clusters=0,
            active_constraint_types=(),
        )

    zmin = float(model["zmin"])
    zmax = float(model["zmax"])
    altitude_span = max(zmax - zmin, 1e-9)
    clearance_target = max(float(model.get("droneSize", 1.0)) + 2.5, zmin + 0.15 * altitude_span)
    drone_size = float(model.get("droneSize", 1.0))
    safe_dist = float(model.get("safeDist", model.get("safe_dist", 10.0)))
    obstacle_boundary = max(drone_size, drone_size + 0.35 * safe_dist)
    obstacles = _collect_obstacles(model)
    turn_limit = float(model.get("maxTurnDeg", 75.0))
    separation = float(model.get("separationMin", model.get("safeDist", 10.0)))
    radius = max(separation, separation * float(pairwise_edge_radius_scale))
    smooth_weight = float(max(0.0, smoothing_weight))
    smoothing_enabled = bool(use_edge_coupling if use_smoothing_edges is None else use_smoothing_edges)
    pairwise_enabled = bool(use_edge_coupling if use_pairwise_edges is None else use_pairwise_edges)

    x_span = max(float(model["xmax"]) - float(model["xmin"]), 1e-9)
    y_span = max(float(model["ymax"]) - float(model["ymin"]), 1e-9)
    fleet_den = max(1, fleet_size - 1)
    point_den = max(1, n_points - 1)
    for uav_idx, path in enumerate(safe_paths):
        ground = _ground(model, path[:, :2])
        rel = path[:, 2] - ground
        for point_idx in range(n_points):
            if point_idx < path.shape[0]:
                x_norm = (float(path[point_idx, 0]) - float(model["xmin"])) / x_span
                y_norm = (float(path[point_idx, 1]) - float(model["ymin"])) / y_span
                z_norm = (float(rel[point_idx]) - zmin) / altitude_span
            else:
                x_norm = y_norm = z_norm = 0.0
            node_features.append(
                [
                    _NODE_WAYPOINT,
                    float(uav_idx) / fleet_den,
                    float(point_idx) / point_den,
                    float(np.clip(x_norm, 0.0, 1.0)),
                    float(np.clip(y_norm, 0.0, 1.0)),
                    float(np.clip(z_norm, -1.0, 2.0)),
                    0.0,
                    0.0,
                ]
            )

    def add_constraint_edge(
        node_type: float,
        source_wp: int,
        target_wp: int,
        weight: float,
        edge_type: int,
        direction: np.ndarray | None = None,
    ) -> None:
        vec = np.zeros(3, dtype=float) if direction is None else np.asarray(direction, dtype=float).reshape(-1)[:3]
        if vec.size < 3:
            vec = np.pad(vec, (0, 3 - vec.size))
        constraint_node = len(node_features)
        node_features.append(
            [
                float(node_type),
                float(source_wp) / fleet_den,
                float(target_wp) / point_den,
                float(np.clip(weight, 0.0, 1e6)),
                float(vec[0]),
                float(vec[1]),
                float(vec[2]),
                0.0,
            ]
        )
        target_node = _waypoint_node(source_wp, target_wp, n_points)
        edge_pairs.append((constraint_node, target_node))
        edge_pairs.append((target_node, constraint_node))
        edge_weights.extend([float(weight), float(weight)])
        edge_types.extend([int(edge_type), int(edge_type)])

    for uav_idx, path in enumerate(safe_paths):
        ground = _ground(model, path[:, :2])
        rel = path[:, 2] - ground
        for point_idx in range(path.shape[0]):
            if point_idx == 0 or point_idx == path.shape[0] - 1:
                continue
            if use_terrain_edges:
                terrain_margin = rel[point_idx] - clearance_target
                if terrain_margin < terrain_boundary_beta * clearance_target:
                    residual = max(0.0, -terrain_margin / max(clearance_target, 1e-9))
                    boundary = max(
                        0.0, (terrain_boundary_beta * clearance_target - terrain_margin) / max(clearance_target, 1e-9)
                    )
                    pressure = residual + 0.20 * boundary
                    tension[uav_idx, point_idx, 2] += pressure * altitude_span
                    add_constraint_edge(
                        _NODE_TERRAIN, uav_idx, point_idx, pressure, _EDGE_TERRAIN, np.array([0.0, 0.0, 1.0])
                    )
                    terrain_edges += 1
                    active.add("terrain")
                if rel[point_idx] < zmin:
                    pressure = (zmin - rel[point_idx]) / altitude_span
                    tension[uav_idx, point_idx, 2] += pressure
                    add_constraint_edge(
                        _NODE_ALTITUDE, uav_idx, point_idx, pressure, _EDGE_ALTITUDE, np.array([0.0, 0.0, 1.0])
                    )
                    terrain_edges += 1
                    active.add("altitude")
                elif rel[point_idx] > zmax:
                    pressure = (rel[point_idx] - zmax) / altitude_span
                    tension[uav_idx, point_idx, 2] -= pressure
                    add_constraint_edge(
                        _NODE_ALTITUDE, uav_idx, point_idx, pressure, _EDGE_ALTITUDE, np.array([0.0, 0.0, -1.0])
                    )
                    terrain_edges += 1
                    active.add("altitude")
            if use_obstacle_edges and obstacles.shape[0] > 0:
                xy = path[point_idx, :2]
                for obstacle in obstacles:
                    center = np.asarray(obstacle[:2], dtype=float)
                    radius_obs = float(obstacle[3])
                    delta = xy - center
                    dist = float(np.linalg.norm(delta))
                    clearance = dist - radius_obs
                    if clearance >= obstacle_boundary:
                        continue
                    direction = delta / max(dist, 1e-9)
                    if not np.all(np.isfinite(direction)) or float(np.linalg.norm(direction)) <= 1e-9:
                        direction = np.array([1.0, 0.0], dtype=float)
                    residual = max(0.0, (drone_size - clearance) / max(drone_size, 1e-9))
                    boundary = max(0.0, (obstacle_boundary - clearance) / max(obstacle_boundary, 1e-9))
                    pressure = residual + 0.20 * boundary
                    tension[uav_idx, point_idx, :2] += pressure * obstacle_boundary * direction
                    add_constraint_edge(
                        _NODE_OBSTACLE,
                        uav_idx,
                        point_idx,
                        pressure,
                        _EDGE_OBSTACLE,
                        np.array([direction[0], direction[1], 0.0]),
                    )
                    obstacle_edges += 1
                    active.add("obstacle")

        if use_turn_edges:
            for point_idx in range(1, path.shape[0] - 1):
                turn = _horizontal_turn(path, point_idx)
                if turn <= min(turn_limit, turn_limit - 0.5):
                    continue
                residual = max(0.0, (turn - turn_limit) / max(turn_limit, 1e-9))
                midpoint = 0.5 * (path[point_idx - 1, :2] + path[point_idx + 1, :2])
                direction = midpoint - path[point_idx, :2]
                tension[uav_idx, point_idx, :2] += residual * direction
                add_constraint_edge(
                    _NODE_TURN, uav_idx, point_idx, residual, _EDGE_TURN, np.array([direction[0], direction[1], 0.0])
                )
                turn_edges += 1
                active.add("turn")

        if smoothing_enabled and path.shape[0] > 2 and smooth_weight > 0.0:
            for point_idx in range(1, path.shape[0] - 1):
                smooth = 0.5 * (path[point_idx - 1] + path[point_idx + 1]) - path[point_idx]
                tension[uav_idx, point_idx] += smooth_weight * smooth
                node = _waypoint_node(uav_idx, point_idx, n_points)
                for neighbor_idx in (point_idx - 1, point_idx + 1):
                    neighbor = _waypoint_node(uav_idx, neighbor_idx, n_points)
                    edge_pairs.append((neighbor, node))
                    edge_weights.append(float(smooth_weight))
                    edge_types.append(_EDGE_SMOOTHING)
                objective_edges += 1
            active.add("smoothing")

    if pairwise_enabled and fleet_size > 1:
        window = max(0, int(temporal_window))
        for u_idx in range(fleet_size - 1):
            for v_idx in range(u_idx + 1, fleet_size):
                u_path = safe_paths[u_idx]
                v_path = safe_paths[v_idx]
                for t_idx in range(1, min(u_path.shape[0], n_points) - 1):
                    lo = max(1, t_idx - window)
                    hi = min(v_path.shape[0] - 1, t_idx + window + 1)
                    for s_idx in range(lo, hi):
                        delta = u_path[t_idx, :2] - v_path[s_idx, :2]
                        dist = float(np.linalg.norm(delta))
                        if dist >= radius:
                            continue
                        direction = delta / max(dist, 1e-9)
                        if not np.all(np.isfinite(direction)) or float(np.linalg.norm(direction)) <= 1e-9:
                            direction = np.array([1.0, 0.0], dtype=float)
                        residual = max(0.0, (separation - dist) / max(separation, 1e-9))
                        boundary = max(0.0, (radius - dist) / max(radius, 1e-9))
                        pressure = residual + 0.15 * boundary
                        shift = pressure * separation * direction
                        tension[u_idx, t_idx, :2] += shift
                        tension[v_idx, s_idx, :2] -= shift
                        u_node = _waypoint_node(u_idx, t_idx, n_points)
                        v_node = _waypoint_node(v_idx, s_idx, n_points)
                        edge_pairs.append((u_node, v_node))
                        edge_pairs.append((v_node, u_node))
                        edge_weights.extend([float(pressure), float(pressure)])
                        edge_types.extend([_EDGE_PAIRWISE, _EDGE_PAIRWISE])
                        pairwise_edges += 1
                        active.add("pairwise")
                        if residual > 0.0:
                            conflict_clusters += 1

    graph_tension = np.zeros_like(tension)
    waypoint_nodes = fleet_size * n_points

    for (source, target), weight, edge_type in zip(edge_pairs, edge_weights, edge_types, strict=False):
        if target < 0 or target >= waypoint_nodes:
            continue
        uav_idx, point_idx = _waypoint_from_node(target, n_points)
        if point_idx <= 0 or point_idx >= safe_paths[uav_idx].shape[0] - 1:
            continue

        if source >= waypoint_nodes:
            feature = np.asarray(node_features[source], dtype=float)
            direction = np.asarray(feature[4:7], dtype=float)
            if edge_type in (_EDGE_TERRAIN, _EDGE_ALTITUDE):
                scale = np.array([1.0, 1.0, altitude_span], dtype=float)
            elif edge_type == _EDGE_OBSTACLE:
                scale = np.array([obstacle_boundary, obstacle_boundary, 1.0], dtype=float)
            else:
                scale = np.ones(3, dtype=float)
            graph_tension[uav_idx, point_idx] += float(weight) * direction * scale
            continue

        source_uav, source_point = _waypoint_from_node(source, n_points)
        if source_uav >= len(safe_paths) or source_point >= safe_paths[source_uav].shape[0]:
            continue
        target_point = safe_paths[uav_idx][point_idx]
        source_point_xyz = safe_paths[source_uav][source_point]
        if edge_type == _EDGE_PAIRWISE:
            direction = target_point[:2] - source_point_xyz[:2]
            dist = float(np.linalg.norm(direction))
            direction = np.array([1.0, 0.0], dtype=float) if dist <= 1e-09 else direction / dist
            graph_tension[uav_idx, point_idx, :2] += float(weight) * separation * direction
        elif edge_type == _EDGE_SMOOTHING:
            graph_tension[uav_idx, point_idx] += float(weight) * (source_point_xyz - target_point)

    if tension.size:
        tension = graph_tension
    scalar = np.linalg.norm(tension, axis=2) if tension.size else np.zeros((fleet_size, n_points), dtype=float)
    if node_features:
        for uav_idx in range(fleet_size):
            for point_idx in range(n_points):
                node_features[_waypoint_node(uav_idx, point_idx, n_points)][6] = float(scalar[uav_idx, point_idx])
    edge_index = np.asarray(edge_pairs, dtype=int).T if edge_pairs else np.zeros((2, 0), dtype=int)
    per_uav = (
        np.mean(scalar, axis=1) + 0.25 * np.max(scalar, axis=1) if scalar.size else np.zeros(fleet_size, dtype=float)
    )
    return ConstraintInteractionGraph(
        tension=tension,
        scalar_tension=scalar,
        per_uav_tension=per_uav,
        node_features=np.asarray(node_features, dtype=float).reshape(-1, 8),
        edge_index=edge_index,
        edge_weight=np.asarray(edge_weights, dtype=float),
        edge_type=np.asarray(edge_types, dtype=int),
        mean_tension=float(np.mean(scalar)) if scalar.size else 0.0,
        max_tension=float(np.max(scalar)) if scalar.size else 0.0,
        terrain_edges=int(terrain_edges),
        obstacle_edges=int(obstacle_edges),
        turn_edges=int(turn_edges),
        pairwise_edges=int(pairwise_edges),
        objective_edges=int(objective_edges),
        conflict_clusters=int(conflict_clusters),
        active_constraint_types=tuple(sorted(active)),
    )
