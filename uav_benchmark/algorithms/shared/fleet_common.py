from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.mission_encoding import decision_size, decision_to_paths
from uav_benchmark.io.matlab import load_mat
from uav_benchmark.problem_generation.fleet_assignments import sample_homogeneous_assignments

# ═══════════════════════════════════════════════════════════════════
# Shared helpers (non-duplicated from engine)
# ═══════════════════════════════════════════════════════════════════


def _build_bounds(model: dict[str, Any], fleet_size: int, n_waypoints: int) -> tuple[np.ndarray, np.ndarray]:
    lower_single = np.array([float(model["xmin"]), float(model["ymin"]), float(model["zmin"])], dtype=float)
    upper_single = np.array([float(model["xmax"]), float(model["ymax"]), float(model["zmax"])], dtype=float)
    total = decision_size(fleet_size, n_waypoints)
    lower = np.tile(lower_single, total // 3)
    upper = np.tile(upper_single, total // 3)
    return lower, upper


def _safe_height(model: dict[str, Any]) -> float | None:
    safe_h = model.get("safeH")
    return None if safe_h is None else float(safe_h)


def _clip_relative_altitude(model: dict[str, Any], z: np.ndarray | float, safe_h: float | None) -> np.ndarray | float:
    clipped = np.clip(z, float(model["zmin"]), float(model["zmax"]))
    if safe_h is None:
        return clipped
    return np.maximum(clipped, safe_h)


def _ground_values(model: dict[str, Any], xy: np.ndarray) -> np.ndarray:
    return np.asarray([_ground_height_bilinear(model, float(x), float(y)) for x, y in xy], dtype=float)


def _decision_to_direct_paths(
    vector: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> list[np.ndarray]:
    """Decode matched-hybrid vectors as direct internal waypoints."""
    block = np.asarray(vector, dtype=float).reshape(fleet_size, n_waypoints, 3)
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    safe_h = _safe_height(model)
    paths: list[np.ndarray] = []
    for uav_idx in range(fleet_size):
        start = np.asarray(starts[uav_idx, :3], dtype=float).copy()
        goal = np.asarray(goals[uav_idx, :3], dtype=float).copy()
        endpoint_ground = _ground_values(model, np.vstack([start[:2], goal[:2]]))
        start[2] = endpoint_ground[0] + float(_clip_relative_altitude(model, start[2], safe_h))
        goal[2] = endpoint_ground[1] + float(_clip_relative_altitude(model, goal[2], safe_h))

        internal = np.asarray(block[uav_idx], dtype=float).copy()
        internal[:, 0] = np.clip(internal[:, 0], float(model["xmin"]), float(model["xmax"]))
        internal[:, 1] = np.clip(internal[:, 1], float(model["ymin"]), float(model["ymax"]))
        internal[:, 2] = _ground_values(model, internal[:, :2]) + _clip_relative_altitude(model, internal[:, 2], safe_h)
        paths.append(np.vstack([start, internal, goal]))
    return paths


def _ensure_fleet_endpoints(
    model: dict[str, Any],
    fleet_size: int,
    seed: int,
    separation_min: float,
) -> tuple[dict[str, Any], int]:
    """Ensure starts/goals are available for the requested fleet size."""
    normalized = dict(model)
    starts_raw = normalized.get("starts")
    goals_raw = normalized.get("goals")

    starts: np.ndarray | None = None
    goals: np.ndarray | None = None
    if starts_raw is not None and goals_raw is not None:
        starts = np.asarray(starts_raw, dtype=float)
        goals = np.asarray(goals_raw, dtype=float)
        if starts.ndim == 1:
            starts = starts.reshape(1, -1)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        starts = starts[:, :3]
        goals = goals[:, :3]

    if starts is None or goals is None or starts.shape[0] < fleet_size or goals.shape[0] < fleet_size:
        assignment = sample_homogeneous_assignments(
            terrain=normalized,
            fleet_size=int(fleet_size),
            seed=int(seed),
            separation_min=float(separation_min),
            mission_prefix="runtime",
        )
        starts = np.asarray(assignment.starts, dtype=float)
        goals = np.asarray(assignment.goals, dtype=float)
    else:
        starts = starts[:fleet_size]
        goals = goals[:fleet_size]

    normalized["starts"] = starts
    normalized["goals"] = goals
    normalized["fleetSize"] = float(fleet_size)
    normalized["separationMin"] = float(separation_min)
    return normalized, int(fleet_size)


def _build_navigation_bounds(
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    max_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)

    def _endpoint_abs(point: np.ndarray, enforce_safe_h: bool) -> np.ndarray:
        px = float(point[0])
        py = float(point[1])
        z_rel = float(point[2])
        if enforce_safe_h and "safeH" in model and model["safeH"] is not None:
            z_rel = max(z_rel, float(model["safeH"]))
        return np.array([px, py, z_rel + _ground_height_bilinear(model, px, py)], dtype=float)

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


def _transformation_matrix(radius: float, phi: float, psi: float) -> np.ndarray:
    cp = math.cos(phi)
    sp = math.sin(phi)
    cs = math.cos(-psi)
    ss = math.sin(-psi)
    rot_z = np.array([[cp, -sp, 0.0, 0.0], [sp, cp, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    rot_y = np.array([[cs, 0.0, ss, 0.0], [0.0, 1.0, 0.0, 0.0], [-ss, 0.0, cs, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    trans_x = np.array(
        [[1.0, 0.0, 0.0, radius], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float
    )
    return rot_z @ rot_y @ trans_x


def _ground_height_bilinear(model: dict[str, Any], x: float, y: float) -> float:
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


def _spherical_to_cart(solution: dict[str, np.ndarray], model: dict[str, Any]) -> dict[str, np.ndarray]:
    n_points = solution["r"].shape[0]
    xs, ys, zs_rel = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
    xf, yf, zf_rel = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
    if "safeH" in model and model["safeH"] is not None:
        zs_rel = max(float(zs_rel), float(model["safeH"]))
    zs = float(zs_rel + _ground_height_bilinear(model, xs, ys))
    zf = float(zf_rel + _ground_height_bilinear(model, xf, yf))
    direction = np.array([xf - xs, yf - ys, zf - zs], dtype=float)
    phi_start = math.atan2(direction[1], direction[0])
    psi_start = math.atan2(direction[2], np.linalg.norm(direction[:2]))
    current = np.array(
        [[1.0, 0.0, 0.0, xs], [0.0, 1.0, 0.0, ys], [0.0, 0.0, 1.0, zs], [0.0, 0.0, 0.0, 1.0]], dtype=float
    )
    current = current @ _transformation_matrix(0.0, phi_start, psi_start)
    x_coord = np.zeros(n_points, dtype=float)
    y_coord = np.zeros(n_points, dtype=float)
    z_abs = np.zeros(n_points, dtype=float)
    for index in range(n_points):
        current = current @ _transformation_matrix(
            float(solution["r"][index]), float(solution["phi"][index]), float(solution["psi"][index])
        )
        x_coord[index] = current[0, 3]
        y_coord[index] = current[1, 3]
        z_abs[index] = current[2, 3]
    x_coord = np.clip(x_coord, float(model["xmin"]), float(model["xmax"]))
    y_coord = np.clip(y_coord, float(model["ymin"]), float(model["ymax"]))
    return {"x": x_coord, "y": y_coord, "z_abs": z_abs}


def _position_to_cart(
    position: dict[str, np.ndarray], model: dict[str, Any], representation: str
) -> dict[str, np.ndarray]:
    if representation == "SC":
        return _spherical_to_cart(position, model)
    return {
        "x": np.clip(position["x"], float(model["xmin"]), float(model["xmax"])),
        "y": np.clip(position["y"], float(model["ymin"]), float(model["ymax"])),
        "z": np.clip(position["z"], float(model["zmin"]), float(model["zmax"])),
    }


def _cart_to_absolute_path(cart: dict[str, np.ndarray], model: dict[str, Any]) -> np.ndarray:
    xs, ys, zs_rel = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
    xf, yf, zf_rel = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
    if "safeH" in model and model["safeH"] is not None:
        zs_rel = max(float(zs_rel), float(model["safeH"]))
    start_abs = float(zs_rel + _ground_height_bilinear(model, xs, ys))
    goal_abs = float(zf_rel + _ground_height_bilinear(model, xf, yf))
    x_all = np.hstack([[xs], cart["x"], [xf]])
    y_all = np.hstack([[ys], cart["y"], [yf]])
    z_abs = np.hstack([[start_abs], cart["z_abs"], [goal_abs]])
    path = np.zeros((x_all.shape[0], 3), dtype=float)
    for index in range(x_all.shape[0]):
        path[index] = [x_all[index], y_all[index], z_abs[index]]
    return path


def _decision_to_paths_spherical(
    vector: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> list[np.ndarray]:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    block = np.asarray(vector, dtype=float).reshape(fleet_size, n_waypoints, 3)
    paths: list[np.ndarray] = []
    for uav_idx in range(fleet_size):
        local_model = dict(model)
        local_model["start"] = starts[uav_idx].reshape(-1)[:3]
        local_model["end"] = goals[uav_idx].reshape(-1)[:3]
        position = {
            "r": block[uav_idx, :, 0],
            "phi": block[uav_idx, :, 1],
            "psi": block[uav_idx, :, 2],
        }
        cart = _position_to_cart(position, local_model, "SC")
        paths.append(_cart_to_absolute_path(cart, local_model))
    return paths


def _normalize_objective_vector(
    objective: np.ndarray,
    details: dict[str, Any],
    model: dict[str, Any],
    fleet_size: int,
) -> np.ndarray:
    del model, fleet_size
    obj = np.asarray(objective, dtype=float).reshape(-1)
    if obj.size != 4:
        return obj
    if np.any(~np.isfinite(obj)):
        return obj

    normalized = np.clip(obj, 0.0, 1.0)

    details["objective_raw"] = np.asarray(obj, dtype=float)
    details["objective_normalized"] = np.asarray(normalized, dtype=float)
    details["objective_scale"] = np.ones(4, dtype=float)
    return details["objective_normalized"]


def _evaluate_population(
    population: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    representation: str = "cart",
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for idx in range(population.shape[0]):
        vector = np.asarray(population[idx], dtype=float).copy()
        if representation == "SC":
            paths = _decision_to_paths_spherical(vector, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        elif representation == "direct":
            paths = _decision_to_direct_paths(vector, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        else:
            paths = decision_to_paths(vector, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        objective, details = evaluate_mission_details(paths, model)
        objective = _normalize_objective_vector(objective, details, model=model, fleet_size=fleet_size)
        details["paths"] = paths
        candidates.append(Candidate(vector=vector, objective=objective, details=details))
    return candidates


def _vectors_from_candidates(candidates: list[Candidate], fallback: np.ndarray) -> np.ndarray:
    try:
        vectors = np.stack([np.asarray(candidate.vector, dtype=float).reshape(-1) for candidate in candidates], axis=0)
    except (TypeError, ValueError):
        return np.asarray(fallback, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    return vectors if vectors.shape == fallback.shape else fallback


def _constraint_violation(candidate: Candidate, model: dict[str, Any]) -> float:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    violation = 0.0

    if objective.size == 0 or np.any(~np.isfinite(objective)):
        non_finite = float(np.sum(~np.isfinite(objective))) if objective.size > 0 else 1.0
        violation += 10.0 * max(1.0, non_finite)

    separation_min = float(model.get("separationMin", model.get("safeDist", 10.0)))
    drone_size = float(model.get("droneSize", 1.0))

    if float(details.get("separationViolation", 0.0)) > 0.5:
        min_sep = float(details.get("minSeparation", np.nan))
        if np.isfinite(min_sep):
            violation += max(0.0, (separation_min - min_sep) / max(separation_min, 1e-9))
        else:
            violation += 1.0

    if float(details.get("collisionViolation", 0.0)) > 0.5:
        min_clearance = float(details.get("minClearance", np.nan))
        if np.isfinite(min_clearance):
            violation += max(0.0, (drone_size - min_clearance) / max(drone_size, 1e-9))
        else:
            violation += 1.0

    if float(details.get("feasible", 1.0)) <= 0.5 and violation <= 0.0:
        violation = 1.0
    return float(max(0.0, violation))


def _constraint_violation_vector(candidates: list[Candidate], model: dict[str, Any]) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=float)
    return np.asarray([_constraint_violation(candidate, model) for candidate in candidates], dtype=float)


def _resume_run_scores(
    run_dir: Path,
    problem_index: int,
    objective_count: int,
    compute_metrics: bool,
) -> np.ndarray | None:
    popobj_path = run_dir / "final_popobj.mat"
    if not popobj_path.exists():
        return None
    try:
        data = load_mat(popobj_path)
        matrix_raw = data.get("PopObj")
        if matrix_raw is None:
            matrix_raw = data.get("final_popobj")
        if matrix_raw is None:
            matrix_raw = data.get("popObj")
        if matrix_raw is None:
            return None
        matrix = np.asarray(matrix_raw, dtype=float)
        if matrix.size == 0:
            return None
        if compute_metrics:
            hv = cal_metric(1, matrix, problem_index, objective_count)
            spacing = cal_metric(2, matrix, problem_index, objective_count)
            return np.array([hv, spacing], dtype=float)
        return np.zeros(2, dtype=float)
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _resolve_run_indices(params: BenchmarkParams) -> tuple[int, ...]:
    raw = params.run_indices
    if raw is None:
        return tuple(range(1, int(params.runs) + 1))
    if isinstance(raw, (list, tuple)):
        indices = [int(item) for item in raw if int(item) >= 1]
    else:
        indices = [int(raw)] if int(raw) >= 1 else []
    if not indices:
        return tuple(range(1, int(params.runs) + 1))
    return tuple(dict.fromkeys(indices))


def _should_write_final_hv(params: BenchmarkParams) -> bool:
    return bool(params.write_final_hv)


def _torch_device_peak_bytes(device_tag: str) -> float:
    if "cuda" not in device_tag:
        return 0.0
    try:
        import torch  # type: ignore[import-not-found]

        device = device_tag.split(":")[-1] if ":" in device_tag else "cuda:0"
        try:
            return float(torch.cuda.max_memory_allocated(device))
        except (RuntimeError, ValueError):
            return 0.0
    except ImportError:
        return 0.0
