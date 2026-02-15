from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.core.dominance import dominates


def determine_domination(costs: np.ndarray) -> np.ndarray:
    n_points = costs.shape[0]
    dominated = np.zeros(n_points, dtype=bool)
    for left_index in range(n_points):
        if dominated[left_index]:
            continue
        for right_index in range(n_points):
            if left_index == right_index:
                continue
            if dominates(costs[right_index], costs[left_index]):
                dominated[left_index] = True
                break
    return dominated


def create_grid(costs: np.ndarray, n_grid: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.min(costs, axis=0)
    maximum = np.max(costs, axis=0)
    delta = maximum - minimum
    minimum = minimum - alpha * delta
    maximum = maximum + alpha * delta
    lb = []
    ub = []
    for index in range(costs.shape[1]):
        ticks = np.linspace(minimum[index], maximum[index], n_grid + 1)
        lb.append(np.hstack([[-np.inf], ticks]))
        ub.append(np.hstack([ticks, [np.inf]]))
    return np.array(lb, dtype=float), np.array(ub, dtype=float)


def find_grid_index(cost: np.ndarray, grid_lb: np.ndarray, grid_ub: np.ndarray) -> tuple[int, np.ndarray]:
    n_obj = cost.shape[0]
    n_grid = grid_lb.shape[1] - 2
    sub_index = np.zeros(n_obj, dtype=int)
    for objective_index in range(n_obj):
        match = np.where(cost[objective_index] < grid_ub[objective_index])[0]
        # MATLAB uses 1-based grid sub-indices; keep parity here.
        sub_index[objective_index] = int(match[0] + 1 if match.size > 0 else n_grid)
    grid_index = int(sub_index[0])
    for objective_index in range(1, n_obj):
        grid_index = (grid_index - 1) * n_grid + sub_index[objective_index]
    return grid_index, sub_index


def roulette_wheel(probabilities: np.ndarray) -> int:
    cumulative = np.cumsum(probabilities)
    random_draw = np.random.rand()
    return int(np.where(random_draw <= cumulative)[0][0])


def normalize_objectives(objectives: np.ndarray) -> np.ndarray:
    if objectives.size == 0:
        return objectives
    minimum = np.min(objectives, axis=0)
    maximum = np.max(objectives, axis=0)
    ranges = maximum - minimum
    ranges[ranges <= 0] = 1.0
    normalized = (objectives - minimum.reshape(1, -1)) / ranges.reshape(1, -1)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normalized / norms


def select_leader_ref(costs: np.ndarray, reference: np.ndarray, n_select: int) -> np.ndarray:
    if costs.shape[0] == 0:
        return np.zeros(0, dtype=int)
    pop_obj = normalize_objectives(costs)
    reference = normalize_objectives(reference)
    cosine = pop_obj @ reference.T
    reference_assignment = np.argmax(cosine, axis=1)
    n_reference = reference.shape[0]
    rho = np.bincount(reference_assignment, minlength=n_reference)
    picks = np.zeros(n_select, dtype=int)
    active = np.ones(n_reference, dtype=bool)
    count = 0
    while count < n_select:
        available = np.where(active)[0]
        if available.size == 0:
            picks[count:] = np.random.randint(0, costs.shape[0], size=n_select - count)
            break
        min_count = np.min(rho[available])
        candidate = available[rho[available] == min_count]
        ref_index = int(np.random.choice(candidate))
        associated = np.where(reference_assignment == ref_index)[0]
        if associated.size > 0:
            picks[count] = int(np.random.choice(associated))
            rho[ref_index] += 1
            count += 1
        else:
            active[ref_index] = False
    return picks


@dataclass(slots=True)
class AtlasConfig:
    enabled: bool = False
    n_topology_bins: int = 24
    n_robust_bins: int = 4
    max_obstacles: int = 3
    hash_levels: int = 6
    objective_weight: float = 0.5
    atlas_weight: float = 0.5


def build_atlas_config(ablation: dict[str, Any]) -> AtlasConfig:
    config = AtlasConfig()
    if not ablation:
        return config
    config.enabled = bool(ablation.get("useTopologyRobustArchive", config.enabled))
    config.n_topology_bins = max(2, int(ablation.get("atlasTopologyBins", config.n_topology_bins)))
    config.n_robust_bins = max(2, int(ablation.get("atlasRobustBins", config.n_robust_bins)))
    config.max_obstacles = max(1, int(ablation.get("atlasMaxObstacles", config.max_obstacles)))
    config.hash_levels = max(2, int(ablation.get("atlasHashLevels", config.hash_levels)))
    config.objective_weight = max(0.0, float(ablation.get("atlasObjectiveWeight", config.objective_weight)))
    config.atlas_weight = max(0.0, float(ablation.get("atlasTopologyWeight", config.atlas_weight)))
    total_weight = config.objective_weight + config.atlas_weight
    if total_weight <= 0:
        config.objective_weight = 0.5
        config.atlas_weight = 0.5
    else:
        config.objective_weight /= total_weight
        config.atlas_weight /= total_weight
    return config


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def extract_obstacles(model: dict[str, Any], max_obstacles: int) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    radii = []
    if "nofly_c" in model and "nofly_r" in model and model["nofly_c"] is not None:
        nofly_center = np.asarray(model["nofly_c"], dtype=float)
        if nofly_center.ndim == 1:
            nofly_center = nofly_center.reshape(1, -1)
        nofly_center = nofly_center[:, :2]
        nofly_radius = np.asarray(model["nofly_r"], dtype=float).reshape(-1)
        if nofly_radius.size == 1:
            nofly_radius = np.repeat(nofly_radius, nofly_center.shape[0])
        centers.append(nofly_center)
        radii.append(nofly_radius[: nofly_center.shape[0]])
    if "threats" in model and model["threats"] is not None:
        threats = np.asarray(model["threats"], dtype=float)
        if threats.ndim == 2 and threats.shape[1] >= 4:
            centers.append(threats[:, :2])
            radii.append(threats[:, 3])
    if not centers:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    merged_centers = np.vstack(centers)
    merged_radii = np.hstack(radii)
    valid = np.all(np.isfinite(merged_centers), axis=1) & np.isfinite(merged_radii) & (merged_radii > 0)
    merged_centers = merged_centers[valid]
    merged_radii = merged_radii[valid]
    if merged_centers.size == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    order = np.argsort(-merged_radii)
    order = order[: min(max_obstacles, order.size)]
    return merged_centers[order], merged_radii[order]


def topology_signature(path_xyz: np.ndarray, model: dict[str, Any], max_obstacles: int) -> np.ndarray:
    signature = np.zeros(4 + 3 * max_obstacles, dtype=float)
    if path_xyz.shape[0] < 2:
        return signature
    xy = np.asarray(path_xyz[:, :2], dtype=float)
    dx = max(1.0, float(model["xmax"]) - float(model["xmin"]))
    dy = max(1.0, float(model["ymax"]) - float(model["ymin"]))
    map_diag = math.sqrt(dx * dx + dy * dy)
    dxy = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(dxy, axis=1)
    path_len_norm = float(np.sum(seg_len) / map_diag)
    heading = np.arctan2(dxy[:, 1], dxy[:, 0])
    if heading.size >= 2:
        turn = wrap_to_pi(np.diff(heading))
        mean_turn = float(np.mean(np.abs(turn)) / np.pi)
        signed_turn = float(np.sum(turn) / (np.pi * max(1, turn.size)))
        turn_std = float(np.std(turn) / np.pi)
    else:
        mean_turn = 0.0
        signed_turn = 0.0
        turn_std = 0.0
    signature[:4] = [path_len_norm, mean_turn, signed_turn, turn_std]

    obstacle_features = np.zeros(3 * max_obstacles, dtype=float)
    centers, radii = extract_obstacles(model, max_obstacles)
    if centers.shape[0] > 0:
        base_dir = xy[-1] - xy[0]
        if np.linalg.norm(base_dir) < 1e-12:
            base_dir = np.array([1.0, 0.0], dtype=float)
        for obstacle_index in range(centers.shape[0]):
            center = centers[obstacle_index]
            radius = radii[obstacle_index]
            dist = np.linalg.norm(xy - center.reshape(1, -1), axis=1)
            nearest_idx = int(np.argmin(dist))
            side_vec = xy[nearest_idx] - center
            side = float(np.sign(base_dir[0] * side_vec[1] - base_dir[1] * side_vec[0]))
            if not np.isfinite(side):
                side = 0.0
            angle = np.unwrap(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
            winding = float((angle[-1] - angle[0]) / (2 * np.pi))
            clearance = float((dist[nearest_idx] - radius) / map_diag)
            base = 3 * obstacle_index
            obstacle_features[base : base + 3] = [side, winding, clearance]
    signature[4:] = obstacle_features
    signature[~np.isfinite(signature)] = 0.0
    return signature


def normalize_signature_for_hash(signature: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(signature, dtype=float)
    if signature.size == 0:
        return normalized
    normalized[0] = np.clip(signature[0], 0.0, 3.0) / 3.0
    if signature.size > 1:
        normalized[1] = np.clip(signature[1], 0.0, 1.0)
    if signature.size > 2:
        normalized[2] = (np.clip(signature[2], -1.0, 1.0) + 1.0) / 2.0
    if signature.size > 3:
        normalized[3] = np.clip(signature[3], 0.0, 1.0)
    for index in range(4, signature.size):
        local = (index - 4) % 3
        if local in (0, 1):
            normalized[index] = (np.clip(signature[index], -1.0, 1.0) + 1.0) / 2.0
        else:
            normalized[index] = (np.clip(signature[index], -0.2, 0.2) + 0.2) / 0.4
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def topology_bin_from_signature(signature: np.ndarray, config: AtlasConfig) -> int:
    n_bins = max(2, int(config.n_topology_bins))
    levels = max(2, int(config.hash_levels))
    quantized = np.floor(normalize_signature_for_hash(signature) * levels)
    quantized = np.clip(quantized, 0, levels - 1).astype(int)
    hash_value = 0
    for index, value in enumerate(quantized, start=1):
        hash_value = (hash_value + (value + 1) * (2 * index + 1)) % n_bins
    return int(hash_value + 1)


def robustness_from_cost(cost: np.ndarray, n_bins: int) -> tuple[float, int]:
    n_bins = max(2, int(n_bins))
    if cost.size < 4:
        return 0.0, 1
    second = float(cost[1])
    fourth = float(cost[3])
    if not np.isfinite(second):
        score = 0.0
    else:
        smooth_penalty = 0.35 * fourth if np.isfinite(fourth) and fourth > 0 else 0.0
        score = 1.0 / (1.0 + max(0.0, second) + smooth_penalty)
    score = float(np.clip(score, 0.0, 1.0))
    bin_index = int(min(n_bins, max(1, math.floor(score * n_bins) + 1)))
    return score, bin_index


def archive_occupancies(grid_indices: np.ndarray, atlas_indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    n_points = grid_indices.shape[0]
    objective_occ = np.ones(n_points, dtype=float)
    atlas_occ = np.ones(n_points, dtype=float)
    if n_points == 0:
        return objective_occ, atlas_occ
    unique_grid, inverse_grid = np.unique(grid_indices, return_inverse=True)
    del unique_grid
    objective_counts = np.bincount(inverse_grid)
    objective_occ = objective_counts[inverse_grid].astype(float)
    if atlas_indices is not None and atlas_indices.size == n_points:
        unique_atlas, inverse_atlas = np.unique(atlas_indices, return_inverse=True)
        del unique_atlas
        atlas_counts = np.bincount(inverse_atlas)
        atlas_occ = atlas_counts[inverse_atlas].astype(float)
    return objective_occ, atlas_occ


def delete_one_with_weights(indices: np.ndarray, gamma: float, objective_weight: float, atlas_weight: float, atlas_indices: np.ndarray | None) -> int:
    obj_occ, atlas_occ = archive_occupancies(indices, atlas_indices)
    occ = objective_weight * obj_occ + atlas_weight * atlas_occ
    probability = np.exp(gamma * occ)
    if np.sum(probability) <= 0 or not np.all(np.isfinite(probability)):
        probability = np.ones_like(probability) / probability.shape[0]
    else:
        probability = probability / np.sum(probability)
    return roulette_wheel(probability)


def select_leader_with_weights(indices: np.ndarray, beta: float, objective_weight: float, atlas_weight: float, atlas_indices: np.ndarray | None) -> int:
    obj_occ, atlas_occ = archive_occupancies(indices, atlas_indices)
    occ = objective_weight * obj_occ + atlas_weight * atlas_occ
    probability = np.exp(-beta * occ)
    if np.sum(probability) <= 0 or not np.all(np.isfinite(probability)):
        probability = np.ones_like(probability) / probability.shape[0]
    else:
        probability = probability / np.sum(probability)
    return roulette_wheel(probability)


def archive_region_count(grid_indices: np.ndarray, atlas_indices: np.ndarray | None) -> int:
    if atlas_indices is not None and atlas_indices.size > 0:
        return max(1, np.unique(atlas_indices).shape[0])
    if grid_indices.size == 0:
        return 1
    return max(1, np.unique(grid_indices).shape[0])


def mutate(
    position: dict[str, np.ndarray],
    best_position: dict[str, np.ndarray],
    delta: float,
    var_min: dict[str, np.ndarray],
    var_max: dict[str, np.ndarray],
    representation: str,
    region_count: int,
) -> dict[str, np.ndarray]:
    beta = math.tanh(delta / max(1, region_count))
    updated: dict[str, np.ndarray] = {}
    if representation == "CC":
        for key in ("x", "y", "z"):
            step = (var_max[key] - var_min[key]) * beta
            updated[key] = np.clip(position[key] + np.random.randn(*position[key].shape) * step, var_min[key], var_max[key])
        return updated
    for key in ("r", "phi", "psi"):
        updated[key] = np.clip(
            position[key] + np.random.randn(*position[key].shape) * best_position[key] * beta,
            var_min[key],
            var_max[key],
        )
    return updated


def transformation_matrix(radius: float, phi: float, psi: float) -> np.ndarray:
    cp = math.cos(phi)
    sp = math.sin(phi)
    cs = math.cos(-psi)
    ss = math.sin(-psi)
    rot_z = np.array([[cp, -sp, 0.0, 0.0], [sp, cp, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    rot_y = np.array([[cs, 0.0, ss, 0.0], [0.0, 1.0, 0.0, 0.0], [-ss, 0.0, cs, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    trans_x = np.array([[1.0, 0.0, 0.0, radius], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ trans_x


def spherical_to_cart(solution: dict[str, np.ndarray], model: dict[str, Any]) -> dict[str, np.ndarray]:
    n_points = solution["r"].shape[0]
    xs, ys, zs = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
    xf, yf, zf = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
    if "safeH" in model and model["safeH"] is not None:
        zs = float(model["safeH"])
        zf = float(model["safeH"])
    direction = np.array([xf - xs, yf - ys, zf - zs], dtype=float)
    phi_start = math.atan2(direction[1], direction[0])
    psi_start = math.atan2(direction[2], np.linalg.norm(direction[:2]))
    current = np.array([[1.0, 0.0, 0.0, xs], [0.0, 1.0, 0.0, ys], [0.0, 0.0, 1.0, zs], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    current = current @ transformation_matrix(0.0, phi_start, psi_start)
    x_coord = np.zeros(n_points, dtype=float)
    y_coord = np.zeros(n_points, dtype=float)
    z_coord = np.zeros(n_points, dtype=float)
    for index in range(n_points):
        current = current @ transformation_matrix(float(solution["r"][index]), float(solution["phi"][index]), float(solution["psi"][index]))
        x_coord[index] = current[0, 3]
        y_coord[index] = current[1, 3]
        z_coord[index] = current[2, 3]
    x_coord = np.clip(x_coord, float(model["xmin"]), float(model["xmax"]))
    y_coord = np.clip(y_coord, float(model["ymin"]), float(model["ymax"]))
    z_coord = np.clip(z_coord, float(model["zmin"]), float(model["zmax"]))
    return {"x": x_coord, "y": y_coord, "z": z_coord}


def position_to_cart(position: dict[str, np.ndarray], model: dict[str, Any], representation: str) -> dict[str, np.ndarray]:
    if representation == "SC":
        return spherical_to_cart(position, model)
    return {
        "x": np.clip(position["x"], float(model["xmin"]), float(model["xmax"])),
        "y": np.clip(position["y"], float(model["ymin"]), float(model["ymax"])),
        "z": np.clip(position["z"], float(model["zmin"]), float(model["zmax"])),
    }


def cart_to_absolute_path(cart: dict[str, np.ndarray], model: dict[str, Any]) -> np.ndarray:
    xs, ys, zs = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
    xf, yf, zf = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
    if "safeH" in model and model["safeH"] is not None:
        zs = float(model["safeH"])
        zf = float(model["safeH"])
    x_all = np.hstack([[xs], cart["x"], [xf]])
    y_all = np.hstack([[ys], cart["y"], [yf]])
    z_rel = np.hstack([[zs], cart["z"], [zf]])
    path = np.zeros((x_all.shape[0], 3), dtype=float)
    for index in range(x_all.shape[0]):
        xi = int(np.clip(round(x_all[index]), 1, int(model["xmax"]))) - 1
        yi = int(np.clip(round(y_all[index]), 1, int(model["ymax"]))) - 1
        ground = float(np.asarray(model["H"], dtype=float)[yi, xi])
        path[index] = [x_all[index], y_all[index], z_rel[index] + ground]
    return path
