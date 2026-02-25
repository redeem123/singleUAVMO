from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.shared.fleet_runner import run_fleet_nmopso
from uav_benchmark.algorithms.shared.mission_stats import build_mission_stats
from uav_benchmark.core.dominance import dominates
from uav_benchmark.core.evaluate_path import evaluate_path
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir


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


def delete_one_with_weights(
    indices: np.ndarray,
    gamma: float,
    objective_weight: float,
    atlas_weight: float,
    atlas_indices: np.ndarray | None,
) -> int:
    obj_occ, atlas_occ = archive_occupancies(indices, atlas_indices)
    occ = objective_weight * obj_occ + atlas_weight * atlas_occ
    probability = np.exp(gamma * occ)
    if np.sum(probability) <= 0 or not np.all(np.isfinite(probability)):
        probability = np.ones_like(probability) / probability.shape[0]
    else:
        probability = probability / np.sum(probability)
    return roulette_wheel(probability)


def select_leader_with_weights(
    indices: np.ndarray,
    beta: float,
    objective_weight: float,
    atlas_weight: float,
    atlas_indices: np.ndarray | None,
) -> int:
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
    direction = np.array([xf - xs, yf - ys, zf - zs], dtype=float)
    phi_start = math.atan2(direction[1], direction[0])
    psi_start = math.atan2(direction[2], np.linalg.norm(direction[:2]))
    current = np.array([[1.0, 0.0, 0.0, xs], [0.0, 1.0, 0.0, ys], [0.0, 0.0, 1.0, zs], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    current = current @ transformation_matrix(0.0, phi_start, psi_start)
    x_coord = np.zeros(n_points, dtype=float)
    y_coord = np.zeros(n_points, dtype=float)
    z_coord = np.zeros(n_points, dtype=float)
    for index in range(n_points):
        current = current @ transformation_matrix(
            float(solution["r"][index]),
            float(solution["phi"][index]),
            float(solution["psi"][index]),
        )
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


@dataclass(slots=True)
class NMOPSOParticle:
    position: dict[str, np.ndarray]
    velocity: dict[str, np.ndarray]
    cost: np.ndarray
    best_position: dict[str, np.ndarray]
    best_cost: np.ndarray
    grid_index: int = 0
    grid_sub_index: np.ndarray | None = None
    topology_signature: np.ndarray | None = None
    topology_bin: int = 1
    robustness_score: float = 0.0
    robustness_bin: int = 1
    atlas_cell_index: int = 1


def _clone_particle(particle: NMOPSOParticle) -> NMOPSOParticle:
    return NMOPSOParticle(
        position={key: value.copy() for key, value in particle.position.items()},
        velocity={key: value.copy() for key, value in particle.velocity.items()},
        cost=particle.cost.copy(),
        best_position={key: value.copy() for key, value in particle.best_position.items()},
        best_cost=particle.best_cost.copy(),
        grid_index=int(particle.grid_index),
        grid_sub_index=None if particle.grid_sub_index is None else particle.grid_sub_index.copy(),
        topology_signature=None if particle.topology_signature is None else particle.topology_signature.copy(),
        topology_bin=int(particle.topology_bin),
        robustness_score=float(particle.robustness_score),
        robustness_bin=int(particle.robustness_bin),
        atlas_cell_index=int(particle.atlas_cell_index),
    )


def _select_member_from_grid_cells(indices: np.ndarray, pressure: float, invert: bool) -> int:
    if indices.size == 0:
        return 0
    unique_cells, inverse = np.unique(indices.astype(int), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    exponents = (-pressure * counts) if invert else (pressure * counts)
    cell_probability = np.exp(exponents)
    if np.sum(cell_probability) <= 0 or not np.all(np.isfinite(cell_probability)):
        cell_probability = np.ones_like(cell_probability) / cell_probability.shape[0]
    else:
        cell_probability = cell_probability / np.sum(cell_probability)
    chosen_cell = unique_cells[int(roulette_wheel(cell_probability))]
    members = np.where(indices.astype(int) == int(chosen_cell))[0]
    if members.size == 0:
        return int(np.random.randint(0, indices.size))
    return int(np.random.choice(members))


def _normalize_representation(value: Any) -> str:
    if isinstance(value, (int, float)):
        return "CC" if int(value) == 0 else "SC"
    representation = str(value).strip().upper()
    if representation in {"CC", "CARTESIAN"}:
        return "CC"
    return "SC"


def _parse_ablation(params: BenchmarkParams) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "name": "",
        "useRepository": True,
        "useGrid": True,
        "useMutation": True,
        "useAdaptiveMutation": True,
        "useRegionMutation": True,
        "mutationProb": 0.1,
        "representation": "SC",
        "useReferenceLeader": False,
        "useTwoLayerRef": False,
        "nRep": 50,
        "nGrid": 5,
        "alpha_grid": 0.1,
        "beta": 2.0,
        "gamma": 2.0,
        "w": 1.0,
        "wdamp": 0.98,
        "c1": 1.5,
        "c2": 1.5,
        "mu": 0.5,
        "delta": 20.0,
        "metricInterval": 100,
        "useTopologyRobustArchive": False,
        "atlasTopologyBins": 24,
        "atlasRobustBins": 4,
        "atlasMaxObstacles": 3,
        "atlasHashLevels": 6,
        "atlasObjectiveWeight": 0.5,
        "atlasTopologyWeight": 0.5,
    }
    if "ablation" in params.extra and isinstance(params.extra["ablation"], dict):
        defaults.update(params.extra["ablation"])
    for key in defaults:
        if key in params.extra:
            defaults[key] = params.extra[key]
    defaults["representation"] = _normalize_representation(defaults.get("representation", "SC"))
    defaults["useReferenceLeader"] = bool(defaults.get("useReferenceLeader", False))
    defaults["useTwoLayerRef"] = bool(defaults.get("useTwoLayerRef", False))
    return defaults


def _build_reference_points(n_points: int, objective_count: int, use_two_layer: bool) -> np.ndarray:
    from uav_benchmark.core.nsga3_ops import uniform_point

    first, _ = uniform_point(n_points, objective_count, "NBI")
    if not use_two_layer:
        return first
    second, _ = uniform_point(max(1, n_points // 2), objective_count, "NBI")
    second = second / 2.0 + 1.0 / (2.0 * objective_count)
    return np.vstack([first, second])


def _nmopso_cost(cart_sol: dict[str, np.ndarray], model: dict[str, Any]) -> np.ndarray:
    xs, ys, zs = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
    xf, yf, zf = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
    if "safeH" in model and model["safeH"] is not None:
        zs = float(model["safeH"])
    x_all = np.hstack([[xs], cart_sol["x"], [xf]])
    y_all = np.hstack([[ys], cart_sol["y"], [yf]])
    z_rel = np.hstack([[zs], cart_sol["z"], [zf]])
    path = np.zeros((x_all.shape[0], 3), dtype=float)
    for index in range(x_all.shape[0]):
        xi = int(np.clip(round(x_all[index]), 1, int(model["xmax"]))) - 1
        yi = int(np.clip(round(y_all[index]), 1, int(model["ymax"]))) - 1
        abs_z = z_rel[index] + float(np.asarray(model["H"], dtype=float)[yi, xi])
        if z_rel[index] < 0:
            return np.array([np.inf, np.inf, np.inf, np.inf], dtype=float)
        path[index] = [x_all[index], y_all[index], abs_z]
    return evaluate_path(path, model)


def _initialize_particle(
    model: dict[str, Any],
    representation: str,
    var_min: dict[str, np.ndarray],
    var_max: dict[str, np.ndarray],
) -> NMOPSOParticle:
    if representation == "SC":
        position = {
            "r": np.random.uniform(var_min["r"], var_max["r"]),
            "psi": np.random.uniform(var_min["psi"], var_max["psi"]),
            "phi": np.random.uniform(var_min["phi"], var_max["phi"]),
        }
        velocity = {key: np.zeros_like(value) for key, value in position.items()}
    else:
        position = {
            "x": np.random.uniform(var_min["x"], var_max["x"]),
            "y": np.random.uniform(var_min["y"], var_max["y"]),
            "z": np.random.uniform(var_min["z"], var_max["z"]),
        }
        velocity = {key: np.zeros_like(value) for key, value in position.items()}
    cart = position_to_cart(position, model, representation)
    cost = _nmopso_cost(cart, model)
    return NMOPSOParticle(
        position={key: value.copy() for key, value in position.items()},
        velocity={key: value.copy() for key, value in velocity.items()},
        cost=cost.copy(),
        best_position={key: value.copy() for key, value in position.items()},
        best_cost=cost.copy(),
    )


def _update_particle_velocity_and_position(
    particle: NMOPSOParticle,
    leader: NMOPSOParticle,
    var_min: dict[str, np.ndarray],
    var_max: dict[str, np.ndarray],
    vel_min: dict[str, np.ndarray],
    vel_max: dict[str, np.ndarray],
    c1: float,
    c2: float,
    inertia: float,
) -> None:
    for key in particle.position.keys():
        particle.velocity[key] = (
            inertia * particle.velocity[key]
            + c1 * np.random.rand(*particle.position[key].shape) * (particle.best_position[key] - particle.position[key])
            + c2 * np.random.rand(*particle.position[key].shape) * (leader.position[key] - particle.position[key])
        )
        particle.velocity[key] = np.clip(particle.velocity[key], vel_min[key], vel_max[key])
        particle.position[key] = particle.position[key] + particle.velocity[key]
        out_of_range = (particle.position[key] < var_min[key]) | (particle.position[key] > var_max[key])
        particle.velocity[key][out_of_range] = -particle.velocity[key][out_of_range]
        particle.position[key] = np.clip(particle.position[key], var_min[key], var_max[key])


def run_nmopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    use_legacy_runner = bool(params.extra.get("legacyPathRunner", False))
    if (not use_legacy_runner) or int(params.fleet_size) > 1:
        return run_fleet_nmopso(model, params)
    objective_count = 4
    model = dict(model)
    model["n"] = 10
    ablation = _parse_ablation(params)
    representation = ablation["representation"]
    use_spherical = representation == "SC"
    atlas_config = build_atlas_config(ablation)
    use_atlas_archive = atlas_config.enabled and bool(ablation["useRepository"])

    n_var = int(model["n"])
    alpha_vel = 0.5
    if use_spherical:
        path_diag = float(np.linalg.norm(np.asarray(model["start"], dtype=float).reshape(-1) - np.asarray(model["end"], dtype=float).reshape(-1)))
        var_max = {
            "r": np.full(n_var, 3.0 * path_diag / n_var, dtype=float),
            "psi": np.full(n_var, np.pi / 4.0, dtype=float),
            "phi": np.full(n_var, np.pi / 4.0, dtype=float),
        }
        var_min = {
            "r": np.full(n_var, (3.0 * path_diag / n_var) / 9.0, dtype=float),
            "psi": -var_max["psi"],
            "phi": -var_max["phi"],
        }
        vel_max = {
            "r": alpha_vel * (var_max["r"] - var_min["r"]),
            "psi": alpha_vel * (var_max["psi"] - var_min["psi"]),
            "phi": alpha_vel * (var_max["phi"] - var_min["phi"]),
        }
        vel_min = {key: -value for key, value in vel_max.items()}
    else:
        var_min = {
            "x": np.full(n_var, float(model["xmin"]), dtype=float),
            "y": np.full(n_var, float(model["ymin"]), dtype=float),
            "z": np.full(n_var, float(model["zmin"]), dtype=float),
        }
        var_max = {
            "x": np.full(n_var, float(model["xmax"]), dtype=float),
            "y": np.full(n_var, float(model["ymax"]), dtype=float),
            "z": np.full(n_var, float(model["zmax"]), dtype=float),
        }
        vel_max = {
            "x": alpha_vel * (var_max["x"] - var_min["x"]),
            "y": alpha_vel * (var_max["y"] - var_min["y"]),
            "z": alpha_vel * (var_max["z"] - var_min["z"]),
        }
        vel_min = {key: -value for key, value in vel_max.items()}

    reference_points = np.zeros((0, objective_count), dtype=float)
    if bool(ablation["useReferenceLeader"]):
        reference_points = _build_reference_points(params.population, objective_count, bool(ablation["useTwoLayerRef"]))
    init_max_tries = 10

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    for run_index in range(1, params.runs + 1):
        run_start = time.perf_counter()
        particles: list[NMOPSOParticle] = []
        for _ in range(init_max_tries):
            particles = [
                _initialize_particle(model, representation, var_min, var_max)
                for _ in range(params.population)
            ]
            init_costs = np.array([particle.cost for particle in particles], dtype=float)
            if np.any(np.all(np.isfinite(init_costs), axis=1)):
                break
        costs = np.array([particle.cost for particle in particles], dtype=float)
        dominated = determine_domination(costs)
        repository = [_clone_particle(particle) for particle, is_dominated in zip(particles, dominated) if not is_dominated]
        if not repository:
            repository = [_clone_particle(particle) for particle in particles]

        inertia = float(ablation["w"])
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            repository_costs = np.array([entry.cost for entry in repository], dtype=float) if repository else np.zeros((0, objective_count), dtype=float)
            if bool(ablation["useReferenceLeader"]) and repository and reference_points.shape[0] > 0:
                leader_indices = select_leader_ref(repository_costs, reference_points, params.population)
            else:
                leader_indices = np.random.randint(0, max(1, len(repository)), size=params.population)

            grid_lb = grid_ub = np.zeros((0, 0), dtype=float)
            repository_grid_index = np.zeros(len(repository), dtype=int)
            repository_atlas_index = np.zeros(len(repository), dtype=int) if use_atlas_archive else None
            if bool(ablation["useRepository"]) and repository:
                repository_costs = np.array([entry.cost for entry in repository], dtype=float)
                if bool(ablation["useGrid"]):
                    grid_lb, grid_ub = create_grid(repository_costs, int(ablation["nGrid"]), float(ablation["alpha_grid"]))
                    for idx, entry in enumerate(repository):
                        entry.grid_index, entry.grid_sub_index = find_grid_index(entry.cost, grid_lb, grid_ub)
                        repository_grid_index[idx] = entry.grid_index
                else:
                    repository_grid_index = np.arange(len(repository), dtype=int) + 1
                if use_atlas_archive:
                    for idx, entry in enumerate(repository):
                        cart = position_to_cart(entry.position, model, representation)
                        path_xyz = cart_to_absolute_path(cart, model)
                        entry.topology_signature = topology_signature(path_xyz, model, atlas_config.max_obstacles)
                        entry.topology_bin = topology_bin_from_signature(entry.topology_signature, atlas_config)
                        entry.robustness_score, entry.robustness_bin = robustness_from_cost(entry.cost, atlas_config.n_robust_bins)
                        entry.atlas_cell_index = (entry.topology_bin - 1) * atlas_config.n_robust_bins + entry.robustness_bin
                        repository_atlas_index[idx] = entry.atlas_cell_index

            for particle_index, particle in enumerate(particles):
                if bool(ablation["useRepository"]) and repository:
                    if bool(ablation["useReferenceLeader"]) and leader_indices.size == params.population:
                        leader = repository[int(leader_indices[particle_index])]
                    elif use_atlas_archive and repository_atlas_index is not None and repository_grid_index.size > 0:
                        leader_idx = select_leader_with_weights(
                            repository_grid_index,
                            float(ablation["beta"]),
                            atlas_config.objective_weight,
                            atlas_config.atlas_weight,
                            repository_atlas_index,
                        )
                        leader = repository[int(leader_idx)]
                    elif bool(ablation["useGrid"]) and repository_grid_index.size > 0:
                        leader = repository[
                            _select_member_from_grid_cells(
                                repository_grid_index,
                                float(ablation["beta"]),
                                invert=True,
                            )
                        ]
                    else:
                        leader = repository[int(np.random.randint(0, len(repository)))]
                else:
                    leader = particle

                _update_particle_velocity_and_position(
                    particle=particle,
                    leader=leader,
                    var_min=var_min,
                    var_max=var_max,
                    vel_min=vel_min,
                    vel_max=vel_max,
                    c1=float(ablation["c1"]),
                    c2=float(ablation["c2"]),
                    inertia=inertia,
                )

                cart_particle = position_to_cart(particle.position, model, representation)
                particle.cost = _nmopso_cost(cart_particle, model)

                mutation_prob = float(ablation["mutationProb"])
                if bool(ablation["useMutation"]):
                    if bool(ablation["useAdaptiveMutation"]):
                        mutation_prob = (1.0 - (generation - 1) / max(1, params.generations - 1)) ** (1.0 / float(ablation["mu"]))
                else:
                    mutation_prob = 0.0
                if np.random.rand() < mutation_prob:
                    region_count = archive_region_count(
                        repository_grid_index if repository_grid_index.size > 0 else np.arange(len(repository), dtype=int) + 1,
                        repository_atlas_index,
                    )
                    mutated = mutate(
                        particle.position,
                        particle.best_position,
                        float(ablation["delta"]),
                        var_min,
                        var_max,
                        representation,
                        region_count,
                    )
                    cart_mutated = position_to_cart(mutated, model, representation)
                    mutated_cost = _nmopso_cost(cart_mutated, model)
                    if dominates(mutated_cost, particle.cost) or (
                        not dominates(particle.cost, mutated_cost) and np.random.rand() < 0.5
                    ):
                        particle.position = {key: value.copy() for key, value in mutated.items()}
                        particle.cost = mutated_cost

                if dominates(particle.cost, particle.best_cost) or (
                    not dominates(particle.best_cost, particle.cost) and np.random.rand() < 0.5
                ):
                    particle.best_position = {key: value.copy() for key, value in particle.position.items()}
                    particle.best_cost = particle.cost.copy()

            if bool(ablation["useRepository"]):
                merged = repository + particles
                merged_costs = np.array([entry.cost for entry in merged], dtype=float)
                merged_dom = determine_domination(merged_costs)
                repository = [_clone_particle(entry) for entry, is_dom in zip(merged, merged_dom) if not is_dom]
                if len(repository) > int(ablation["nRep"]):
                    if bool(ablation["useGrid"]):
                        repo_costs = np.array([entry.cost for entry in repository], dtype=float)
                        grid_lb, grid_ub = create_grid(repo_costs, int(ablation["nGrid"]), float(ablation["alpha_grid"]))
                        repo_grid = np.zeros(len(repository), dtype=int)
                        for idx, entry in enumerate(repository):
                            entry.grid_index, entry.grid_sub_index = find_grid_index(entry.cost, grid_lb, grid_ub)
                            repo_grid[idx] = entry.grid_index
                        repo_atlas = np.zeros(len(repository), dtype=int) if use_atlas_archive else None
                        if use_atlas_archive:
                            for idx, entry in enumerate(repository):
                                cart = position_to_cart(entry.position, model, representation)
                                sig = topology_signature(cart_to_absolute_path(cart, model), model, atlas_config.max_obstacles)
                                topo_bin = topology_bin_from_signature(sig, atlas_config)
                                _, robust_bin = robustness_from_cost(entry.cost, atlas_config.n_robust_bins)
                                entry.atlas_cell_index = (topo_bin - 1) * atlas_config.n_robust_bins + robust_bin
                                repo_atlas[idx] = entry.atlas_cell_index
                        while len(repository) > int(ablation["nRep"]):
                            if use_atlas_archive and repo_atlas is not None:
                                delete_index = delete_one_with_weights(
                                    repo_grid,
                                    float(ablation["gamma"]),
                                    atlas_config.objective_weight,
                                    atlas_config.atlas_weight,
                                    repo_atlas,
                                )
                            else:
                                delete_index = _select_member_from_grid_cells(
                                    repo_grid,
                                    float(ablation["gamma"]),
                                    invert=False,
                                )
                            repository.pop(int(delete_index))
                            repo_grid = np.delete(repo_grid, int(delete_index))
                            if repo_atlas is not None:
                                repo_atlas = np.delete(repo_atlas, int(delete_index))
                    else:
                        repository = list(np.random.choice(repository, size=int(ablation["nRep"]), replace=False))

            report_costs = np.array([entry.cost for entry in (repository if repository else particles)], dtype=float)
            if params.compute_metrics:
                if generation == 1 or generation == params.generations or generation % int(ablation["metricInterval"]) == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, report_costs, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, report_costs, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]
            inertia *= float(ablation["wdamp"])

        run_dir = results_path / f"Run_{run_index}"
        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        final_members = repository if repository else particles
        final_costs = np.array([entry.cost for entry in final_members], dtype=float)
        save_run_popobj(run_dir / "final_popobj.mat", final_costs, params.problem_index, objective_count)
        saved_paths: list[np.ndarray] = []
        for member_index, member in enumerate(final_members, start=1):
            cart = position_to_cart(member.position, model, representation)
            path_xyz = cart_to_absolute_path(cart, model)
            saved_paths.append(path_xyz)
            save_bp(run_dir / f"bp_{member_index}.mat", path_xyz, member.cost)
        if saved_paths:
            finite_cost = np.where(np.isfinite(final_costs), final_costs, 1e9)
            best_idx = int(np.argmin(np.sum(finite_cost, axis=1)))
            save_mat(run_dir / "fleet_paths.mat", {"uav1": np.asarray(saved_paths[best_idx], dtype=float)})
        mission_stats, feasible_mask = build_mission_stats(saved_paths, model)
        save_mat(run_dir / "mission_stats.mat", mission_stats)
        feasible_count = int(np.sum(feasible_mask))
        save_mat(
            run_dir / "run_stats.mat",
            {
                "runtimeSec": float(time.perf_counter() - run_start),
                "feasibleCount": feasible_count,
                "solutionCount": int(final_costs.shape[0]),
            },
        )

        if params.compute_metrics:
            run_scores[run_index - 1] = np.array(
                [
                    cal_metric(1, final_costs, params.problem_index, objective_count),
                    cal_metric(2, final_costs, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores
