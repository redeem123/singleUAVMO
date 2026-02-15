from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uav_benchmark.io.matlab import load_terrain_struct, save_mat
from uav_benchmark.problem_generation.multi_uav_assignments import sample_homogeneous_assignments


@dataclass(slots=True)
class TerrainSpec:
    terrain_size: int = 200
    seed: int | None = None


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mesh(terrain_size: int) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.arange(1, terrain_size + 1), np.arange(1, terrain_size + 1))


def _save(path: Path, terrain_struct: dict) -> None:
    save_mat(path, {"terrainStruct": terrain_struct})


def generate_city(spec: TerrainSpec, with_nofly: bool = False) -> dict:
    rng = _rng(spec.seed)
    terrain_size = spec.terrain_size
    num_buildings = 100 if with_nofly else 120
    max_height = 305.0
    x_grid, y_grid = _mesh(terrain_size)
    base = (
        np.sin(0.1 * x_grid) * np.sin(0.1 * y_grid)
        + 0.5 * np.sin(0.2 * x_grid) * np.sin(0.2 * y_grid)
        + 0.25 * np.sin(0.3 * x_grid) * np.sin(0.3 * y_grid)
        + 0.125 * np.sin(0.4 * x_grid) * np.sin(0.4 * y_grid)
    )
    base = (base - np.min(base)) / (np.max(base) - np.min(base))
    base = base * max_height / 10.0

    nofly_center = np.array([terrain_size / 2, terrain_size / 2], dtype=float)
    nofly_radius = 20.0
    nofly_height = 400.0
    nofly_mask = np.zeros_like(base, dtype=bool)
    if with_nofly:
        distance = np.sqrt((x_grid - nofly_center[0]) ** 2 + (y_grid - nofly_center[1]) ** 2)
        nofly_mask = distance <= nofly_radius
        base[nofly_mask] = nofly_height

    min_size = 3
    max_size = 15
    for _ in range(num_buildings):
        width = int(rng.integers(min_size, max_size + 1))
        depth = int(rng.integers(min_size, max_size + 1))
        height = float(rng.integers(1, int(max_height) + 1))
        while True:
            sx = int(rng.integers(0, terrain_size - width + 1))
            sy = int(rng.integers(0, terrain_size - depth + 1))
            if not with_nofly or not np.any(nofly_mask[sy : sy + depth, sx : sx + width]):
                break
        avg_height = float(np.mean(base[sy : sy + depth, sx : sx + width]))
        base[sy : sy + depth, sx : sx + width] = avg_height + height

    terrain = {
        "start": np.array([1, 1, 1], dtype=float),
        "end": np.array([200, 200, 100], dtype=float),
        "n": 20.0,
        "xmin": 1.0,
        "xmax": float(terrain_size),
        "ymin": 1.0,
        "ymax": 200.0,
        "zmin": 1.0,
        "zmax": 120.0,
        "X": x_grid[0],
        "Y": y_grid[:, 0],
        "H": base,
        "safeH": 90.0,
        "theta": 30.0,
    }
    if with_nofly:
        terrain["nofly_c"] = nofly_center.reshape(1, 2)
        terrain["nofly_r"] = np.array([nofly_radius], dtype=float)
        terrain["nofly_h"] = np.array([nofly_height], dtype=float)
    return terrain


def generate_suburban(spec: TerrainSpec, with_nofly: bool = False) -> dict:
    rng = _rng(spec.seed)
    terrain_size = spec.terrain_size
    num_buildings = 110 if with_nofly else 120
    max_height = 20.0
    x_grid, y_grid = _mesh(terrain_size)
    base = 10.0 * (np.sin(0.005 * x_grid) + np.sin(0.005 * y_grid) + 0.5 * np.sin(0.03 * x_grid) * np.sin(0.03 * y_grid))
    base = base - np.min(base)
    base = base / np.max(base)
    base = base * max_height * 2

    nofly_center = np.array([terrain_size / 2, terrain_size / 2], dtype=float)
    nofly_radius = 20.0
    nofly_height = 60.0
    nofly_mask = np.zeros_like(base, dtype=bool)
    if with_nofly:
        distance = np.sqrt((x_grid - nofly_center[0]) ** 2 + (y_grid - nofly_center[1]) ** 2)
        nofly_mask = distance <= nofly_radius
        base[nofly_mask] = nofly_height

    width = 10
    depth = 10
    for _ in range(num_buildings):
        building_height = float(rng.integers(5, int(max_height) + 1))
        while True:
            sx = int(rng.integers(0, terrain_size - width + 1))
            sy = int(rng.integers(0, terrain_size - depth + 1))
            if not with_nofly or not np.any(nofly_mask[sy : sy + depth, sx : sx + width]):
                break
        avg_height = float(np.mean(base[sy : sy + depth, sx : sx + width]))
        base[sy : sy + depth, sx : sx + width] = avg_height + building_height

    terrain = {
        "start": np.array([1, 1, 1], dtype=float),
        "end": np.array([200, 200, 40], dtype=float),
        "n": 10.0 if with_nofly else 7.0,
        "xmin": 1.0,
        "xmax": float(terrain_size),
        "ymin": 1.0,
        "ymax": 200.0,
        "zmin": 0.0,
        "zmax": 50.0,
        "X": x_grid[0],
        "Y": y_grid[:, 0],
        "H": base,
        "safeH": 5.0,
        "theta": 30.0,
    }
    if with_nofly:
        terrain["nofly_c"] = nofly_center.reshape(1, 2)
        terrain["nofly_r"] = np.array([nofly_radius], dtype=float)
        terrain["nofly_h"] = np.array([nofly_height], dtype=float)
    return terrain


def generate_mountain(spec: TerrainSpec, with_nofly: bool = False) -> dict:
    rng = _rng(spec.seed)
    terrain_size = spec.terrain_size
    num_peaks = 200
    peak_height = 400.0 if with_nofly else 300.0
    valley_depth = 0.0
    x_grid, y_grid = _mesh(terrain_size)
    base = 8.0 * (
        np.sin(0.03 * x_grid) * np.sin(0.03 * y_grid)
        + 0.5 * np.sin(0.07 * x_grid) * np.sin(0.07 * y_grid)
        + 0.25 * np.sin(0.1 * x_grid) * np.sin(0.1 * y_grid)
        + 0.1 * rng.random((terrain_size, terrain_size))
    )
    for _ in range(num_peaks):
        peak_x = int(rng.integers(20 if with_nofly else 17, terrain_size - (20 if with_nofly else 17)))
        peak_y = int(rng.integers(20 if with_nofly else 17, terrain_size - (20 if with_nofly else 17)))
        radius = int(rng.integers(15, 31))
        x_start = max(0, peak_x - radius)
        x_end = min(terrain_size - 1, peak_x + radius)
        y_start = max(0, peak_y - radius)
        y_end = min(terrain_size - 1, peak_y + radius)
        for x_pos in range(x_start, x_end + 1):
            for y_pos in range(y_start, y_end + 1):
                dist = np.sqrt((x_pos - peak_x) ** 2 + (y_pos - peak_y) ** 2)
                if dist <= radius:
                    base[y_pos, x_pos] += peak_height * np.exp(-(dist**2) / (2 * (radius / 2.0) ** 2))
    base = (base - np.min(base)) / (np.max(base) - np.min(base))
    base = base * (peak_height - valley_depth) + valley_depth

    terrain = {
        "start": np.array([1, 1, 1], dtype=float),
        "end": np.array([200, 200, 1], dtype=float),
        "n": 7.0,
        "xmin": 1.0,
        "xmax": float(terrain_size),
        "ymin": 1.0,
        "ymax": float(terrain_size),
        "zmin": valley_depth,
        "zmax": peak_height,
        "X": x_grid[0],
        "Y": y_grid[:, 0],
        "H": base,
        "safeH": 10.0,
        "theta": 30.0,
    }
    if with_nofly:
        center = np.array([terrain_size / 2 + terrain_size / 3.5, terrain_size / 2 + terrain_size / 3.5], dtype=float)
        radius = 25.0
        height = peak_height + 10.0
        distance = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)
        base[distance <= radius] = height
        terrain["H"] = base
        terrain["nofly_c"] = center.reshape(1, 2)
        terrain["nofly_r"] = np.array([radius], dtype=float)
        terrain["nofly_h"] = np.array([height], dtype=float)
    return terrain


def save_city(path: Path, with_nofly: bool = False, seed: int | None = None) -> None:
    _save(path, generate_city(TerrainSpec(seed=seed), with_nofly=with_nofly))


def save_suburban(path: Path, with_nofly: bool = False, seed: int | None = None) -> None:
    _save(path, generate_suburban(TerrainSpec(seed=seed), with_nofly=with_nofly))


def save_mountain(path: Path, with_nofly: bool = False, seed: int | None = None) -> None:
    _save(path, generate_mountain(TerrainSpec(seed=seed), with_nofly=with_nofly))


def make_multi_uav_terrain(
    terrain: dict,
    fleet_size: int,
    seed: int,
    separation_min: float,
    mission_prefix: str = "paper_medium",
) -> dict:
    assignment = sample_homogeneous_assignments(
        terrain=terrain,
        fleet_size=fleet_size,
        seed=seed,
        separation_min=separation_min,
        mission_prefix=mission_prefix,
    )
    output = dict(terrain)
    output["starts"] = assignment.starts
    output["goals"] = assignment.goals
    output["fleetSize"] = float(assignment.fleet_size)
    output["separationMin"] = float(separation_min)
    output["missionId"] = assignment.mission_id
    return output


def save_multi_uav_scenarios(
    project_root: Path,
    base_problem_names: list[str],
    fleet_sizes: tuple[int, ...],
    seed: int,
    separation_min: float,
    mission_prefix: str = "paper_medium",
) -> list[Path]:
    project_root = project_root.resolve()
    output_paths: list[Path] = []
    problems_dir = project_root / "problems"
    for base_name in base_problem_names:
        terrain_file = problems_dir / f"terrainStruct_{base_name}.mat"
        if not terrain_file.exists():
            continue
        terrain = load_terrain_struct(terrain_file)
        for fleet_size in fleet_sizes:
            if fleet_size <= 1:
                continue
            multi = make_multi_uav_terrain(
                terrain=terrain,
                fleet_size=fleet_size,
                seed=seed + int(fleet_size),
                separation_min=separation_min,
                mission_prefix=mission_prefix,
            )
            out_file = problems_dir / f"terrainStruct_{base_name}_uav{fleet_size}.mat"
            save_mat(out_file, {"terrainStruct": multi})
            output_paths.append(out_file)
    return output_paths
