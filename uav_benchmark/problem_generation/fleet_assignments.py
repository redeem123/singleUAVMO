from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class FleetAssignment:
    starts: np.ndarray
    goals: np.ndarray
    fleet_size: int
    mission_id: str


def _sample_xy(rng: np.random.Generator, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[float, float]:
    return float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax))


def _enforce_min_distance(points: list[np.ndarray], candidate: np.ndarray, separation_min: float) -> bool:
    if not points:
        return True
    return all(np.linalg.norm(point[:2] - candidate[:2]) >= separation_min for point in points)


def _sample_endpoint(
    rng: np.random.Generator,
    x_low: float,
    x_high: float,
    ymin: float,
    ymax: float,
    z_low: float,
    z_high: float,
) -> np.ndarray:
    x_coord, y_coord = _sample_xy(rng, x_low, x_high, ymin, ymax)
    z_coord = float(rng.uniform(z_low, z_high))
    return np.array([x_coord, y_coord, z_coord], dtype=float)


def _complete_point_set(
    rng: np.random.Generator,
    points: list[np.ndarray],
    target_count: int,
    separation_min: float,
    x_low: float,
    x_high: float,
    ymin: float,
    ymax: float,
    z_low: float,
    z_high: float,
) -> list[np.ndarray]:
    if len(points) >= target_count:
        return points[:target_count]
    if separation_min <= 0:
        completed = list(points)
        while len(completed) < target_count:
            completed.append(_sample_endpoint(rng, x_low, x_high, ymin, ymax, z_low, z_high))
        return completed

    candidate_count = max(2048, target_count * 1024)
    candidates = np.vstack(
        [_sample_endpoint(rng, x_low, x_high, ymin, ymax, z_low, z_high) for _ in range(candidate_count)]
    )
    chosen = [np.asarray(point, dtype=float).copy() for point in points]
    while len(chosen) < target_count:
        chosen_xy = np.asarray(chosen, dtype=float)[:, :2]
        diff = candidates[:, np.newaxis, :2] - chosen_xy[np.newaxis, :, :]
        min_dist = np.min(np.linalg.norm(diff, axis=2), axis=1)
        best_index = int(np.argmax(min_dist))
        if float(min_dist[best_index]) + 1e-9 < separation_min:
            break
        chosen.append(candidates[best_index].copy())
        keep_mask = min_dist + 1e-9 >= separation_min
        keep_mask[best_index] = False
        candidates = candidates[keep_mask]
        if candidates.size == 0 and len(chosen) < target_count:
            break

    if len(chosen) < target_count:
        raise ValueError(
            "Unable to generate fleet endpoints satisfying separation_min="
            f"{float(separation_min):.3f} for fleet_size={int(target_count)} "
            f"within x=[{x_low:.3f}, {x_high:.3f}], y=[{ymin:.3f}, {ymax:.3f}]."
        )
    return chosen[:target_count]


def sample_homogeneous_assignments(
    terrain: dict[str, Any],
    fleet_size: int,
    seed: int,
    separation_min: float,
    mission_prefix: str = "mission",
) -> FleetAssignment:
    rng = np.random.default_rng(seed)
    start = np.asarray(terrain["start"], dtype=float).reshape(-1)[:3]
    end = np.asarray(terrain["end"], dtype=float).reshape(-1)[:3]
    xmin = float(terrain["xmin"])
    xmax = float(terrain["xmax"])
    ymin = float(terrain["ymin"])
    ymax = float(terrain["ymax"])
    zmin = float(terrain["zmin"])
    zmax = float(terrain["zmax"])

    starts: list[np.ndarray] = [start.copy()]
    goals: list[np.ndarray] = [end.copy()]
    max_tries = 2000
    start_x_high = max(xmin + 1.0, xmin + 0.35 * (xmax - xmin))
    goal_x_low = min(xmax - 1.0, xmin + 0.65 * (xmax - xmin))
    start_z_low = max(zmin, start[2])
    goal_z_low = max(zmin, end[2])

    while len(starts) < fleet_size and max_tries > 0:
        max_tries -= 1
        s = _sample_endpoint(rng, xmin, start_x_high, ymin, ymax, start_z_low, zmax)
        g = _sample_endpoint(rng, goal_x_low, xmax, ymin, ymax, goal_z_low, zmax)
        if not _enforce_min_distance(starts, s, separation_min):
            continue
        if not _enforce_min_distance(goals, g, separation_min):
            continue
        starts.append(s)
        goals.append(g)

    starts = _complete_point_set(
        rng=rng,
        points=starts,
        target_count=fleet_size,
        separation_min=separation_min,
        x_low=xmin,
        x_high=start_x_high,
        ymin=ymin,
        ymax=ymax,
        z_low=start_z_low,
        z_high=zmax,
    )
    goals = _complete_point_set(
        rng=rng,
        points=goals,
        target_count=fleet_size,
        separation_min=separation_min,
        x_low=goal_x_low,
        x_high=xmax,
        ymin=ymin,
        ymax=ymax,
        z_low=goal_z_low,
        z_high=zmax,
    )

    mission_id = f"{mission_prefix}_k{fleet_size}_s{seed}"
    return FleetAssignment(
        starts=np.asarray(starts[:fleet_size], dtype=float),
        goals=np.asarray(goals[:fleet_size], dtype=float),
        fleet_size=fleet_size,
        mission_id=mission_id,
    )
