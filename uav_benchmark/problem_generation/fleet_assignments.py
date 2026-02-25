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
    for point in points:
        if np.linalg.norm(point[:2] - candidate[:2]) < separation_min:
            return False
    return True


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

    while len(starts) < fleet_size and max_tries > 0:
        max_tries -= 1
        sx, sy = _sample_xy(rng, xmin, max(xmin + 1.0, xmin + 0.35 * (xmax - xmin)), ymin, ymax)
        gx, gy = _sample_xy(rng, min(xmax - 1.0, xmin + 0.65 * (xmax - xmin)), xmax, ymin, ymax)
        sz = float(rng.uniform(max(zmin, start[2]), min(zmax, max(start[2], zmax))))
        gz = float(rng.uniform(max(zmin, end[2]), min(zmax, max(end[2], zmax))))
        s = np.array([sx, sy, sz], dtype=float)
        g = np.array([gx, gy, gz], dtype=float)
        if not _enforce_min_distance(starts, s, separation_min):
            continue
        if not _enforce_min_distance(goals, g, separation_min):
            continue
        starts.append(s)
        goals.append(g)

    # If strict sampling failed due crowded space, pad with jittered offsets.
    jitter_scale = max(1.0, separation_min * 0.5)
    while len(starts) < fleet_size:
        ref_s = starts[-1] if starts else start
        ref_g = goals[-1] if goals else end
        s = ref_s.copy()
        g = ref_g.copy()
        s[:2] += rng.normal(0.0, jitter_scale, size=2)
        g[:2] += rng.normal(0.0, jitter_scale, size=2)
        s[0] = float(np.clip(s[0], xmin, xmax))
        s[1] = float(np.clip(s[1], ymin, ymax))
        g[0] = float(np.clip(g[0], xmin, xmax))
        g[1] = float(np.clip(g[1], ymin, ymax))
        s[2] = float(np.clip(s[2], zmin, zmax))
        g[2] = float(np.clip(g[2], zmin, zmax))
        starts.append(s)
        goals.append(g)

    mission_id = f"{mission_prefix}_k{fleet_size}_s{seed}"
    return FleetAssignment(
        starts=np.asarray(starts[:fleet_size], dtype=float),
        goals=np.asarray(goals[:fleet_size], dtype=float),
        fleet_size=fleet_size,
        mission_id=mission_id,
    )

