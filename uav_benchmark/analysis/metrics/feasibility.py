from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.io.matlab import load_mat


def normalize_pop_obj(pop_obj: np.ndarray, objective_count: int | None = 4) -> np.ndarray:
    pop = np.asarray(pop_obj, dtype=float)
    if pop.ndim == 1:
        pop = pop.reshape(1, -1)
    if (
        objective_count is not None
        and pop.ndim == 2
        and pop.shape[1] != objective_count
        and pop.shape[0] == objective_count
    ):
        pop = pop.T
    return pop


def full_length_bool_mask(values: Any, size: int, threshold: float = 0.5) -> np.ndarray | None:
    mask = np.asarray(values, dtype=float).reshape(-1)
    if size <= 0 or mask.size != size:
        return None
    return mask > threshold


def finite_rows_mask(pop_obj: np.ndarray) -> np.ndarray:
    pop = np.asarray(pop_obj, dtype=float)
    if pop.ndim != 2 or pop.shape[0] <= 0:
        return np.zeros(0, dtype=bool)
    return np.all(np.isfinite(pop), axis=1)


def load_mission_feasible_mask(
    run_dir: Path,
    count: int,
    base_mask: np.ndarray | None = None,
    violation_keys: tuple[str, ...] = ("turnViolation", "separationViolation", "collisionViolation"),
) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=bool)

    if base_mask is None:
        mask = np.ones(count, dtype=bool)
    else:
        mask = np.asarray(base_mask, dtype=bool).reshape(-1)
        mask = np.ones(count, dtype=bool) if mask.size != count else mask.copy()

    fallback = mask.copy()
    mission_file = run_dir / "mission_stats.mat"
    if not mission_file.exists():
        return fallback

    try:
        payload = load_mat(mission_file)
    except (OSError, RuntimeError, ValueError):
        return fallback

    mission_mask_applied = False
    if "feasible" in payload:
        feasible = full_length_bool_mask(payload["feasible"], count)
        if feasible is not None:
            mask &= feasible
            mission_mask_applied = True

    for violation_key in violation_keys:
        if violation_key not in payload:
            continue
        violation = full_length_bool_mask(payload[violation_key], count)
        if violation is None:
            continue
        mask &= ~violation
        mission_mask_applied = True

    return mask if mission_mask_applied else fallback
