from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np

from uav_benchmark.exceptions import ModelValidationError

TerrainModel: TypeAlias = dict[str, Any]
ReadonlyTerrainModel: TypeAlias = Mapping[str, Any]

_REQUIRED_BOUNDS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _label(context: str, key: str) -> str:
    return f"{context}.{key}" if context else key


def _as_float(model: ReadonlyTerrainModel, key: str, context: str) -> float:
    if key not in model or model[key] is None:
        raise ModelValidationError(f"{_label(context, key)} is required")
    try:
        return float(np.asarray(model[key]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ModelValidationError(f"{_label(context, key)} must be numeric") from exc


def _as_point_matrix(raw: Any, key: str, context: str) -> np.ndarray:
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{_label(context, key)} must be numeric") from exc
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] < 3 or matrix.shape[0] < 1:
        raise ModelValidationError(f"{_label(context, key)} must have shape (N, >=3)")
    if not np.all(np.isfinite(matrix[:, :3])):
        raise ModelValidationError(f"{_label(context, key)} must contain finite coordinates")
    return matrix


def _validate_height_map(model: ReadonlyTerrainModel, context: str) -> None:
    if "H" not in model or model["H"] is None:
        raise ModelValidationError(f"{_label(context, 'H')} is required")
    try:
        height_map = np.asarray(model["H"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{_label(context, 'H')} must be numeric") from exc
    if height_map.ndim != 2 or height_map.size == 0:
        raise ModelValidationError(f"{_label(context, 'H')} must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(height_map)):
        raise ModelValidationError(f"{_label(context, 'H')} must contain finite values")


def _validate_bounds(model: ReadonlyTerrainModel, context: str) -> None:
    values = {key: _as_float(model, key, context) for key in _REQUIRED_BOUNDS}
    if values["xmin"] >= values["xmax"]:
        raise ModelValidationError(f"{_label(context, 'xmin')} must be < {_label(context, 'xmax')}")
    if values["ymin"] >= values["ymax"]:
        raise ModelValidationError(f"{_label(context, 'ymin')} must be < {_label(context, 'ymax')}")
    if values["zmin"] >= values["zmax"]:
        raise ModelValidationError(f"{_label(context, 'zmin')} must be < {_label(context, 'zmax')}")


def _validate_endpoints(model: ReadonlyTerrainModel, context: str) -> None:
    starts = model.get("starts")
    goals = model.get("goals")
    if starts is not None or goals is not None:
        if starts is None or goals is None:
            raise ModelValidationError(f"{context} must define both starts and goals")
        starts_matrix = _as_point_matrix(starts, "starts", context)
        goals_matrix = _as_point_matrix(goals, "goals", context)
        if starts_matrix.shape[0] != goals_matrix.shape[0]:
            raise ModelValidationError(f"{context}.starts and {context}.goals must have the same row count")
        if "fleetSize" in model and model["fleetSize"] is not None:
            fleet_size = int(_as_float(model, "fleetSize", context))
            if fleet_size != starts_matrix.shape[0]:
                raise ModelValidationError(f"{_label(context, 'fleetSize')} must match starts/goals rows")
        return
    if "start" not in model or "end" not in model:
        raise ModelValidationError(f"{context} must define start/end or starts/goals")
    _as_point_matrix(model["start"], "start", context)
    _as_point_matrix(model["end"], "end", context)


def validate_terrain_model(model: ReadonlyTerrainModel, *, context: str = "terrain") -> TerrainModel:
    """Validate the terrain/mission model contract and return a mutable copy."""
    if not isinstance(model, Mapping):
        raise ModelValidationError(f"{context} must be a mapping")
    _validate_height_map(model, context)
    _validate_bounds(model, context)
    _validate_endpoints(model, context)

    if "n" in model and model["n"] is not None and int(_as_float(model, "n", context)) < 1:
        raise ModelValidationError(f"{_label(context, 'n')} must be >= 1")
    if "safeDist" in model and model["safeDist"] is not None and _as_float(model, "safeDist", context) < 0:
        raise ModelValidationError(f"{_label(context, 'safeDist')} must be >= 0")
    if "droneSize" in model and model["droneSize"] is not None and _as_float(model, "droneSize", context) < 0:
        raise ModelValidationError(f"{_label(context, 'droneSize')} must be >= 0")
    if (
        "separationMin" in model
        and model["separationMin"] is not None
        and _as_float(model, "separationMin", context) < 0
    ):
        raise ModelValidationError(f"{_label(context, 'separationMin')} must be >= 0")
    if "maxTurnDeg" in model and model["maxTurnDeg"] is not None and _as_float(model, "maxTurnDeg", context) < 0:
        raise ModelValidationError(f"{_label(context, 'maxTurnDeg')} must be >= 0")

    return dict(model)
