from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MatlabDependencyError(RuntimeError):
    """Raised when MATLAB IO dependency is not available."""


def _require_scipy():
    try:
        from scipy.io import loadmat, savemat  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency gate
        raise MatlabDependencyError(
            "scipy is required for .mat I/O. Install requirements-python.txt first."
        ) from exc
    return loadmat, savemat


def _to_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _to_scalar(value.reshape(-1)[0])
        return value
    return value.item() if hasattr(value, "item") else value


def _mat_struct_to_dict(value: Any) -> Any:
    if hasattr(value, "_fieldnames") and getattr(value, "_fieldnames", None):
        result = {}
        for field_name in value._fieldnames:
            result[field_name] = _mat_struct_to_dict(getattr(value, field_name))
        return result
    if isinstance(value, np.ndarray) and value.dtype.names:
        if value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return [_mat_struct_to_dict(item) for item in value.reshape(-1)]
    if hasattr(value, "dtype") and getattr(value, "dtype", None) is not None and value.dtype.names:
        result = {}
        for field_name in value.dtype.names:
            result[field_name] = _mat_struct_to_dict(value[field_name])
        return result
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.size == 1:
                return _mat_struct_to_dict(value.reshape(-1)[0])
            return np.array([_mat_struct_to_dict(item) for item in value.reshape(-1)], dtype=object)
        return value
    return _to_scalar(value)


def load_mat(path: str | Path) -> dict[str, Any]:
    loadmat, _ = _require_scipy()
    data = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("__"):
            continue
        clean[key] = _mat_struct_to_dict(value)
    return clean


def save_mat(path: str | Path, payload: dict[str, Any]) -> None:
    _, savemat = _require_scipy()
    savemat(str(path), payload)


def ensure_column_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    return array


def load_terrain_struct(path: str | Path) -> dict[str, Any]:
    data = load_mat(path)
    if "terrainStruct" not in data:
        raise KeyError(f"{path} does not contain terrainStruct")
    terrain = data["terrainStruct"]
    if not isinstance(terrain, dict):
        terrain = _mat_struct_to_dict(terrain)
    normalized = dict(terrain)
    for key in ("start", "end", "X", "Y", "nofly_r", "nofly_h"):
        if key in normalized:
            normalized[key] = ensure_column_vector(np.asarray(normalized[key], dtype=float))
    for key in ("starts", "goals"):
        if key in normalized and normalized[key] is not None:
            matrix = np.asarray(normalized[key], dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            normalized[key] = matrix
    if "nofly_c" in normalized:
        normalized["nofly_c"] = np.asarray(normalized["nofly_c"], dtype=float)
        if normalized["nofly_c"].ndim == 1:
            normalized["nofly_c"] = normalized["nofly_c"].reshape(1, -1)
    if "threats" in normalized and normalized["threats"] is not None:
        normalized["threats"] = np.asarray(normalized["threats"], dtype=float)
    if "H" in normalized:
        normalized["H"] = np.asarray(normalized["H"], dtype=float)
    for key in (
        "n",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "safeH",
        "safeDist",
        "droneSize",
        "fleetSize",
        "separationMin",
        "maxTurnDeg",
    ):
        if key in normalized and normalized[key] is not None:
            normalized[key] = float(np.asarray(normalized[key]).reshape(-1)[0])
    # Backward compatibility: promote start/end to starts/goals for K=1 missions.
    if "starts" not in normalized and "start" in normalized:
        start = np.asarray(normalized["start"], dtype=float).reshape(-1)
        normalized["starts"] = start.reshape(1, -1)
    if "goals" not in normalized and "end" in normalized:
        end = np.asarray(normalized["end"], dtype=float).reshape(-1)
        normalized["goals"] = end.reshape(1, -1)
    if "fleetSize" not in normalized and "starts" in normalized:
        normalized["fleetSize"] = float(np.asarray(normalized["starts"], dtype=float).shape[0])
    if "separationMin" not in normalized:
        normalized["separationMin"] = float(
            normalized.get("safeDist", normalized.get("safe_dist", 10.0))
        )
    if "missionId" in normalized and not isinstance(normalized["missionId"], str):
        normalized["missionId"] = str(_to_scalar(normalized["missionId"]))
    return normalized


def save_run_popobj(path: str | Path, pop_obj: np.ndarray, problem_index: int, objective_count: int) -> None:
    payload = {
        "PopObj": np.asarray(pop_obj, dtype=float),
        "problemIndex": int(problem_index),
        "M": int(objective_count),
    }
    save_mat(path, payload)


def save_bp(path: str | Path, path_xyz: np.ndarray, objectives: np.ndarray) -> None:
    payload = {"dt_sv": {"path": np.asarray(path_xyz, dtype=float), "objs": np.asarray(objectives, dtype=float)}}
    save_mat(path, payload)
