from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.io.matlab import load_mat, load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain

_PAPER_CONTROLLER_CONFIG = {
    "sacHiddenDim": 128,
    "sacWarmupSteps": 8,
    "sacBatchSize": 24,
    "sacReplayCapacity": 4096,
    "sacUpdatesPerStep": 2,
    "sacScratchPolicyMixStart": 0.40,
    "sacScratchPolicyMixEnd": 0.95,
    "sacLoadedPolicyMixStart": 0.75,
    "sacLoadedPolicyMixEnd": 1.00,
    "sacPolicyMixAnnealSteps": 256,
}

STAGE_PRESETS: dict[str, dict[str, Any]] = {
    "stage1": {
        "trainProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat"],
        "evalProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat"],
        "fleetSizes": [2],
        "trainSeeds": [11, 13, 17, 19, 23, 29, 31, 37],
        "evalSeeds": [101, 103],
        "epochs": 2,
        "generations": 10,
        "population": 8,
        "separationMin": 10.0,
    },
    "stage2": {
        "trainProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "evalProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "fleetSizes": [3],
        "trainSeeds": [11, 13, 17, 19, 23, 29, 31, 37],
        "evalSeeds": [101, 103],
        "epochs": 2,
        "generations": 16,
        "population": 12,
        "separationMin": 10.0,
    },
    "stage3": {
        "trainProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "evalProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "fleetSizes": [5],
        "trainSeeds": [11, 17, 23, 31],
        "evalSeeds": [101, 103],
        "epochs": 1,
        "generations": 20,
        "population": 14,
        "separationMin": 10.0,
    },
    "paper_stage1": {
        "trainProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat"],
        "evalProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "fleetSizes": [2],
        "trainSeeds": [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
        "evalSeeds": [101, 103, 107],
        "epochs": 3,
        "generations": 16,
        "population": 12,
        "separationMin": 10.0,
        "controllerConfig": dict(_PAPER_CONTROLLER_CONFIG),
    },
    "paper_stage2": {
        "trainProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "evalProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "fleetSizes": [3],
        "trainSeeds": [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
        "evalSeeds": [101, 103, 107],
        "epochs": 3,
        "generations": 24,
        "population": 16,
        "separationMin": 10.0,
        "controllerConfig": dict(_PAPER_CONTROLLER_CONFIG),
    },
    "paper_stage3": {
        "trainProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "evalProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "fleetSizes": [5],
        "trainSeeds": [11, 13, 17, 19, 23, 29, 31, 37],
        "evalSeeds": [101, 103, 107],
        "epochs": 3,
        "generations": 32,
        "population": 20,
        "separationMin": 10.0,
        "controllerConfig": dict(_PAPER_CONTROLLER_CONFIG),
    },
    "paper_stage4": {
        "trainProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "evalProblems": ["terrainStruct_c_100.mat", "terrainStruct_m_100.mat", "terrainStruct_s_120.mat"],
        "fleetSizes": [8],
        "trainSeeds": [11, 13, 17, 19, 23, 29, 31, 37],
        "evalSeeds": [101, 103, 107],
        "epochs": 2,
        "generations": 40,
        "population": 24,
        "separationMin": 10.0,
        "controllerConfig": dict(_PAPER_CONTROLLER_CONFIG),
    },
    # Legacy preset name retained for compatibility. Keep this curriculum on
    # multi-UAV cases only: SAC-SMOPSO delegates fleet=1 to plain NMOPSO, so
    # single-UAV episodes do not train the controller or encoder.
    "paper_mixed_12": {
        "trainProblems": [
            "terrainStruct_c_100.mat",
            "terrainStruct_m_100.mat",
            "terrainStruct_s_120.mat",
        ],
        "evalProblems": [
            "terrainStruct_c_100.mat",
            "terrainStruct_m_100.mat",
            "terrainStruct_s_120.mat",
        ],
        "fleetSizes": [2],
        "trainSeeds": [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
        "evalSeeds": [101, 103, 107],
        "epochs": 3,
        "generations": 24,
        "population": 16,
        "separationMin": 10.0,
        "controllerConfig": dict(_PAPER_CONTROLLER_CONFIG),
    },
}


def stage_preset(name: str | None) -> dict[str, Any]:
    if name is None:
        return {}
    key = str(name).strip().lower()
    if key not in STAGE_PRESETS:
        raise ValueError(f"Unknown SAC-SMOPSO stage preset: {name}")
    return dict(STAGE_PRESETS[key])


def validate_multi_uav_fleet_sizes(fleet_sizes: list[int] | tuple[int, ...]) -> list[int]:
    values = [int(size) for size in fleet_sizes]
    if not values:
        raise ValueError("At least one fleet size is required.")
    invalid = [size for size in values if size <= 1]
    if invalid:
        raise ValueError(f"SAC-SMOPSO checkpointed workflows require fleet sizes > 1, got {invalid}.")
    return values


def string_field(value: Any) -> str:
    flat = np.asarray(value).reshape(-1)
    return str(flat[0]) if flat.size > 0 else ""


def float_field(value: Any, default: float = 0.0) -> float:
    flat = np.asarray(value, dtype=float).reshape(-1)
    return float(flat[0]) if flat.size > 0 else float(default)


def mean_field(payload: dict[str, Any], key: str) -> float:
    values = np.asarray(payload.get(key, np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size > 0 else 0.0


def load_protocol(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load protocol files. Install pyyaml.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol file must contain a mapping: {path}")
    return dict(payload)


def configure_torch_runtime(controller_module: Any) -> None:
    if not bool(getattr(controller_module, "_TORCH_AVAILABLE", False)):
        return
    torch_module = getattr(controller_module, "torch", None)
    if torch_module is None:
        return
    cpu_count = max(1, int(os.cpu_count() or 1))
    with contextlib.suppress(RuntimeError, TypeError, ValueError):
        torch_module.set_num_threads(cpu_count)
    with contextlib.suppress(RuntimeError, TypeError, ValueError):
        interop = max(1, min(4, cpu_count // 2 or 1))
        torch_module.set_num_interop_threads(interop)


def policy_backend_tag(
    gpu_mode: str,
    *,
    controller_module: Any | None = None,
    heuristic_fallback: bool = False,
) -> str:
    from uav_benchmark.utils.gpu import resolve_gpu

    gpu_info = resolve_gpu(gpu_mode)
    if gpu_info.enabled and gpu_info.backend == "torch":
        if "cuda" in gpu_info.device:
            return "torch:cuda"
        if "mps" in gpu_info.device:
            return "torch:mps"
    torch_available = True
    if controller_module is not None:
        torch_available = bool(getattr(controller_module, "_TORCH_AVAILABLE", False))
    if heuristic_fallback and not torch_available:
        return "heuristic:numpy"
    return "torch:cpu"


def scenario_label(problem_file: Path, fleet_size: int, seed: int) -> str:
    return f"{problem_file.stem}_uav{int(fleet_size)}_seed{int(seed)}"


def prepare_terrain(problem_file: Path, fleet_size: int, seed: int, separation_min: float) -> dict[str, Any]:
    terrain = load_terrain_struct(problem_file)
    if int(terrain.get("fleetSize", 1)) > 1:
        return terrain
    return make_fleet_terrain(
        terrain=terrain,
        fleet_size=int(fleet_size),
        seed=1000 + int(seed),
        separation_min=float(separation_min),
        mission_prefix="sac_pretrain",
    )


def first_positive_generation(values: np.ndarray, threshold: float = 0.5) -> int:
    vector = np.asarray(values, dtype=float).reshape(-1)
    hits = np.flatnonzero(np.isfinite(vector) & (vector > float(threshold)))
    return int(hits[0] + 1) if hits.size > 0 else 0


def load_rollout_summary(run_dir: Path) -> dict[str, float]:
    summary = {
        "firstFeasibleGeneration": 0.0,
        "finalTraceFeasible": 0.0,
        "finalTraceConflict": 0.0,
        "meanReward": 0.0,
    }
    feasible_path = run_dir / "rl_feasible.mat"
    if feasible_path.exists():
        payload = load_mat(feasible_path)
        values = np.asarray(payload.get("rl_feasible", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        summary["firstFeasibleGeneration"] = float(first_positive_generation(values))
        if values.size > 0:
            summary["finalTraceFeasible"] = float(values[-1])
    conflict_path = run_dir / "rl_conflict.mat"
    if conflict_path.exists():
        payload = load_mat(conflict_path)
        values = np.asarray(payload.get("rl_conflict", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        if values.size > 0:
            summary["finalTraceConflict"] = float(values[-1])
    reward_path = run_dir / "rl_reward.mat"
    if reward_path.exists():
        payload = load_mat(reward_path)
        values = np.asarray(payload.get("rl_reward", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            summary["meanReward"] = float(np.mean(finite))
    return summary


def aggregate_records(records: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    if not records:
        return {f"{key}Mean": 0.0 for key in keys}
    summary: dict[str, float] = {}
    for key in keys:
        summary[f"{key}Mean"] = float(np.mean([float(record.get(key, 0.0)) for record in records]))
    return summary
