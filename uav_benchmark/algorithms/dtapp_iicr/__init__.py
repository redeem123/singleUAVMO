from __future__ import annotations

import csv
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import (
    _ensure_fleet_endpoints,
    _normalize_objective_vector,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
)
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.mission_encoding import paths_to_decision
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir

_OBJECTIVE_COUNT = 4
_PATH_PATTERN = re.compile(r"\((-?\d+),(-?\d+),(-?\d+)\)")


def _python_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _reference_root() -> Path:
    return _python_root() / "research" / "reference_code" / "uav_comparators" / "4DPlanning"


def _dtapp_executable(params: BenchmarkParams) -> Path:
    raw = params.extra.get("dtappExecutable", params.extra.get("dtapp_executable"))
    executable = Path(str(raw)).expanduser() if raw else _reference_root() / "build" / "preflight_plan"
    if not executable.exists():
        raise RuntimeError(
            f"DTAPP-IICR executable not found at {executable}. Build the reference code with CMake first."
        )
    return executable


def _ground_height(model: dict[str, Any], x: float, y: float) -> float:
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


def _grid_dimensions(model: dict[str, Any]) -> tuple[int, int]:
    width = int(round(float(model["xmax"]) - float(model["xmin"]) + 1.0))
    height = int(round(float(model["ymax"]) - float(model["ymin"]) + 1.0))
    return max(2, width), max(2, height)


def _to_grid_xy(model: dict[str, Any], point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x0 = int(round(float(point[0]) - float(model["xmin"])))
    y0 = int(round(float(point[1]) - float(model["ymin"])))
    return int(np.clip(x0, 0, width - 1)), int(np.clip(y0, 0, height - 1))


def _to_grid_z(model: dict[str, Any], point: np.ndarray, depth: int) -> int:
    safe_h = float(model.get("safeH", model.get("zmin", 0.0)) or 0.0)
    zmin = float(model.get("zmin", safe_h))
    zmax = float(model.get("zmax", max(safe_h + 1.0, zmin + 1.0)))
    relative = float(point[2])
    lower = max(zmin, safe_h)
    if zmax <= lower:
        return 0
    scaled = (relative - lower) / (zmax - lower)
    return int(np.clip(round(scaled * (depth - 1)), 0, depth - 1))


def _from_grid_point(model: dict[str, Any], x0: int, y0: int, z0: int, depth: int) -> np.ndarray:
    x = float(model["xmin"]) + float(x0)
    y = float(model["ymin"]) + float(y0)
    safe_h = float(model.get("safeH", model.get("zmin", 0.0)) or 0.0)
    zmin = float(model.get("zmin", safe_h))
    zmax = float(model.get("zmax", max(safe_h + 1.0, zmin + 1.0)))
    lower = max(zmin, safe_h)
    rel = lower if depth <= 1 else lower + (float(z0) / float(depth - 1)) * max(0.0, zmax - lower)
    return np.array([x, y, _ground_height(model, x, y) + rel], dtype=float)


def _write_dtapp_inputs(
    model: dict[str, Any], params: BenchmarkParams, run_dir: Path, fleet_size: int
) -> tuple[Path, Path, int]:
    width, height = _grid_dimensions(model)
    depth = max(2, int(params.extra.get("dtappDepth", params.extra.get("dtapp_depth", 8))))
    work_dir = run_dir / "_dtapp_iicr"
    ensure_dir(work_dir)
    map_path = work_dir / "benchmark.3dmap"
    scenario_path = work_dir / "benchmark.3dscen"

    # The shared Python evaluator remains authoritative for terrain and mission
    # constraints. DTAPP receives an obstacle-free voxel map that preserves the
    # benchmark start/goal layout and inter-agent timing problem.
    map_path.write_text(f"{width} {height} {depth}\n0\n0\n", encoding="utf-8")

    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    start_time = float(params.extra.get("dtappStartTime", params.extra.get("dtapp_start_time", 0.0)))
    radius = float(
        params.extra.get("dtappRadius", params.extra.get("dtapp_radius", max(0.5, float(model.get("droneSize", 1.0)))))
    )
    speed = float(params.extra.get("dtappSpeed", params.extra.get("dtapp_speed", 4.0)))
    lines = [f"{width} {height} {depth}", str(fleet_size)]
    for idx in range(fleet_size):
        sx, sy = _to_grid_xy(model, starts[idx], width, height)
        gx, gy = _to_grid_xy(model, goals[idx], width, height)
        sz = _to_grid_z(model, starts[idx], depth)
        gz = _to_grid_z(model, goals[idx], depth)
        lines.append(f"{sx} {sy} {sz} {gx} {gy} {gz} {start_time:g} {radius:g} {speed:g} 0")
    scenario_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return map_path, scenario_path, depth


def _parse_paths(path_file: Path, model: dict[str, Any], fleet_size: int, depth: int) -> list[np.ndarray]:
    if not path_file.exists():
        return []
    paths: list[np.ndarray] = []
    for line in path_file.read_text(encoding="utf-8", errors="replace").splitlines():
        matches = _PATH_PATTERN.findall(line)
        if not matches:
            continue
        points = [_from_grid_point(model, int(x), int(y), int(z), depth) for x, y, z in matches]
        # Drop consecutive duplicates caused by scheduled waiting at start.
        deduped: list[np.ndarray] = []
        for point in points:
            if deduped and float(np.linalg.norm(point - deduped[-1])) <= 1e-9:
                continue
            deduped.append(point)
        if len(deduped) >= 2:
            paths.append(np.vstack(deduped))
    return paths[:fleet_size]


def _fallback_paths(model: dict[str, Any], fleet_size: int) -> list[np.ndarray]:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    paths: list[np.ndarray] = []
    for idx in range(fleet_size):
        start = np.asarray(starts[idx, :3], dtype=float)
        goal = np.asarray(goals[idx, :3], dtype=float)
        start_abs = np.array([start[0], start[1], _ground_height(model, start[0], start[1]) + start[2]], dtype=float)
        goal_abs = np.array([goal[0], goal[1], _ground_height(model, goal[0], goal[1]) + goal[2]], dtype=float)
        paths.append(np.vstack([start_abs, goal_abs]))
    return paths


def _read_success(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    raw = rows[-1].get("success", "0")
    try:
        return float(raw) > 0.5
    except ValueError:
        return str(raw).strip().lower() in {"true", "yes", "success"}


def _run_dtapp_reference(
    *,
    model: dict[str, Any],
    params: BenchmarkParams,
    run_idx: int,
    run_dir: Path,
    fleet_size: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    map_path, scenario_path, depth = _write_dtapp_inputs(model, params, run_dir, fleet_size)
    work_dir = run_dir / "_dtapp_iicr"
    output_prefix = work_dir / "dtapp"
    path_prefix = work_dir / "paths.txt"
    stats_prefix = work_dir / "stats.csv"
    max_iterations = int(
        params.extra.get("dtappMaxIterations", params.extra.get("dtapp_max_iterations", params.generations))
    )
    cutoff_time = float(params.extra.get("dtappCutoffTime", params.extra.get("dtapp_cutoff_time", 480.0)))
    command = [
        str(_dtapp_executable(params)),
        "-m",
        str(map_path),
        "-a",
        str(scenario_path),
        "-k",
        str(fleet_size),
        "-t",
        str(cutoff_time),
        "-o",
        str(output_prefix),
        "--outputPaths",
        str(path_prefix),
        "--stats",
        str(stats_prefix),
        "--roundTrip",
        str(bool(params.extra.get("dtappRoundTrip", params.extra.get("dtapp_round_trip", False)))).lower(),
        "--deliveryWaitTime",
        str(float(params.extra.get("dtappDeliveryWaitTime", params.extra.get("dtapp_delivery_wait_time", 0.0)))),
        "--initLNS",
        str(bool(params.extra.get("dtappInitLns", params.extra.get("dtapp_init_lns", True)))).lower(),
        "--enablePruning",
        str(bool(params.extra.get("dtappEnablePruning", params.extra.get("dtapp_enable_pruning", True)))).lower(),
        "--maxIterations",
        str(max_iterations),
        "--screen",
        str(int(params.extra.get("dtappScreen", params.extra.get("dtapp_screen", 0)))),
        "--seed",
        str(int((params.seed or 0) + run_idx)),
    ]
    completed = subprocess.run(command, cwd=str(_reference_root()), text=True, capture_output=True, check=False)
    (work_dir / "dtapp_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (work_dir / "dtapp_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result_csv = Path(f"{output_prefix}-LNS.csv")
    success = completed.returncode == 0 and _read_success(result_csv)
    paths = _parse_paths(path_prefix, model, fleet_size, depth)
    if len(paths) != fleet_size:
        paths = _fallback_paths(model, fleet_size)
    metadata = {
        "dtappReturnCode": int(completed.returncode),
        "dtappSuccess": bool(success),
        "dtappMapPath": str(map_path),
        "dtappScenarioPath": str(scenario_path),
        "dtappResultCsv": str(result_csv),
        "dtappPathFile": str(path_prefix),
        "dtappCutoffTime": float(cutoff_time),
        "dtappMaxIterations": int(max_iterations),
        "dtappNativePopulation": False,
        "dtappNativeGenerations": False,
    }
    return paths, metadata


def run_dtapp_iicr(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 42
    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + 1,
        separation_min=float(params.separation_min),
    )

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in _resolve_run_indices(params):
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=_OBJECTIVE_COUNT,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue

        ensure_dir(run_dir)
        run_start = time.perf_counter()
        paths, metadata = _run_dtapp_reference(
            model=model, params=params, run_idx=run_idx, run_dir=run_dir, fleet_size=fleet_size
        )
        decision = paths_to_decision(paths, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        objective, details = evaluate_mission_details(paths, model)
        objective = _normalize_objective_vector(objective, details, model=model, fleet_size=fleet_size)
        details["paths"] = paths
        candidates = [Candidate(vector=decision, objective=objective, details=details)]
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=candidates,
            problem_index=params.problem_index,
            objective_count=_OBJECTIVE_COUNT,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="cpp:dtapp-iicr-reference",
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": "DTAPP-IICR",
                "optimizerBackend": "Official C++ DTAPP-IICR reference planner with Python benchmark re-scoring",
                "pythonProblemEvaluation": True,
                "benchmarkObjectiveDuringSearch": False,
                "finalPathReevaluatedByPython": True,
                "population": int(params.population),
                "generations": int(params.generations),
                "nativePopulationLoop": False,
                "nativeGenerationLoop": False,
                **metadata,
            },
        )
        if params.compute_metrics:
            final_obj = _candidate_matrix(candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_obj, params.problem_index, _OBJECTIVE_COUNT),
                    cal_metric(2, final_obj, params.problem_index, _OBJECTIVE_COUNT),
                ],
                dtype=float,
            )

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


__all__ = ["run_dtapp_iicr"]
