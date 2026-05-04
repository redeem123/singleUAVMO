from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import (
    _ensure_fleet_endpoints,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
)
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.mission_encoding import paths_to_decision
from uav_benchmark.io.matlab import load_mat, save_mat
from uav_benchmark.io.results import ensure_dir

_OBJECTIVE_COUNT = 4
_AUTHOR_CONTROL_POINTS = 7


def _python_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bridge_root() -> Path:
    return _python_root() / "uav_benchmark" / "platemo_bridge" / "matlab"


def _fair_shim_root() -> Path:
    return _bridge_root() / "moead_awa_astar_fair_shims"


def _reference_root() -> Path:
    return (
        _python_root()
        / "research"
        / "reference_code"
        / "uav_comparators"
        / "Heuristic-Driven-Evolutionary-UAV-Path-Planning"
        / "MOEAD_AWA_Astar"
    )


def _matlab_executable(params: BenchmarkParams) -> str:
    raw = str(params.extra.get("matlabExecutable", params.extra.get("matlab_executable", "matlab")))
    executable = shutil.which(raw) if Path(raw).name == raw else raw
    if not executable:
        raise RuntimeError("MATLAB executable not found. Install MATLAB or pass matlabExecutable=/path/to/matlab.")
    return str(executable)


def _reference_problem_index(problem_name: str) -> int:
    token = str(problem_name).lower()
    if token.startswith("c_"):
        return 1
    if token.startswith("m_"):
        return 2
    return 3


def _model_for_reference(model: dict[str, Any], start: np.ndarray, goal: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("H", "X", "Y", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "safeH"):
        out[key] = model[key]
    out["start"] = np.asarray(start, dtype=float).reshape(1, -1)
    out["end"] = np.asarray(goal, dtype=float).reshape(1, -1)
    return out


def _paths_from_stack(path_stack: np.ndarray) -> list[np.ndarray]:
    stack = np.asarray(path_stack, dtype=float)
    if stack.ndim != 3 or stack.shape[2] < 3:
        return []
    paths: list[np.ndarray] = []
    for idx in range(stack.shape[0]):
        path = stack[idx, :, :3]
        path = path[np.all(np.isfinite(path), axis=1)]
        if path.shape[0] >= 2:
            paths.append(path)
    return paths


def _run_reference_matlab(
    *,
    model: dict[str, Any],
    params: BenchmarkParams,
    run_idx: int,
    run_dir: Path,
    n_waypoints: int,
) -> list[np.ndarray]:
    reference_root = _reference_root()
    if not reference_root.exists():
        raise RuntimeError(f"Heuristic-driven MOEA/D-AWA+A* reference code not found at {reference_root}")

    work_dir = run_dir / "_moead_awa_astar_reference"
    ensure_dir(work_dir)
    output_path = work_dir / "moead_awa_astar_result.mat"
    config_path = work_dir / "moead_awa_astar_config.mat"
    context_path = work_dir / "moead_awa_astar_context.pkl"
    with context_path.open("wb") as handle:
        pickle.dump(
            {"model": model, "fleet_size": 1, "n_waypoints": int(n_waypoints)},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    reference_model = _model_for_reference(model, starts[0], goals[0])
    control_points = int(params.extra.get("moeadAwaAstarControlPoints", _AUTHOR_CONTROL_POINTS))
    if control_points != _AUTHOR_CONTROL_POINTS:
        raise RuntimeError("The author MOEA/D-AWA+A* implementation supports exactly 7 control points.")
    config = {
        "referenceRoot": str(reference_root),
        "fairShimRoot": str(_fair_shim_root()),
        "workDir": str(work_dir),
        "outputPath": str(output_path),
        "pythonExecutable": sys.executable,
        "pythonPath": str(_python_root()),
        "contextPath": str(context_path),
        "model": reference_model,
        "start": starts[0].reshape(1, -1),
        "goal": goals[0].reshape(1, -1),
        "N": int(params.population),
        "M": int(_OBJECTIVE_COUNT),
        "generations": int(params.generations),
        "pathNodeCount": int(control_points),
        "problemIndexForReference": int(_reference_problem_index(params.problem_name)),
        "seed": int((params.seed or 0) + int(run_idx)),
    }
    save_mat(config_path, config)

    bridge_root = str(_bridge_root()).replace("'", "''")
    config_arg = str(config_path).replace("'", "''")
    statement = f"addpath('{bridge_root}'); run_moead_awa_astar_reference_bridge('{config_arg}')"
    completed = subprocess.run(
        [_matlab_executable(params), "-batch", statement],
        cwd=str(_python_root()),
        text=True,
        capture_output=True,
        check=False,
    )
    (work_dir / "matlab_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (work_dir / "matlab_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MOEA/D-AWA+A* MATLAB run failed. See {work_dir / 'matlab_stderr.txt'}")
    result = load_mat(output_path)
    return _paths_from_stack(np.asarray(result.get("PathStack", np.zeros((0, 0, 3))), dtype=float))


def run_moead_awa_astar(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    model = dict(model)
    n_waypoints = int(model.get("n", _AUTHOR_CONTROL_POINTS))
    if n_waypoints != _AUTHOR_CONTROL_POINTS:
        raise RuntimeError("The author MOEA/D-AWA+A* implementation supports exactly 7 internal waypoints.")
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    if requested_fleet != 1:
        raise RuntimeError("MOEA/D-AWA+A* official reference adapter supports only single-UAV problems.")
    seed_value = int(params.seed) if params.seed is not None else 42
    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=1,
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
        paths = _run_reference_matlab(
            model=model, params=params, run_idx=run_idx, run_dir=run_dir, n_waypoints=n_waypoints
        )
        decisions = np.asarray(
            [paths_to_decision([path], model=model, fleet_size=fleet_size, n_waypoints=n_waypoints) for path in paths],
            dtype=float,
        )
        candidates = _evaluate_population(decisions, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=candidates,
            problem_index=params.problem_index,
            objective_count=_OBJECTIVE_COUNT,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="matlab:moead-awa-astar-reference",
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": "MOEAD-AWA-ASTAR",
                "optimizerBackend": "Official MATLAB MOEA/D-AWA with A*-guided crossover/mutation and benchmark objective shim",
                "pythonProblemEvaluation": True,
                "benchmarkObjectiveDuringSearch": True,
                "matlabEvaluationBatching": True,
                "finalPathReevaluatedByPython": True,
                "officialReferenceSupportsFleet": False,
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


__all__ = ["run_moead_awa_astar"]
