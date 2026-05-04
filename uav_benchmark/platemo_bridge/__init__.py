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
    _build_bounds,
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
from uav_benchmark.io.matlab import load_mat, save_mat
from uav_benchmark.io.results import ensure_dir

_OBJECTIVE_COUNT = 4
_ALGORITHM_CLASS = {
    "CMOEA-CD": "CMOEACD",
    "APSEA": "APSEA",
    "C-TSEA": "CTSEA",
    "ToP": "ToP",
    "CMOCSO": "CMOCSO",
    "C-TAEA": "CTAEA",
    "Two_Arch2": "Two_Arch2",
    "CMOEA-MS": "CMOEAMS",
    "CMOEA-MSG": "CMOEAMSG",
    "CCMO": "CCMO",
    "URCMO": "URCMO",
}
_REFERENCE_LEGACY_ALGORITHM = {
    "EMMOP": {
        "algorithmFunction": "EMMOP",
        "referenceRoot": "EMMOP",
        "fairShimFolder": "emmop_fair_shims",
        "optimizerObjectives": _OBJECTIVE_COUNT,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _python_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bridge_root() -> Path:
    return Path(__file__).resolve().parent / "matlab"


def _platemo_root() -> Path:
    return _repo_root() / "matlabimplementation" / "reference_code" / "PlatEMO" / "PlatEMO"


def _uav_comparator_root(name: str) -> Path:
    return _python_root() / "research" / "reference_code" / "uav_comparators" / name


def _matlab_executable(params: BenchmarkParams) -> str:
    raw = str(params.extra.get("matlabExecutable", params.extra.get("matlab_executable", "matlab")))
    executable = shutil.which(raw) if Path(raw).name == raw else raw
    if not executable:
        raise RuntimeError(
            "MATLAB executable not found. Install MATLAB or pass extra matlabExecutable=/path/to/matlab."
        )
    return str(executable)


def _model_for_matlab_bridge(model: dict[str, Any], params: BenchmarkParams, fleet_size: int) -> dict[str, Any]:
    prepared = dict(model)
    prepared["fleetSize"] = float(fleet_size)
    prepared["separationMin"] = float(params.separation_min)
    prepared["maxTurnDeg"] = float(params.max_turn_deg)
    for key in (
        "hardCollisionConstraint",
        "fleetSeparationWeight",
        "collisionStep",
        "safeDist",
        "droneSize",
    ):
        if key in params.extra:
            prepared[key] = params.extra[key]
    return prepared


def _write_context(context_path: Path, model: dict[str, Any], fleet_size: int, n_waypoints: int) -> None:
    with context_path.open("wb") as handle:
        pickle.dump(
            {"model": model, "fleet_size": int(fleet_size), "n_waypoints": int(n_waypoints)},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _run_matlab_optimizer(
    *,
    algorithm_label: str,
    params: BenchmarkParams,
    run_idx: int,
    run_dir: Path,
    lower: np.ndarray,
    upper: np.ndarray,
    context_path: Path,
) -> np.ndarray:
    work_dir = run_dir / "_matlab_bridge"
    ensure_dir(work_dir)
    output_path = work_dir / "platemo_result.mat"
    config_path = work_dir / "platemo_config.mat"
    algorithm_class = _ALGORITHM_CLASS[algorithm_label]
    max_fe = int(max(1, params.population) * (max(1, params.generations) + 1))
    config = {
        "algorithmClass": algorithm_class,
        "platemoRoot": str(_platemo_root()),
        "bridgeRoot": str(_bridge_root()),
        "workDir": str(work_dir),
        "outputPath": str(output_path),
        "pythonExecutable": sys.executable,
        "pythonPath": str(_python_root()),
        "contextPath": str(context_path),
        "N": int(params.population),
        "M": int(_OBJECTIVE_COUNT),
        "D": int(lower.size),
        "maxFE": int(max_fe),
        "lower": np.asarray(lower, dtype=float).reshape(1, -1),
        "upper": np.asarray(upper, dtype=float).reshape(1, -1),
        "seed": int((params.seed or 0) + int(run_idx)),
    }
    save_mat(config_path, config)

    matlab = _matlab_executable(params)
    bridge_root = str(_bridge_root()).replace(chr(39), chr(39) + chr(39))
    config_arg = str(config_path).replace(chr(39), chr(39) + chr(39))
    statement = f"addpath('{bridge_root}'); run_platemo_bridge('{config_arg}')"
    completed = subprocess.run(
        [matlab, "-batch", statement],
        cwd=str(_python_root()),
        text=True,
        capture_output=True,
        check=False,
    )
    (work_dir / "matlab_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (work_dir / "matlab_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"MATLAB PlatEMO run failed for {algorithm_label} Run_{run_idx}. See {work_dir / 'matlab_stderr.txt'}"
        )
    if not output_path.exists():
        raise RuntimeError(f"MATLAB PlatEMO run did not write {output_path}")
    result = load_mat(output_path)
    pop_dec = np.asarray(result.get("PopDec", np.zeros((0, lower.size))), dtype=float)
    if pop_dec.ndim == 1:
        pop_dec = pop_dec.reshape(1, -1)
    return pop_dec


def _run_reference_legacy_optimizer(
    *,
    algorithm_label: str,
    params: BenchmarkParams,
    run_idx: int,
    run_dir: Path,
    lower: np.ndarray,
    upper: np.ndarray,
    context_path: Path,
) -> np.ndarray:
    work_dir = run_dir / "_reference_bridge"
    ensure_dir(work_dir)
    output_path = work_dir / "reference_result.mat"
    config_path = work_dir / "reference_config.mat"
    algorithm = _REFERENCE_LEGACY_ALGORITHM[algorithm_label]
    optimizer_objectives = int(
        params.extra.get("referenceObjectiveCount", algorithm.get("optimizerObjectives", _OBJECTIVE_COUNT))
    )
    reference_root = _uav_comparator_root(str(algorithm["referenceRoot"]))
    if not reference_root.exists():
        raise RuntimeError(f"Reference code for {algorithm_label} not found at {reference_root}")
    fair_shim_root = _bridge_root() / str(algorithm.get("fairShimFolder", ""))

    max_fe = int(max(1, params.population) * (max(1, params.generations) + 1))
    config = {
        "algorithmFunction": str(algorithm["algorithmFunction"]),
        "referenceRoot": str(reference_root),
        "fairShimRoot": str(fair_shim_root) if fair_shim_root.exists() else "",
        "platemoRoot": str(_platemo_root()),
        "bridgeRoot": str(_bridge_root()),
        "workDir": str(work_dir),
        "outputPath": str(output_path),
        "pythonExecutable": sys.executable,
        "pythonPath": str(_python_root()),
        "contextPath": str(context_path),
        "N": int(params.population),
        "M": int(optimizer_objectives),
        "D": int(lower.size),
        "maxFE": int(max_fe),
        "lower": np.asarray(lower, dtype=float).reshape(1, -1),
        "upper": np.asarray(upper, dtype=float).reshape(1, -1),
        "seed": int((params.seed or 0) + int(run_idx)),
    }
    save_mat(config_path, config)

    matlab = _matlab_executable(params)
    bridge_root = str(_bridge_root()).replace(chr(39), chr(39) + chr(39))
    config_arg = str(config_path).replace(chr(39), chr(39) + chr(39))
    statement = f"addpath('{bridge_root}'); run_reference_legacy_bridge('{config_arg}')"
    completed = subprocess.run(
        [matlab, "-batch", statement],
        cwd=str(_python_root()),
        text=True,
        capture_output=True,
        check=False,
    )
    (work_dir / "matlab_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (work_dir / "matlab_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"MATLAB reference-code run failed for {algorithm_label} Run_{run_idx}. "
            f"See {work_dir / 'matlab_stderr.txt'}"
        )
    if not output_path.exists():
        raise RuntimeError(f"MATLAB reference-code run did not write {output_path}")
    result = load_mat(output_path)
    pop_dec = np.asarray(result.get("PopDec", np.zeros((0, lower.size))), dtype=float)
    if pop_dec.ndim == 1:
        pop_dec = pop_dec.reshape(1, -1)
    return pop_dec


def run_platemo_algorithm(model: dict[str, Any], params: BenchmarkParams, algorithm_label: str) -> np.ndarray:
    objective_count = _OBJECTIVE_COUNT
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 42
    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model = _model_for_matlab_bridge(model, params, fleet_size)
    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)

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
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue

        ensure_dir(run_dir)
        run_start = time.perf_counter()
        context_path = run_dir / "_matlab_bridge_context.pkl"
        _write_context(context_path, model, fleet_size, n_waypoints)
        pop_dec = _run_matlab_optimizer(
            algorithm_label=algorithm_label,
            params=params,
            run_idx=run_idx,
            run_dir=run_dir,
            lower=lower,
            upper=upper,
            context_path=context_path,
        )
        candidates = _evaluate_population(pop_dec, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="matlab:platemo",
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": algorithm_label,
                "optimizerBackend": "MATLAB PlatEMO",
                "pythonProblemEvaluation": True,
                "benchmarkObjectiveDuringSearch": True,
                "matlabEvaluationBatching": True,
                "finalPathReevaluatedByPython": True,
            },
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_obj, params.problem_index, objective_count),
                    cal_metric(2, final_obj, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_reference_legacy_algorithm(model: dict[str, Any], params: BenchmarkParams, algorithm_label: str) -> np.ndarray:
    objective_count = _OBJECTIVE_COUNT
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 42
    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model = _model_for_matlab_bridge(model, params, fleet_size)
    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)

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
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue

        ensure_dir(run_dir)
        run_start = time.perf_counter()
        context_path = run_dir / "_reference_bridge_context.pkl"
        _write_context(context_path, model, fleet_size, n_waypoints)
        pop_dec = _run_reference_legacy_optimizer(
            algorithm_label=algorithm_label,
            params=params,
            run_idx=run_idx,
            run_dir=run_dir,
            lower=lower,
            upper=upper,
            context_path=context_path,
        )
        candidates = _evaluate_population(pop_dec, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="matlab:reference-legacy",
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": algorithm_label,
                "optimizerBackend": "MATLAB reference code",
                "pythonProblemEvaluation": True,
                "benchmarkObjectiveDuringSearch": True,
                "referenceRoot": str(
                    _uav_comparator_root(_REFERENCE_LEGACY_ALGORITHM[algorithm_label]["referenceRoot"])
                ),
                "fairShimRoot": str(
                    _bridge_root() / str(_REFERENCE_LEGACY_ALGORITHM[algorithm_label].get("fairShimFolder", ""))
                ),
            },
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_obj, params.problem_index, objective_count),
                    cal_metric(2, final_obj, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


__all__ = ["run_platemo_algorithm", "run_reference_legacy_algorithm"]
