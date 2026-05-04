from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms import (
    ALL_REGISTRY as _ALL_ALGORITHM_REGISTRY,
)
from uav_benchmark.algorithms import (
    EXPERIMENTAL_REGISTRY as _EXPERIMENTAL_ALGORITHM_REGISTRY,
)
from uav_benchmark.algorithms import (
    REGISTRY as _ALGORITHM_REGISTRY,
)
from uav_benchmark.benchmark_selection import (
    _ALGORITHM_ORDER,
    _ALGORITHM_SEED_OFFSET,
    _TORCH_ACCELERATED_ALGORITHMS,
    _allow_experimental_algorithms,
    _base_problem_name,
    _fleet_from_problem_name,
    _requested_algorithms,
    _requested_problem_names,
    _variant_tasks_for_algorithm,
)
from uav_benchmark.benchmark_selection import (
    _normalize_algorithm_name as _normalize_algorithm_name,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.exceptions import ArtifactReadError
from uav_benchmark.io.matlab import load_mat, load_terrain_struct, save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.model_contracts import validate_terrain_model
from uav_benchmark.problem_generation.generate import save_fleet_scenarios
from uav_benchmark.utils.random import seed_everything

LOGGER = logging.getLogger(__name__)
AlgorithmRunner = Callable[[dict, BenchmarkParams], Any]
BenchmarkTask = tuple[Path, int, str, str, BenchmarkParams]
LegacyBenchmarkTask = tuple[Path, int, str, BenchmarkParams]
_DEFAULT_MAX_WORKERS = 14
_PAPER_MEDIUM_BASE_PROBLEM_NAMES = (
    "c_100",
    "c_150",
    "c_100_20_nofly",
    "c_70_40_nofly",
    "m_100",
    "m_200",
    "m_100_30c_nofly",
    "m_200_20c_nofly",
    "s_120",
    "s_180",
    "s_110_20_nofly",
    "s_80_40_nofly",
)


def _seed_for_task(base_seed: int, problem_index: int, algorithm_name: str) -> int:
    return int(base_seed) + int(problem_index) * 100 + int(_ALGORITHM_SEED_OFFSET.get(algorithm_name, 0))


def _seed_for_run(base_seed: int, problem_index: int, algorithm_name: str, run_index: int) -> int:
    return _seed_for_task(base_seed, problem_index, algorithm_name) + int(run_index)


def _can_parallelize_runs(algorithm_name: str, params: BenchmarkParams) -> bool:
    # Torch-backed adaptive methods should stay in the current interpreter so GPU/MPS
    # state is not lost across multiprocessing worker boundaries.
    return not (str(algorithm_name) in _TORCH_ACCELERATED_ALGORITHMS and str(params.gpu_mode).strip().lower() != "off")


def _next_dispatchable_task(
    pending_by_task: list[list[int]],
    active_by_task: list[int],
    limit_by_task: list[int],
    start_index: int = 0,
) -> int | None:
    n_tasks = len(pending_by_task)
    if n_tasks <= 0:
        return None
    for offset in range(n_tasks):
        task_index = (int(start_index) + offset) % n_tasks
        if pending_by_task[task_index] and active_by_task[task_index] < limit_by_task[task_index]:
            return task_index
    return None


def _max_parallel_worker_slots(tasks: Sequence[BenchmarkTask | LegacyBenchmarkTask]) -> int:
    """Upper bound on run-level concurrency across all tasks."""
    slots = 0
    for task in tasks:
        _problem_file, _problem_index, _algorithm_label, runner_name, run_params = _unpack_benchmark_task(task)
        if _can_parallelize_runs(runner_name, run_params):
            slots += max(1, int(run_params.runs))
        else:
            slots += 1
    return max(1, int(slots))


def _run_tasks_in_current_process(
    *,
    tasks: list[BenchmarkTask],
    run_indices: tuple[int, ...],
    params: BenchmarkParams,
    task_problem_name: list[str],
) -> None:
    for task_index, task in enumerate(tasks):
        problem_file, problem_index, algorithm_label, runner_name, run_params = task
        for run_index in run_indices:
            _execute_task_run(problem_file, problem_index, algorithm_label, runner_name, run_params, run_index)
        _write_grouped_run_hv_summary(
            params=params,
            algorithm_label=algorithm_label,
            problem_name=task_problem_name[task_index],
            problem_index=problem_index,
        )


def _safe_module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return "unavailable"


def _git_info(project_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        return subprocess.check_output(
            args,
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    info: dict[str, Any] = {
        "available": False,
        "commit": "",
        "branch": "",
        "isDirty": False,
    }
    try:
        commit = _run(["git", "rev-parse", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = _run(["git", "status", "--porcelain"])
        info["available"] = True
        info["commit"] = commit
        info["branch"] = branch
        info["isDirty"] = bool(status)
    except (subprocess.SubprocessError, OSError):
        LOGGER.debug("Git metadata unavailable for %s", project_root)
    return info


def _params_manifest(params: BenchmarkParams) -> dict[str, Any]:
    return {
        "generations": int(params.generations),
        "population": int(params.population),
        "runs": int(params.runs),
        "computeMetrics": bool(params.compute_metrics),
        "safeDist": float(params.safe_dist),
        "droneSize": float(params.drone_size),
        "seed": int(params.seed) if params.seed is not None else None,
        "mode": str(params.mode),
        "fleetSize": int(params.fleet_size),
        "fleetSizes": [int(item) for item in params.fleet_sizes],
        "separationMin": float(params.separation_min),
        "maxTurnDeg": float(params.max_turn_deg),
        "evaluationBudget": int(params.evaluation_budget),
        "scenarioSet": str(params.scenario_set),
        "gpuMode": str(params.gpu_mode),
        "runIndices": [int(item) for item in params.run_indices] if params.run_indices else [],
        "writeFinalHv": bool(params.write_final_hv),
        "resultsDir": str(params.results_dir.resolve()),
        "extra": dict(params.extra) if isinstance(params.extra, dict) else {},
    }


def _plan_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_benchmark_manifest(
    project_root: Path,
    params: BenchmarkParams,
    fleet_sizes: tuple[int, ...],
    tasks: list[BenchmarkTask],
    n_workers: int,
    created_utc: str | None = None,
) -> dict[str, Any]:
    created = created_utc or datetime.now(timezone.utc).isoformat()
    base_seed = int(params.seed) if params.seed is not None else 42
    task_plan: list[dict[str, Any]] = []
    problems: list[str] = []
    algorithms_resolved: list[str] = []
    for problem_file, problem_index, algorithm_label, runner_name, _run_params in tasks:
        problem = _problem_name(problem_file)
        fleet_size = _fleet_from_problem_name(problem) or int(params.fleet_size)
        problems.append(problem)
        algorithms_resolved.append(algorithm_label)
        task_plan.append(
            {
                "problem": problem,
                "problemIndex": int(problem_index),
                "algorithm": str(algorithm_label),
                "runner": str(runner_name),
                "fleetSize": int(fleet_size),
                "seedOffset": int(_ALGORITHM_SEED_OFFSET.get(runner_name, 0)),
                "effectiveSeed": int(_seed_for_task(base_seed, problem_index, runner_name)),
            }
        )
    plan_payload = {
        "parameters": _params_manifest(params),
        "fleetSizesResolved": [int(size) for size in fleet_sizes],
        "problemsResolved": sorted(dict.fromkeys(problems)),
        "algorithmsResolved": list(dict.fromkeys(algorithms_resolved)),
        "taskPlan": task_plan,
        "workers": int(n_workers),
    }
    return {
        "schemaVersion": 1,
        "createdUtc": created,
        "projectRoot": str(project_root.resolve()),
        "planHashSha256": _plan_hash(plan_payload),
        "plan": plan_payload,
        "git": _git_info(project_root),
        "environment": {
            "pythonVersion": sys.version.split()[0],
            "platform": platform.platform(),
            "numpyVersion": _safe_module_version("numpy"),
            "scipyVersion": _safe_module_version("scipy"),
        },
    }


def _write_benchmark_manifest(
    project_root: Path,
    params: BenchmarkParams,
    fleet_sizes: tuple[int, ...],
    tasks: list[BenchmarkTask],
    n_workers: int,
) -> Path:
    manifest = _build_benchmark_manifest(
        project_root=project_root,
        params=params,
        fleet_sizes=fleet_sizes,
        tasks=tasks,
        n_workers=n_workers,
    )
    ensure_dir(params.results_dir)
    manifest_path = params.results_dir / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _unpack_benchmark_task(task: BenchmarkTask | LegacyBenchmarkTask) -> BenchmarkTask:
    if len(task) == 5:
        problem_file, problem_index, algorithm_label, runner_name, run_params = task
        return problem_file, problem_index, algorithm_label, runner_name, run_params
    problem_file, problem_index, algorithm_name, run_params = task
    return problem_file, problem_index, algorithm_name, algorithm_name, run_params


def _resolved_run_indices(params: BenchmarkParams) -> tuple[int, ...]:
    raw = params.run_indices
    if raw is None:
        return tuple(range(1, int(params.runs) + 1))
    if isinstance(raw, (list, tuple)):
        indices = [int(item) for item in raw if int(item) >= 1]
    else:
        indices = [int(raw)] if int(raw) >= 1 else []
    if not indices:
        return tuple(range(1, int(params.runs) + 1))
    return tuple(dict.fromkeys(indices))


def _configure_worker_python_executable() -> str:
    executable = str(Path(sys.executable).absolute())
    try:
        multiprocessing.set_executable(executable)
    except (RuntimeError, OSError):
        return executable
    return executable


def _problem_name(problem_file: Path) -> str:
    name = problem_file.stem
    if name.startswith("terrainStruct_"):
        name = name.replace("terrainStruct_", "", 1)
    return name


def _algorithm_map(
    include_algorithms: tuple[str, ...] = (),
    *,
    allow_experimental: bool = False,
) -> list[tuple[str, AlgorithmRunner]]:
    if not include_algorithms:
        registry = dict(_ALGORITHM_REGISTRY)
    else:
        registry = dict(_ALGORITHM_REGISTRY)
        if allow_experimental:
            registry.update(_EXPERIMENTAL_ALGORITHM_REGISTRY)
        missing = [name for name in include_algorithms if name not in registry]
        blocked = [name for name in missing if name in _EXPERIMENTAL_ALGORITHM_REGISTRY]
        unknown = [name for name in missing if name not in _ALL_ALGORITHM_REGISTRY]
        if blocked:
            raise ValueError(
                "Experimental algorithms require an explicit opt-in. "
                "Set allowExperimentalAlgorithms=true to use: " + ", ".join(blocked)
            )
        if unknown:
            raise ValueError("Unknown algorithm(s): " + ", ".join(unknown))

    mapping = [(name, registry[name]) for name in _ALGORITHM_ORDER if name in registry]
    extra = [(name, runner) for name, runner in registry.items() if name not in _ALGORITHM_ORDER]
    mapping.extend(sorted(extra, key=lambda item: item[0]))
    if not include_algorithms:
        return mapping
    include_set = set(include_algorithms)
    return [item for item in mapping if item[0] in include_set]


def _resolve_algorithm_runner(name: str) -> AlgorithmRunner:
    if name in _ALGORITHM_REGISTRY:
        return _ALGORITHM_REGISTRY[name]
    if name in _EXPERIMENTAL_ALGORITHM_REGISTRY:
        return _EXPERIMENTAL_ALGORITHM_REGISTRY[name]
    return _ALL_ALGORITHM_REGISTRY[name]


def _resolve_benchmark_seed(params: BenchmarkParams) -> BenchmarkParams:
    if params.seed is not None:
        return params

    import secrets

    resolved_seed = secrets.randbelow(2**31)
    LOGGER.info(
        "No --seed provided; using randomly generated seed=%d. Pass --seed %d to reproduce this run.",
        resolved_seed,
        resolved_seed,
    )
    return replace(params, seed=resolved_seed)


def _resolved_fleet_sizes(params: BenchmarkParams) -> tuple[int, ...]:
    raw_fleet_sizes = params.fleet_sizes if params.fleet_sizes else (int(params.fleet_size),)
    return tuple(dict.fromkeys(max(1, int(size)) for size in raw_fleet_sizes))


def _maybe_generate_fleet_scenarios(
    project_root: Path,
    params: BenchmarkParams,
    fleet_sizes: tuple[int, ...],
) -> list[Path]:
    if params.scenario_set != "paper_medium":
        return []
    generated = save_fleet_scenarios(
        project_root=project_root,
        base_problem_names=list(_PAPER_MEDIUM_BASE_PROBLEM_NAMES),
        fleet_sizes=tuple(int(size) for size in fleet_sizes),
        seed=int(params.seed) if params.seed is not None else 42,
        separation_min=float(params.separation_min),
        mission_prefix="paper_medium",
        output_dir=params.results_dir / "generated_problems",
    )
    for path in generated:
        load_terrain_struct(path)
    return generated


def _problem_files_by_name(*groups: list[Path]) -> list[Path]:
    by_name: dict[str, Path] = {}
    for group in groups:
        for path in group:
            by_name[_problem_name(path)] = path
    return sorted(by_name.values(), key=lambda path: _problem_name(path))


def _select_problem_files(
    all_problem_files: list[Path],
    params: BenchmarkParams,
    fleet_sizes: tuple[int, ...],
) -> list[Path]:
    requested_fleets = set(int(size) for size in fleet_sizes)
    explicit_uav1_bases = {
        _base_problem_name(_problem_name(path))
        for path in all_problem_files
        if _fleet_from_problem_name(_problem_name(path)) == 1
    }

    problem_files: list[Path] = []
    for path in all_problem_files:
        problem_name = _problem_name(path)
        fleet = _fleet_from_problem_name(problem_name)
        if fleet is None:
            if 1 in requested_fleets and _base_problem_name(problem_name) not in explicit_uav1_bases:
                problem_files.append(path)
            continue
        if fleet in requested_fleets:
            problem_files.append(path)

    if not problem_files:
        problem_files = [path for path in all_problem_files if "_uav" in path.stem]

    requested_problem_names = _requested_problem_names(params.extra)
    if requested_problem_names:
        requested_set = set(requested_problem_names)
        problem_files = [
            path
            for path in problem_files
            if _problem_name(path) in requested_set or _base_problem_name(_problem_name(path)) in requested_set
        ]
    return problem_files


def _build_benchmark_tasks(problem_files: list[Path], params: BenchmarkParams) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    requested = _requested_algorithms(params.extra)
    allow_experimental = _allow_experimental_algorithms(params.extra)
    algo_map = _algorithm_map(requested, allow_experimental=allow_experimental)
    for problem_index, problem_file in enumerate(problem_files, start=1):
        problem_name = _problem_name(problem_file)
        base_problem = _base_problem_name(problem_name)
        run_params = replace(params, problem_name=base_problem)
        for algorithm_name, _runner in algo_map:
            tasks.extend(
                (
                    problem_file,
                    problem_index,
                    algorithm_label,
                    runner_name,
                    variant_params,
                )
                for algorithm_label, runner_name, variant_params in _variant_tasks_for_algorithm(
                    algorithm_name,
                    run_params,
                )
            )
    return tasks


def _worker_count(tasks: list[BenchmarkTask], params: BenchmarkParams) -> int:
    worker_cap = (
        int(params.extra.get("maxWorkers", _DEFAULT_MAX_WORKERS))
        if isinstance(params.extra, dict)
        else _DEFAULT_MAX_WORKERS
    )
    cpu_count = os.cpu_count() or 1
    max_parallel_tasks = _max_parallel_worker_slots(tasks)
    if worker_cap > 0:
        return min(max_parallel_tasks, worker_cap, cpu_count)
    return min(max_parallel_tasks, cpu_count)


def _execute_task_run(
    problem_file: Path,
    problem_index: int,
    algorithm_label: str,
    runner_name: str,
    params: BenchmarkParams,
    run_index: int,
) -> None:
    """Worker function that executes exactly one run index."""
    base_seed = int(params.seed) if params.seed is not None else 42
    seed_everything(_seed_for_run(base_seed, problem_index, runner_name, run_index))

    terrain = load_terrain_struct(problem_file)
    terrain["safeDist"] = params.safe_dist
    terrain["droneSize"] = params.drone_size
    terrain["separationMin"] = params.separation_min
    terrain["maxTurnDeg"] = params.max_turn_deg
    terrain = validate_terrain_model(terrain, context=str(problem_file))

    name = _problem_name(problem_file)
    fleet_size = _fleet_from_problem_name(name) or int(params.fleet_size)
    run_params = replace(
        params,
        problem_name=name,
        problem_index=problem_index,
        fleet_size=fleet_size,
        run_indices=(int(run_index),),
        write_final_hv=False,
        algorithm=algorithm_label,
    )

    runner = _resolve_algorithm_runner(runner_name)
    algo_params = replace(
        run_params,
        results_dir=params.results_dir / algorithm_label,
        algorithm=algorithm_label,
    )
    ensure_dir(algo_params.results_dir)
    print(f"[PID {os.getpid()}] Starting {algorithm_label} / {name} / Run_{int(run_index)}")
    runner(terrain, algo_params)
    print(f"[PID {os.getpid()}] Finished {algorithm_label} / {name} / Run_{int(run_index)}")


def _write_grouped_run_hv_summary(
    params: BenchmarkParams,
    algorithm_label: str,
    problem_name: str,
    problem_index: int,
) -> None:
    if not params.compute_metrics:
        return
    results_path = params.results_dir / algorithm_label / problem_name
    ensure_dir(results_path)
    scores: list[list[float]] = []
    completed_run_indices: list[int] = []
    objective_count = 4
    for run_index in _resolved_run_indices(params):
        popobj_path = results_path / f"Run_{run_index}" / "final_popobj.mat"
        if not popobj_path.exists():
            continue
        try:
            data = load_mat(popobj_path)
            matrix_raw = data.get("PopObj")
            matrix = np.asarray(matrix_raw, dtype=float) if matrix_raw is not None else np.zeros((0, 0), dtype=float)
            if matrix.size == 0:
                continue
            scores.append(
                [
                    cal_metric(1, matrix, problem_index, objective_count),
                    cal_metric(2, matrix, problem_index, objective_count),
                ]
            )
            completed_run_indices.append(int(run_index))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping unreadable metric artifact %s: %s", popobj_path, exc)
            continue
        except ArtifactReadError as exc:
            LOGGER.warning("Skipping invalid metric artifact %s: %s", popobj_path, exc)
            continue
    save_mat(
        results_path / "final_hv.mat",
        {
            "bestScores": np.asarray(scores, dtype=float).reshape(-1, 2),
            "runIndices": np.asarray(completed_run_indices, dtype=int).reshape(-1, 1),
        },
    )


def run_benchmark(project_root: Path, params: BenchmarkParams) -> None:
    _configure_worker_python_executable()
    params = _resolve_benchmark_seed(params)

    problems_dir = project_root / "problems"
    fleet_sizes = _resolved_fleet_sizes(params)
    ensure_dir(params.results_dir)
    generated_problem_files = _maybe_generate_fleet_scenarios(project_root, params, fleet_sizes)
    all_problem_files = _problem_files_by_name(
        sorted(problems_dir.glob("*.mat")),
        generated_problem_files,
    )
    problem_files = _select_problem_files(all_problem_files, params, fleet_sizes)

    tasks = _build_benchmark_tasks(problem_files, params)

    if not tasks:
        manifest_path = _write_benchmark_manifest(
            project_root=project_root,
            params=params,
            fleet_sizes=fleet_sizes,
            tasks=tasks,
            n_workers=0,
        )
        print(f"benchmark_manifest={manifest_path}")
        print("No benchmark tasks found for the selected mode/scenario settings.")
        return

    n_workers = _worker_count(tasks, params)
    manifest_path = _write_benchmark_manifest(
        project_root=project_root,
        params=params,
        fleet_sizes=fleet_sizes,
        tasks=tasks,
        n_workers=n_workers,
    )
    print(f"benchmark_manifest={manifest_path}")
    print(f"Running {len(tasks)} tasks in grouped_runs mode (max workers={n_workers})")

    run_indices = _resolved_run_indices(params)
    task_pending_runs: list[list[int]] = [list(run_indices) for _ in tasks]
    task_active_runs = [0 for _ in tasks]
    task_finalized = [False for _ in tasks]
    task_run_limit: list[int] = []
    task_problem_name: list[str] = []

    for task_index, task in enumerate(tasks, start=1):
        problem_file, _problem_index, algorithm_label, runner_name, run_params = task
        problem_name = _problem_name(problem_file)
        task_problem_name.append(problem_name)
        parallel_runs = _can_parallelize_runs(runner_name, run_params)
        run_workers = min(n_workers, max(1, int(params.runs))) if parallel_runs else 1
        task_run_limit.append(run_workers)
        print(
            f"Task {task_index}/{len(tasks)}: {algorithm_label} / {problem_name} "
            f"using up to {run_workers} worker(s) across {len(run_indices)} run(s)"
        )

    if all(not _can_parallelize_runs(runner_name, run_params) for _, _, _, runner_name, run_params in tasks):
        _run_tasks_in_current_process(
            tasks=tasks,
            run_indices=run_indices,
            params=params,
            task_problem_name=task_problem_name,
        )
        return

    in_flight: list[tuple[AsyncResult, int, int]] = []
    dispatch_cursor = 0
    with multiprocessing.Pool(processes=n_workers) as pool:
        while True:
            while len(in_flight) < n_workers:
                dispatch_task_index = _next_dispatchable_task(
                    pending_by_task=task_pending_runs,
                    active_by_task=task_active_runs,
                    limit_by_task=task_run_limit,
                    start_index=dispatch_cursor,
                )
                if dispatch_task_index is None:
                    break
                run_index = task_pending_runs[dispatch_task_index].pop(0)
                problem_file, problem_index, algorithm_label, runner_name, run_params = tasks[dispatch_task_index]
                result = pool.apply_async(
                    _execute_task_run,
                    args=(problem_file, problem_index, algorithm_label, runner_name, run_params, run_index),
                )
                task_active_runs[dispatch_task_index] += 1
                in_flight.append((result, dispatch_task_index, run_index))
                dispatch_cursor = (dispatch_task_index + 1) % max(1, len(tasks))

            if not in_flight:
                break

            completed_any = False
            remaining: list[tuple[AsyncResult, int, int]] = []
            for result, dispatch_task_index, _run_index in in_flight:
                if result.ready():
                    result.get()
                    completed_any = True
                    task_active_runs[dispatch_task_index] -= 1
                    if (
                        not task_pending_runs[dispatch_task_index]
                        and task_active_runs[dispatch_task_index] == 0
                        and not task_finalized[dispatch_task_index]
                    ):
                        _problem_file, problem_index, algorithm_label, _runner_name, _run_params = tasks[
                            dispatch_task_index
                        ]
                        _write_grouped_run_hv_summary(
                            params=params,
                            algorithm_label=algorithm_label,
                            problem_name=task_problem_name[dispatch_task_index],
                            problem_index=problem_index,
                        )
                        task_finalized[dispatch_task_index] = True
                else:
                    remaining.append((result, dispatch_task_index, _run_index))
            in_flight = remaining

            if not completed_any:
                in_flight[0][0].wait(timeout=0.05)

    for task_index, task in enumerate(tasks):
        if task_finalized[task_index]:
            continue
        _problem_file, problem_index, algorithm_label, _runner_name, _run_params = task
        _write_grouped_run_hv_summary(
            params=params,
            algorithm_label=algorithm_label,
            problem_name=task_problem_name[task_index],
            problem_index=problem_index,
        )
        task_finalized[task_index] = True


def run_nmopso_ablation(project_root: Path, params: BenchmarkParams) -> None:
    extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    extra["ablationStudy"] = True
    extra["legacyPathRunner"] = True
    params = replace(params, extra=extra)
    seed_everything(params.seed)
    problems_dir = project_root / "problems"
    problem_files = [
        path for path in sorted(problems_dir.glob("*.mat")) if _fleet_from_problem_name(_problem_name(path)) is None
    ]
    ensure_dir(params.results_dir)
    runner = _ALGORITHM_REGISTRY.get("NMOPSO")
    if runner is None:
        raise RuntimeError("NMOPSO is not registered in the algorithm registry.")
    for problem_index, problem_file in enumerate(problem_files, start=1):
        terrain = load_terrain_struct(problem_file)
        terrain["safeDist"] = params.safe_dist
        terrain["droneSize"] = params.drone_size
        terrain = validate_terrain_model(terrain, context=str(problem_file))
        name = _problem_name(problem_file)
        run_params = replace(params, problem_name=name, problem_index=problem_index)
        runner(terrain, run_params)
