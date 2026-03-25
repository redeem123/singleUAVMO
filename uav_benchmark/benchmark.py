from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uav_benchmark.algorithms import REGISTRY as _ALGORITHM_REGISTRY
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import load_mat, load_terrain_struct, save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.problem_generation.generate import save_fleet_scenarios
from uav_benchmark.utils.random import seed_everything

AlgorithmRunner = Callable[[dict, BenchmarkParams], Any]
_DEFAULT_MAX_WORKERS = 14

_ALGORITHM_SEED_OFFSET: dict[str, int] = {
    "NMOPSO": 11,
    "MOPSO": 23,
    "SMPSO": 29,
    "NSGA-II": 37,
    "NSGA-III": 41,
    "MOEAD": 43,
    "SPEA2": 47,
    "MFO-SPEA2": 61,
    "GCNMOEA": 71,
    "CMOSMA": 59,
    "FASTR-MOEA": 73,
    "TACTIC-MOEA": 79,
    "MO-MFEA": 53,
    "MO-MFEA-II": 67,
    "MOGWO": 83,
    "MOGWO-NO-ATTENTION": 89,
    "MOGWO-STANDARD-GWO": 101,
    "APEX-SHADE": 97,
    "TSKAC-NSGA-II": 107,
}

def _seed_for_task(base_seed: int, problem_index: int, algorithm_name: str) -> int:
    return int(base_seed) + int(problem_index) * 100 + int(_ALGORITHM_SEED_OFFSET.get(algorithm_name, 0))


def _seed_for_run(base_seed: int, problem_index: int, algorithm_name: str, run_index: int) -> int:
    return _seed_for_task(base_seed, problem_index, algorithm_name) + int(run_index)


def _can_parallelize_runs(algorithm_name: str, params: BenchmarkParams) -> bool:
    # Per-run dispatch uses isolated worker processes (multiprocessing.Pool), so all
    # algorithms are safe to parallelize at the run level. Stateful algorithms (e.g.,
    # RL-NMOPSO) rebuild their controller fresh per worker process call. Always True.
    return True


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


def _max_parallel_worker_slots(tasks: list[tuple[Path, int, str, BenchmarkParams]]) -> int:
    """Upper bound on run-level concurrency across all tasks."""
    slots = 0
    for _problem_file, _problem_index, algorithm_name, run_params in tasks:
        if _can_parallelize_runs(algorithm_name, run_params):
            slots += max(1, int(run_params.runs))
        else:
            slots += 1
    return max(1, int(slots))


def _safe_module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
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
    except Exception:
        pass
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
    problem_files: list[Path],
    algorithms: tuple[str, ...],
    n_workers: int,
    created_utc: str | None = None,
) -> dict[str, Any]:
    created = created_utc or datetime.now(timezone.utc).isoformat()
    base_seed = int(params.seed) if params.seed is not None else 42
    problems = [_problem_name(path) for path in problem_files]
    task_plan: list[dict[str, Any]] = []
    for problem_index, problem in enumerate(problems, start=1):
        fleet_size = _fleet_from_problem_name(problem) or int(params.fleet_size)
        for algorithm in algorithms:
            task_plan.append(
                {
                    "problem": problem,
                    "problemIndex": int(problem_index),
                    "algorithm": str(algorithm),
                    "fleetSize": int(fleet_size),
                    "seedOffset": int(_ALGORITHM_SEED_OFFSET.get(algorithm, 0)),
                    "effectiveSeed": int(_seed_for_task(base_seed, problem_index, algorithm)),
                }
            )
    plan_payload = {
        "parameters": _params_manifest(params),
        "fleetSizesResolved": [int(size) for size in fleet_sizes],
        "problemsResolved": problems,
        "algorithmsResolved": list(algorithms),
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
    problem_files: list[Path],
    algorithms: tuple[str, ...],
    n_workers: int,
) -> Path:
    manifest = _build_benchmark_manifest(
        project_root=project_root,
        params=params,
        fleet_sizes=fleet_sizes,
        problem_files=problem_files,
        algorithms=algorithms,
        n_workers=n_workers,
    )
    ensure_dir(params.results_dir)
    manifest_path = params.results_dir / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _normalize_algorithm_name(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"nmopso"}:
        return "NMOPSO"
    if key in {"mopso"}:
        return "MOPSO"
    if key in {"smpso", "sm-pso", "sm_pso"}:
        return "SMPSO"
    if key in {"nsga-ii", "nsga2", "nsga_ii"}:
        return "NSGA-II"
    if key in {"nsga-iii", "nsga3", "nsga_iii"}:
        return "NSGA-III"
    if key in {"moead", "moea/d", "moea-d", "moea_d"}:
        return "MOEAD"
    if key in {"spea2", "spea-2", "spea_2"}:
        return "SPEA2"
    if key in {"mfo-spea2", "mfospea2", "mfo_spea2", "mfo-spea-2"}:
        return "MFO-SPEA2"
    if key in {"gcnmoea", "gcn-moea", "gcn_moea"}:
        return "GCNMOEA"
    if key in {"cmosma", "cmo-sma", "cmo_sma"}:
        return "CMOSMA"
    if key in {"fastr-moea", "fastr_moea", "fastrmoea"}:
        return "FASTR-MOEA"
    if key in {"tactic-moea", "tactic_moea", "tacticmoea"}:
        return "TACTIC-MOEA"
    if key in {"mo-mfea", "momfea"}:
        return "MO-MFEA"
    if key in {"mo-mfea-ii", "momfea2", "momfea-ii"}:
        return "MO-MFEA-II"
    if key in {"mogwo", "a2mogwo", "a2-mogwo"}:
        return "MOGWO"
    if key in {
        "mogwo-no-attention",
        "mogwo_no_attention",
        "a2mogwo-no-attention",
        "a2mogwo_no_attention",
        "a2-mogwo-no-attention",
        "a2mogwo-noattention",
    }:
        return "MOGWO-NO-ATTENTION"
    if key in {
        "mogwo-standard-gwo",
        "mogwo_standard_gwo",
        "a2mogwo-standard-gwo",
        "a2mogwo_standard_gwo",
        "a2-mogwo-standard-gwo",
    }:
        return "MOGWO-STANDARD-GWO"
    if key in {"apex-shade", "apexshade", "apex_shade"}:
        return "APEX-SHADE"
    if key in {
        "tskac-nsga-ii",
        "tskac_nsga_ii",
        "tskacnsga2",
        "tskac-nsga2",
        "tskac-nsgaii",
        "tskacnsgaii",
    }:
        return "TSKAC-NSGA-II"
    return str(name).strip()


def _requested_algorithms(extra: dict[str, Any]) -> tuple[str, ...]:
    raw = extra.get("algorithms")
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    else:
        return ()
    normalized = [_normalize_algorithm_name(token) for token in tokens]
    if not normalized:
        return ()
    # Keep order while removing duplicates
    return tuple(dict.fromkeys(normalized))


def _requested_problem_names(extra: dict[str, Any]) -> tuple[str, ...]:
    raw = extra.get("problemNames")
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    else:
        return ()
    if not tokens:
        return ()
    return tuple(dict.fromkeys(tokens))


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


def _problem_name(problem_file: Path) -> str:
    name = problem_file.stem
    if name.startswith("terrainStruct_"):
        name = name.replace("terrainStruct_", "", 1)
    return name


def _algorithm_map(include_algorithms: tuple[str, ...] = ()) -> list[tuple[str, AlgorithmRunner]]:
    _known_order = [
        "NMOPSO",
        "MOPSO",
        "SMPSO",
        "NSGA-II",
        "NSGA-III",
        "MOEAD",
        "SPEA2",
        "MFO-SPEA2",
        "GCNMOEA",
        "CMOSMA",
        "FASTR-MOEA",
        "TACTIC-MOEA",
        "MO-MFEA",
        "MO-MFEA-II",
        "MOGWO",
        "MOGWO-NO-ATTENTION",
        "MOGWO-STANDARD-GWO",
        "APEX-SHADE",
        "TSKAC-NSGA-II",
    ]
    mapping = [(name, _ALGORITHM_REGISTRY[name]) for name in _known_order if name in _ALGORITHM_REGISTRY]
    extra = [(name, runner) for name, runner in _ALGORITHM_REGISTRY.items() if name not in _known_order]
    mapping.extend(sorted(extra, key=lambda item: item[0]))
    if not include_algorithms:
        return mapping
    include_set = set(include_algorithms)
    return [item for item in mapping if item[0] in include_set]


def _fleet_from_problem_name(problem_name: str) -> int | None:
    match = re.search(r"_uav(\d+)$", problem_name)
    if not match:
        return None
    return int(match.group(1))


def _base_problem_name(problem_name: str) -> str:
    return re.sub(r"_uav\d+$", "", problem_name)


def _execute_task_run(
    problem_file: Path,
    problem_index: int,
    algorithm_name: str,
    params: BenchmarkParams,
    run_index: int,
) -> None:
    """Worker function that executes exactly one run index."""
    base_seed = int(params.seed) if params.seed is not None else 42
    seed_everything(_seed_for_run(base_seed, problem_index, algorithm_name, run_index))

    terrain = load_terrain_struct(problem_file)
    terrain["safeDist"] = params.safe_dist
    terrain["droneSize"] = params.drone_size
    terrain["separationMin"] = params.separation_min
    terrain["maxTurnDeg"] = params.max_turn_deg

    name = _problem_name(problem_file)
    fleet_size = _fleet_from_problem_name(name) or int(params.fleet_size)
    run_params = replace(
        params,
        problem_name=name,
        problem_index=problem_index,
        fleet_size=fleet_size,
        run_indices=(int(run_index),),
        write_final_hv=False,
    )

    runner = _ALGORITHM_REGISTRY[algorithm_name]
    algo_params = replace(
        run_params,
        results_dir=params.results_dir / algorithm_name,
        algorithm=algorithm_name,
    )
    ensure_dir(algo_params.results_dir)
    print(f"[PID {os.getpid()}] Starting {algorithm_name} / {name} / Run_{int(run_index)}")
    runner(terrain, algo_params)
    print(f"[PID {os.getpid()}] Finished {algorithm_name} / {name} / Run_{int(run_index)}")


def _write_grouped_run_hv_summary(
    params: BenchmarkParams,
    algorithm_name: str,
    problem_name: str,
    problem_index: int,
) -> None:
    if not params.compute_metrics:
        return
    results_path = params.results_dir / algorithm_name / problem_name
    ensure_dir(results_path)
    scores = np.zeros((params.runs, 2), dtype=float)
    objective_count = 4
    for run_index in range(1, params.runs + 1):
        popobj_path = results_path / f"Run_{run_index}" / "final_popobj.mat"
        if not popobj_path.exists():
            continue
        try:
            data = load_mat(popobj_path)
            matrix_raw = data.get("PopObj")
            matrix = np.asarray(matrix_raw, dtype=float) if matrix_raw is not None else np.zeros((0, 0), dtype=float)
            if matrix.size == 0:
                continue
            scores[run_index - 1, 0] = cal_metric(1, matrix, problem_index, objective_count)
            scores[run_index - 1, 1] = cal_metric(2, matrix, problem_index, objective_count)
        except Exception:
            continue
    save_mat(results_path / "final_hv.mat", {"bestScores": scores})


def run_benchmark(project_root: Path, params: BenchmarkParams) -> None:
    # Resolve seed=None to a random seed so all workers share the same base.
    if params.seed is None:
        import secrets
        import logging
        resolved_seed = secrets.randbelow(2**31)
        logging.getLogger(__name__).info(
            "No --seed provided; using randomly generated seed=%d. "
            "Pass --seed %d to reproduce this run.", resolved_seed, resolved_seed
        )
        params = replace(params, seed=resolved_seed)

    problems_dir = project_root / "problems"
    all_problem_files = sorted(problems_dir.glob("*.mat"))
    raw_fleet_sizes = params.fleet_sizes if params.fleet_sizes else (int(params.fleet_size),)
    fleet_sizes = tuple(dict.fromkeys(max(1, int(size)) for size in raw_fleet_sizes))
    base_names = [
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
    ]
    if params.scenario_set == "paper_medium":
        save_fleet_scenarios(
            project_root=project_root,
            base_problem_names=base_names,
            fleet_sizes=tuple(int(size) for size in fleet_sizes),
            seed=int(params.seed) if params.seed is not None else 42,
            separation_min=float(params.separation_min),
            mission_prefix="paper_medium",
        )
        all_problem_files = sorted(problems_dir.glob("*.mat"))

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
            # Base scenario files (without _uav suffix) represent legacy-path cases.
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
    ensure_dir(params.results_dir)

    # Build task list: all (problem, algorithm) combinations
    tasks: list[tuple[Path, int, str, BenchmarkParams]] = []
    requested = _requested_algorithms(params.extra)
    algo_map = _algorithm_map(requested)
    algo_names = tuple(name for name, _runner in algo_map)
    for problem_index, problem_file in enumerate(problem_files, start=1):
        problem_name = _problem_name(problem_file)
        base_problem = _base_problem_name(problem_name)
        run_params = replace(params, problem_name=base_problem)
        for algorithm_name, _runner in algo_map:
            tasks.append((problem_file, problem_index, algorithm_name, run_params))

    if not tasks:
        manifest_path = _write_benchmark_manifest(
            project_root=project_root,
            params=params,
            fleet_sizes=fleet_sizes,
            problem_files=problem_files,
            algorithms=algo_names,
            n_workers=0,
        )
        print(f"benchmark_manifest={manifest_path}")
        print("No benchmark tasks found for the selected mode/scenario settings.")
        return

    worker_cap = int(params.extra.get("maxWorkers", _DEFAULT_MAX_WORKERS)) if isinstance(params.extra, dict) else _DEFAULT_MAX_WORKERS
    max_parallel_tasks = _max_parallel_worker_slots(tasks)
    if worker_cap > 0:
        n_workers = min(max_parallel_tasks, worker_cap, os.cpu_count() or 1)
    else:
        n_workers = min(max_parallel_tasks, os.cpu_count() or 1)
    manifest_path = _write_benchmark_manifest(
        project_root=project_root,
        params=params,
        fleet_sizes=fleet_sizes,
        problem_files=problem_files,
        algorithms=algo_names,
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
        problem_file, _problem_index, algorithm_name, run_params = task
        problem_name = _problem_name(problem_file)
        task_problem_name.append(problem_name)
        parallel_runs = _can_parallelize_runs(algorithm_name, run_params)
        run_workers = min(n_workers, max(1, int(params.runs))) if parallel_runs else 1
        task_run_limit.append(run_workers)
        print(
            f"Task {task_index}/{len(tasks)}: {algorithm_name} / {problem_name} "
            f"using up to {run_workers} worker(s) across {len(run_indices)} run(s)"
        )

    in_flight: list[tuple[multiprocessing.pool.AsyncResult, int, int]] = []
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
                problem_file, problem_index, algorithm_name, run_params = tasks[dispatch_task_index]
                result = pool.apply_async(
                    _execute_task_run,
                    args=(problem_file, problem_index, algorithm_name, run_params, run_index),
                )
                task_active_runs[dispatch_task_index] += 1
                in_flight.append((result, dispatch_task_index, run_index))
                dispatch_cursor = (dispatch_task_index + 1) % max(1, len(tasks))

            if not in_flight:
                break

            completed_any = False
            remaining: list[tuple[multiprocessing.pool.AsyncResult, int, int]] = []
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
                        _problem_file, problem_index, algorithm_name, _run_params = tasks[dispatch_task_index]
                        _write_grouped_run_hv_summary(
                            params=params,
                            algorithm_name=algorithm_name,
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
        _problem_file, problem_index, algorithm_name, _run_params = task
        _write_grouped_run_hv_summary(
            params=params,
            algorithm_name=algorithm_name,
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
        path
        for path in sorted(problems_dir.glob("*.mat"))
        if _fleet_from_problem_name(_problem_name(path)) is None
    ]
    ensure_dir(params.results_dir)
    runner = _ALGORITHM_REGISTRY.get("NMOPSO")
    if runner is None:
        raise RuntimeError("NMOPSO is not registered in the algorithm registry.")
    for problem_index, problem_file in enumerate(problem_files, start=1):
        terrain = load_terrain_struct(problem_file)
        terrain["safeDist"] = params.safe_dist
        terrain["droneSize"] = params.drone_size
        name = _problem_name(problem_file)
        run_params = replace(params, problem_name=name, problem_index=problem_index)
        runner(terrain, run_params)
