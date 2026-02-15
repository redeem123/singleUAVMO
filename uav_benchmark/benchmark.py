from __future__ import annotations

import multiprocessing
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uav_benchmark.algorithms import (
    run_ctmea,
    run_momfea,
    run_momfea2,
    run_mopso,
    run_nmopso,
    run_nsga2,
    run_nsga3,
    run_rl_nmopso,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.problem_generation.generate import save_multi_uav_scenarios
from uav_benchmark.utils.random import seed_everything

AlgorithmRunner = Callable[[dict, BenchmarkParams], Any]

# Lookup table so workers can resolve algorithm runners by name
# (function objects are not always picklable across processes).
_RUNNER_BY_NAME: dict[str, AlgorithmRunner] = {
    "NMOPSO": run_nmopso,
    "RL-NMOPSO": run_rl_nmopso,
    "MOPSO": run_mopso,
    "NSGA-II": run_nsga2,
    "NSGA-III": run_nsga3,
    "MO-MFEA": run_momfea,
    "MO-MFEA-II": run_momfea2,
    "CTM-EA": run_ctmea,
}

_ALGORITHM_SEED_OFFSET: dict[str, int] = {
    "NMOPSO": 11,
    "RL-NMOPSO": 19,
    "MOPSO": 23,
    "NSGA-II": 37,
    "NSGA-III": 41,
    "MO-MFEA": 53,
    "MO-MFEA-II": 67,
    "CTM-EA": 79,
}


def _problem_name(problem_file: Path) -> str:
    name = problem_file.stem
    if name.startswith("terrainStruct_"):
        name = name.replace("terrainStruct_", "", 1)
    return name


def _algorithm_map() -> list[tuple[str, AlgorithmRunner]]:
    return [
        ("NMOPSO", run_nmopso),
        ("MOPSO", run_mopso),
        ("NSGA-II", run_nsga2),
        ("NSGA-III", run_nsga3),
    ]


def _algorithm_map_for_mode(mode: str) -> list[tuple[str, AlgorithmRunner]]:
    if str(mode).lower() == "multi":
        return [
            ("RL-NMOPSO", run_rl_nmopso),
            ("NMOPSO", run_nmopso),
            ("MOPSO", run_mopso),
            ("NSGA-II", run_nsga2),
            ("NSGA-III", run_nsga3),
        ]
    return _algorithm_map()


def _fleet_from_problem_name(problem_name: str) -> int | None:
    match = re.search(r"_uav(\d+)$", problem_name)
    if not match:
        return None
    return int(match.group(1))


def _base_problem_name(problem_name: str) -> str:
    return re.sub(r"_uav\d+$", "", problem_name)


def _execute_task(
    problem_file: Path,
    problem_index: int,
    algorithm_name: str,
    params: BenchmarkParams,
) -> None:
    """Worker function executed in a child process."""
    # Each worker gets its own reproducible seed
    base_seed = int(params.seed) if params.seed is not None else 0
    seed_offset = _ALGORITHM_SEED_OFFSET.get(algorithm_name, 0)
    seed_everything(base_seed + problem_index * 100 + seed_offset)

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
    )

    runner = _RUNNER_BY_NAME[algorithm_name]
    algo_params = replace(
        run_params,
        results_dir=params.results_dir / algorithm_name,
        algorithm=algorithm_name,
    )
    ensure_dir(algo_params.results_dir)
    print(f"[PID {os.getpid()}] Starting {algorithm_name} / {name}")
    runner(terrain, algo_params)
    print(f"[PID {os.getpid()}] Finished {algorithm_name} / {name}")


def run_benchmark(project_root: Path, params: BenchmarkParams) -> None:
    problems_dir = project_root / "problems"
    all_problem_files = sorted(problems_dir.glob("*.mat"))
    mode = str(params.mode).lower()
    problem_files = all_problem_files
    if mode == "single":
        problem_files = [path for path in all_problem_files if "_uav" not in path.stem]
    else:
        fleet_sizes = params.fleet_sizes or ((int(params.fleet_size),) if int(params.fleet_size) > 1 else (3, 5, 8))
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
            save_multi_uav_scenarios(
                project_root=project_root,
                base_problem_names=base_names,
                fleet_sizes=tuple(int(size) for size in fleet_sizes),
                seed=int(params.seed) if params.seed is not None else 0,
                separation_min=float(params.separation_min),
                mission_prefix="paper_medium",
            )
            all_problem_files = sorted(problems_dir.glob("*.mat"))
        suffixes = tuple(f"_uav{int(size)}" for size in fleet_sizes)
        problem_files = [path for path in all_problem_files if any(path.stem.endswith(suffix) for suffix in suffixes)]
        if not problem_files:
            problem_files = [path for path in all_problem_files if "_uav" in path.stem]
    ensure_dir(params.results_dir)

    # Build task list: all (problem, algorithm) combinations
    tasks: list[tuple[Path, int, str, BenchmarkParams]] = []
    algo_map = _algorithm_map_for_mode(mode)
    for problem_index, problem_file in enumerate(problem_files, start=1):
        problem_name = _problem_name(problem_file)
        base_problem = _base_problem_name(problem_name)
        run_params = replace(params, problem_name=base_problem)
        for algorithm_name, _runner in algo_map:
            tasks.append((problem_file, problem_index, algorithm_name, run_params))

    if not tasks:
        print("No benchmark tasks found for the selected mode/scenario settings.")
        return

    n_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"Running {len(tasks)} tasks across {n_workers} workers")

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.starmap(_execute_task, tasks)


def run_nmopso_ablation(project_root: Path, params: BenchmarkParams) -> None:
    seed_everything(params.seed)
    problems_dir = project_root / "problems"
    problem_files = sorted(problems_dir.glob("*.mat"))
    ensure_dir(params.results_dir)
    for problem_index, problem_file in enumerate(problem_files, start=1):
        terrain = load_terrain_struct(problem_file)
        terrain["safeDist"] = params.safe_dist
        terrain["droneSize"] = params.drone_size
        name = _problem_name(problem_file)
        run_params = replace(params, problem_name=name, problem_index=problem_index)
        run_nmopso(terrain, run_params)
