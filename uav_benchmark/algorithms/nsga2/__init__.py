from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.shared.fleet_runner import run_fleet_nsga2, _resolve_run_indices
from uav_benchmark.algorithms.shared.mission_stats import build_mission_stats
from uav_benchmark.core.chromosome import Chromosome
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import environmental_selection, f_operator, tournament_selection
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir


def run_nsga2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    use_legacy_runner = bool(params.extra.get("legacyPathRunner", False))
    if (not use_legacy_runner) or int(params.fleet_size) > 1:
        return run_fleet_nsga2(model, params)
    objective_count = 4
    model = dict(model)
    model["n"] = 10
    boundary = np.array(
        [
            [float(model["xmax"]), float(model["ymax"]), float(model["zmax"])],
            [float(model["xmin"]), float(model["ymin"]), float(model["zmin"])],
        ],
        dtype=float,
    )

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    use_constraints = bool(params.extra.get("useConstraints", False))

    for run_index in _resolve_run_indices(params):
        run_score = _nsga2_legacy_run(
            model=model,
            params=params,
            run_index=run_index,
            objective_count=objective_count,
            boundary=boundary,
            results_path=results_path,
            use_constraints=use_constraints,
        )
        if params.compute_metrics and run_score.size == 2:
            run_scores[run_index - 1] = run_score
    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def _nsga2_legacy_run(
    model: dict[str, Any],
    params: BenchmarkParams,
    run_index: int,
    objective_count: int,
    boundary: np.ndarray,
    results_path: Path,
    use_constraints: bool,
) -> np.ndarray:
    run_start = time.perf_counter()
    population = []
    for _ in range(params.population):
        chromosome = Chromosome.new(model)
        chromosome.initialize(model)
        chromosome.evaluate(model)
        population.append(chromosome)
    population, front_no, crowding = environmental_selection(
        population,
        params.population,
        objective_count,
        use_constraints=use_constraints,
    )

    hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    for generation in range(1, params.generations + 1):
        mating_pool = tournament_selection(2, params.population, front_no, -crowding)
        offspring = f_operator(population, mating_pool, boundary, model)
        merged = list(population) + list(offspring)
        population, front_no, crowding = environmental_selection(
            merged,
            params.population,
            objective_count,
            use_constraints=use_constraints,
        )

        if params.compute_metrics:
            objective_matrix = np.array([candidate.objs for candidate in population], dtype=float)
            metric_interval = int(params.extra.get("metricInterval", 20))
            if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                hv_history[generation - 1, 0] = cal_metric(1, objective_matrix, params.problem_index, objective_count)
                hv_history[generation - 1, 1] = cal_metric(2, objective_matrix, params.problem_index, objective_count)
            elif generation > 1:
                hv_history[generation - 1] = hv_history[generation - 2]

    run_dir = results_path / f"Run_{run_index}"
    ensure_dir(run_dir)
    if params.compute_metrics:
        save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})

    objective_matrix = np.array([candidate.objs for candidate in population], dtype=float)
    save_run_popobj(run_dir / "final_popobj.mat", objective_matrix, params.problem_index, objective_count)
    saved_paths: list[np.ndarray] = []
    for candidate_index, candidate in enumerate(population, start=1):
        saved_paths.append(np.asarray(candidate.path, dtype=float))
        save_bp(run_dir / f"bp_{candidate_index}.mat", candidate.path, candidate.objs)
    if saved_paths:
        finite_cost = np.where(np.isfinite(objective_matrix), objective_matrix, 1e9)
        best_idx = int(np.argmin(np.sum(finite_cost, axis=1)))
        save_mat(run_dir / "fleet_paths.mat", {"uav1": np.asarray(saved_paths[best_idx], dtype=float)})
    mission_stats, feasible_mask = build_mission_stats(saved_paths, model)
    save_mat(run_dir / "mission_stats.mat", mission_stats)
    feasible_count = int(np.sum(feasible_mask))
    run_stats = {
        "runtimeSec": float(time.perf_counter() - run_start),
        "feasibleCount": feasible_count,
        "solutionCount": int(objective_matrix.shape[0]),
    }
    save_mat(run_dir / "run_stats.mat", run_stats)

    from uav_benchmark.io.results import save_run_summary_json
    save_run_summary_json(run_dir / "run_summary.json", params, run_stats)

    if not params.compute_metrics:
        return np.zeros(0, dtype=float)
    return np.array(
        [
            cal_metric(1, objective_matrix, params.problem_index, objective_count),
            cal_metric(2, objective_matrix, params.problem_index, objective_count),
        ],
        dtype=float,
    )
