from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.multi_uav import run_multi_nsga3
from uav_benchmark.core.chromosome import Chromosome
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import f_operator, tournament_selection
from uav_benchmark.core.nsga3_ops import environmental_selection_nsga3, uniform_point
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir


def run_nsga3(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    if str(params.mode).lower() == "multi":
        return run_multi_nsga3(model, params)
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
    reference_method = str(params.extra.get("refPointMethod", "")).strip() or "NBI"
    reference_points, adjusted_population = uniform_point(params.population, objective_count, reference_method)
    if adjusted_population != params.population:
        from dataclasses import replace
        params = replace(params, population=adjusted_population)
    use_constraints = bool(params.extra.get("useConstraints", False))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    for run_index in range(1, params.runs + 1):
        run_score = _nsga3_single_run(
            model=model,
            params=params,
            run_index=run_index,
            objective_count=objective_count,
            boundary=boundary,
            reference_points=reference_points,
            results_path=results_path,
            use_constraints=use_constraints,
        )
        if params.compute_metrics and run_score.size == 2:
            run_scores[run_index - 1] = run_score
    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def _constraint_vector(population: list[Chromosome]) -> np.ndarray:
    return np.array([candidate.cons for candidate in population], dtype=float)


def _zmin(objective_matrix: np.ndarray, constraint_vector: np.ndarray | None) -> np.ndarray:
    if constraint_vector is None:
        return np.min(objective_matrix, axis=0)
    feasible = constraint_vector <= 0
    if np.any(feasible):
        return np.min(objective_matrix[feasible], axis=0)
    return np.min(objective_matrix, axis=0)


def _nsga3_single_run(
    model: dict[str, Any],
    params: BenchmarkParams,
    run_index: int,
    objective_count: int,
    boundary: np.ndarray,
    reference_points: np.ndarray,
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

    objective_matrix = np.array([candidate.objs for candidate in population], dtype=float)
    constraint_vector = _constraint_vector(population) if use_constraints else None
    zmin = _zmin(objective_matrix, constraint_vector)

    hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    for generation in range(1, params.generations + 1):
        if use_constraints:
            cv = _constraint_vector(population)
        else:
            cv = np.zeros(params.population, dtype=float)
        mating_pool = tournament_selection(2, params.population, cv)
        offspring = f_operator(population, mating_pool, boundary, model)
        offspring_objective = np.array([candidate.objs for candidate in offspring], dtype=float)
        offspring_constraint = _constraint_vector(offspring) if use_constraints else None
        zmin = np.minimum(zmin, _zmin(offspring_objective, offspring_constraint))
        population = environmental_selection_nsga3(
            list(population) + list(offspring),
            params.population,
            reference_points,
            zmin,
            use_constraints=use_constraints,
        )

        if params.compute_metrics:
            objective_matrix = np.array([candidate.objs for candidate in population], dtype=float)
            if generation == 1 or generation == params.generations or generation % 50 == 0:
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
    for candidate_index, candidate in enumerate(population, start=1):
        save_bp(run_dir / f"bp_{candidate_index}.mat", candidate.path, candidate.objs)
    feasible_count = int(np.sum(np.all(np.isfinite(objective_matrix), axis=1)))
    save_mat(
        run_dir / "run_stats.mat",
        {
            "runtimeSec": float(time.perf_counter() - run_start),
            "feasibleCount": feasible_count,
            "solutionCount": int(objective_matrix.shape[0]),
        },
    )

    if not params.compute_metrics:
        return np.zeros(0, dtype=float)
    return np.array(
        [
            cal_metric(1, objective_matrix, params.problem_index, objective_count),
            cal_metric(2, objective_matrix, params.problem_index, objective_count),
        ],
        dtype=float,
    )
