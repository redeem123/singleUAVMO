from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from uav_benchmark.algorithms.shared.fleet_artifacts import _save_fleet_artifacts
from uav_benchmark.algorithms.shared.fleet_common import (
    _build_bounds,
    _ensure_fleet_endpoints,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _should_write_final_hv,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort, tournament_selection
from uav_benchmark.core.nsga3_ops import environmental_selection_nsga3, uniform_point
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.utils.random import seed_everything


@dataclass(slots=True)
class _NSGA3Candidate:
    objs: np.ndarray
    cons: float
    index: int


def _sbx_mutation(parents: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    n_parents, n_dims = parents.shape
    if n_parents % 2 == 1:
        parents = np.vstack([parents, parents[np.random.randint(0, n_parents)]])
        n_parents += 1
    half = n_parents // 2
    p1 = parents[:half]
    p2 = parents[half:]
    dis_c = 20.0
    pro_m = 1.0 / max(1, n_dims)

    mu = np.random.rand(*p1.shape)
    beta = np.where(
        mu <= 0.5,
        (2.0 * mu) ** (1.0 / (dis_c + 1.0)),
        (2.0 - 2.0 * mu) ** (-1.0 / (dis_c + 1.0)),
    )
    beta *= np.where(np.random.rand(*beta.shape) < 0.5, 1.0, -1.0)
    c1 = (p1 + p2) * 0.5 + beta * (p1 - p2) * 0.5
    c2 = (p1 + p2) * 0.5 - beta * (p1 - p2) * 0.5
    offspring = np.vstack([c1, c2])
    mutation_mask = np.random.rand(*offspring.shape) < pro_m
    mutation = np.random.normal(0.0, 1.0, size=offspring.shape) * 0.05 * (upper - lower)
    offspring = np.where(mutation_mask, offspring + mutation, offspring)
    return np.clip(offspring, lower, upper)


def _run_fleet_nsga2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix

    objective_count = 4
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
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)
    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dimensions = int(lower.size)
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    run_indices = _resolve_run_indices(params)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in run_indices:
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
        run_start = time.perf_counter()
        seed_everything(seed_value + run_idx)
        population = np.random.uniform(lower, upper, size=(params.population, dimensions))
        candidates = _evaluate_population(population, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )

        for generation in range(1, params.generations + 1):
            obj = _candidate_matrix(candidates)
            front_no, _ = n_d_sort(obj.copy(), None, params.population)
            crowd = crowding_distance(obj, front_no)
            mating = tournament_selection(2, params.population, front_no, -crowd)
            offspring = _sbx_mutation(population[mating], lower, upper)
            off_candidates = _evaluate_population(offspring, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

            merged_vectors = np.vstack([population, offspring])
            merged_candidates = candidates + off_candidates
            merged_obj = _candidate_matrix(merged_candidates)
            merged_front, _ = n_d_sort(merged_obj.copy(), None, params.population)
            merged_crowd = crowding_distance(merged_obj, merged_front)

            selected = []
            for front in np.unique(merged_front[np.isfinite(merged_front)]):
                idx = np.where(merged_front == front)[0]
                if len(selected) + len(idx) <= params.population:
                    selected.extend(idx.tolist())
                else:
                    order = idx[np.argsort(-merged_crowd[idx])]
                    need = params.population - len(selected)
                    selected.extend(order[:need].tolist())
                    break
            selected = np.asarray(selected, dtype=int)
            population = merged_vectors[selected]
            candidates = [merged_candidates[int(i)] for i in selected]

            if params.compute_metrics:
                final_obj = _candidate_matrix(candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, final_obj, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, final_obj, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        final_candidates = candidates
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            rl_trace=None,
            run_metadata={
                "algorithmName": str(params.algorithm or "NSGA-II"),
                "optimizerBackend": "NSGA-II native Python fleet optimizer",
                "pythonProblemEvaluation": True,
                "benchmarkObjectiveDuringSearch": True,
                "nativePopulationLoop": True,
                "nativeGenerationLoop": True,
                "constraintHandling": "Deb feasibility-first via shared constraint vector",
            },
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(final_candidates)
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


# ═══════════════════════════════════════════════════════════════════
# NSGA-III runner
# ═══════════════════════════════════════════════════════════════════


def _run_fleet_nsga3(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix, _finite_min

    objective_count = 4
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
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)
    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dimensions = int(lower.size)
    metric_interval = int(params.extra.get("metricInterval", 20))

    reference_method = str(params.extra.get("refPointMethod", "")).strip() or "NBI"
    reference_points, adjusted_population = uniform_point(params.population, objective_count, reference_method)
    population_size = int(adjusted_population)

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    run_indices = _resolve_run_indices(params)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in run_indices:
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
        run_start = time.perf_counter()
        seed_everything(seed_value + run_idx)
        population = np.random.uniform(lower, upper, size=(population_size, dimensions))
        candidates = _evaluate_population(population, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        obj = _candidate_matrix(candidates)
        zmin = _finite_min(obj)
        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )

        for generation in range(1, params.generations + 1):
            constraints = np.zeros(population_size, dtype=float)
            mating = tournament_selection(2, population_size, constraints)
            offspring = _sbx_mutation(population[mating], lower, upper)
            off_candidates = _evaluate_population(offspring, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

            merged_vectors = np.vstack([population, offspring])
            merged_candidates = candidates + off_candidates
            merged_obj = _candidate_matrix(merged_candidates)
            if merged_obj.size > 0:
                zmin = np.minimum(zmin, _finite_min(merged_obj))

            wrapped = [_NSGA3Candidate(objs=merged_obj[idx], cons=0.0, index=idx) for idx in range(merged_obj.shape[0])]
            selected_wrapped = environmental_selection_nsga3(
                cast(Any, wrapped),
                population_size,
                reference_points,
                zmin,
                use_constraints=False,
            )
            selected = np.asarray([item.index for item in cast(Any, selected_wrapped)], dtype=int)
            if selected.size < population_size:
                remainder = np.setdiff1d(np.arange(merged_vectors.shape[0], dtype=int), selected, assume_unique=False)
                if remainder.size > 0:
                    need = population_size - selected.size
                    fill = remainder[:need]
                    selected = np.hstack([selected, fill])
            elif selected.size > population_size:
                selected = selected[:population_size]

            population = merged_vectors[selected]
            candidates = [merged_candidates[int(idx)] for idx in selected]

            if params.compute_metrics:
                final_obj = _candidate_matrix(candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, final_obj, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, final_obj, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            rl_trace=None,
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


def run_fleet_nsga2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_nsga2(model=model, params=params)


def run_fleet_nsga3(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_nsga3(model=model, params=params)
