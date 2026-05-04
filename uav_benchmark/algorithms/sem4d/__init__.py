from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_benchmark.algorithms.sem4d.core import (
    _build_aux_model,
    _candidate_from_paths,
    _extra_int,
    _shield_config,
)
from uav_benchmark.algorithms.sem4d.evolution import (
    _evaluate_initial_population,
    _make_offspring,
    _save_shield_artifact,
    _select_candidates,
    _select_individuals,
    _target_evaluate_population,
    _task_ids_for_fleet,
)
from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _ensure_fleet_endpoints,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
)
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.utils.random import seed_everything


def run_sem4d(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """Run SEM-4D: shielded evolutionary multitasking for fleet path planning."""
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
    model["safeDist"] = float(params.safe_dist)
    model["droneSize"] = float(params.drone_size)
    model["maxTurnDeg"] = float(params.max_turn_deg)
    model["hardCollisionConstraint"] = True
    aux_model = _build_aux_model(model, params)
    config = _shield_config(model, params)
    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dimensions = int(lower.size)
    task_ids = _task_ids_for_fleet(fleet_size, params)
    metric_interval = max(1, _extra_int(params, "metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    run_indices = _resolve_run_indices(params)
    resume_existing = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in run_indices:
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing:
            resumed = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resumed is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resumed
                continue

        run_start = time.perf_counter()
        seed_everything(seed_value + run_idx)
        vectors = np.random.uniform(lower, upper, size=(params.population, dimensions))
        initial_tasks = np.resize(task_ids, params.population)
        np.random.shuffle(initial_tasks)
        population = _evaluate_initial_population(
            vectors=vectors,
            task_ids=initial_tasks,
            model=model,
            aux_model=aux_model,
            params=params,
            config=config,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
        )
        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )

        for generation in range(1, params.generations + 1):
            child_vectors, child_tasks = _make_offspring(
                population=population,
                lower=lower,
                upper=upper,
                fleet_size=fleet_size,
                n_waypoints=n_waypoints,
                task_ids=task_ids,
                params=params,
            )
            children = _evaluate_initial_population(
                vectors=child_vectors,
                task_ids=child_tasks,
                model=model,
                aux_model=aux_model,
                params=params,
                config=config,
                fleet_size=fleet_size,
                n_waypoints=n_waypoints,
            )
            population = _select_individuals(population + children, params.population)

            if params.compute_metrics:
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    target_candidates = _target_evaluate_population(
                        population,
                        model=model,
                        aux_model=aux_model,
                        params=params,
                        config=config,
                        fleet_size=fleet_size,
                        n_waypoints=n_waypoints,
                    )
                    matrix = _candidate_matrix(target_candidates)
                    hv_history[generation - 1, 0] = cal_metric(1, matrix, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, matrix, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        final_candidates = _target_evaluate_population(
            population,
            model=model,
            aux_model=aux_model,
            params=params,
            config=config,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
        )
        final_candidates = _select_candidates(final_candidates, params.population)

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})

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
                "algorithmName": str(params.algorithm or "SEM-4D"),
                "sem4dShieldIterations": float(config.iterations),
                "sem4dTimeSamples": float(config.time_samples),
                "sem4dTaskCount": float(task_ids.size),
                "sem4dEnergyMax": float(config.energy_max) if config.energy_max is not None else 0.0,
            },
        )
        _save_shield_artifact(run_dir, final_candidates)

        if params.compute_metrics:
            final_matrix = _candidate_matrix(final_candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_matrix, params.problem_index, objective_count),
                    cal_metric(2, final_matrix, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


__all__ = ["_candidate_from_paths", "_shield_config", "run_sem4d"]
