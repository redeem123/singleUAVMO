from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
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
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort, tournament_selection
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


@dataclass(slots=True)
class MTIndividual:
    vector: np.ndarray
    objective: np.ndarray
    task_id: int


def _build_aux_model(model: dict[str, Any], params: BenchmarkParams) -> dict[str, Any]:
    aux_model = dict(model)
    safe_dist_scale = float(params.extra.get("mfeaAuxSafeDistScale", 0.5))
    if "safeDist" in aux_model and aux_model["safeDist"] is not None:
        aux_model["safeDist"] = max(1.0, float(aux_model["safeDist"]) * safe_dist_scale)
    elif "safe_dist" in aux_model and aux_model["safe_dist"] is not None:
        aux_model["safe_dist"] = max(1.0, float(aux_model["safe_dist"]) * safe_dist_scale)
    else:
        aux_model["safeDist"] = 10.0

    nofly_scale = float(params.extra.get("mfeaAuxNoFlyScale", 0.8))
    if "nofly_r" in aux_model and aux_model["nofly_r"] is not None:
        aux_model["nofly_r"] = np.asarray(aux_model["nofly_r"], dtype=float) * nofly_scale
    return aux_model


def _evaluate_task_candidates(
    vectors: np.ndarray,
    task_id: int,
    model: dict[str, Any],
    aux_model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> list[Candidate]:
    """Evaluate a batch of decision vectors for a given task using the canonical fleet evaluator."""
    active_model = model if task_id == 2 else aux_model
    return _evaluate_population(vectors, active_model, fleet_size=fleet_size, n_waypoints=n_waypoints)


def _make_offspring(
    population: list[MTIndividual],
    model: dict[str, Any],
    aux_model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    crossover_rate: float,
    mutation_std: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[MTIndividual]:
    pop_obj = np.array([item.objective for item in population], dtype=float)
    front_no, _ = n_d_sort(pop_obj.copy(), None, len(population))
    mating_pool = tournament_selection(2, len(population), front_no)
    offspring: list[MTIndividual] = []
    for pair_index in range(0, len(mating_pool), 2):
        p1 = population[int(mating_pool[pair_index])]
        p2 = population[int(mating_pool[(pair_index + 1) % len(mating_pool)])]
        alpha = np.random.rand(p1.vector.shape[0])
        if np.random.rand() < crossover_rate:
            child_vec = alpha * p1.vector + (1.0 - alpha) * p2.vector
        else:
            child_vec = p1.vector.copy()
        child_vec += np.random.randn(*child_vec.shape) * mutation_std * (upper - lower)
        child_vec = np.clip(child_vec, lower, upper)
        if p1.task_id == p2.task_id:
            child_task = p1.task_id
        else:
            child_task = int(np.random.choice([p1.task_id, p2.task_id]))
        offspring.append(MTIndividual(vector=child_vec, objective=np.full(4, np.inf), task_id=child_task))
    return offspring


def run_momfea_core(model: dict[str, Any], params: BenchmarkParams, algorithm_name: str) -> np.ndarray:
    """Core MO-MFEA runner using the canonical fleet evaluator.

    When ``algorithm_name`` is ``"MOMFEAII"``, adaptive Random Mating
    Probability (RMP) is used: the crossover rate decays over generations
    and the mutation standard deviation increases, following the key
    innovation of MO-MFEA-II (learned inter-task transfer).
    """
    use_adaptive = algorithm_name.upper().replace("-", "") in {"MOMFEAII", "MOMFEA2"}
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
    model["maxTurnDeg"] = float(params.max_turn_deg)
    model["hardCollisionConstraint"] = True
    aux_model = _build_aux_model(model, params)

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dim = lower.size

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = (
        np.zeros((params.runs, 2), dtype=float)
        if params.compute_metrics
        else np.zeros((0, 2), dtype=float)
    )

    crossover_rate = float(params.extra.get("mfeaRMP", 0.9))
    mutation_std = float(params.extra.get("mfeaMutationStd", 0.05))
    max_fe = int(params.extra.get("maxFE", params.population * (params.generations + 1)))
    generations = max(1, max_fe // max(1, params.population))
    metric_interval = int(params.extra.get("metricInterval", 20))

    run_indices = _resolve_run_indices(params)
    resume_existing = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in run_indices:
        run_start = time.perf_counter()
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

        # ── Initialize population ──────────────────────────────────────
        pop_size = params.population
        init_vectors = np.random.uniform(lower, upper, size=(pop_size, dim))
        init_task_ids = np.random.choice([1, 2], size=pop_size)

        # Evaluate task-2 (primary) and task-1 (auxiliary) individuals separately
        task2_mask = init_task_ids == 2
        task1_mask = ~task2_mask

        pop_cands_t2 = _evaluate_task_candidates(
            init_vectors[task2_mask], 2, model, aux_model, fleet_size, n_waypoints
        ) if np.any(task2_mask) else []
        pop_cands_t1 = _evaluate_task_candidates(
            init_vectors[task1_mask], 1, model, aux_model, fleet_size, n_waypoints
        ) if np.any(task1_mask) else []

        population: list[MTIndividual] = []
        t2_iter = iter(pop_cands_t2)
        t1_iter = iter(pop_cands_t1)
        for i in range(pop_size):
            if init_task_ids[i] == 2:
                cand = next(t2_iter)
            else:
                cand = next(t1_iter)
            population.append(MTIndividual(
                vector=init_vectors[i].copy(),
                objective=cand.objective.copy(),
                task_id=int(init_task_ids[i]),
            ))

        hv_hist = (
            np.zeros((generations, 2), dtype=float)
            if params.compute_metrics
            else np.zeros((0, 2), dtype=float)
        )

        # ── Generation loop ────────────────────────────────────────────
        for gen in range(1, generations + 1):
            if use_adaptive:
                progress = gen / max(1, generations - 1)
                gen_crossover = crossover_rate * (1.0 - 0.5 * progress)
                gen_mutation = mutation_std * (1.0 + progress)
            else:
                gen_crossover = crossover_rate
                gen_mutation = mutation_std

            offspring = _make_offspring(
                population, model, aux_model, fleet_size, n_waypoints,
                gen_crossover, gen_mutation, lower, upper,
            )

            # Evaluate offspring with their assigned task
            off_t2 = [(i, o) for i, o in enumerate(offspring) if o.task_id == 2]
            off_t1 = [(i, o) for i, o in enumerate(offspring) if o.task_id == 1]

            if off_t2:
                vecs_t2 = np.stack([o.vector for _, o in off_t2])
                cands_t2 = _evaluate_task_candidates(vecs_t2, 2, model, aux_model, fleet_size, n_waypoints)
                for (idx, _), cand in zip(off_t2, cands_t2):
                    offspring[idx].objective = cand.objective.copy()

            if off_t1:
                vecs_t1 = np.stack([o.vector for _, o in off_t1])
                cands_t1 = _evaluate_task_candidates(vecs_t1, 1, model, aux_model, fleet_size, n_waypoints)
                for (idx, _), cand in zip(off_t1, cands_t1):
                    offspring[idx].objective = cand.objective.copy()

            merged = population + offspring
            merged_obj = np.array([item.objective for item in merged], dtype=float)
            front_no, _ = n_d_sort(merged_obj.copy(), None, pop_size)
            order = np.argsort(front_no, kind="mergesort")
            population = [merged[index] for index in order[:pop_size]]

            if params.compute_metrics and hv_hist.shape[0] > 0:
                target = [ind for ind in population if ind.task_id == 2]
                if target:
                    t_obj = np.array([ind.objective for ind in target], dtype=float)
                    if gen == 1 or gen == generations or gen % metric_interval == 0:
                        hv_hist[gen - 1, 0] = cal_metric(1, t_obj, params.problem_index, objective_count)
                        hv_hist[gen - 1, 1] = cal_metric(2, t_obj, params.problem_index, objective_count)
                    elif gen > 1:
                        hv_hist[gen - 1] = hv_hist[gen - 2]

        # ── Finalize run ───────────────────────────────────────────────
        target_population = [ind for ind in population if ind.task_id == 2]
        if not target_population:
            target_population = population

        # Re-evaluate final population with primary model to get Candidate objects
        final_vecs = np.stack([ind.vector for ind in target_population])
        final_candidates = _evaluate_task_candidates(
            final_vecs, 2, model, aux_model, fleet_size, n_waypoints
        )

        ensure_dir(run_dir)
        if params.compute_metrics and hv_hist.shape[0] > 0:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_hist})

        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": algorithm_name,
                "representation": "cart",
                "requestedPopulation": float(params.population),
                "effectivePopulation": float(len(target_population)),
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


def run_momfea(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_momfea_core(model, params, "MOMFEA")


def run_momfea2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_momfea_core(model, params, "MOMFEAII")
