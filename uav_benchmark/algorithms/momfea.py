from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_path import evaluate_path
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort, tournament_selection
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir


@dataclass(slots=True)
class MTIndividual:
    decision: np.ndarray
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


def _decode_uav_path(norm_decision: np.ndarray, model: dict[str, Any], n_control: int) -> np.ndarray:
    n_control = max(3, int(round(n_control)))
    n_mid = n_control - 2
    needed = 3 * n_mid
    decision = np.asarray(norm_decision, dtype=float).reshape(-1)
    if decision.size < needed:
        decision = np.hstack([decision, np.full(needed - decision.size, 0.5, dtype=float)])
    elif decision.size > needed:
        decision = decision[:needed]
    mid = decision.reshape(n_mid, 3)
    mid = np.clip(mid, 0.0, 1.0)
    x_coord = float(model["xmin"]) + mid[:, 0] * (float(model["xmax"]) - float(model["xmin"]))
    y_coord = float(model["ymin"]) + mid[:, 1] * (float(model["ymax"]) - float(model["ymin"]))
    z_alpha = mid[:, 2]
    order = np.argsort(x_coord, kind="mergesort")
    x_coord = x_coord[order]
    y_coord = y_coord[order]
    z_alpha = z_alpha[order]

    safe_h = float(model.get("safeH", 0.0))
    z_coord = np.zeros(n_mid, dtype=float)
    for index in range(n_mid):
        xi = int(np.clip(round(x_coord[index]), 1, np.asarray(model["H"]).shape[1])) - 1
        yi = int(np.clip(round(y_coord[index]), 1, np.asarray(model["H"]).shape[0])) - 1
        ground = float(np.asarray(model["H"], dtype=float)[yi, xi])
        min_z = max(float(model["zmin"]), ground + safe_h)
        max_z = float(model["zmax"])
        z_coord[index] = min_z if max_z <= min_z else min_z + z_alpha[index] * (max_z - min_z)

    path = np.zeros((n_control, 3), dtype=float)
    path[0] = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
    path[-1] = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
    path[1:-1] = np.column_stack([x_coord, y_coord, z_coord])
    path[:, 0] = np.clip(path[:, 0], float(model["xmin"]), float(model["xmax"]))
    path[:, 1] = np.clip(path[:, 1], float(model["ymin"]), float(model["ymax"]))
    path[:, 2] = np.clip(path[:, 2], float(model["zmin"]), float(model["zmax"]))
    return path


def _evaluate_task(decision: np.ndarray, task_id: int, model: dict[str, Any], aux_model: dict[str, Any], n_control: int) -> np.ndarray:
    active_model = model if task_id == 2 else aux_model
    path = _decode_uav_path(decision, active_model, n_control)
    objective = evaluate_path(path, active_model)
    if np.any(~np.isfinite(objective)):
        return np.full(4, np.inf, dtype=float)
    return objective


def _initialize_population(model: dict[str, Any], aux_model: dict[str, Any], n_control: int, pop_size: int) -> list[MTIndividual]:
    n_var = 3 * (n_control - 2)
    population = []
    for _ in range(pop_size):
        decision = np.random.rand(n_var)
        task_id = int(np.random.choice([1, 2]))
        objective = _evaluate_task(decision, task_id, model, aux_model, n_control)
        population.append(MTIndividual(decision=decision, objective=objective, task_id=task_id))
    return population


def _make_offspring(
    population: list[MTIndividual],
    model: dict[str, Any],
    aux_model: dict[str, Any],
    n_control: int,
    crossover_rate: float,
    mutation_std: float,
) -> list[MTIndividual]:
    pop_obj = np.array([item.objective for item in population], dtype=float)
    front_no, _ = n_d_sort(pop_obj.copy(), None, len(population))
    mating_pool = tournament_selection(2, len(population), front_no)
    offspring: list[MTIndividual] = []
    for pair_index in range(0, len(mating_pool), 2):
        p1 = population[int(mating_pool[pair_index])]
        p2 = population[int(mating_pool[(pair_index + 1) % len(mating_pool)])]
        alpha = np.random.rand(p1.decision.shape[0])
        if np.random.rand() < crossover_rate:
            child_decision = alpha * p1.decision + (1.0 - alpha) * p2.decision
        else:
            child_decision = p1.decision.copy()
        child_decision += np.random.randn(*child_decision.shape) * mutation_std
        child_decision = np.clip(child_decision, 0.0, 1.0)
        if p1.task_id == p2.task_id:
            child_task = p1.task_id
        else:
            child_task = int(np.random.choice([p1.task_id, p2.task_id]))
        child_objective = _evaluate_task(child_decision, child_task, model, aux_model, n_control)
        offspring.append(MTIndividual(decision=child_decision, objective=child_objective, task_id=child_task))
    return offspring


def _select_target_task(population: list[MTIndividual], objective_count: int) -> tuple[np.ndarray, np.ndarray]:
    target = [individual for individual in population if individual.task_id == 2]
    if not target:
        target = population
    pop_dec = np.array([np.hstack([individual.decision, individual.task_id]) for individual in target], dtype=float)
    pop_obj = np.array([individual.objective[:objective_count] for individual in target], dtype=float)
    return pop_dec, pop_obj


def run_momfea_core(model: dict[str, Any], params: BenchmarkParams, algorithm_name: str) -> np.ndarray:
    """Core MO-MFEA runner.

    When ``algorithm_name`` is ``"MOMFEAII"``, adaptive Random Mating
    Probability (RMP) is used: the crossover rate decays over generations
    and the mutation standard deviation increases, following the key
    innovation of MO-MFEA-II (learned inter-task transfer).
    """
    use_adaptive = algorithm_name.upper().replace("-", "") in {"MOMFEAII", "MOMFEA2"}
    objective_count = 4
    model = dict(model)
    model["n"] = 10
    n_control = int(model["n"])
    aux_model = _build_aux_model(model, params)
    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    crossover_rate = float(params.extra.get("mfeaRMP", 0.9))
    mutation_std = float(params.extra.get("mfeaMutationStd", 0.05))
    max_fe = int(params.extra.get("maxFE", params.population * (params.generations + 1)))
    generations = max(1, max_fe // max(1, params.population))

    for run_index in range(1, params.runs + 1):
        population = _initialize_population(model, aux_model, n_control, params.population)
        for gen in range(generations):
            # MO-MFEA-II: adaptive RMP — crossover rate decays, mutation grows
            if use_adaptive:
                progress = gen / max(1, generations - 1)
                gen_crossover = crossover_rate * (1.0 - 0.5 * progress)
                gen_mutation = mutation_std * (1.0 + progress)
            else:
                gen_crossover = crossover_rate
                gen_mutation = mutation_std

            offspring = _make_offspring(population, model, aux_model, n_control, gen_crossover, gen_mutation)
            merged = population + offspring
            merged_obj = np.array([item.objective for item in merged], dtype=float)
            front_no, _ = n_d_sort(merged_obj.copy(), None, params.population)
            order = np.argsort(front_no, kind="mergesort")
            population = [merged[index] for index in order[: params.population]]

        pop_dec, pop_obj = _select_target_task(population, objective_count)
        run_dir = results_path / f"Run_{run_index}"
        ensure_dir(run_dir)
        save_run_popobj(run_dir / "final_popobj.mat", pop_obj, params.problem_index, objective_count)
        for solution_index in range(pop_obj.shape[0]):
            path = _decode_uav_path(pop_dec[solution_index, :-1], model, n_control)
            save_bp(run_dir / f"bp_{solution_index + 1}.mat", path, pop_obj[solution_index])
        if params.compute_metrics:
            run_scores[run_index - 1] = np.array(
                [
                    cal_metric(1, pop_obj, params.problem_index, objective_count),
                    cal_metric(2, pop_obj, params.problem_index, objective_count),
                ],
                dtype=float,
            )
    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_momfea(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_momfea_core(model, params, "MOMFEA")


def run_momfea2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_momfea_core(model, params, "MOMFEAII")
