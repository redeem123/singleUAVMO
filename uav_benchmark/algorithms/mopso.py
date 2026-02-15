from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.multi_uav import run_multi_mopso
from uav_benchmark.core.evaluate_path import evaluate_path
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir


@dataclass(slots=True)
class Position:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(slots=True)
class Particle:
    position: Position
    velocity: Position
    cost: np.ndarray
    is_dominated: bool = False
    grid_index: int = 0
    grid_sub_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def _cost_function(position: Position, model: dict[str, Any]) -> np.ndarray:
    infinite_cost = np.array([np.inf, np.inf, np.inf, np.inf], dtype=float)
    xs, ys, zs = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
    xf, yf, zf = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
    if "safeH" in model and model["safeH"] is not None:
        zs = float(model["safeH"])
        zf = float(model["safeH"])
    x_all = np.hstack([[xs], position.x, [xf]])
    y_all = np.hstack([[ys], position.y, [yf]])
    z_rel = np.hstack([[zs], position.z, [zf]])
    path = np.zeros((x_all.shape[0], 3), dtype=float)
    for index in range(x_all.shape[0]):
        xi = int(np.clip(round(x_all[index]), 1, int(model["xmax"]))) - 1
        yi = int(np.clip(round(y_all[index]), 1, int(model["ymax"]))) - 1
        abs_z = z_rel[index] + float(np.asarray(model["H"], dtype=float)[yi, xi])
        if z_rel[index] < 0:
            return infinite_cost
        path[index] = [x_all[index], y_all[index], abs_z]
    return evaluate_path(path, model)


def _cost_matrix(particles: list[Particle]) -> np.ndarray:
    if not particles:
        return np.zeros((0, 4), dtype=float)
    return np.array([particle.cost for particle in particles], dtype=float)


def _copy_position(position: Position) -> Position:
    return Position(
        x=position.x.copy(),
        y=position.y.copy(),
        z=position.z.copy(),
    )


def _copy_particle(particle: Particle) -> Particle:
    return Particle(
        position=_copy_position(particle.position),
        velocity=_copy_position(particle.velocity),
        cost=particle.cost.copy(),
        is_dominated=particle.is_dominated,
        grid_index=particle.grid_index,
        grid_sub_index=particle.grid_sub_index.copy(),
    )


def _update_archive(archive: list[Particle], max_size: int, divisions: int) -> list[Particle]:
    if not archive:
        return []
    objective_matrix = _cost_matrix(archive)
    front_no, _ = n_d_sort(objective_matrix.copy(), None, 1)
    archive = [particle for particle, front in zip(archive, front_no) if front == 1]
    if len(archive) <= max_size:
        return archive
    objective_matrix = _cost_matrix(archive)
    delete_mask = _delete_indices(objective_matrix, len(archive) - max_size, divisions)
    archive = [particle for idx, particle in enumerate(archive) if not delete_mask[idx]]
    return archive


def _delete_indices(pop_obj: np.ndarray, count: int, divisions: int) -> np.ndarray:
    n_points = pop_obj.shape[0]
    max_values = np.max(pop_obj, axis=0)
    min_values = np.min(pop_obj, axis=0)
    with np.errstate(invalid="ignore"):
        step = (max_values - min_values) / divisions
    with np.errstate(divide="ignore", invalid="ignore"):
        grid = np.floor((pop_obj - min_values) / step)
    grid[np.isnan(grid)] = 0
    grid[np.isposinf(grid)] = divisions - 1
    grid[np.isneginf(grid)] = 0
    grid = grid.astype(int)
    grid[grid >= divisions] = divisions - 1
    grid[grid < 0] = 0
    _, site = np.unique(grid, axis=0, return_inverse=True)
    crowded = np.bincount(site, minlength=np.max(site) + 1).astype(int)
    delete_mask = np.zeros(n_points, dtype=bool)
    while np.sum(delete_mask) < count:
        candidate_grids = np.where(crowded == np.max(crowded))[0]
        target_grid = int(np.random.choice(candidate_grids))
        inside = np.where(site == target_grid)[0]
        pick = int(np.random.choice(inside))
        delete_mask[pick] = True
        site[pick] = -1
        crowded[target_grid] -= 1
    return delete_mask


def _roulette_wheel(n_select: int, crowded: np.ndarray) -> np.ndarray:
    fitness = crowded.astype(float)
    fitness = fitness - min(np.min(fitness), 0) + 1e-6
    cumulative = np.cumsum(1.0 / fitness)
    cumulative = cumulative / np.max(cumulative)
    choices = []
    for _ in range(n_select):
        draw = np.random.rand()
        choices.append(int(np.where(draw <= cumulative)[0][0]))
    return np.array(choices, dtype=int)


def _rep_selection(archive: list[Particle], population_size: int, divisions: int) -> np.ndarray:
    pop_obj = _cost_matrix(archive)
    n_points = pop_obj.shape[0]
    max_values = np.max(pop_obj, axis=0)
    min_values = np.min(pop_obj, axis=0)
    with np.errstate(invalid="ignore"):
        step = (max_values - min_values) / divisions
    with np.errstate(divide="ignore", invalid="ignore"):
        grid = np.floor((pop_obj - min_values.reshape(1, -1)) / step.reshape(1, -1))
    grid[np.isnan(grid)] = 0
    grid[np.isposinf(grid)] = divisions - 1
    grid[np.isneginf(grid)] = 0
    grid = grid.astype(int)
    grid[grid >= divisions] = divisions - 1
    grid[grid < 0] = 0
    _, site = np.unique(grid, axis=0, return_inverse=True)
    crowded = np.bincount(site, minlength=np.max(site) + 1)
    grid_choice = _roulette_wheel(population_size, crowded)
    selected = np.zeros(population_size, dtype=int)
    for index in range(population_size):
        inside = np.where(site == grid_choice[index])[0]
        selected[index] = int(np.random.choice(inside))
    return selected


def _operator(
    particles: list[Particle],
    pbest: list[Particle],
    gbest: list[Particle],
    inertia: float,
    lower: Position,
    upper: Position,
) -> list[Particle]:
    if not particles:
        return particles
    n_var = particles[0].position.x.shape[0]
    n_pop = len(particles)
    decision = np.zeros((n_pop, 3 * n_var), dtype=float)
    velocity = np.zeros((n_pop, 3 * n_var), dtype=float)
    pbest_decision = np.zeros((n_pop, 3 * n_var), dtype=float)
    gbest_decision = np.zeros((n_pop, 3 * n_var), dtype=float)
    for index, particle in enumerate(particles):
        decision[index, :n_var] = particle.position.x
        decision[index, n_var : 2 * n_var] = particle.position.y
        decision[index, 2 * n_var :] = particle.position.z
        velocity[index, :n_var] = particle.velocity.x
        velocity[index, n_var : 2 * n_var] = particle.velocity.y
        velocity[index, 2 * n_var :] = particle.velocity.z
        pbest_decision[index, :n_var] = pbest[index].position.x
        pbest_decision[index, n_var : 2 * n_var] = pbest[index].position.y
        pbest_decision[index, 2 * n_var :] = pbest[index].position.z
        gbest_decision[index, :n_var] = gbest[index].position.x
        gbest_decision[index, n_var : 2 * n_var] = gbest[index].position.y
        gbest_decision[index, 2 * n_var :] = gbest[index].position.z
    rand_one = np.random.rand(*decision.shape)
    rand_two = np.random.rand(*decision.shape)
    # PlatEMO MOPSO update (OperatorPSO): W*V + r1*(Pbest-X) + r2*(Gbest-X)
    new_velocity = inertia * velocity + rand_one * (pbest_decision - decision) + rand_two * (gbest_decision - decision)
    new_decision = decision + new_velocity
    x_coord = np.clip(new_decision[:, :n_var], lower.x, upper.x)
    y_coord = np.clip(new_decision[:, n_var : 2 * n_var], lower.y, upper.y)
    z_coord = np.clip(new_decision[:, 2 * n_var :], lower.z, upper.z)
    for index, particle in enumerate(particles):
        particle.position = Position(x=x_coord[index], y=y_coord[index], z=z_coord[index])
        particle.velocity = Position(
            x=new_velocity[index, :n_var],
            y=new_velocity[index, n_var : 2 * n_var],
            z=new_velocity[index, 2 * n_var :],
        )
    return particles


def _update_pbest(pbest: list[Particle], population: list[Particle]) -> list[Particle]:
    if not pbest or not population:
        return pbest
    pbest_obj = _cost_matrix(pbest)
    pop_obj = _cost_matrix(population)
    if pbest_obj.size == 0 or pop_obj.size == 0:
        return pbest

    with np.errstate(invalid="ignore"):
        temp = pbest_obj - pop_obj
    dominate = np.any(temp < 0, axis=1).astype(int) - np.any(temp > 0, axis=1).astype(int)

    replace = np.where(dominate == -1)[0]
    for index in replace:
        pbest[index] = _copy_particle(population[index])

    tie = np.where(dominate == 0)[0]
    if tie.size > 0:
        pick = np.random.rand(tie.size) < 0.5
        for index in tie[pick]:
            pbest[index] = _copy_particle(population[index])
    return pbest


def _adjust_min_z(model: dict[str, Any], min_z: float, max_z: float) -> float:
    if "safeH" in model and model["safeH"] is not None:
        min_z = max(min_z, float(model["safeH"]))
    else:
        drone = float(model.get("droneSize", model.get("drone_size", 1.0)))
        min_z = max(min_z, drone + 1e-3)
    return min(min_z, max_z)


def run_mopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    if str(params.mode).lower() == "multi":
        return run_multi_mopso(model, params)
    objective_count = 4
    model = dict(model)
    model["n"] = 10
    n_var = int(model["n"])
    var_shape = (n_var,)
    lower = Position(
        x=np.full(var_shape, float(model["xmin"]), dtype=float),
        y=np.full(var_shape, float(model["ymin"]), dtype=float),
        z=np.full(var_shape, _adjust_min_z(model, float(model["zmin"]), float(model["zmax"])), dtype=float),
    )
    upper = Position(
        x=np.full(var_shape, float(model["xmax"]), dtype=float),
        y=np.full(var_shape, float(model["ymax"]), dtype=float),
        z=np.full(var_shape, float(model["zmax"]), dtype=float),
    )

    divisions = int(params.extra.get("div", 10))
    inertia = float(params.extra.get("w", params.extra.get("inertia", 0.4)))
    metric_interval = int(params.extra.get("metricInterval", 100))
    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)

    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    for run_index in range(1, params.runs + 1):
        run_start = time.perf_counter()
        particles = []
        for _ in range(params.population):
            position = Position(
                x=np.random.uniform(lower.x, upper.x),
                y=np.random.uniform(lower.y, upper.y),
                z=np.random.uniform(lower.z, upper.z),
            )
            velocity = Position(x=np.zeros(var_shape), y=np.zeros(var_shape), z=np.zeros(var_shape))
            cost = _cost_function(position, model)
            particles.append(
                Particle(
                    position=position,
                    velocity=velocity,
                    cost=cost,
                )
            )
        archive = _update_archive(list(particles), params.population, divisions)
        personal_best = [_copy_particle(particle) for particle in particles]
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            if archive:
                rep_idx = _rep_selection(archive, params.population, divisions)
                global_best = [archive[index] for index in rep_idx]
            else:
                global_best = personal_best
            particles = _operator(particles, personal_best, global_best, inertia, lower, upper)
            for particle in particles:
                particle.cost = _cost_function(particle.position, model)
            archive = _update_archive(archive + particles, params.population, divisions)
            personal_best = _update_pbest(personal_best, particles)

            objective_matrix = _cost_matrix(archive) if archive else _cost_matrix(particles)
            if params.compute_metrics:
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, objective_matrix, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, objective_matrix, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        run_dir = results_path / f"Run_{run_index}"
        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        final_objectives = _cost_matrix(archive) if archive else _cost_matrix(particles)
        save_run_popobj(run_dir / "final_popobj.mat", final_objectives, params.problem_index, objective_count)
        save_members = archive if archive else particles
        for member_index, member in enumerate(save_members, start=1):
            x_full = np.hstack([[model["start"][0]], member.position.x, [model["end"][0]]])
            y_full = np.hstack([[model["start"][1]], member.position.y, [model["end"][1]]])
            start_z = float(model.get("safeH", np.asarray(model["start"], dtype=float).reshape(-1)[2]))
            end_z = float(model.get("safeH", np.asarray(model["end"], dtype=float).reshape(-1)[2]))
            z_rel = np.hstack([[start_z], member.position.z, [end_z]])
            z_abs = np.zeros_like(z_rel)
            for point_index in range(z_abs.shape[0]):
                xi = int(np.clip(round(x_full[point_index]), 1, int(model["xmax"]))) - 1
                yi = int(np.clip(round(y_full[point_index]), 1, int(model["ymax"]))) - 1
                z_abs[point_index] = z_rel[point_index] + float(np.asarray(model["H"], dtype=float)[yi, xi])
            path_xyz = np.column_stack([x_full, y_full, z_abs])
            save_bp(run_dir / f"bp_{member_index}.mat", path_xyz, member.cost)
        feasible_count = int(np.sum(np.all(np.isfinite(final_objectives), axis=1)))
        save_mat(
            run_dir / "run_stats.mat",
            {
                "runtimeSec": float(time.perf_counter() - run_start),
                "feasibleCount": feasible_count,
                "solutionCount": int(final_objectives.shape[0]),
            },
        )

        if params.compute_metrics:
            run_scores[run_index - 1] = np.array(
                [
                    cal_metric(1, final_objectives, params.problem_index, objective_count),
                    cal_metric(2, final_objectives, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores
