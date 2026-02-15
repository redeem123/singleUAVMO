from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.algorithms.multi_uav import run_multi_nmopso
from uav_benchmark.core.evaluate_path import evaluate_path
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.algorithms import nmopso_utils


@dataclass(slots=True)
class NMOPSOParticle:
    position: dict[str, np.ndarray]
    velocity: dict[str, np.ndarray]
    cost: np.ndarray
    best_position: dict[str, np.ndarray]
    best_cost: np.ndarray
    grid_index: int = 0
    grid_sub_index: np.ndarray | None = None
    topology_signature: np.ndarray | None = None
    topology_bin: int = 1
    robustness_score: float = 0.0
    robustness_bin: int = 1
    atlas_cell_index: int = 1


def _clone_particle(particle: NMOPSOParticle) -> NMOPSOParticle:
    return NMOPSOParticle(
        position={key: value.copy() for key, value in particle.position.items()},
        velocity={key: value.copy() for key, value in particle.velocity.items()},
        cost=particle.cost.copy(),
        best_position={key: value.copy() for key, value in particle.best_position.items()},
        best_cost=particle.best_cost.copy(),
        grid_index=int(particle.grid_index),
        grid_sub_index=None if particle.grid_sub_index is None else particle.grid_sub_index.copy(),
        topology_signature=None if particle.topology_signature is None else particle.topology_signature.copy(),
        topology_bin=int(particle.topology_bin),
        robustness_score=float(particle.robustness_score),
        robustness_bin=int(particle.robustness_bin),
        atlas_cell_index=int(particle.atlas_cell_index),
    )


def _select_member_from_grid_cells(indices: np.ndarray, pressure: float, invert: bool) -> int:
    if indices.size == 0:
        return 0
    unique_cells, inverse = np.unique(indices.astype(int), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    exponents = (-pressure * counts) if invert else (pressure * counts)
    cell_probability = np.exp(exponents)
    if np.sum(cell_probability) <= 0 or not np.all(np.isfinite(cell_probability)):
        cell_probability = np.ones_like(cell_probability) / cell_probability.shape[0]
    else:
        cell_probability = cell_probability / np.sum(cell_probability)
    chosen_cell = unique_cells[int(nmopso_utils.roulette_wheel(cell_probability))]
    members = np.where(indices.astype(int) == int(chosen_cell))[0]
    if members.size == 0:
        return int(np.random.randint(0, indices.size))
    return int(np.random.choice(members))


def _normalize_representation(value: Any) -> str:
    if isinstance(value, (int, float)):
        return "CC" if int(value) == 0 else "SC"
    representation = str(value).strip().upper()
    if representation in {"CC", "CARTESIAN"}:
        return "CC"
    return "SC"


def _parse_ablation(params: BenchmarkParams) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "name": "",
        "useRepository": True,
        "useGrid": True,
        "useMutation": True,
        "useAdaptiveMutation": True,
        "useRegionMutation": True,
        "mutationProb": 0.1,
        "representation": "SC",
        "useReferenceLeader": False,
        "useTwoLayerRef": False,
        "nRep": 50,
        "nGrid": 5,
        "alpha_grid": 0.1,
        "beta": 2.0,
        "gamma": 2.0,
        "w": 1.0,
        "wdamp": 0.98,
        "c1": 1.5,
        "c2": 1.5,
        "mu": 0.5,
        "delta": 20.0,
        "metricInterval": 100,
        "useTopologyRobustArchive": False,
        "atlasTopologyBins": 24,
        "atlasRobustBins": 4,
        "atlasMaxObstacles": 3,
        "atlasHashLevels": 6,
        "atlasObjectiveWeight": 0.5,
        "atlasTopologyWeight": 0.5,
    }
    if "ablation" in params.extra and isinstance(params.extra["ablation"], dict):
        defaults.update(params.extra["ablation"])
    for key in defaults:
        if key in params.extra:
            defaults[key] = params.extra[key]
    defaults["representation"] = _normalize_representation(defaults.get("representation", "SC"))
    defaults["useReferenceLeader"] = bool(defaults.get("useReferenceLeader", False))
    defaults["useTwoLayerRef"] = bool(defaults.get("useTwoLayerRef", False))
    return defaults


def _build_reference_points(n_points: int, objective_count: int, use_two_layer: bool) -> np.ndarray:
    from uav_benchmark.core.nsga3_ops import uniform_point

    first, _ = uniform_point(n_points, objective_count, "NBI")
    if not use_two_layer:
        return first
    second, _ = uniform_point(max(1, n_points // 2), objective_count, "NBI")
    second = second / 2.0 + 1.0 / (2.0 * objective_count)
    return np.vstack([first, second])


def _nmopso_cost(cart_sol: dict[str, np.ndarray], model: dict[str, Any]) -> np.ndarray:
    xs, ys, zs = np.asarray(model["start"], dtype=float).reshape(-1)[:3]
    xf, yf, zf = np.asarray(model["end"], dtype=float).reshape(-1)[:3]
    if "safeH" in model and model["safeH"] is not None:
        zs = float(model["safeH"])
        zf = float(model["safeH"])
    x_all = np.hstack([[xs], cart_sol["x"], [xf]])
    y_all = np.hstack([[ys], cart_sol["y"], [yf]])
    z_rel = np.hstack([[zs], cart_sol["z"], [zf]])
    path = np.zeros((x_all.shape[0], 3), dtype=float)
    for index in range(x_all.shape[0]):
        xi = int(np.clip(round(x_all[index]), 1, int(model["xmax"]))) - 1
        yi = int(np.clip(round(y_all[index]), 1, int(model["ymax"]))) - 1
        abs_z = z_rel[index] + float(np.asarray(model["H"], dtype=float)[yi, xi])
        if z_rel[index] < 0:
            return np.array([np.inf, np.inf, np.inf, np.inf], dtype=float)
        path[index] = [x_all[index], y_all[index], abs_z]
    return evaluate_path(path, model)


def _initialize_particle(
    model: dict[str, Any],
    representation: str,
    var_min: dict[str, np.ndarray],
    var_max: dict[str, np.ndarray],
) -> NMOPSOParticle:
    if representation == "SC":
        position = {
            "r": np.random.uniform(var_min["r"], var_max["r"]),
            "psi": np.random.uniform(var_min["psi"], var_max["psi"]),
            "phi": np.random.uniform(var_min["phi"], var_max["phi"]),
        }
        velocity = {key: np.zeros_like(value) for key, value in position.items()}
    else:
        position = {
            "x": np.random.uniform(var_min["x"], var_max["x"]),
            "y": np.random.uniform(var_min["y"], var_max["y"]),
            "z": np.random.uniform(var_min["z"], var_max["z"]),
        }
        velocity = {key: np.zeros_like(value) for key, value in position.items()}
    cart = nmopso_utils.position_to_cart(position, model, representation)
    cost = _nmopso_cost(cart, model)
    return NMOPSOParticle(
        position={key: value.copy() for key, value in position.items()},
        velocity={key: value.copy() for key, value in velocity.items()},
        cost=cost.copy(),
        best_position={key: value.copy() for key, value in position.items()},
        best_cost=cost.copy(),
    )


def _update_particle_velocity_and_position(
    particle: NMOPSOParticle,
    leader: NMOPSOParticle,
    var_min: dict[str, np.ndarray],
    var_max: dict[str, np.ndarray],
    vel_min: dict[str, np.ndarray],
    vel_max: dict[str, np.ndarray],
    c1: float,
    c2: float,
    inertia: float,
) -> None:
    for key in particle.position.keys():
        particle.velocity[key] = (
            inertia * particle.velocity[key]
            + c1 * np.random.rand(*particle.position[key].shape) * (particle.best_position[key] - particle.position[key])
            + c2 * np.random.rand(*particle.position[key].shape) * (leader.position[key] - particle.position[key])
        )
        particle.velocity[key] = np.clip(particle.velocity[key], vel_min[key], vel_max[key])
        particle.position[key] = particle.position[key] + particle.velocity[key]
        out_of_range = (particle.position[key] < var_min[key]) | (particle.position[key] > var_max[key])
        particle.velocity[key][out_of_range] = -particle.velocity[key][out_of_range]
        particle.position[key] = np.clip(particle.position[key], var_min[key], var_max[key])


def run_nmopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    if str(params.mode).lower() == "multi":
        return run_multi_nmopso(model, params)
    objective_count = 4
    model = dict(model)
    model["n"] = 10
    ablation = _parse_ablation(params)
    representation = ablation["representation"]
    use_spherical = representation == "SC"
    atlas_config = nmopso_utils.build_atlas_config(ablation)
    use_atlas_archive = atlas_config.enabled and bool(ablation["useRepository"])

    n_var = int(model["n"])
    alpha_vel = 0.5
    if use_spherical:
        path_diag = float(np.linalg.norm(np.asarray(model["start"], dtype=float).reshape(-1) - np.asarray(model["end"], dtype=float).reshape(-1)))
        var_max = {
            "r": np.full(n_var, 3.0 * path_diag / n_var, dtype=float),
            "psi": np.full(n_var, np.pi / 4.0, dtype=float),
            "phi": np.full(n_var, np.pi / 4.0, dtype=float),
        }
        var_min = {
            "r": np.full(n_var, (3.0 * path_diag / n_var) / 9.0, dtype=float),
            "psi": -var_max["psi"],
            "phi": -var_max["phi"],
        }
        vel_max = {
            "r": alpha_vel * (var_max["r"] - var_min["r"]),
            "psi": alpha_vel * (var_max["psi"] - var_min["psi"]),
            "phi": alpha_vel * (var_max["phi"] - var_min["phi"]),
        }
        vel_min = {key: -value for key, value in vel_max.items()}
    else:
        var_min = {
            "x": np.full(n_var, float(model["xmin"]), dtype=float),
            "y": np.full(n_var, float(model["ymin"]), dtype=float),
            "z": np.full(n_var, float(model["zmin"]), dtype=float),
        }
        var_max = {
            "x": np.full(n_var, float(model["xmax"]), dtype=float),
            "y": np.full(n_var, float(model["ymax"]), dtype=float),
            "z": np.full(n_var, float(model["zmax"]), dtype=float),
        }
        vel_max = {
            "x": alpha_vel * (var_max["x"] - var_min["x"]),
            "y": alpha_vel * (var_max["y"] - var_min["y"]),
            "z": alpha_vel * (var_max["z"] - var_min["z"]),
        }
        vel_min = {key: -value for key, value in vel_max.items()}

    reference_points = np.zeros((0, objective_count), dtype=float)
    if bool(ablation["useReferenceLeader"]):
        reference_points = _build_reference_points(params.population, objective_count, bool(ablation["useTwoLayerRef"]))
    init_max_tries = 10

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    for run_index in range(1, params.runs + 1):
        run_start = time.perf_counter()
        particles: list[NMOPSOParticle] = []
        for _ in range(init_max_tries):
            particles = [
                _initialize_particle(model, representation, var_min, var_max)
                for _ in range(params.population)
            ]
            init_costs = np.array([particle.cost for particle in particles], dtype=float)
            if np.any(np.all(np.isfinite(init_costs), axis=1)):
                break
        costs = np.array([particle.cost for particle in particles], dtype=float)
        dominated = nmopso_utils.determine_domination(costs)
        repository = [_clone_particle(particle) for particle, is_dominated in zip(particles, dominated) if not is_dominated]
        if not repository:
            repository = [_clone_particle(particle) for particle in particles]

        inertia = float(ablation["w"])
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            repository_costs = np.array([entry.cost for entry in repository], dtype=float) if repository else np.zeros((0, objective_count), dtype=float)
            if bool(ablation["useReferenceLeader"]) and repository and reference_points.shape[0] > 0:
                leader_indices = nmopso_utils.select_leader_ref(repository_costs, reference_points, params.population)
            else:
                leader_indices = np.random.randint(0, max(1, len(repository)), size=params.population)

            grid_lb = grid_ub = np.zeros((0, 0), dtype=float)
            repository_grid_index = np.zeros(len(repository), dtype=int)
            repository_atlas_index = np.zeros(len(repository), dtype=int) if use_atlas_archive else None
            if bool(ablation["useRepository"]) and repository:
                repository_costs = np.array([entry.cost for entry in repository], dtype=float)
                if bool(ablation["useGrid"]):
                    grid_lb, grid_ub = nmopso_utils.create_grid(repository_costs, int(ablation["nGrid"]), float(ablation["alpha_grid"]))
                    for idx, entry in enumerate(repository):
                        entry.grid_index, entry.grid_sub_index = nmopso_utils.find_grid_index(entry.cost, grid_lb, grid_ub)
                        repository_grid_index[idx] = entry.grid_index
                else:
                    repository_grid_index = np.arange(len(repository), dtype=int) + 1
                if use_atlas_archive:
                    for idx, entry in enumerate(repository):
                        cart = nmopso_utils.position_to_cart(entry.position, model, representation)
                        path_xyz = nmopso_utils.cart_to_absolute_path(cart, model)
                        entry.topology_signature = nmopso_utils.topology_signature(path_xyz, model, atlas_config.max_obstacles)
                        entry.topology_bin = nmopso_utils.topology_bin_from_signature(entry.topology_signature, atlas_config)
                        entry.robustness_score, entry.robustness_bin = nmopso_utils.robustness_from_cost(entry.cost, atlas_config.n_robust_bins)
                        entry.atlas_cell_index = (entry.topology_bin - 1) * atlas_config.n_robust_bins + entry.robustness_bin
                        repository_atlas_index[idx] = entry.atlas_cell_index

            for particle_index, particle in enumerate(particles):
                if bool(ablation["useRepository"]) and repository:
                    if bool(ablation["useReferenceLeader"]) and leader_indices.size == params.population:
                        leader = repository[int(leader_indices[particle_index])]
                    elif use_atlas_archive and repository_atlas_index is not None and repository_grid_index.size > 0:
                        leader_idx = nmopso_utils.select_leader_with_weights(
                            repository_grid_index,
                            float(ablation["beta"]),
                            atlas_config.objective_weight,
                            atlas_config.atlas_weight,
                            repository_atlas_index,
                        )
                        leader = repository[int(leader_idx)]
                    elif bool(ablation["useGrid"]) and repository_grid_index.size > 0:
                        leader = repository[
                            _select_member_from_grid_cells(
                                repository_grid_index,
                                float(ablation["beta"]),
                                invert=True,
                            )
                        ]
                    else:
                        leader = repository[int(np.random.randint(0, len(repository)))]
                else:
                    leader = particle

                _update_particle_velocity_and_position(
                    particle=particle,
                    leader=leader,
                    var_min=var_min,
                    var_max=var_max,
                    vel_min=vel_min,
                    vel_max=vel_max,
                    c1=float(ablation["c1"]),
                    c2=float(ablation["c2"]),
                    inertia=inertia,
                )

                cart_particle = nmopso_utils.position_to_cart(particle.position, model, representation)
                particle.cost = _nmopso_cost(cart_particle, model)

                mutation_prob = float(ablation["mutationProb"])
                if bool(ablation["useMutation"]):
                    if bool(ablation["useAdaptiveMutation"]):
                        mutation_prob = (1.0 - (generation - 1) / max(1, params.generations - 1)) ** (1.0 / float(ablation["mu"]))
                else:
                    mutation_prob = 0.0
                if np.random.rand() < mutation_prob:
                    region_count = nmopso_utils.archive_region_count(
                        repository_grid_index if repository_grid_index.size > 0 else np.arange(len(repository), dtype=int) + 1,
                        repository_atlas_index,
                    )
                    mutated = nmopso_utils.mutate(
                        particle.position,
                        particle.best_position,
                        float(ablation["delta"]),
                        var_min,
                        var_max,
                        representation,
                        region_count,
                    )
                    cart_mutated = nmopso_utils.position_to_cart(mutated, model, representation)
                    mutated_cost = _nmopso_cost(cart_mutated, model)
                    if nmopso_utils.dominates(mutated_cost, particle.cost) or (
                        not nmopso_utils.dominates(particle.cost, mutated_cost) and np.random.rand() < 0.5
                    ):
                        particle.position = {key: value.copy() for key, value in mutated.items()}
                        particle.cost = mutated_cost

                if nmopso_utils.dominates(particle.cost, particle.best_cost) or (
                    not nmopso_utils.dominates(particle.best_cost, particle.cost) and np.random.rand() < 0.5
                ):
                    particle.best_position = {key: value.copy() for key, value in particle.position.items()}
                    particle.best_cost = particle.cost.copy()

            if bool(ablation["useRepository"]):
                merged = repository + particles
                merged_costs = np.array([entry.cost for entry in merged], dtype=float)
                merged_dom = nmopso_utils.determine_domination(merged_costs)
                repository = [_clone_particle(entry) for entry, is_dom in zip(merged, merged_dom) if not is_dom]
                if len(repository) > int(ablation["nRep"]):
                    if bool(ablation["useGrid"]):
                        repo_costs = np.array([entry.cost for entry in repository], dtype=float)
                        grid_lb, grid_ub = nmopso_utils.create_grid(repo_costs, int(ablation["nGrid"]), float(ablation["alpha_grid"]))
                        repo_grid = np.zeros(len(repository), dtype=int)
                        for idx, entry in enumerate(repository):
                            entry.grid_index, entry.grid_sub_index = nmopso_utils.find_grid_index(entry.cost, grid_lb, grid_ub)
                            repo_grid[idx] = entry.grid_index
                        repo_atlas = np.zeros(len(repository), dtype=int) if use_atlas_archive else None
                        if use_atlas_archive:
                            for idx, entry in enumerate(repository):
                                cart = nmopso_utils.position_to_cart(entry.position, model, representation)
                                sig = nmopso_utils.topology_signature(nmopso_utils.cart_to_absolute_path(cart, model), model, atlas_config.max_obstacles)
                                topo_bin = nmopso_utils.topology_bin_from_signature(sig, atlas_config)
                                _, robust_bin = nmopso_utils.robustness_from_cost(entry.cost, atlas_config.n_robust_bins)
                                entry.atlas_cell_index = (topo_bin - 1) * atlas_config.n_robust_bins + robust_bin
                                repo_atlas[idx] = entry.atlas_cell_index
                        while len(repository) > int(ablation["nRep"]):
                            if use_atlas_archive and repo_atlas is not None:
                                delete_index = nmopso_utils.delete_one_with_weights(
                                    repo_grid,
                                    float(ablation["gamma"]),
                                    atlas_config.objective_weight,
                                    atlas_config.atlas_weight,
                                    repo_atlas,
                                )
                            else:
                                delete_index = _select_member_from_grid_cells(
                                    repo_grid,
                                    float(ablation["gamma"]),
                                    invert=False,
                                )
                            repository.pop(int(delete_index))
                            repo_grid = np.delete(repo_grid, int(delete_index))
                            if repo_atlas is not None:
                                repo_atlas = np.delete(repo_atlas, int(delete_index))
                    else:
                        repository = list(np.random.choice(repository, size=int(ablation["nRep"]), replace=False))

            report_costs = np.array([entry.cost for entry in (repository if repository else particles)], dtype=float)
            if params.compute_metrics:
                if generation == 1 or generation == params.generations or generation % int(ablation["metricInterval"]) == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, report_costs, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, report_costs, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]
            inertia *= float(ablation["wdamp"])

        run_dir = results_path / f"Run_{run_index}"
        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        final_members = repository if repository else particles
        final_costs = np.array([entry.cost for entry in final_members], dtype=float)
        save_run_popobj(run_dir / "final_popobj.mat", final_costs, params.problem_index, objective_count)
        for member_index, member in enumerate(final_members, start=1):
            cart = nmopso_utils.position_to_cart(member.position, model, representation)
            path_xyz = nmopso_utils.cart_to_absolute_path(cart, model)
            save_bp(run_dir / f"bp_{member_index}.mat", path_xyz, member.cost)
        feasible_count = int(np.sum(np.all(np.isfinite(final_costs), axis=1)))
        save_mat(
            run_dir / "run_stats.mat",
            {
                "runtimeSec": float(time.perf_counter() - run_start),
                "feasibleCount": feasible_count,
                "solutionCount": int(final_costs.shape[0]),
            },
        )

        if params.compute_metrics:
            run_scores[run_index - 1] = np.array(
                [
                    cal_metric(1, final_costs, params.problem_index, objective_count),
                    cal_metric(2, final_costs, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores
