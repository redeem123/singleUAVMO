from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_benchmark.algorithms.nmopso.legacy_core import (
    AtlasConfig,
    NMOPSOParticle,
    _build_reference_points,
    _clone_particle,
    _initialize_particle,
    _nmopso_cost,
    _normalize_representation,
    _parse_ablation,
    _select_member_from_grid_cells,
    _update_particle_velocity_and_position,
    archive_occupancies,
    archive_region_count,
    build_atlas_config,
    cart_to_absolute_path,
    create_grid,
    delete_one_with_weights,
    determine_domination,
    extract_obstacles,
    find_grid_index,
    mutate,
    normalize_objectives,
    normalize_signature_for_hash,
    position_to_cart,
    robustness_from_cost,
    roulette_wheel,
    select_leader_ref,
    select_leader_with_weights,
    spherical_to_cart,
    topology_bin_from_signature,
    topology_signature,
    transformation_matrix,
    wrap_to_pi,
)
from uav_benchmark.algorithms.shared.fleet_runner import run_fleet_nmopso
from uav_benchmark.algorithms.shared.mission_stats import build_mission_stats
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.dominance import dominates
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir

__all__ = [
    "AtlasConfig",
    "NMOPSOParticle",
    "_normalize_representation",
    "archive_occupancies",
    "archive_region_count",
    "build_atlas_config",
    "cart_to_absolute_path",
    "create_grid",
    "delete_one_with_weights",
    "determine_domination",
    "extract_obstacles",
    "find_grid_index",
    "mutate",
    "normalize_objectives",
    "normalize_signature_for_hash",
    "position_to_cart",
    "robustness_from_cost",
    "roulette_wheel",
    "run_nmopso",
    "select_leader_ref",
    "select_leader_with_weights",
    "spherical_to_cart",
    "topology_bin_from_signature",
    "topology_signature",
    "transformation_matrix",
    "wrap_to_pi",
]


def run_nmopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    use_legacy_runner = bool(params.extra.get("legacyPathRunner", False))
    if (not use_legacy_runner) or int(params.fleet_size) > 1:
        return run_fleet_nmopso(model, params)
    objective_count = 4
    model = dict(model)
    model["n"] = 10
    ablation = _parse_ablation(params)
    representation = ablation["representation"]
    use_spherical = representation == "SC"
    atlas_config = build_atlas_config(ablation)
    use_atlas_archive = atlas_config.enabled and bool(ablation["useRepository"])

    n_var = int(model["n"])
    alpha_vel = 0.5
    if use_spherical:
        path_diag = float(
            np.linalg.norm(
                np.asarray(model["start"], dtype=float).reshape(-1) - np.asarray(model["end"], dtype=float).reshape(-1)
            )
        )
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
                _initialize_particle(model, representation, var_min, var_max) for _ in range(params.population)
            ]
            init_costs = np.array([particle.cost for particle in particles], dtype=float)
            if np.any(np.all(np.isfinite(init_costs), axis=1)):
                break
        costs = np.array([particle.cost for particle in particles], dtype=float)
        dominated = determine_domination(costs)
        repository = [
            _clone_particle(particle)
            for particle, is_dominated in zip(particles, dominated, strict=False)
            if not is_dominated
        ]
        if not repository:
            repository = [_clone_particle(particle) for particle in particles]

        inertia = float(ablation["w"])
        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )

        for generation in range(1, params.generations + 1):
            repository_costs = (
                np.array([entry.cost for entry in repository], dtype=float)
                if repository
                else np.zeros((0, objective_count), dtype=float)
            )
            if bool(ablation["useReferenceLeader"]) and repository and reference_points.shape[0] > 0:
                leader_indices = select_leader_ref(repository_costs, reference_points, params.population)
            else:
                leader_indices = np.random.randint(0, max(1, len(repository)), size=params.population)

            grid_lb = grid_ub = np.zeros((0, 0), dtype=float)
            repository_grid_index = np.zeros(len(repository), dtype=int)
            repository_atlas_index = np.zeros(len(repository), dtype=int) if use_atlas_archive else None
            if bool(ablation["useRepository"]) and repository:
                repository_costs = np.array([entry.cost for entry in repository], dtype=float)
                if bool(ablation["useGrid"]):
                    grid_lb, grid_ub = create_grid(
                        repository_costs, int(ablation["nGrid"]), float(ablation["alpha_grid"])
                    )
                    for idx, entry in enumerate(repository):
                        entry.grid_index, entry.grid_sub_index = find_grid_index(entry.cost, grid_lb, grid_ub)
                        repository_grid_index[idx] = entry.grid_index
                else:
                    repository_grid_index = np.arange(len(repository), dtype=int) + 1
                if use_atlas_archive and repository_atlas_index is not None:
                    for idx, entry in enumerate(repository):
                        cart = position_to_cart(entry.position, model, representation)
                        path_xyz = cart_to_absolute_path(cart, model)
                        entry.topology_signature = topology_signature(path_xyz, model, atlas_config.max_obstacles)
                        entry.topology_bin = topology_bin_from_signature(entry.topology_signature, atlas_config)
                        entry.robustness_score, entry.robustness_bin = robustness_from_cost(
                            entry.cost, atlas_config.n_robust_bins
                        )
                        entry.atlas_cell_index = (
                            entry.topology_bin - 1
                        ) * atlas_config.n_robust_bins + entry.robustness_bin
                        repository_atlas_index[idx] = entry.atlas_cell_index

            for particle_index, particle in enumerate(particles):
                if bool(ablation["useRepository"]) and repository:
                    if bool(ablation["useReferenceLeader"]) and leader_indices.size == params.population:
                        leader = repository[int(leader_indices[particle_index])]
                    elif use_atlas_archive and repository_atlas_index is not None and repository_grid_index.size > 0:
                        leader_idx = select_leader_with_weights(
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

                cart_particle = position_to_cart(particle.position, model, representation)
                particle.cost = _nmopso_cost(cart_particle, model)

                mutation_prob = float(ablation["mutationProb"])
                if bool(ablation["useMutation"]):
                    if bool(ablation["useAdaptiveMutation"]):
                        mutation_prob = (1.0 - (generation - 1) / max(1, params.generations - 1)) ** (
                            1.0 / float(ablation["mu"])
                        )
                else:
                    mutation_prob = 0.0
                if np.random.rand() < mutation_prob:
                    region_count = archive_region_count(
                        repository_grid_index
                        if repository_grid_index.size > 0
                        else np.arange(len(repository), dtype=int) + 1,
                        repository_atlas_index,
                    )
                    mutated = mutate(
                        particle.position,
                        particle.best_position,
                        float(ablation["delta"]),
                        var_min,
                        var_max,
                        representation,
                        region_count,
                    )
                    cart_mutated = position_to_cart(mutated, model, representation)
                    mutated_cost = _nmopso_cost(cart_mutated, model)
                    if dominates(mutated_cost, particle.cost) or (
                        not dominates(particle.cost, mutated_cost) and np.random.rand() < 0.5
                    ):
                        particle.position = {key: value.copy() for key, value in mutated.items()}
                        particle.cost = mutated_cost

                if dominates(particle.cost, particle.best_cost) or (
                    not dominates(particle.best_cost, particle.cost) and np.random.rand() < 0.5
                ):
                    particle.best_position = {key: value.copy() for key, value in particle.position.items()}
                    particle.best_cost = particle.cost.copy()

            if bool(ablation["useRepository"]):
                merged = repository + particles
                merged_costs = np.array([entry.cost for entry in merged], dtype=float)
                merged_dom = determine_domination(merged_costs)
                repository = [
                    _clone_particle(entry) for entry, is_dom in zip(merged, merged_dom, strict=False) if not is_dom
                ]
                if len(repository) > int(ablation["nRep"]):
                    if bool(ablation["useGrid"]):
                        repo_costs = np.array([entry.cost for entry in repository], dtype=float)
                        grid_lb, grid_ub = create_grid(
                            repo_costs, int(ablation["nGrid"]), float(ablation["alpha_grid"])
                        )
                        repo_grid = np.zeros(len(repository), dtype=int)
                        for idx, entry in enumerate(repository):
                            entry.grid_index, entry.grid_sub_index = find_grid_index(entry.cost, grid_lb, grid_ub)
                            repo_grid[idx] = entry.grid_index
                        repo_atlas = np.zeros(len(repository), dtype=int) if use_atlas_archive else None
                        if use_atlas_archive and repo_atlas is not None:
                            for idx, entry in enumerate(repository):
                                cart = position_to_cart(entry.position, model, representation)
                                sig = topology_signature(
                                    cart_to_absolute_path(cart, model), model, atlas_config.max_obstacles
                                )
                                topo_bin = topology_bin_from_signature(sig, atlas_config)
                                _, robust_bin = robustness_from_cost(entry.cost, atlas_config.n_robust_bins)
                                entry.atlas_cell_index = (topo_bin - 1) * atlas_config.n_robust_bins + robust_bin
                                repo_atlas[idx] = entry.atlas_cell_index
                        while len(repository) > int(ablation["nRep"]):
                            if use_atlas_archive and repo_atlas is not None:
                                delete_index = delete_one_with_weights(
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
                        keep_indices = np.random.choice(len(repository), size=int(ablation["nRep"]), replace=False)
                        repository = [repository[int(index)] for index in keep_indices]

            report_costs = np.array([entry.cost for entry in (repository if repository else particles)], dtype=float)
            if params.compute_metrics:
                if (
                    generation == 1
                    or generation == params.generations
                    or generation % int(ablation["metricInterval"]) == 0
                ):
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
        saved_paths: list[np.ndarray] = []
        for member_index, member in enumerate(final_members, start=1):
            cart = position_to_cart(member.position, model, representation)
            path_xyz = cart_to_absolute_path(cart, model)
            saved_paths.append(path_xyz)
            save_bp(run_dir / f"bp_{member_index}.mat", path_xyz, member.cost)
        if saved_paths:
            finite_cost = np.where(np.isfinite(final_costs), final_costs, 1e9)
            best_idx = int(np.argmin(np.sum(finite_cost, axis=1)))
            save_mat(run_dir / "fleet_paths.mat", {"uav1": np.asarray(saved_paths[best_idx], dtype=float)})
        mission_stats, feasible_mask = build_mission_stats(saved_paths, model)
        save_mat(run_dir / "mission_stats.mat", mission_stats)
        feasible_count = int(np.sum(feasible_mask))
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
