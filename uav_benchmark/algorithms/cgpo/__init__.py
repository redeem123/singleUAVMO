"""Constraint-Graph Policy Optimizer (CGPO).

Published contribution: a *three-mechanism* graph-aware multi-objective
evolutionary algorithm for fleet UAV path planning.

    1. **CIG** (Constraint Interaction Graph) -- a per-generation typed graph
       over fleet waypoints whose tension field encodes terrain, altitude,
       obstacle, turn, smoothing, and pairwise-separation pressures.
    2. **PPF** (Pareto Pressure Field) -- softmax parent-selection pressure
       that combines NSGA-II rank, crowding, normalised constraint violation,
       and a graph-tension-aware boundary stratum.
    3. **OVO** (Orchestrated Variation Operator) -- tension-weighted parent
       blend with anisotropic Gaussian perturbation and an optional
       waypoint-aligned fleet coordination pass.

Selection follows standard NSGA-II constraint-domination semantics.  The
initial population is uniformly random subject to domain bounds; no
constructive priors are used.

For paper-grade comparisons, this is the algorithm registered as ``CGPO``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_benchmark.algorithms.cgpo.cig import build_constraint_interaction_graph
from uav_benchmark.algorithms.cgpo.core import (
    _CGPO_CANDIDATE_EVAL_KEY,
    _CGPO_CONTROL_KEY,
    _CGPO_PROXY_EVAL_KEY,
    _OBJECTIVE_COUNT,
    _AblationControls,
    _candidate_from_paths,
    _clone_paths,
    _constraint_pressure,
    _controls,
    _controls_from_extra,
    _float_any,
    _graph_kwargs,
    _initial_population,
    _is_feasible,
    _project_fleet,
    _random_paths,
    _safe_objective_matrix,
    _select_candidates,
    _shape_feasible_geometry,
    _update_archives,
)
from uav_benchmark.algorithms.cgpo.ovo import OVOTrace, orchestrated_variation
from uav_benchmark.algorithms.cgpo.ppf import compute_pareto_pressure_field
from uav_benchmark.algorithms.cgpo.trace import CGPOTrace
from uav_benchmark.algorithms.shared.fleet_runner import (
    _ensure_fleet_endpoints,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
)
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.utils.random import seed_everything

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_cgpo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    model = dict(model)
    controls = _controls_from_extra(dict(params.extra) if isinstance(params.extra, dict) else {})
    model[_CGPO_CONTROL_KEY] = controls
    n_waypoints = int(model.get("n", 7))
    n_points = max(3, n_waypoints + 2)
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
    model["hardCollisionConstraint"] = bool(params.extra.get("hardCollisionConstraint", True))

    archive_size = int(params.extra.get("nRep", max(params.population, 100)))
    metric_interval = int(params.extra.get("metricInterval", 10))
    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    run_indices = _resolve_run_indices(params)
    resume_existing = bool(params.extra.get("resumeExistingRuns", True))

    # PPF hyperparameters
    extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    ppf_boundary_epsilon = _float_any(extra, ("cgpoPpfBoundaryEpsilon", "boundary_epsilon"), 0.05)
    ppf_temperature = _float_any(extra, ("cgpoPpfPressureTemperature", "pressure_temperature"), 1.0)
    ppf_diversity_weight = _float_any(extra, ("cgpoPpfDiversityWeight", "diversity_weight"), 0.25)
    ppf_boundary_weight = _float_any(extra, ("cgpoPpfBoundaryWeight", "boundary_weight"), 0.25)
    ppf_rank_weight = _float_any(extra, ("cgpoPpfRankWeight", "rank_weight"), 0.75)
    ppf_violation_weight = _float_any(extra, ("cgpoPpfViolationWeight", "violation_weight"), 1.10)
    model["_cgpoSmoothInitRatio"] = float(
        np.clip(_float_any(extra, ("cgpoSmoothInitRatio", "smooth_init_ratio"), 0.75), 0.0, 1.0)
    )
    model["_cgpoFeasibilityShapingStrength"] = float(
        np.clip(_float_any(extra, ("cgpoFeasibilityShapingStrength", "feasibility_shaping_strength"), 0.45), 0.0, 1.0)
    )

    for run_idx in run_indices:
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing:
            resumed = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=_OBJECTIVE_COUNT,
                compute_metrics=params.compute_metrics,
            )
            if resumed is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resumed
                continue

        run_start = time.perf_counter()
        seed_everything(seed_value + run_idx)
        rng = np.random.default_rng(seed_value + run_idx)
        model[_CGPO_CANDIDATE_EVAL_KEY] = 0
        model[_CGPO_PROXY_EVAL_KEY] = 0

        population = _initial_population(model, fleet_size, n_points, params.population, rng)
        initial_feasible_count = int(sum(1 for c in population if _is_feasible(c, model)))
        initial_feasible_ratio = float(initial_feasible_count / max(1, len(population)))
        feasible_archive: list[Candidate] = []
        relaxed_archive: list[Candidate] = []
        unconstrained_archive: list[Candidate] = []
        feasible_archive, relaxed_archive, unconstrained_archive = _update_archives(
            population, feasible_archive, relaxed_archive, unconstrained_archive, model, archive_size
        )
        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )
        feasibility_pressure_trace = np.zeros(params.generations, dtype=float)
        feasible_trace = np.zeros(params.generations, dtype=float)
        conflict_trace = np.zeros(params.generations, dtype=float)
        cgpo_trace = CGPOTrace()

        for generation in range(1, params.generations + 1):
            offspring = _step(
                generation=generation,
                generations=params.generations,
                population=population,
                feasible_archive=feasible_archive,
                relaxed_archive=relaxed_archive,
                unconstrained_archive=unconstrained_archive,
                model=model,
                rng=rng,
                params=params,
                fleet_size=fleet_size,
                n_points=n_points,
                cgpo_trace=cgpo_trace,
                feasibility_pressure_trace=feasibility_pressure_trace,
                feasible_trace=feasible_trace,
                conflict_trace=conflict_trace,
                ppf_boundary_epsilon=ppf_boundary_epsilon,
                ppf_temperature=ppf_temperature,
                ppf_diversity_weight=ppf_diversity_weight,
                ppf_boundary_weight=ppf_boundary_weight,
                ppf_rank_weight=ppf_rank_weight,
                ppf_violation_weight=ppf_violation_weight,
            )
            combined = population + offspring + feasible_archive
            population = _select_candidates(combined, model, params.population)
            feasible_archive, relaxed_archive, unconstrained_archive = _update_archives(
                population, feasible_archive, relaxed_archive, unconstrained_archive, model, archive_size
            )

            if params.compute_metrics:
                metric_source = feasible_archive if feasible_archive else population
                matrix = _candidate_matrix(metric_source)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = (
                        cal_metric(1, matrix, params.problem_index, _OBJECTIVE_COUNT) if matrix.size else 0.0
                    )
                    hv_history[generation - 1, 1] = (
                        cal_metric(2, matrix, params.problem_index, _OBJECTIVE_COUNT) if matrix.size else 0.0
                    )
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        final_pool = feasible_archive + population
        final_candidates = _select_candidates(final_pool, model, archive_size)
        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        cgpo_trace_arrays = cgpo_trace.as_trace()
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=_OBJECTIVE_COUNT,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            rl_trace={
                "feasible": feasible_trace,
                "conflict": conflict_trace,
                "repair": feasibility_pressure_trace,
                "feasibility_pressure": feasibility_pressure_trace,
                **cgpo_trace_arrays,
            },
            run_metadata=_run_metadata(
                params,
                controls,
                fleet_size,
                model,
                initial_feasible_count,
                initial_feasible_ratio,
                feasible_archive,
                relaxed_archive,
                unconstrained_archive,
            ),
        )
        for trace_name, trace_values in cgpo_trace_arrays.items():
            save_mat(run_dir / f"cgpo_{trace_name}.mat", {f"cgpo_{trace_name}": np.asarray(trace_values, dtype=float)})
        if params.compute_metrics:
            matrix = _candidate_matrix(final_candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, matrix, params.problem_index, _OBJECTIVE_COUNT) if matrix.size else 0.0,
                    cal_metric(2, matrix, params.problem_index, _OBJECTIVE_COUNT) if matrix.size else 0.0,
                ],
                dtype=float,
            )

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


# ---------------------------------------------------------------------------
# Per-generation step (CIG -> PPF -> OVO)
# ---------------------------------------------------------------------------


def _step(
    generation: int,
    generations: int,
    population: list[Candidate],
    feasible_archive: list[Candidate],
    relaxed_archive: list[Candidate],
    unconstrained_archive: list[Candidate],
    model: dict[str, Any],
    rng: np.random.Generator,
    params: BenchmarkParams,
    fleet_size: int,
    n_points: int,
    cgpo_trace: CGPOTrace,
    feasibility_pressure_trace: np.ndarray,
    feasible_trace: np.ndarray,
    conflict_trace: np.ndarray,
    ppf_boundary_epsilon: float,
    ppf_temperature: float,
    ppf_diversity_weight: float,
    ppf_boundary_weight: float,
    ppf_rank_weight: float,
    ppf_violation_weight: float,
) -> list[Candidate]:
    controls = _controls(model)
    selection_pool = population + feasible_archive
    if not selection_pool:
        selection_pool = population

    # CIG: build the constraint-interaction graph once per pool member.
    graph_kwargs = _graph_kwargs(model)
    pool_graphs = [
        build_constraint_interaction_graph(
            candidate.details.get("paths", []),
            model,
            **graph_kwargs,
        )
        for candidate in selection_pool
    ]
    pool_objective = _safe_objective_matrix(selection_pool)
    pool_cv = np.asarray([_constraint_pressure(c, model) for c in selection_pool], dtype=float)
    pool_feasible = np.asarray([_is_feasible(c, model) for c in selection_pool], dtype=bool)

    # PPF: derive parent-selection pressure from the graph + violations + ranks.
    pressure = compute_pareto_pressure_field(
        objective=pool_objective,
        violations=pool_cv,
        feasible=pool_feasible,
        graphs=pool_graphs,
        enabled=controls.ppf_pressure_enabled,
        boundary_epsilon=ppf_boundary_epsilon,
        pressure_temperature=ppf_temperature,
        diversity_weight=ppf_diversity_weight,
        boundary_weight=ppf_boundary_weight,
        rank_weight=ppf_rank_weight,
        violation_weight=ppf_violation_weight,
    )
    feasible_ratio = float(np.mean(pool_feasible)) if pool_feasible.size else 0.0
    conflict_mean = (
        float(np.mean([float(c.details.get("conflictRate", 0.0)) for c in selection_pool])) if selection_pool else 0.0
    )
    mean_feasibility_pressure = (
        float(np.mean(pressure.feasibility_pressure)) if pressure.feasibility_pressure.size else 0.0
    )
    feasibility_pressure_trace[generation - 1] = mean_feasibility_pressure
    feasible_trace[generation - 1] = feasible_ratio
    conflict_trace[generation - 1] = conflict_mean

    # OVO: tension-aware parent blend + perturbation.
    offspring: list[Candidate] = []
    ovo_scales: list[float] = []
    ovo_clusters: list[float] = []
    while len(offspring) < params.population:
        if len(selection_pool) < 2 or pressure.parent_probability.size != len(selection_pool):
            parent_indices = rng.integers(0, len(selection_pool), size=2)
        else:
            parent_indices = rng.choice(
                len(selection_pool),
                size=2,
                replace=True,
                p=pressure.parent_probability,
            )
        parent_a_idx = int(parent_indices[0])
        parent_b_idx = int(parent_indices[1])
        exploration_scale = (
            float(pressure.exploration_scale[parent_a_idx]) if pressure.exploration_scale.size > parent_a_idx else 0.075
        )

        if controls.ovo_variation_enabled:
            raw_paths, ovo_trace = orchestrated_variation(
                selection_pool[parent_a_idx],
                selection_pool[parent_b_idx],
                pool_graphs[parent_a_idx],
                pool_graphs[parent_b_idx],
                model,
                rng,
                exploration_scale=exploration_scale,
                use_coordination=controls.ovo_coordination_enabled,
            )
        else:
            parent = selection_pool[parent_a_idx]
            raw_paths = _mutate_paths(
                parent.details.get("paths", []),
                model,
                rng,
                scale=exploration_scale,
            )
            ovo_trace = OVOTrace(
                perturbation_scale=float(exploration_scale),
                perturbed_waypoints=0,
                coordinated_clusters=0,
                parent_blend_entropy=0.0,
            )
        if not raw_paths:
            raw_paths = _random_paths(model, fleet_size, n_points, rng)

        # Domain projection (always) -- this is just bounds clipping, not repair.
        paths = _project_fleet(raw_paths, model)
        child_graph = build_constraint_interaction_graph(paths, model, **graph_kwargs)
        paths = _shape_feasible_geometry(
            paths,
            model,
            strength=float(model.get("_cgpoFeasibilityShapingStrength", 0.45)),
            graph_tension=child_graph.tension,
            pairwise=controls.ovo_coordination_enabled,
        )

        ovo_scales.append(float(ovo_trace.perturbation_scale))
        ovo_clusters.append(float(ovo_trace.coordinated_clusters))
        offspring.append(_candidate_from_paths(paths, model))

    if controls.trace_enabled:
        cgpo_trace.cig_mean_tension.append(
            float(np.mean([g.mean_tension for g in pool_graphs])) if pool_graphs else 0.0
        )
        cgpo_trace.cig_max_tension.append(float(np.max([g.max_tension for g in pool_graphs])) if pool_graphs else 0.0)
        cgpo_trace.cig_terrain_edges.append(
            float(np.mean([g.terrain_edges for g in pool_graphs])) if pool_graphs else 0.0
        )
        cgpo_trace.cig_obstacle_edges.append(
            float(np.mean([g.obstacle_edges for g in pool_graphs])) if pool_graphs else 0.0
        )
        cgpo_trace.cig_turn_edges.append(float(np.mean([g.turn_edges for g in pool_graphs])) if pool_graphs else 0.0)
        cgpo_trace.cig_smoothing_edges.append(
            float(np.mean([g.objective_edges for g in pool_graphs])) if pool_graphs else 0.0
        )
        cgpo_trace.cig_pairwise_edges.append(
            float(np.mean([g.pairwise_edges for g in pool_graphs])) if pool_graphs else 0.0
        )
        cgpo_trace.ppf_feasibility_pressure.append(mean_feasibility_pressure)
        cgpo_trace.ppf_boundary_mass.append(float(pressure.boundary_mass))
        cgpo_trace.ppf_pressure_entropy.append(float(pressure.pressure_entropy))
        cgpo_trace.ovo_perturbation_scale.append(float(np.mean(ovo_scales)) if ovo_scales else 0.0)
        cgpo_trace.ovo_coordinated_clusters.append(float(np.mean(ovo_clusters)) if ovo_clusters else 0.0)
        cgpo_trace.offspring_feasible_ratio.append(
            float(np.mean([1.0 if _is_feasible(c, model) else 0.0 for c in offspring])) if offspring else 0.0
        )
        cgpo_trace.candidate_evaluations.append(float(model.get(_CGPO_CANDIDATE_EVAL_KEY, 0)))
        cgpo_trace.gfp_projection_norm.append(0.0)
        cgpo_trace.gfp_violation_delta.append(0.0)
        cgpo_trace.gfp_acceptance_rate.append(0.0)
        cgpo_trace.projection_proxy_evaluations.append(float(model.get(_CGPO_PROXY_EVAL_KEY, 0)))

    return offspring


def _mutate_paths(
    paths: list[np.ndarray], model: dict[str, Any], rng: np.random.Generator, scale: float
) -> list[np.ndarray]:
    """Lightweight Gaussian mutation used as the ``no_ovo_variation`` fallback."""
    mutated = _clone_paths(paths)
    span = max(float(model["xmax"]) - float(model["xmin"]), float(model["ymax"]) - float(model["ymin"]))
    z_span = float(model["zmax"]) - float(model["zmin"])
    for path in mutated:
        if path.shape[0] <= 2:
            continue
        xy_noise = rng.normal(0.0, max(1e-9, scale) * span, size=(path.shape[0] - 2, 2))
        z_noise = rng.normal(0.0, max(1e-9, scale) * z_span, size=path.shape[0] - 2)
        path[1:-1, :2] += xy_noise
        path[1:-1, 2] += z_noise
    return _project_fleet(mutated, model)


def _run_metadata(
    params: BenchmarkParams,
    controls: _AblationControls,
    fleet_size: int,
    model: dict[str, Any],
    initial_feasible_count: int,
    initial_feasible_ratio: float,
    feasible_archive: list[Candidate],
    relaxed_archive: list[Candidate],
    unconstrained_archive: list[Candidate],
) -> dict[str, Any]:
    return {
        "algorithmName": str(params.algorithm or "CGPO"),
        "optimizerBackend": "CGPO native Python fleet optimizer",
        "pythonProblemEvaluation": True,
        "benchmarkObjectiveDuringSearch": True,
        "nativePopulationLoop": True,
        "nativeGenerationLoop": True,
        "finalReporting": "shared_multi_objective_benchmark",
        "cgpoMethodName": str(params.extra.get("cgpoMethodName", "Constraint-Graph Policy Optimizer")),
        "cgpoPaperRole": str(params.extra.get("cgpoPaperRole", "ablation_or_experimental")),
        "cgpoAblationVariant": str(params.extra.get("cgpoAblationVariant", "full")),
        "generations": int(params.generations),
        "population": int(params.population),
        "seed": int(
            (int(params.seed) if params.seed is not None else 42) + int(params.extra.get("_runIndexOffset", 0))
        ),
        "cgpoInitialFeasibleCount": int(initial_feasible_count),
        "cgpoInitialFeasibleRatio": float(initial_feasible_ratio),
        "cgpoFeasibleArchiveSize": int(len(feasible_archive)),
        "cgpoRelaxedArchiveSize": int(len(relaxed_archive)),
        "cgpoUnconstrainedArchiveSize": int(len(unconstrained_archive)),
        "cgpoFleetSize": int(fleet_size),
        "cgpoCandidateEvaluations": int(model.get(_CGPO_CANDIDATE_EVAL_KEY, 0)),
        "cgpoProjectionProxyEvaluations": int(model.get(_CGPO_PROXY_EVAL_KEY, 0)),
        "cgpoTotalEvaluationCalls": int(model.get(_CGPO_CANDIDATE_EVAL_KEY, 0))
        + int(model.get(_CGPO_PROXY_EVAL_KEY, 0)),
        "cgpoTotalMissionEvaluations": int(model.get(_CGPO_CANDIDATE_EVAL_KEY, 0)),
        "cgpoControls": {
            "cigEdgeCouplingEnabled": bool(controls.cig_edge_coupling_enabled),
            "cigTerrainEdgesEnabled": bool(controls.cig_terrain_edges_enabled),
            "cigObstacleEdgesEnabled": bool(controls.cig_obstacle_edges_enabled),
            "cigTurnEdgesEnabled": bool(controls.cig_turn_edges_enabled),
            "cigSmoothingEdgesEnabled": bool(controls.cig_smoothing_edges_enabled),
            "cigPairwiseEdgesEnabled": bool(controls.cig_pairwise_edges_enabled),
            "ppfPressureEnabled": bool(controls.ppf_pressure_enabled),
            "ovoVariationEnabled": bool(controls.ovo_variation_enabled),
            "ovoCoordinationEnabled": bool(controls.ovo_coordination_enabled),
            "traceEnabled": bool(controls.trace_enabled),
        },
    }


__all__ = [
    "_AblationControls",
    "_CGPO_CONTROL_KEY",
    "_controls_from_extra",
    "_initial_population",
    "run_cgpo",
]
