from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_benchmark.algorithms.apex_shade import _cdp_sort_indices, _obl_population
from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _constraint_violation_vector,
    _ensure_fleet_endpoints,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
)
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.algorithms.tactic_moea.topology import (
    _ISSUE_CLEARANCE,
    _ISSUE_PAIR,
    _ISSUE_TURN,
    ConflictTopology,
    _build_obstacle_matrix,
    _extract_topology,
    _is_feasible,
    _select_next_population,
    _topology_edit_vector,
    _update_topology_archive,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


def run_tactic_moea(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """TACTIC-MOEA: Topology-Aware Conflict-Targeted Iterative Correction MOEA."""
    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 211

    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)
    model["hardCollisionConstraint"] = True
    model["_tacticObstacleMatrix"] = _build_obstacle_matrix(model)

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dim = lower.size
    pop_size = max(6, int(params.population))
    archive_size = int(params.extra.get("nRep", pop_size))
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    run_indices = _resolve_run_indices(params)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in run_indices:
        np.random.seed(seed_value + int(run_idx) + 1000 * int(params.problem_index))
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
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
        archive: list[Candidate] = []

        population = np.random.uniform(lower, upper, size=(pop_size, dim))
        opposition = _obl_population(population, lower, upper)
        init_pool = np.vstack([population, opposition])
        init_candidates = _evaluate_population(init_pool, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        init_cv = _constraint_violation_vector(init_candidates, model)
        keep = _cdp_sort_indices(_candidate_matrix(init_candidates), init_cv)[:pop_size]
        population = init_pool[keep]
        candidates = [init_candidates[int(idx)] for idx in keep]
        archive = _update_topology_archive([], candidates, archive_size, model)

        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )
        pair_edits = 0.0
        clearance_edits = 0.0
        turn_edits = 0.0
        trade_edits = 0.0
        feasible_ratio_sum = 0.0
        unique_topology_sum = 0.0
        archive_topology_sum = 0.0

        for generation in range(1, params.generations + 1):
            progress = float(generation) / float(max(1, params.generations))
            offspring_vectors: list[np.ndarray] = []
            topologies_used: list[ConflictTopology] = []

            for parent in candidates:
                child_vector, topology = _topology_edit_vector(
                    parent=parent,
                    archive=archive,
                    model=model,
                    fleet_size=fleet_size,
                    n_waypoints=n_waypoints,
                    lower=lower,
                    upper=upper,
                    progress=progress,
                )
                offspring_vectors.append(np.asarray(child_vector, dtype=float).copy())
                topologies_used.append(topology)
                if topology.issue_code == _ISSUE_PAIR:
                    pair_edits += 1.0
                elif topology.issue_code == _ISSUE_CLEARANCE:
                    clearance_edits += 1.0
                elif topology.issue_code == _ISSUE_TURN:
                    turn_edits += 1.0
                else:
                    trade_edits += 1.0

            offspring_matrix = (
                np.stack(offspring_vectors, axis=0) if offspring_vectors else np.zeros((0, dim), dtype=float)
            )
            offspring_candidates = (
                _evaluate_population(
                    offspring_matrix,
                    model,
                    fleet_size=fleet_size,
                    n_waypoints=n_waypoints,
                )
                if offspring_matrix.size > 0
                else []
            )

            candidates = _select_next_population(list(candidates) + list(offspring_candidates), pop_size, model)
            if len(candidates) < pop_size and candidates:
                while len(candidates) < pop_size:
                    candidates.append(candidates[int(np.random.randint(0, len(candidates)))])
            population = np.stack([np.asarray(candidate.vector, dtype=float) for candidate in candidates], axis=0)
            archive = _update_topology_archive(
                archive, list(candidates) + list(offspring_candidates), archive_size, model
            )

            feasible_ratio_sum += float(
                np.mean(np.asarray([1.0 if _is_feasible(candidate) else 0.0 for candidate in candidates], dtype=float))
            )
            unique_topology_sum += float(len({_extract_topology(candidate, model).key for candidate in candidates}))
            archive_topology_sum += (
                float(len({_extract_topology(candidate, model).key for candidate in archive})) if archive else 0.0
            )

            if params.compute_metrics:
                matrix = _candidate_matrix(archive if archive else candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, matrix, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, matrix, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})

        final_candidates = archive if archive else candidates
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": "TACTIC-MOEA",
                "representation": "cart",
                "requestedPopulation": float(params.population),
                "effectivePopulation": float(pop_size),
                "archiveSize": float(archive_size),
                "tacticPairEdits": float(pair_edits),
                "tacticClearanceEdits": float(clearance_edits),
                "tacticTurnEdits": float(turn_edits),
                "tacticTradeEdits": float(trade_edits),
                "tacticFeasibleRatioMean": float(feasible_ratio_sum / max(1, params.generations)),
                "tacticUniqueTopologyMean": float(unique_topology_sum / max(1, params.generations)),
                "tacticArchiveTopologyMean": float(archive_topology_sum / max(1, params.generations)),
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


__all__ = [
    "ConflictTopology",
    "run_tactic_moea",
    "_extract_topology",
]
