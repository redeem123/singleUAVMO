from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_benchmark.algorithms.apex_shade import (
    SHADEMemory,
    _cdp_sort_indices,
    _cdp_wins,
    _elite_local_search,
    _obl_population,
    _update_pareto_archive,
)
from uav_benchmark.algorithms.fastr_moea.operators import (
    _ALL_OPERATORS,
    _OP_BRIDGE,
    _OP_DE,
    _OP_REPAIR,
    _allowed_operators,
    _bridge_child,
    _build_relaxed_model,
    _de_child,
    _objective_diversity,
    _operator_bias,
    _operator_reward,
    _OperatorBandit,
    _OperatorChoice,
    _repair_child,
    _select_transfer_vectors,
    _stage_partition,
    _strict_feasible_ratio,
    _strict_objective_sum,
)
from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _constraint_violation,
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
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.r2_archive import uniform_weight_vectors
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


def run_fastr_moea(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """FASTR-MOEA: Fleet-Aware Staged Transfer-and-Repair MOEA."""
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
    model["hardCollisionConstraint"] = True

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dim = lower.size
    pop_size = max(8, int(params.population))
    archive_size = int(params.extra.get("nRep", pop_size))
    metric_interval = int(params.extra.get("metricInterval", 20))
    transfer_base = float(params.extra.get("fastrTransferBase", 0.08))
    transfer_pressure = float(params.extra.get("fastrTransferPressure", 0.18))
    late_local_search_start = float(params.extra.get("fastrLocalSearchStart", 0.55))
    late_local_search_share = float(params.extra.get("fastrLocalSearchShare", 0.06))

    r2_weights = uniform_weight_vectors(
        n_obj=objective_count, n_divisions=int(params.extra.get("fastrR2Divisions", 15))
    )
    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    run_indices = _resolve_run_indices(params)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in run_indices:
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
        bandit = _OperatorBandit(exploration=float(params.extra.get("fastrBanditExplore", 0.30)))
        shade = SHADEMemory(H=int(params.extra.get("fastrHistory", 10)))
        strict_z_ideal = np.full(objective_count, np.inf, dtype=float)
        relaxed_z_ideal = np.full(objective_count, np.inf, dtype=float)
        strict_archive: list[Candidate] = []
        relaxed_archive: list[Candidate] = []

        population = np.random.uniform(lower, upper, size=(pop_size, dim))
        opposition = _obl_population(population, lower, upper)
        init_pool = np.vstack([population, opposition])

        init_candidates = _evaluate_population(init_pool, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        init_cv = _constraint_violation_vector(init_candidates, model)
        keep = _cdp_sort_indices(_candidate_matrix(init_candidates), init_cv)[:pop_size]
        population = init_pool[keep]
        candidates = [init_candidates[int(idx)] for idx in keep]

        relaxed_init_model = _build_relaxed_model(model, params, progress=0.0)
        relaxed_init_candidates = _evaluate_population(
            init_pool, relaxed_init_model, fleet_size=fleet_size, n_waypoints=n_waypoints
        )
        strict_archive, strict_z_ideal = _update_pareto_archive(
            [],
            candidates,
            archive_size,
            r2_weights,
            strict_z_ideal,
            model,
        )
        relaxed_archive, relaxed_z_ideal = _update_pareto_archive(
            [],
            relaxed_init_candidates,
            archive_size,
            r2_weights,
            relaxed_z_ideal,
            relaxed_init_model,
        )

        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )
        operator_usage = {name: 0.0 for name in _ALL_OPERATORS}
        stage_explore_sum = 0.0
        stage_exploit_sum = 0.0
        stage_repair_sum = 0.0
        feasible_ratio_sum = 0.0
        diversity_sum = 0.0
        transfer_sum = 0.0

        for generation in range(1, params.generations + 1):
            progress = float(generation) / float(max(1, params.generations))
            relaxed_model = _build_relaxed_model(model, params, progress=progress)
            relaxed_current = _evaluate_population(
                population, relaxed_model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
            relaxed_archive, relaxed_z_ideal = _update_pareto_archive(
                relaxed_archive,
                relaxed_current,
                archive_size,
                r2_weights,
                relaxed_z_ideal,
                relaxed_model,
            )

            feasible_ratio = _strict_feasible_ratio(candidates)
            diversity = _objective_diversity(strict_archive if strict_archive else candidates)
            split = _stage_partition(candidates, model, progress, feasible_ratio, diversity)
            pressure = 1.0 - feasible_ratio

            stage_explore_sum += split.explore_share
            stage_exploit_sum += split.exploit_share
            stage_repair_sum += split.repair_share
            feasible_ratio_sum += feasible_ratio
            diversity_sum += diversity

            offspring_vectors: list[np.ndarray] = []
            offspring_meta: list[_OperatorChoice] = []

            for stage_name, indices in (
                ("explore", split.explore),
                ("exploit", split.exploit),
                ("repair", split.repair),
            ):
                allowed = _allowed_operators(stage_name)
                bias = _operator_bias(stage_name, progress, pressure, diversity)
                for parent_idx in indices.tolist():
                    op_name = bandit.select(allowed, bias)
                    if op_name == _OP_DE:
                        child_vec, f_value, cr_value = _de_child(
                            parent_idx=parent_idx,
                            population=population,
                            candidates=candidates,
                            strict_archive=strict_archive,
                            relaxed_archive=relaxed_archive,
                            shade=shade,
                            lower=lower,
                            upper=upper,
                            model=model,
                            pressure=pressure,
                        )
                    elif op_name == _OP_BRIDGE:
                        child_vec = _bridge_child(
                            parent_idx=parent_idx,
                            population=population,
                            candidates=candidates,
                            strict_archive=strict_archive,
                            relaxed_archive=relaxed_archive,
                            model=model,
                            fleet_size=fleet_size,
                            n_waypoints=n_waypoints,
                            lower=lower,
                            upper=upper,
                            pressure=pressure,
                        )
                        f_value = None
                        cr_value = None
                    else:
                        child_vec = _repair_child(
                            parent_idx=parent_idx,
                            population=population,
                            strict_archive=strict_archive,
                            model=model,
                            fleet_size=fleet_size,
                            n_waypoints=n_waypoints,
                            lower=lower,
                            upper=upper,
                        )
                        f_value = None
                        cr_value = None
                    offspring_vectors.append(np.asarray(child_vec, dtype=float).copy())
                    offspring_meta.append(
                        _OperatorChoice(
                            name=op_name,
                            parent_idx=int(parent_idx),
                            stage=stage_name,
                            f_value=f_value,
                            cr_value=cr_value,
                        )
                    )
                    operator_usage[op_name] += 1.0

            offspring_matrix = (
                np.stack(offspring_vectors, axis=0) if offspring_vectors else np.zeros((0, dim), dtype=float)
            )
            offspring_candidates = (
                _evaluate_population(offspring_matrix, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
                if offspring_matrix.size > 0
                else []
            )

            success_f: list[float] = []
            success_cr: list[float] = []
            success_delta: list[float] = []

            for meta, child in zip(offspring_meta, offspring_candidates, strict=False):
                parent = candidates[int(meta.parent_idx)]
                reward = _operator_reward(child, parent, model)
                bandit.update(meta.name, reward)

                child_cv = float(max(0.0, _constraint_violation(child, model)))
                parent_cv = float(max(0.0, _constraint_violation(parent, model)))
                if _cdp_wins(child_cv, parent_cv, child.objective, parent.objective):
                    population[int(meta.parent_idx)] = np.asarray(child.vector, dtype=float).copy()
                    candidates[int(meta.parent_idx)] = child
                    if meta.name == _OP_DE and meta.f_value is not None and meta.cr_value is not None:
                        success_f.append(float(meta.f_value))
                        success_cr.append(float(meta.cr_value))
                        success_delta.append(max(1e-6, reward + 1.0))

            transfer_count = 0
            transfer_ratio = transfer_base + transfer_pressure * pressure
            if relaxed_archive:
                transfer_count = max(1, int(round(pop_size * transfer_ratio)))
            transfer_vectors = _select_transfer_vectors(relaxed_archive, strict_archive, transfer_count)
            transfer_candidates = (
                _evaluate_population(transfer_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
                if transfer_vectors.size > 0
                else []
            )
            transfer_sum += float(len(transfer_candidates))

            if transfer_candidates:
                worst_order = sorted(
                    range(len(candidates)),
                    key=lambda idx: (
                        0 if _constraint_violation(candidates[idx], model) > 0.0 else 1,
                        -float(_constraint_violation(candidates[idx], model)),
                        -_strict_objective_sum(candidates[idx]),
                    ),
                )
                for target_idx, transfer in zip(worst_order, transfer_candidates, strict=False):
                    current = candidates[int(target_idx)]
                    transfer_cv = float(max(0.0, _constraint_violation(transfer, model)))
                    current_cv = float(max(0.0, _constraint_violation(current, model)))
                    if _cdp_wins(transfer_cv, current_cv, transfer.objective, current.objective):
                        population[int(target_idx)] = np.asarray(transfer.vector, dtype=float).copy()
                        candidates[int(target_idx)] = transfer

            local_search_candidates: list[Candidate] = []
            has_feasible_archive = any(
                float(candidate.details.get("feasible", 0.0)) > 0.5 for candidate in strict_archive
            )
            if strict_archive and has_feasible_archive and progress >= late_local_search_start:
                local_trials = max(0, int(round(pop_size * late_local_search_share)))
                if local_trials > 0:
                    sigma = max(0.02, 0.10 * (1.0 - progress))
                    local_vectors = _elite_local_search(strict_archive, lower, upper, local_trials, sigma)
                    local_search_candidates = _evaluate_population(
                        local_vectors,
                        model,
                        fleet_size=fleet_size,
                        n_waypoints=n_waypoints,
                    )

            if success_f:
                shade.update(success_f, success_cr, success_delta)

            archive_pool = (
                list(candidates)
                + list(offspring_candidates)
                + list(transfer_candidates)
                + list(local_search_candidates)
            )
            strict_archive, strict_z_ideal = _update_pareto_archive(
                strict_archive,
                archive_pool,
                archive_size,
                r2_weights,
                strict_z_ideal,
                model,
            )

            if params.compute_metrics:
                matrix = _candidate_matrix(strict_archive if strict_archive else candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, matrix, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, matrix, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})

        final_candidates = strict_archive if strict_archive else candidates
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": "FASTR-MOEA",
                "representation": "cart",
                "requestedPopulation": float(params.population),
                "effectivePopulation": float(pop_size),
                "archiveSize": float(archive_size),
                "fastrExploreShareMean": float(stage_explore_sum / max(1, params.generations)),
                "fastrExploitShareMean": float(stage_exploit_sum / max(1, params.generations)),
                "fastrRepairShareMean": float(stage_repair_sum / max(1, params.generations)),
                "fastrFeasibleRatioMean": float(feasible_ratio_sum / max(1, params.generations)),
                "fastrArchiveDiversityMean": float(diversity_sum / max(1, params.generations)),
                "fastrTransferMean": float(transfer_sum / max(1, params.generations)),
                "fastrOperatorCountDE": float(operator_usage[_OP_DE]),
                "fastrOperatorCountBridge": float(operator_usage[_OP_BRIDGE]),
                "fastrOperatorCountRepair": float(operator_usage[_OP_REPAIR]),
                "fastrOperatorValueDE": float(bandit.values.get(_OP_DE, 0.0)),
                "fastrOperatorValueBridge": float(bandit.values.get(_OP_BRIDGE, 0.0)),
                "fastrOperatorValueRepair": float(bandit.values.get(_OP_REPAIR, 0.0)),
                "fastrOperatorPullsDE": float(bandit.counts.get(_OP_DE, 0)),
                "fastrOperatorPullsBridge": float(bandit.counts.get(_OP_BRIDGE, 0)),
                "fastrOperatorPullsRepair": float(bandit.counts.get(_OP_REPAIR, 0)),
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
    "run_fastr_moea",
    "_build_relaxed_model",
]
