from __future__ import annotations

"""SMPSO runner adapted for this benchmark.

Core update rules follow the PlatEMO SMPSO operator provided by the user:
- velocity update with constriction coefficient,
- deterministic boundary-back velocity damping,
- polynomial mutation.
"""

import copy
import time
from typing import Any

import numpy as np

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
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort, tournament_selection
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


def _clone_candidate(candidate: Candidate, vector: np.ndarray | None = None) -> Candidate:
    cloned_details = copy.deepcopy(candidate.details) if isinstance(candidate.details, dict) else {}
    return Candidate(
        vector=np.asarray(vector if vector is not None else candidate.vector, dtype=float).copy(),
        objective=np.asarray(candidate.objective, dtype=float).copy(),
        details=cloned_details,
    )


def _update_gbest(pool: list[Candidate], archive_size: int) -> tuple[list[Candidate], np.ndarray]:
    if not pool:
        return [], np.zeros(0, dtype=float)
    objective_matrix = _candidate_matrix(pool)
    if objective_matrix.size == 0:
        return [], np.zeros(0, dtype=float)

    front_no, _ = n_d_sort(objective_matrix.copy(), None, 1)
    first_front = np.where(front_no == 1)[0]
    if first_front.size == 0:
        return [], np.zeros(0, dtype=float)

    non_dominated = [_clone_candidate(pool[int(idx)]) for idx in first_front.tolist()]
    front_obj = _candidate_matrix(non_dominated)
    if front_obj.size == 0:
        return [], np.zeros(0, dtype=float)

    crowd = crowding_distance(front_obj, np.ones(front_obj.shape[0], dtype=float))
    order = np.argsort(-crowd, kind="mergesort")
    keep = order[: min(max(1, int(archive_size)), len(non_dominated))]
    return [non_dominated[int(idx)] for idx in keep.tolist()], crowd[keep]


def _update_pbest(
    pbest_vectors: np.ndarray,
    pbest_candidates: list[Candidate],
    population: np.ndarray,
    candidates: list[Candidate],
) -> tuple[np.ndarray, list[Candidate]]:
    if pbest_vectors.size == 0 or population.size == 0 or not pbest_candidates or not candidates:
        return pbest_vectors, pbest_candidates

    pbest_obj = _candidate_matrix(pbest_candidates)
    current_obj = _candidate_matrix(candidates)
    if pbest_obj.shape != current_obj.shape:
        return pbest_vectors, pbest_candidates

    # Matches PlatEMO SMPSO UpdatePbest:
    # replace = ~all(Population.objs>=Pbest.objs,2)
    replace = ~np.all(current_obj >= pbest_obj, axis=1)
    replace_idx = np.where(replace)[0]
    if replace_idx.size > 0:
        pbest_vectors[replace_idx] = population[replace_idx]
        for idx in replace_idx.tolist():
            pbest_candidates[idx] = _clone_candidate(candidates[idx], vector=population[idx])
    return pbest_vectors, pbest_candidates


def _smpso_operator(
    population: np.ndarray,
    velocity: np.ndarray,
    pbest: np.ndarray,
    leaders: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mutation_prob: float = 0.15,
    dis_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    n_pop, n_dim = population.shape
    if n_pop == 0 or n_dim == 0:
        return population, velocity

    lower_m = np.broadcast_to(np.asarray(lower, dtype=float).reshape(1, -1), (n_pop, n_dim))
    upper_m = np.broadcast_to(np.asarray(upper, dtype=float).reshape(1, -1), (n_pop, n_dim))
    span = upper_m - lower_m
    span_safe = np.where(span > 0.0, span, 1.0)

    # SMPSO velocity update (PlatEMO Operator.m)
    w = np.random.uniform(0.1, 0.5, size=(n_pop, 1))
    r1 = np.random.rand(n_pop, 1)
    r2 = np.random.rand(n_pop, 1)
    c1 = np.random.uniform(1.5, 2.5, size=(n_pop, 1))
    c2 = np.random.uniform(1.5, 2.5, size=(n_pop, 1))

    off_vel = w * velocity + c1 * r1 * (pbest - population) + c2 * r2 * (leaders - population)
    phi = np.maximum(4.0, c1 + c2)
    denominator = np.abs(2.0 - phi - np.sqrt(np.maximum(phi * phi - 4.0 * phi, 0.0)))
    denominator = np.where(denominator > 0.0, denominator, 1.0)
    off_vel = off_vel * (2.0 / denominator)
    delta = 0.5 * span
    off_vel = np.clip(off_vel, -delta, delta)
    off_dec = population + off_vel

    # Deterministic back
    repair = (off_dec < lower_m) | (off_dec > upper_m)
    if np.any(repair):
        off_vel[repair] = 0.001 * off_vel[repair]
    off_dec = np.clip(off_dec, lower_m, upper_m)

    # Polynomial mutation
    gate_prob = float(np.clip(mutation_prob, 0.0, 1.0))
    site1 = np.random.rand(n_pop, 1) < gate_prob
    site2 = np.random.rand(n_pop, n_dim) < (1.0 / max(1, n_dim))
    site = site1 & site2
    if np.any(site):
        mu = np.random.rand(n_pop, n_dim)
        dis_pow = float(dis_m) + 1.0
        inv_dis_pow = 1.0 / dis_pow

        temp = site & (mu <= 0.5)
        if np.any(temp):
            base = 2.0 * mu + (1.0 - 2.0 * mu) * np.power(np.clip(1.0 - (off_dec - lower_m) / span_safe, 0.0, 1.0), dis_pow)
            delta_q = np.power(np.clip(base, 0.0, None), inv_dis_pow) - 1.0
            off_dec[temp] = off_dec[temp] + span[temp] * delta_q[temp]

        temp = site & (mu > 0.5)
        if np.any(temp):
            base = 2.0 * (1.0 - mu) + 2.0 * (mu - 0.5) * np.power(
                np.clip(1.0 - (upper_m - off_dec) / span_safe, 0.0, 1.0),
                dis_pow,
            )
            delta_q = 1.0 - np.power(np.clip(base, 0.0, None), inv_dis_pow)
            off_dec[temp] = off_dec[temp] + span[temp] * delta_q[temp]

    off_dec = np.clip(off_dec, lower_m, upper_m)
    return off_dec, off_vel


def _run_fleet_smpso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
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
    archive_size = int(params.extra.get("nRep", params.population))
    metric_interval = int(params.extra.get("metricInterval", 20))
    mutation_prob = float(params.extra.get("mutationProb", 0.15))
    dis_m = float(params.extra.get("disM", 20.0))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    def _evaluate(vectors: np.ndarray) -> list[Candidate]:
        return _evaluate_population(
            vectors,
            model=model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            representation="cart",
        )

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
        population = np.random.uniform(lower, upper, size=(params.population, dimensions))
        velocity = np.zeros_like(population)
        candidates = _evaluate(population)

        pbest_vectors = population.copy()
        pbest_candidates = [_clone_candidate(candidates[idx], vector=population[idx]) for idx in range(len(candidates))]
        gbest, crowd_dis = _update_gbest(pbest_candidates, archive_size)
        if not gbest:
            gbest = [_clone_candidate(candidates[idx], vector=population[idx]) for idx in range(min(len(candidates), archive_size))]
            crowd_dis = np.ones(len(gbest), dtype=float)

        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            if gbest:
                gbest_vectors = np.stack([candidate.vector for candidate in gbest], axis=0)
                if gbest_vectors.shape[0] == 1:
                    leader_indices = np.zeros(params.population, dtype=int)
                else:
                    leader_indices = tournament_selection(2, params.population, -np.asarray(crowd_dis, dtype=float))
                leaders = gbest_vectors[leader_indices]
            else:
                leaders = pbest_vectors.copy()

            population, velocity = _smpso_operator(
                population=population,
                velocity=velocity,
                pbest=pbest_vectors,
                leaders=leaders,
                lower=lower,
                upper=upper,
                mutation_prob=mutation_prob,
                dis_m=dis_m,
            )
            candidates = _evaluate(population)

            gbest, crowd_dis = _update_gbest(gbest + candidates, archive_size)
            if not gbest:
                gbest = [_clone_candidate(candidates[idx], vector=population[idx]) for idx in range(min(len(candidates), archive_size))]
                crowd_dis = np.ones(len(gbest), dtype=float)

            pbest_vectors, pbest_candidates = _update_pbest(
                pbest_vectors=pbest_vectors,
                pbest_candidates=pbest_candidates,
                population=population,
                candidates=candidates,
            )

            if params.compute_metrics:
                report_candidates = gbest if gbest else candidates
                report_matrix = _candidate_matrix(report_candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    if report_matrix.size > 0:
                        hv_history[generation - 1, 0] = cal_metric(1, report_matrix, params.problem_index, objective_count)
                        hv_history[generation - 1, 1] = cal_metric(2, report_matrix, params.problem_index, objective_count)
                    else:
                        hv_history[generation - 1] = 0.0
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        final_candidates = gbest if gbest else candidates

        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
        )

        if params.compute_metrics:
            final_matrix = _candidate_matrix(final_candidates)
            if final_matrix.size > 0:
                run_scores[run_idx - 1] = np.array(
                    [
                        cal_metric(1, final_matrix, params.problem_index, objective_count),
                        cal_metric(2, final_matrix, params.problem_index, objective_count),
                    ],
                    dtype=float,
                )
            else:
                run_scores[run_idx - 1] = 0.0

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_smpso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_smpso(model, params)
