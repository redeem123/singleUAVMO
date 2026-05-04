from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _constraint_violation,
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
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.utils.random import seed_everything

_OBJECTIVE_COUNT = 4
_SINGLE_CONSTRAINT_STRATEGY = 0


@dataclass(slots=True)
class AdvsSelection:
    selected: np.ndarray
    strategy: int
    reason: str


@dataclass(slots=True)
class JadeState:
    archive: np.ndarray
    mu_f: float
    mu_cr: float


def _empty_int_selection() -> AdvsSelection:
    return AdvsSelection(np.zeros(0, dtype=int), 3, "empty")


def _pad_vector(values: np.ndarray, size: int, dtype: Any) -> np.ndarray:
    result = np.zeros(int(size), dtype=dtype)
    values = np.asarray(values, dtype=dtype).reshape(-1)
    result[: values.size] = values
    return result


def _as_2d_archive(archive: np.ndarray, width: int | None = None) -> np.ndarray:
    arr = np.asarray(archive, dtype=float)
    if arr.size == 0:
        resolved_width = width
        if resolved_width is None:
            resolved_width = arr.shape[1] if arr.ndim == 2 else 0
        return np.zeros((0, int(resolved_width)), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def advs_probabilities(conflict_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(conflict_counts, dtype=float).reshape(-1)
    if counts.size == 0:
        return np.zeros(0, dtype=float)
    positive = np.maximum(counts, 0.0)
    if float(np.sum(positive)) <= 0.0:
        return np.full(positive.shape, 1.0 / float(positive.size), dtype=float)
    return positive / float(np.sum(positive))


def update_strategy_weights(
    weights: np.ndarray,
    strategy: int,
    improved: bool,
    rho: float = 0.05,
) -> np.ndarray:
    updated = np.asarray(weights, dtype=float).reshape(-1).copy()
    if strategy < 1 or strategy > updated.size:
        return np.clip(updated, 0.1, 10.0)
    index = int(strategy) - 1
    if improved:
        updated[index] = min((1.0 - float(rho)) * updated[index] + float(rho) * 0.5, 10.0)
    else:
        updated[index] = max((1.0 - float(rho)) * updated[index] - float(rho) * 0.1, 0.1)
    return np.clip(updated, 0.1, 10.0)


def select_advs_variables(
    single_uav_infeasible: np.ndarray,
    conflict_counts: np.ndarray,
    weights: np.ndarray,
    nsel: int,
    rng: np.random.Generator,
    forced_strategy: int | None = None,
) -> AdvsSelection:
    infeasible = np.asarray(single_uav_infeasible, dtype=bool).reshape(-1)
    counts = np.asarray(conflict_counts, dtype=float).reshape(-1)
    fleet_size = max(infeasible.size, counts.size)
    if fleet_size <= 0:
        return _empty_int_selection()
    if infeasible.size != fleet_size:
        infeasible = _pad_vector(infeasible, fleet_size, bool)
    if counts.size != fleet_size:
        counts = _pad_vector(counts, fleet_size, float)

    failing = np.flatnonzero(infeasible)
    if failing.size > 0:
        return AdvsSelection(failing.astype(int), _SINGLE_CONSTRAINT_STRATEGY, "single_constraints")

    limit = max(0, min(int(nsel), int(fleet_size)))
    if limit == 0:
        return _empty_int_selection()

    has_conflicts = bool(np.any(counts > 0.0))
    strategy = _choose_advs_strategy(weights, rng, forced_strategy) if has_conflicts else 3

    if strategy == 1:
        order = np.argsort(-counts, kind="mergesort")
        selected = order[:limit]
        return AdvsSelection(np.asarray(selected, dtype=int), 1, "greedy_conflict")

    if strategy == 2:
        probs = advs_probabilities(counts)
        if probs.size != fleet_size or float(np.sum(probs)) <= 0.0:
            selected = rng.choice(fleet_size, size=limit, replace=False)
        else:
            selected = rng.choice(fleet_size, size=limit, replace=False, p=probs)
        return AdvsSelection(np.asarray(selected, dtype=int), 2, "roulette_conflict")

    selected = rng.choice(fleet_size, size=limit, replace=False)
    return AdvsSelection(np.asarray(selected, dtype=int), 3, "random_refinement")


def _choose_advs_strategy(
    weights: np.ndarray,
    rng: np.random.Generator,
    forced_strategy: int | None,
) -> int:
    if forced_strategy is not None:
        return int(forced_strategy)
    strategy_weights = np.asarray(weights, dtype=float).reshape(-1)
    if strategy_weights.size != 3 or np.sum(strategy_weights) <= 0.0:
        strategy_weights = np.ones(3, dtype=float)
    probs = strategy_weights / float(np.sum(strategy_weights))
    return int(rng.choice(np.array([1, 2, 3], dtype=int), p=probs))


def trim_archive(archive: np.ndarray, max_size: int, rng: np.random.Generator) -> np.ndarray:
    arr = _as_2d_archive(archive)
    if arr.size == 0:
        return arr
    if arr.shape[0] <= int(max_size):
        return arr.copy()
    keep = rng.choice(arr.shape[0], size=int(max_size), replace=False)
    return arr[np.asarray(keep, dtype=int)].copy()


def adapt_jade_means(
    mu_f: float,
    mu_cr: float,
    successful_f: np.ndarray,
    successful_cr: np.ndarray,
    c: float = 0.1,
) -> tuple[float, float]:
    sf = np.asarray(successful_f, dtype=float).reshape(-1)
    scr = np.asarray(successful_cr, dtype=float).reshape(-1)
    next_f = float(mu_f)
    next_cr = float(mu_cr)
    if sf.size > 0 and float(np.sum(sf * sf)) > 0.0 and float(np.sum(sf)) > 0.0:
        lehmer = float(np.sum(sf * sf) / np.sum(sf))
        next_f = (1.0 - float(c)) * next_f + float(c) * lehmer
    if scr.size > 0:
        next_cr = (1.0 - float(c)) * next_cr + float(c) * float(np.mean(scr))
    return float(np.clip(next_f, 1e-6, 1.0)), float(np.clip(next_cr, 0.0, 1.0))


def _sample_f(mu_f: float, rng: np.random.Generator) -> float:
    for _ in range(32):
        candidate = float(mu_f) + 0.1 * float(rng.standard_cauchy())
        if candidate > 0.0:
            return float(min(candidate, 1.0))
    return 0.5


def jade_current_to_pbest_trials(
    population: np.ndarray,
    fitness: np.ndarray,
    archive: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mu_f: float,
    mu_cr: float,
    p_best_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop = np.asarray(population, dtype=float)
    fit = np.asarray(fitness, dtype=float).reshape(-1)
    low = np.asarray(lower, dtype=float).reshape(-1)
    high = np.asarray(upper, dtype=float).reshape(-1)
    n_pop, dim = pop.shape
    archive_arr = np.asarray(archive, dtype=float)
    archive_arr = _as_2d_archive(archive_arr, width=dim)

    p_count = max(2, int(np.ceil(float(p_best_rate) * float(n_pop))))
    p_count = min(n_pop, p_count)
    ordered = np.argsort(fit, kind="mergesort")
    union = np.vstack([pop, archive_arr]) if archive_arr.size else pop
    trials = np.zeros_like(pop)
    sampled_f = np.zeros(n_pop, dtype=float)
    sampled_cr = np.zeros(n_pop, dtype=float)
    all_indices = np.arange(n_pop)

    for idx in range(n_pop):
        f_val = _sample_f(mu_f, rng)
        cr_val = float(np.clip(rng.normal(float(mu_cr), 0.1), 0.0, 1.0))
        sampled_f[idx] = f_val
        sampled_cr[idx] = cr_val

        pbest = pop[int(rng.choice(ordered[:p_count]))]
        candidates_r1 = all_indices[all_indices != idx]
        r1_idx = int(rng.choice(candidates_r1)) if candidates_r1.size else idx

        union_indices = np.arange(union.shape[0])
        forbidden = {idx, r1_idx}
        valid = np.asarray([j for j in union_indices if j not in forbidden], dtype=int)
        r2_idx = int(rng.choice(valid)) if valid.size else int(rng.integers(0, union.shape[0]))

        mutant = pop[idx] + f_val * (pbest - pop[idx]) + f_val * (pop[r1_idx] - union[r2_idx])
        mutant = np.clip(mutant, low, high)
        cross = rng.random(dim) <= cr_val
        cross[int(rng.integers(0, dim))] = True
        trial = np.where(cross, mutant, pop[idx])
        trials[idx] = np.clip(trial, low, high)

    return trials, sampled_f, sampled_cr


def _context_vector(context: np.ndarray) -> np.ndarray:
    return np.asarray(context, dtype=float).reshape(-1)


def _scalar_fitness(candidate: Candidate, model: dict[str, Any]) -> float:
    obj = np.asarray(candidate.objective, dtype=float).reshape(-1)
    finite_obj = np.where(np.isfinite(obj), obj, 1e3)
    objective_sum = float(np.sum(finite_obj)) if finite_obj.size else 1e3
    violation = _constraint_violation(candidate, model)
    feasible_bonus = 0.0 if float(candidate.details.get("feasible", 0.0)) > 0.5 else 1.0
    return float(objective_sum + 100.0 * violation + 10.0 * feasible_bonus)


def _evaluate_context(
    context: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> tuple[Candidate, float]:
    candidate = _evaluate_population(
        _context_vector(context).reshape(1, -1),
        model=model,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
        representation="direct",
    )[0]
    return candidate, _scalar_fitness(candidate, model)


def _evaluate_member(
    context: np.ndarray,
    uav_idx: int,
    member: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> tuple[Candidate, float]:
    trial_context = np.asarray(context, dtype=float).copy()
    trial_context[int(uav_idx)] = np.asarray(member, dtype=float)
    return _evaluate_context(trial_context, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)


def _member_fitnesses(
    subpopulation: np.ndarray,
    context: np.ndarray,
    uav_idx: int,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> np.ndarray:
    return np.asarray(
        [
            _evaluate_member(context, uav_idx, member, model, fleet_size, n_waypoints)[1]
            for member in np.asarray(subpopulation, dtype=float)
        ],
        dtype=float,
    )


def _conflict_counts(details: dict[str, Any], fleet_size: int) -> np.ndarray:
    counts = np.zeros(int(fleet_size), dtype=float)
    conflict_log = np.asarray(details.get("conflictLog", np.zeros((0, 5), dtype=float)), dtype=float)
    if conflict_log.ndim == 1 and conflict_log.size >= 5:
        conflict_log = conflict_log.reshape(1, -1)
    if conflict_log.ndim != 2 or conflict_log.shape[1] < 5:
        return counts
    rows = conflict_log[conflict_log[:, 4] > 0.0]
    for row in rows:
        i = int(row[1])
        j = int(row[2])
        if 0 <= i < fleet_size:
            counts[i] += 1.0
        if 0 <= j < fleet_size:
            counts[j] += 1.0
    return counts


def _single_uav_infeasible(
    context: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> np.ndarray:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    flags = np.zeros(int(fleet_size), dtype=bool)
    for idx in range(int(fleet_size)):
        local_model = dict(model)
        local_model["starts"] = starts[[idx], :3]
        local_model["goals"] = goals[[idx], :3]
        local_model["fleetSize"] = 1.0
        candidate = _evaluate_population(
            np.asarray(context[idx], dtype=float).reshape(1, -1),
            model=local_model,
            fleet_size=1,
            n_waypoints=n_waypoints,
            representation="direct",
        )[0]
        flags[idx] = bool(_constraint_violation(candidate, local_model) > 1e-12)
    return flags


def _dubins_refine_member(
    member: np.ndarray,
    model: dict[str, Any],
    uav_idx: int,
    lower: np.ndarray,
    upper: np.ndarray,
    n_waypoints: int,
) -> np.ndarray:
    points = np.asarray(member, dtype=float).reshape(int(n_waypoints), 3)
    if points.shape[0] <= 1:
        return np.clip(points.reshape(-1), lower, upper)
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    start = starts[int(uav_idx), :3]
    goal = goals[int(uav_idx), :3]
    extended = np.vstack([start, points, goal])
    smoothed = points.copy()
    for idx in range(points.shape[0]):
        left = extended[idx]
        center = extended[idx + 1]
        right = extended[idx + 2]
        smoothed[idx] = 0.25 * left + 0.5 * center + 0.25 * right
    # Keep the refinement conservative: one smoothing pass and exact benchmark
    # clipping. The stage is labeled as an approximation in saved metadata.
    refined = 0.6 * points + 0.4 * smoothed
    return np.clip(refined.reshape(-1), lower, upper)


def _better_or_preserves_feasibility(
    current_candidate: Candidate,
    current_fitness: float,
    trial_candidate: Candidate,
    trial_fitness: float,
) -> bool:
    current_feasible = float(current_candidate.details.get("feasible", 0.0)) > 0.5
    trial_feasible = float(trial_candidate.details.get("feasible", 0.0)) > 0.5
    if trial_feasible and not current_feasible:
        return True
    if trial_feasible == current_feasible and trial_fitness <= current_fitness:
        return True
    return bool(trial_fitness < current_fitness)


def _evaluate_trial_member(
    *,
    trial_member: np.ndarray,
    context: np.ndarray,
    uav_idx: int,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    lower: np.ndarray,
    upper: np.ndarray,
    phase: int,
) -> tuple[np.ndarray, Candidate, float]:
    if int(phase) != 2:
        candidate, fitness = _evaluate_member(context, uav_idx, trial_member, model, fleet_size, n_waypoints)
        return trial_member, candidate, fitness

    refined = _dubins_refine_member(trial_member, model, uav_idx, lower, upper, n_waypoints)
    refined_candidate, refined_fit = _evaluate_member(context, uav_idx, refined, model, fleet_size, n_waypoints)
    trial_candidate, trial_fit = _evaluate_member(context, uav_idx, trial_member, model, fleet_size, n_waypoints)
    if _better_or_preserves_feasibility(trial_candidate, trial_fit, refined_candidate, refined_fit):
        return refined, refined_candidate, refined_fit
    return trial_member, trial_candidate, trial_fit


def _jade_evolve_subpopulation(
    *,
    subpopulation: np.ndarray,
    fitness: np.ndarray,
    context: np.ndarray,
    uav_idx: int,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    lower: np.ndarray,
    upper: np.ndarray,
    state: JadeState,
    inner_iterations: int,
    p_best_rate: float,
    rng: np.random.Generator,
    phase: int,
) -> tuple[np.ndarray, np.ndarray, JadeState, np.ndarray, float, Candidate]:
    pop = np.asarray(subpopulation, dtype=float).copy()
    fit = np.asarray(fitness, dtype=float).reshape(-1).copy()
    archive = _as_2d_archive(state.archive, width=pop.shape[1])

    best_idx = int(np.argmin(fit))
    best_member = pop[best_idx].copy()
    best_candidate, best_fit = _evaluate_member(context, uav_idx, best_member, model, fleet_size, n_waypoints)
    mu_f = float(state.mu_f)
    mu_cr = float(state.mu_cr)

    for _ in range(max(1, int(inner_iterations))):
        trials, sampled_f, sampled_cr = jade_current_to_pbest_trials(
            pop, fit, archive, lower, upper, mu_f, mu_cr, p_best_rate, rng
        )
        successful_f: list[float] = []
        successful_cr: list[float] = []
        next_pop = pop.copy()
        next_fit = fit.copy()
        added_archive: list[np.ndarray] = []

        for idx, trial in enumerate(trials):
            current_member = pop[idx]
            current_candidate, current_fit = _evaluate_member(
                context, uav_idx, current_member, model, fleet_size, n_waypoints
            )
            trial_member = np.asarray(trial, dtype=float)
            trial_member, trial_candidate, trial_fit = _evaluate_trial_member(
                trial_member=trial_member,
                context=context,
                uav_idx=uav_idx,
                model=model,
                fleet_size=fleet_size,
                n_waypoints=n_waypoints,
                lower=lower,
                upper=upper,
                phase=phase,
            )

            if _better_or_preserves_feasibility(current_candidate, current_fit, trial_candidate, trial_fit):
                next_pop[idx] = trial_member
                next_fit[idx] = trial_fit
                added_archive.append(current_member.copy())
                successful_f.append(float(sampled_f[idx]))
                successful_cr.append(float(sampled_cr[idx]))
                if trial_fit < best_fit:
                    best_fit = float(trial_fit)
                    best_member = trial_member.copy()
                    best_candidate = trial_candidate

        pop = next_pop
        fit = next_fit
        if added_archive:
            archive = (
                np.vstack([archive, np.stack(added_archive, axis=0)])
                if archive.size
                else np.stack(added_archive, axis=0)
            )
            archive = trim_archive(archive, pop.shape[0], rng)
        mu_f, mu_cr = adapt_jade_means(mu_f, mu_cr, np.asarray(successful_f), np.asarray(successful_cr), c=0.1)

    return pop, fit, JadeState(archive=archive, mu_f=mu_f, mu_cr=mu_cr), best_member, float(best_fit), best_candidate


def _initialize_subpopulations(
    lower: np.ndarray,
    upper: np.ndarray,
    fleet_size: int,
    population: int,
    dim_per_uav: int,
    rng: np.random.Generator,
) -> np.ndarray:
    low = np.asarray(lower, dtype=float).reshape(int(fleet_size), int(dim_per_uav))
    high = np.asarray(upper, dtype=float).reshape(int(fleet_size), int(dim_per_uav))
    subpops = np.zeros((int(fleet_size), int(population), int(dim_per_uav)), dtype=float)
    for uav_idx in range(int(fleet_size)):
        span = high[uav_idx] - low[uav_idx]
        subpops[uav_idx] = low[uav_idx].reshape(1, -1) + rng.random((int(population), int(dim_per_uav))) * span.reshape(
            1, -1
        )
    return subpops


def _refresh_fitnesses(
    subpops: np.ndarray,
    context: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> np.ndarray:
    return np.vstack(
        [
            _member_fitnesses(subpops[uav_idx], context, uav_idx, model, fleet_size, n_waypoints)
            for uav_idx in range(int(fleet_size))
        ]
    )


def _initialize_context(
    subpops: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> tuple[np.ndarray, Candidate, float, np.ndarray]:
    context = np.asarray(subpops[:, 0, :], dtype=float).copy()
    candidate, best_fitness = _evaluate_context(context, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    fitnesses = np.zeros((int(fleet_size), subpops.shape[1]), dtype=float)
    for uav_idx in range(int(fleet_size)):
        fitnesses[uav_idx] = _member_fitnesses(subpops[uav_idx], context, uav_idx, model, fleet_size, n_waypoints)
        best_idx = int(np.argmin(fitnesses[uav_idx]))
        if fitnesses[uav_idx, best_idx] < best_fitness:
            context[uav_idx] = subpops[uav_idx, best_idx]
            candidate, best_fitness = _evaluate_context(
                context, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints
            )
    fitnesses = _refresh_fitnesses(subpops, context, model, fleet_size, n_waypoints)
    candidate, best_fitness = _evaluate_context(context, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    return context, candidate, float(best_fitness), fitnesses


def _assembled_candidates(
    subpops: np.ndarray,
    context: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    population: int,
) -> list[Candidate]:
    vectors = [_context_vector(context)]
    count = min(int(population) - 1, subpops.shape[1])
    for member_idx in range(max(0, count)):
        context_member = np.asarray(context, dtype=float).copy()
        for uav_idx in range(int(fleet_size)):
            context_member[uav_idx] = subpops[uav_idx, member_idx]
        vectors.append(_context_vector(context_member))
    candidates = _evaluate_population(
        np.stack(vectors, axis=0),
        model=model,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
        representation="direct",
    )
    order = np.argsort(
        np.asarray([_scalar_fitness(candidate, model) for candidate in candidates], dtype=float), kind="mergesort"
    )
    return [candidates[int(idx)] for idx in order[: int(population)]]


def _record_hv(candidates: list[Candidate], problem_index: int, objective_count: int) -> float:
    matrix = _candidate_matrix(candidates)
    if matrix.size == 0:
        return 0.0
    return float(cal_metric(1, matrix, problem_index, objective_count))


def _extra_int(params: BenchmarkParams, key: str, default: int) -> int:
    return int(params.extra.get(key, params.extra.get(key[0].lower() + key[1:], default)))


def _extra_float(params: BenchmarkParams, key: str, default: float) -> float:
    return float(params.extra.get(key, params.extra.get(key[0].lower() + key[1:], default)))


def run_ccea_advs(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
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

    dim_per_uav = int(3 * n_waypoints)
    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    lower_by_uav = lower.reshape(fleet_size, dim_per_uav)
    upper_by_uav = upper.reshape(fleet_size, dim_per_uav)

    niter1_default = max(1, int(params.generations) // 2)
    niter1 = _extra_int(params, "cceaAdvsNiter1", niter1_default)
    niter2 = _extra_int(params, "cceaAdvsNiter2", max(1, int(params.generations) - niter1_default))
    total_iterations = max(1, int(niter1) + int(niter2))
    inner_iterations = max(1, _extra_int(params, "cceaAdvsJadeInnerIterations", 20))
    rho_weight = _extra_float(params, "cceaAdvsRhoWeight", 0.05)
    mu_f0 = _extra_float(params, "cceaAdvsMuF", 0.5)
    mu_cr0 = _extra_float(params, "cceaAdvsMuCR", 0.5)
    p_best_rate = _extra_float(params, "cceaAdvsPBestRate", 0.06)
    nsel = max(1, _extra_int(params, "cceaAdvsNsel", 8))
    metric_interval = max(1, _extra_int(params, "metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in _resolve_run_indices(params):
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=_OBJECTIVE_COUNT,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue

        ensure_dir(run_dir)
        run_start = time.perf_counter()
        seed_everything(seed_value + run_idx)
        rng = np.random.default_rng(seed_value + 1009 * run_idx)

        subpops = _initialize_subpopulations(lower, upper, fleet_size, params.population, dim_per_uav, rng)
        context, context_candidate, context_fitness, fitnesses = _initialize_context(
            subpops, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints
        )
        jade_states = [
            JadeState(archive=np.zeros((0, dim_per_uav), dtype=float), mu_f=mu_f0, mu_cr=mu_cr0)
            for _ in range(fleet_size)
        ]
        weights = np.ones(3, dtype=float)

        trace_selected: list[np.ndarray] = []
        trace_strategy: list[float] = []
        trace_weights: list[np.ndarray] = []
        trace_fitness: list[float] = []
        trace_feasible: list[float] = []
        trace_conflict: list[float] = []
        trace_phase: list[float] = []
        hv_history: list[float] = []

        for iteration in range(1, total_iterations + 1):
            phase = 1 if iteration <= int(niter1) else 2
            before_fitness = float(context_fitness)
            single_flags = _single_uav_infeasible(context, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            conflicts = _conflict_counts(context_candidate.details, fleet_size=fleet_size)
            selection = select_advs_variables(single_flags, conflicts, weights, nsel=nsel, rng=rng)

            selected_ids = selection.selected.astype(int)
            improved = False
            for uav_idx in selected_ids:
                pop, fit, state, best_member, best_fit, _best_candidate = _jade_evolve_subpopulation(
                    subpopulation=subpops[int(uav_idx)],
                    fitness=fitnesses[int(uav_idx)],
                    context=context,
                    uav_idx=int(uav_idx),
                    model=model,
                    fleet_size=fleet_size,
                    n_waypoints=n_waypoints,
                    lower=lower_by_uav[int(uav_idx)],
                    upper=upper_by_uav[int(uav_idx)],
                    state=jade_states[int(uav_idx)],
                    inner_iterations=inner_iterations,
                    p_best_rate=p_best_rate,
                    rng=rng,
                    phase=phase,
                )
                subpops[int(uav_idx)] = pop
                fitnesses[int(uav_idx)] = fit
                jade_states[int(uav_idx)] = state
                if best_fit < context_fitness:
                    context[int(uav_idx)] = best_member
                    context_candidate, context_fitness = _evaluate_context(
                        context, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints
                    )
                    improved = True
                    fitnesses = _refresh_fitnesses(subpops, context, model, fleet_size, n_waypoints)

            weights = update_strategy_weights(weights, selection.strategy, improved, rho=rho_weight)

            selected_record = np.full(max(1, nsel), -1.0, dtype=float)
            fill = min(selected_record.size, selected_ids.size)
            selected_record[:fill] = selected_ids[:fill].astype(float)
            trace_selected.append(selected_record)
            trace_strategy.append(float(selection.strategy))
            trace_weights.append(weights.copy())
            trace_fitness.append(float(context_fitness))
            trace_feasible.append(float(context_candidate.details.get("feasible", 0.0)))
            trace_conflict.append(float(context_candidate.details.get("conflictRate", 0.0)))
            trace_phase.append(float(phase))

            if iteration == 1 or iteration == total_iterations or iteration % metric_interval == 0:
                hv_history.append(_record_hv([context_candidate], params.problem_index, _OBJECTIVE_COUNT))

            if context_fitness > before_fitness and selection.strategy == _SINGLE_CONSTRAINT_STRATEGY:
                context_candidate, context_fitness = _evaluate_context(
                    context, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints
                )

        final_candidates = _assembled_candidates(
            subpops, context, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints, population=params.population
        )
        save_mat(run_dir / "gen_hv.mat", {"gen_hv": np.asarray(hv_history, dtype=float)})
        save_mat(
            run_dir / "ccea_advs_trace.mat",
            {
                "selectedUavIds": np.vstack(trace_selected)
                if trace_selected
                else np.zeros((0, max(1, nsel)), dtype=float),
                "selectedStrategy": np.asarray(trace_strategy, dtype=float),
                "strategyWeights": np.vstack(trace_weights) if trace_weights else np.zeros((0, 3), dtype=float),
                "contextFitness": np.asarray(trace_fitness, dtype=float),
                "feasibleRatio": np.asarray(trace_feasible, dtype=float),
                "conflictRate": np.asarray(trace_conflict, dtype=float),
                "phaseIndex": np.asarray(trace_phase, dtype=float),
            },
        )

        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=_OBJECTIVE_COUNT,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": "CCEA-ADVS",
                "optimizerBackend": "clean-room CCEA-ADVS with shared Python UAV evaluator",
                "cleanRoomImplementation": True,
                "benchmarkObjectiveDuringSearch": True,
                "population": int(params.population),
                "generations": int(total_iterations),
                "cceaAdvsNiter1": int(niter1),
                "cceaAdvsNiter2": int(niter2),
                "cceaAdvsJadeInnerIterations": int(inner_iterations),
                "cceaAdvsRhoWeight": float(rho_weight),
                "cceaAdvsMuF": float(mu_f0),
                "cceaAdvsMuCR": float(mu_cr0),
                "cceaAdvsPBestRate": float(p_best_rate),
                "cceaAdvsNsel": int(nsel),
                "dubinsRefinement": "benchmark_approximation",
                "internalSelectionScalar": "normalized_objective_sum_plus_constraint_violation",
                "finalReporting": "shared_multi_objective_benchmark",
            },
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(final_candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_obj, params.problem_index, _OBJECTIVE_COUNT),
                    cal_metric(2, final_obj, params.problem_index, _OBJECTIVE_COUNT),
                ],
                dtype=float,
            )

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


__all__ = [
    "AdvsSelection",
    "JadeState",
    "adapt_jade_means",
    "advs_probabilities",
    "jade_current_to_pbest_trials",
    "run_ccea_advs",
    "select_advs_variables",
    "trim_archive",
    "update_strategy_weights",
]
