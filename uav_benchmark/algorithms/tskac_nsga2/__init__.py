from __future__ import annotations

import time
from typing import Any

import numpy as np

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
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir


def _assistant_model(model: dict[str, Any]) -> dict[str, Any]:
    aux = dict(model)
    if "threats" in aux:
        aux["threats"] = np.zeros((0, 4), dtype=float)
    if "nofly_r" in aux and aux["nofly_r"] is not None:
        radii = np.asarray(aux["nofly_r"], dtype=float).reshape(-1)
        aux["nofly_r"] = np.zeros_like(radii)
    return aux


def _select_nsga2(
    vectors: np.ndarray,
    candidates: list,
    model: dict[str, Any],
    n_keep: int,
    obj1_only: bool = False,
) -> tuple[np.ndarray, list]:
    if vectors.size == 0 or not candidates or n_keep <= 0:
        return np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=float), []
    obj_full = _candidate_matrix(candidates)
    obj = obj_full[:, [0]] if obj1_only else obj_full
    con = _constraint_violation_vector(candidates, model).reshape(-1, 1)
    front_no, _ = n_d_sort(obj.copy(), con, n_keep)
    crowd = crowding_distance(obj, front_no)
    selected: list[int] = []
    finite_fronts = np.unique(front_no[np.isfinite(front_no)])
    for front in finite_fronts:
        idx = np.where(front_no == front)[0]
        if len(selected) + idx.size <= n_keep:
            selected.extend(idx.tolist())
            continue
        order = idx[np.argsort(-crowd[idx])]
        need = n_keep - len(selected)
        selected.extend(order[:need].tolist())
        break
    pick = np.asarray(selected, dtype=int)
    return vectors[pick], [candidates[int(i)] for i in pick]


def _variation(
    vectors: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    crossover_prob: float,
    mutation_prob: float,
) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    n, d = vectors.shape
    order = np.random.permutation(n)
    parents = vectors[order]
    if n % 2 == 1:
        parents = np.vstack([parents, parents[np.random.randint(0, n)]])
    half = parents.shape[0] // 2
    p1 = parents[:half]
    p2 = parents[half:]
    alpha = np.random.uniform(0.0, 1.0, size=(half, d))
    pair_mask = (np.random.rand(half, 1) < float(crossover_prob)).astype(float)
    child1 = pair_mask * (alpha * p1 + (1.0 - alpha) * p2) + (1.0 - pair_mask) * p1
    child2 = pair_mask * ((1.0 - alpha) * p1 + alpha * p2) + (1.0 - pair_mask) * p2
    offspring = np.vstack([child1, child2])[:n]
    mutation_mask = np.random.rand(n, d) < float(mutation_prob)
    mutation = np.random.normal(0.0, 1.0, size=(n, d)) * 0.05 * (upper - lower)
    offspring = np.where(mutation_mask, offspring + mutation, offspring)
    return np.clip(offspring, lower, upper)


def _apply_eq20_top20(
    vectors: np.ndarray,
    candidates: list,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if vectors.size == 0 or not candidates:
        return vectors
    matrix = _candidate_matrix(candidates)
    order = np.argsort(matrix[:, 0], kind="mergesort")
    top_k = max(1, int(np.ceil(0.2 * vectors.shape[0])))
    idx = order[:top_k]
    moved = vectors.copy()
    q = np.random.uniform(0.0, 1.0, size=(top_k, vectors.shape[1]))
    moved[idx] = moved[idx] + q * (upper - lower)
    return np.clip(moved, lower, upper)


def _indicator_stats(dom_obj1: np.ndarray, inf_obj1: np.ndarray) -> tuple[float, float, int]:
    if dom_obj1.size == 0 or inf_obj1.size == 0:
        return 0.0, 0.0, 0
    a = float(np.min(dom_obj1))
    b = float(np.max(dom_obj1))
    c = float(np.min(inf_obj1))
    d = float(np.max(inf_obj1))
    dom_span = max(1e-12, b - a)
    inf_span = max(1e-12, d - c)
    # Relative-distance indicator in Sec. 4.2.4 compares spread relation.
    relative_distance = dom_span / inf_span
    mid = 0.5 * (c + d)
    num_fo = float(np.sum(inf_obj1 <= mid))
    num_la = float(np.sum(inf_obj1 > mid))
    set_optimality = float("inf") if num_la <= 0.0 else (num_fo / num_la)
    if relative_distance > 1.0 and set_optimality > 1.0:
        return relative_distance, set_optimality, 0
    if relative_distance <= 1.0 and set_optimality > 1.0:
        return relative_distance, set_optimality, 1
    if relative_distance > 1.0 and set_optimality <= 1.0:
        return relative_distance, set_optimality, 2
    return relative_distance, set_optimality, 3


def _objective1_stats(candidates: list) -> tuple[float, float]:
    if not candidates:
        return float("inf"), 0.0
    matrix = _candidate_matrix(candidates)
    if matrix.size == 0:
        return float("inf"), 0.0
    obj1 = matrix[:, 0]
    finite = np.isfinite(obj1)
    if not np.any(finite):
        return float("inf"), 0.0
    return float(np.mean(obj1[finite])), float(np.mean(finite.astype(float)))


def _safe_delta(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after):
        return 0.0
    scale = max(1e-9, abs(before))
    return (before - after) / scale


def _reward(
    before_candidates: list,
    after_candidates: list,
    rel_before: float,
    rel_after: float,
    set_before: float,
    set_after: float,
) -> float:
    mean_before, feasible_before = _objective1_stats(before_candidates)
    mean_after, feasible_after = _objective1_stats(after_candidates)
    delta_obj1 = _safe_delta(mean_before, mean_after)
    delta_rel = _safe_delta(rel_before, rel_after)
    delta_set = _safe_delta(set_before, set_after)
    delta_feas = feasible_after - feasible_before
    # Keep reward aligned with paper focus on Obj1 while reflecting indicator changes.
    return float(0.70 * delta_obj1 + 0.15 * delta_rel + 0.15 * delta_set + 0.20 * delta_feas)


def _apply_action(
    action: int,
    inferior: np.ndarray,
    dominant: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    generation: int,
    total: int,
    gamma: float,
    creativity: float,
) -> np.ndarray:
    if inferior.size == 0:
        return inferior
    x = inferior.copy()
    if action == 0:
        # Action 1 (paper): directed guided mode, Eq. (23).
        dom_mean = np.mean(dominant, axis=0) if dominant.size else np.mean(inferior, axis=0)
        f = np.random.uniform(0.0, 1.0, size=(x.shape[0], 1))
        h = 1.0 - f
        x = f * x + h * dom_mean.reshape(1, -1)
    elif action == 1:
        # Action 2 (paper): random swing mode, Eq. (22).
        progress = 1.0 - (float(generation) / max(1.0, float(total)))
        r = np.random.binomial(1, 0.5, size=x.shape)
        sign = np.where(r > 0, -1.0, 1.0)
        gamma_rand = float(gamma) * np.random.uniform(0.0, 1.0, size=x.shape)
        x = x + sign * gamma_rand * max(0.0, progress) * (upper - x)
    else:
        # Action 3 (paper): potential dominance exploration mode, Eq. (24).
        mv = np.mean(x, axis=0)
        rbest = dominant[np.random.randint(0, dominant.shape[0])] if dominant.size else mv
        xi = np.random.uniform(0.0, 1.0, size=(x.shape[0], 1))
        p = xi * x + (1.0 - xi) * rbest.reshape(1, -1)
        mu = np.clip(np.random.uniform(1e-6, 1.0, size=(x.shape[0], 1)), 1e-6, 1.0)
        eps = float(creativity)
        x = p + eps * np.abs(mv.reshape(1, -1) - x) * np.log(1.0 / mu)
    return np.clip(x, lower, upper)


def run_tskac_nsga2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    use_legacy_runner = bool(params.extra.get("legacyPathRunner", False))
    if use_legacy_runner and int(params.fleet_size) <= 1:
        from uav_benchmark.algorithms.nsga2 import run_nsga2

        return run_nsga2(model, params)

    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 0
    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)
    aux_model = _assistant_model(model)

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    pop_size = max(2, int(params.population))
    dimensions = int(lower.size)
    metric_interval = int(params.extra.get("metricInterval", 20))

    # Paper defaults/protocol.
    max_it = max(1, int(params.generations))
    kappa = float(params.extra.get("tskacKappa", 0.06))
    kappa = float(np.clip(kappa, 1e-6, 1.0))
    crossover_prob = float(params.extra.get("tskacCrossoverProb", 0.8))
    mutation_prob = float(params.extra.get("tskacMutationProb", 1.0 / 12.0))
    gamma = float(params.extra.get("tskacGamma", 0.5))
    phi = float(params.extra.get("tskacPhi", 0.5))
    rho1_raw = float(params.extra.get("tskacRhoStart", 0.9))
    rho2_raw = float(params.extra.get("tskacRhoEnd", 0.1))
    rho_norm = max(1e-9, rho1_raw + rho2_raw)
    rho_start = rho1_raw / rho_norm
    rho_end = rho2_raw / rho_norm
    epsilon = float(params.extra.get("tskacEpsilonGreedy", 0.10))
    creativity = float(params.extra.get("tskacCreativity", 0.8))
    dominant_share_start = float(params.extra.get("tskacDominantShareStart", 0.5))
    dominant_share_end = float(params.extra.get("tskacDominantShareEnd", 0.8))
    dominant_share_start = float(np.clip(dominant_share_start, 0.2, 0.95))
    dominant_share_end = float(np.clip(dominant_share_end, dominant_share_start, 0.98))

    # Stage-control factor follows Eq. (18): MaxIt1 ~= MaxIt^2 / Z * kappa.
    if max_it <= 1:
        stage1_gens = 1
        stage2_gens = 0
    else:
        threshold = (float(max_it * max_it) / float(pop_size)) * kappa
        stage1_gens = int(np.floor(threshold))
        stage1_gens = max(1, min(stage1_gens, max_it - 1))
        stage2_gens = max_it - stage1_gens

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

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
        q_table = np.zeros((4, 3), dtype=float)
        main_vectors = np.random.uniform(lower, upper, size=(pop_size, dimensions))
        assistant_vectors = np.random.uniform(lower, upper, size=(pop_size, dimensions))
        main_candidates = _evaluate_population(main_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        assistant_candidates = _evaluate_population(assistant_vectors, aux_model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        hv_history = np.zeros((max_it, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, stage1_gens + 1):
            off_main = _variation(main_vectors, lower, upper, crossover_prob, mutation_prob)
            off_assist = _variation(assistant_vectors, lower, upper, crossover_prob, mutation_prob)
            off_main_cand = _evaluate_population(off_main, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            off_assist_main_cand = _evaluate_population(off_assist, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            merged_main_v = np.vstack([main_vectors, off_main, off_assist])
            merged_main_c = list(main_candidates) + list(off_main_cand) + list(off_assist_main_cand)
            main_vectors, main_candidates = _select_nsga2(merged_main_v, merged_main_c, model, pop_size)

            off_assist_cand = _evaluate_population(off_assist, aux_model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            off_main_assist_cand = _evaluate_population(off_main, aux_model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            merged_assist_v = np.vstack([assistant_vectors, off_assist, off_main])
            merged_assist_c = list(assistant_candidates) + list(off_assist_cand) + list(off_main_assist_cand)
            assistant_vectors, assistant_candidates = _select_nsga2(
                merged_assist_v, merged_assist_c, aux_model, pop_size, obj1_only=True
            )

            main_vectors = _apply_eq20_top20(main_vectors, main_candidates, lower, upper)
            assistant_vectors = _apply_eq20_top20(assistant_vectors, assistant_candidates, lower, upper)
            main_candidates = _evaluate_population(main_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            assistant_candidates = _evaluate_population(assistant_vectors, aux_model, fleet_size=fleet_size, n_waypoints=n_waypoints)

            if params.compute_metrics:
                matrix = _candidate_matrix(main_candidates)
                if generation == 1 or generation == max_it or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, matrix, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, matrix, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        if stage2_gens > 0:
            for local in range(1, stage2_gens + 1):
                generation = stage1_gens + local
                progress = float(local - 1) / max(1.0, float(stage2_gens - 1))
                dominant_share = dominant_share_start + (dominant_share_end - dominant_share_start) * progress
                split = int(round(pop_size * dominant_share))
                split = max(1, min(split, pop_size - 1))

                obj = _candidate_matrix(main_candidates)
                order = np.argsort(obj[:, 0], kind="mergesort")
                dominant_vectors = main_vectors[order[:split]]
                inferior_vectors = main_vectors[order[split:]]
                dom_cand = _evaluate_population(dominant_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

                dom_off = _variation(dominant_vectors, lower, upper, crossover_prob, mutation_prob)
                dom_off_cand = _evaluate_population(dom_off, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
                dom_merge_v = np.vstack([dominant_vectors, dom_off])
                dom_merge_c = list(dom_cand) + list(dom_off_cand)
                dominant_vectors, dom_cand = _select_nsga2(dom_merge_v, dom_merge_c, model, split)
                dominant_vectors = _apply_eq20_top20(dominant_vectors, dom_cand, lower, upper)
                dom_cand = _evaluate_population(dominant_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

                inf_cand = _evaluate_population(inferior_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
                rel_before, set_before, state = _indicator_stats(_candidate_matrix(dom_cand)[:, 0], _candidate_matrix(inf_cand)[:, 0])
                if np.random.rand() < epsilon:
                    action = int(np.random.randint(0, 3))
                else:
                    action = int(np.argmax(q_table[state]))
                acted = _apply_action(
                    action=action,
                    inferior=inferior_vectors,
                    dominant=dominant_vectors,
                    lower=lower,
                    upper=upper,
                    generation=generation,
                    total=max_it,
                    gamma=gamma,
                    creativity=creativity,
                )
                inf_off = _variation(acted, lower, upper, crossover_prob, mutation_prob)
                inf_off_cand = _evaluate_population(inf_off, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
                inf_merge_v = np.vstack([inferior_vectors, inf_off, acted])
                inf_merge_c = list(inf_cand) + list(inf_off_cand) + list(
                    _evaluate_population(acted, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
                )
                keep_inf = max(1, pop_size - split)
                inferior_vectors, inf_cand_new = _select_nsga2(inf_merge_v, inf_merge_c, model, keep_inf)

                rel_after, set_after, next_state = _indicator_stats(
                    _candidate_matrix(dom_cand)[:, 0], _candidate_matrix(inf_cand_new)[:, 0]
                )
                reward = _reward(
                    before_candidates=inf_cand,
                    after_candidates=inf_cand_new,
                    rel_before=rel_before,
                    rel_after=rel_after,
                    set_before=set_before,
                    set_after=set_after,
                )
                rho = rho_start + (rho_end - rho_start) * progress
                q_table[state, action] = (1.0 - rho) * q_table[state, action] + rho * (
                    reward + phi * np.max(q_table[next_state])
                )

                main_vectors = np.vstack([dominant_vectors, inferior_vectors])
                main_candidates = _evaluate_population(main_vectors, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
                main_vectors, main_candidates = _select_nsga2(main_vectors, main_candidates, model, pop_size)

                if params.compute_metrics:
                    matrix = _candidate_matrix(main_candidates)
                    if generation == 1 or generation == max_it or generation % metric_interval == 0:
                        hv_history[generation - 1, 0] = cal_metric(1, matrix, params.problem_index, objective_count)
                        hv_history[generation - 1, 1] = cal_metric(2, matrix, params.problem_index, objective_count)
                    elif generation > 1:
                        hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=main_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(main_candidates)
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
