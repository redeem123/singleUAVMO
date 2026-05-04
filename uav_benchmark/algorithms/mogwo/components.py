from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.mogwo.archive import (
    _build_grid,
    _candidate_is_feasible,
    _candidate_objective_context,
    _candidate_objective_sum,
    _objective_rank_matrix,
    _occupancy_per_solution,
    _repair_candidate_rank,
    _select_archive_members,
    _stack_candidate_contexts,
    _stack_candidate_vectors,
)
from uav_benchmark.algorithms.mogwo.constants import (
    _ATTN_EPS,
    _CAUCHY_SCALE_BASE,
    _COMPONENT_SEED_FRACTION,
    _EXPLORER_RATIO_MAX,
    _EXPLORER_RATIO_MIN,
    _RELAX_INFUSION_MAX,
    _RELAX_INFUSION_MIN,
    _RELAX_PROGRESS_POWER,
    _RELAX_SHARE_MAX,
    _RELAX_SHARE_MIN,
    _REPAIR_RATE,
)
from uav_benchmark.algorithms.shared.fleet_runner import (
    _constraint_violation,
    _evaluate_population,
    _sbx_mutation,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.chromosome import Chromosome
from uav_benchmark.core.mission_encoding import paths_to_decision
from uav_benchmark.core.nsga2_ops import n_d_sort


def _single_uav_seed_model(
    model: dict[str, Any],
    *,
    uav_idx: int,
    n_waypoints: int,
) -> dict[str, Any]:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    local_model = dict(model)
    local_model["start"] = starts[uav_idx].reshape(-1)[:3]
    local_model["end"] = goals[uav_idx].reshape(-1)[:3]
    # ``Chromosome`` uses the total number of points including endpoints.
    local_model["n"] = int(n_waypoints) + 2
    return local_model


def _terrain_seed_population(
    model: dict[str, Any],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    pop_size: int,
    fleet_size: int,
    n_waypoints: int,
    seed_fraction: float = _COMPONENT_SEED_FRACTION,
) -> tuple[np.ndarray, float]:
    seed_count = int(np.clip(round(float(pop_size) * float(seed_fraction)), 0, pop_size))
    if pop_size > 0 and seed_fraction > 0.0:
        seed_count = max(1, seed_count)
    if seed_count <= 0:
        return np.zeros((0, lower.size), dtype=float), 0.0

    seeds = np.zeros((seed_count, lower.size), dtype=float)
    for seed_idx in range(seed_count):
        seeded_paths: list[np.ndarray] = []
        for uav_idx in range(fleet_size):
            local_model = _single_uav_seed_model(
                model,
                uav_idx=uav_idx,
                n_waypoints=n_waypoints,
            )
            chromosome = Chromosome.new(local_model)
            chromosome.initialize(local_model)
            seeded_paths.append(np.asarray(chromosome.path, dtype=float))
        decision = paths_to_decision(
            seeded_paths,
            model=model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
        )
        seeds[seed_idx] = np.clip(np.asarray(decision, dtype=float).reshape(-1), lower, upper)

    return seeds, float(seed_count / max(1, pop_size))


def _adaptive_explorer_ratio(feasible_ratio: float, diversity_level: float) -> float:
    if feasible_ratio < 0.15:
        return 0.0
    if feasible_ratio >= 0.80 and diversity_level >= 0.45:
        return 0.0
    pressure = 1.0 - float(np.clip(feasible_ratio, 0.0, 1.0))
    low_div = 1.0 - float(np.clip(diversity_level, 0.0, 1.0))
    ratio = _EXPLORER_RATIO_MIN + 0.24 * pressure + 0.12 * low_div
    return float(np.clip(ratio, _EXPLORER_RATIO_MIN, _EXPLORER_RATIO_MAX))


def _feedback_relaxation_threshold(
    *,
    candidates: list[Candidate],
    archive_unconstrained: list[Candidate],
    model: dict[str, Any],
    feasible_ratio: float,
    previous_feasible_ratio: float,
    generation: int,
    max_generations: int,
) -> float:
    combined = list(candidates) + list(archive_unconstrained)
    if not combined:
        return 0.0

    cv = np.asarray([_constraint_violation(candidate, model) for candidate in combined], dtype=float)
    cv = np.where(np.isfinite(cv), np.maximum(cv, 0.0), 1.0)
    positive = cv[cv > 0.0]
    if positive.size <= 0:
        return 0.0

    pressure = 1.0 - float(np.clip(feasible_ratio, 0.0, 1.0))
    if pressure <= 1e-6:
        return 0.0
    regression = max(0.0, float(previous_feasible_ratio) - float(feasible_ratio))
    progress = float(np.clip(float(generation) / float(max(1, max_generations)), 0.0, 1.0))

    share = _RELAX_SHARE_MIN + 0.34 * pressure + 0.26 * regression
    share *= max(0.08, 1.0 - progress**_RELAX_PROGRESS_POWER)
    share = float(np.clip(share, _RELAX_SHARE_MIN, _RELAX_SHARE_MAX))

    epsilon = float(np.quantile(positive, share))
    tail_share = float(np.clip(0.12 + 0.10 * pressure + 0.08 * regression, 0.12, 0.26))
    band_cap = float(np.expm1(np.quantile(np.log1p(positive), tail_share)))
    epsilon = min(epsilon, band_cap)
    epsilon = max(float(np.min(positive)), epsilon)
    if progress >= 0.80:
        epsilon *= float(np.clip(1.10 - progress, 0.10, 0.30))
    return float(max(0.0, epsilon))


def _update_relaxed_constraint_archive(
    archive: list[Candidate],
    new_cands: list[Candidate],
    strict_archive: list[Candidate],
    unconstrained_archive: list[Candidate],
    model: dict[str, Any],
    max_size: int,
    divisions: int,
    relaxation_eps: float,
) -> list[Candidate]:
    if relaxation_eps <= _ATTN_EPS:
        return []

    seed_strict = list(strict_archive[: max(1, min(len(strict_archive), max_size // 2 or 1))])
    seed_unconstrained = list(unconstrained_archive[: max(1, min(len(unconstrained_archive), max_size // 2 or 1))])
    all_cands = list(archive) + list(new_cands) + seed_strict + seed_unconstrained
    if not all_cands:
        return []

    cv_all = np.asarray([_constraint_violation(candidate, model) for candidate in all_cands], dtype=float)
    cv_all = np.where(np.isfinite(cv_all), np.maximum(cv_all, 0.0), 1.0)
    keep_mask = cv_all <= float(relaxation_eps) + _ATTN_EPS
    if not np.any(keep_mask):
        return []

    filtered = [candidate for candidate, keep in zip(all_cands, keep_mask, strict=False) if keep]
    filtered_cv = cv_all[keep_mask]
    total = len(filtered)
    if total <= 0:
        return []

    obj_all = np.stack([candidate.objective for candidate in filtered], axis=0)
    obj_rank = _objective_rank_matrix(obj_all)

    relaxed_con = np.maximum(0.0, filtered_cv - float(relaxation_eps))[:, None]
    fronts, _ = n_d_sort(obj_rank.copy(), relaxed_con, total)
    return _select_archive_members(filtered, obj_rank, fronts, max_size, divisions)


def _relaxed_archive_infusion_ratio(feasible_ratio: float, relaxation_eps: float) -> float:
    if relaxation_eps <= _ATTN_EPS:
        return 0.0
    pressure = 1.0 - float(np.clip(feasible_ratio, 0.0, 1.0))
    ratio = 0.22 + 0.54 * pressure
    return float(np.clip(ratio, _RELAX_INFUSION_MIN, _RELAX_INFUSION_MAX))


def _adaptive_archive_explorer(
    *,
    pack_positions: np.ndarray,
    leaders: np.ndarray,
    convergence_archive: list[Candidate],
    diversity_archive: list[Candidate],
    relaxation_archive: list[Candidate],
    lower: np.ndarray,
    upper: np.ndarray,
    offspring_count: int,
    feasible_ratio: float,
    diversity_level: float,
    relaxation_share: float = 0.0,
) -> np.ndarray:
    if offspring_count <= 0:
        return np.zeros((0, lower.size), dtype=float)

    conv_matrix = _stack_candidate_vectors(convergence_archive, lower.size)
    div_matrix = _stack_candidate_vectors(diversity_archive, lower.size)
    relax_matrix = _stack_candidate_vectors(relaxation_archive, lower.size)
    if conv_matrix.size == 0:
        conv_matrix = np.asarray(pack_positions, dtype=float)
    if div_matrix.size == 0:
        div_matrix = conv_matrix
    if relaxation_share > _ATTN_EPS and relax_matrix.size > 0 and feasible_ratio < 0.55:
        div_matrix = relax_matrix

    parent_blocks = [
        np.asarray(pack_positions, dtype=float),
        conv_matrix,
        div_matrix,
        np.asarray(leaders, dtype=float),
    ]
    if relaxation_share > _ATTN_EPS and relax_matrix.size > 0:
        parent_blocks.insert(2, relax_matrix)
    parent_pool = np.vstack(parent_blocks)
    explorer_pool = _sbx_mutation(parent_pool, lower, upper)
    if explorer_pool.shape[0] < offspring_count:
        repeats = int(np.ceil(float(offspring_count) / float(max(1, explorer_pool.shape[0]))))
        explorer_pool = np.vstack([explorer_pool for _ in range(repeats)])
    explorers = explorer_pool[:offspring_count].copy()

    guide_blocks = [div_matrix, conv_matrix, np.asarray(leaders, dtype=float)]
    anchor_blocks = [conv_matrix, np.asarray(leaders, dtype=float)]
    if relaxation_share > _ATTN_EPS and relax_matrix.size > 0:
        guide_blocks.insert(0, relax_matrix)
        anchor_blocks.insert(0, relax_matrix)
    guide_pool = np.vstack(guide_blocks)
    anchor_pool = np.vstack(anchor_blocks)
    guide_ids = np.random.randint(0, guide_pool.shape[0], size=offspring_count)
    anchor_ids = np.random.randint(0, anchor_pool.shape[0], size=offspring_count)

    pressure = 1.0 - float(np.clip(feasible_ratio, 0.0, 1.0))
    low_div = 1.0 - float(np.clip(diversity_level, 0.0, 1.0))
    tail_scale = (_CAUCHY_SCALE_BASE + 0.05 * pressure + 0.04 * low_div) * (upper - lower)
    cauchy = np.clip(np.random.standard_cauchy(size=explorers.shape), -2.5, 2.5)

    explorers += 0.30 * np.random.rand(*explorers.shape) * (guide_pool[guide_ids] - explorers)
    explorers += (0.12 + 0.18 * pressure) * np.random.rand(*explorers.shape) * (anchor_pool[anchor_ids] - explorers)
    if relaxation_share > _ATTN_EPS and relax_matrix.size > 0:
        relax_ids = np.random.randint(0, relax_matrix.shape[0], size=offspring_count)
        relax_pull = 0.18 + 0.24 * pressure + 0.20 * float(np.clip(relaxation_share, 0.0, 1.0))
        explorers += relax_pull * np.random.rand(*explorers.shape) * (relax_matrix[relax_ids] - explorers)
    explorers += cauchy * tail_scale[None, :]
    return np.clip(explorers, lower, upper)


def _topology_relay_guides(
    *,
    pack_positions: np.ndarray,
    candidates: list[Candidate],
    archive: list[Candidate],
    archive_unconstrained: list[Candidate],
    relaxation_archive: list[Candidate],
    model: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    feasible_ratio: float,
    diversity_level: float,
    relaxation_eps: float = 0.0,
    max_pool: int = 12,
) -> tuple[np.ndarray, float, float]:
    if pack_positions.shape[0] <= 0 or not candidates:
        return np.zeros_like(pack_positions), 0.0, 0.0

    combined = list(archive_unconstrained) + list(archive)
    if relaxation_eps > _ATTN_EPS and relaxation_archive:
        relaxed = [
            candidate
            for candidate in relaxation_archive
            if _constraint_violation(candidate, model) <= relaxation_eps + _ATTN_EPS
        ]
        if feasible_ratio < 0.60 and relaxed:
            combined = list(relaxed) + list(archive)
            if len(combined) < max_pool:
                remaining = max(0, int(max_pool) - len(combined))
                supplement = sorted(
                    archive_unconstrained,
                    key=lambda candidate: _constraint_violation(candidate, model),
                )[:remaining]
                combined.extend(supplement)
        else:
            combined = relaxed + combined
    if not combined:
        combined = list(candidates)
    if not combined:
        return np.asarray(pack_positions, dtype=float).copy(), 0.0, 0.0

    pool_size = max(1, min(int(max_pool), len(combined)))
    if feasible_ratio < 0.45:
        cv = np.asarray([_constraint_violation(candidate, model) for candidate in combined], dtype=float)
        cv = np.where(np.isfinite(cv), np.maximum(cv, 0.0), 1.0)
        order = np.argsort(cv, kind="mergesort")[:pool_size]
        relax_bonus = 0.10 if relaxation_eps > _ATTN_EPS and relaxation_archive else 0.0
        activation = float(np.clip(0.35 + 0.65 * (0.45 - feasible_ratio) / 0.45 + relax_bonus, 0.35, 1.0))
        sparsity = np.zeros(pool_size, dtype=float)
    else:
        feasible_pool = [candidate for candidate in combined if _candidate_is_feasible(candidate)]
        if relaxation_eps > _ATTN_EPS and len(feasible_pool) < pool_size:
            feasible_pool.extend(
                candidate
                for candidate in combined
                if not _candidate_is_feasible(candidate)
                and _constraint_violation(candidate, model) <= relaxation_eps + _ATTN_EPS
            )
        if not feasible_pool:
            feasible_pool = combined
        pool_size = max(1, min(pool_size, len(feasible_pool)))
        pool_obj = np.stack([candidate.objective for candidate in feasible_pool], axis=0)
        grid, _, _ = _build_grid(pool_obj, max(4, min(10, pool_size)))
        occ = _occupancy_per_solution(grid)
        order = np.argsort(occ, kind="mergesort")[:pool_size]
        combined = feasible_pool
        activation = float(np.clip(0.20 + 0.70 * (1.0 - diversity_level), 0.20, 0.85))
        occ_sel = occ[order].astype(float, copy=False)
        occ_span = float(np.ptp(occ_sel)) if occ_sel.size > 0 else 0.0
        sparsity = np.zeros(pool_size, dtype=float) if occ_span <= _ATTN_EPS else (np.max(occ_sel) - occ_sel) / occ_span

    pool = [combined[int(idx)] for idx in np.asarray(order, dtype=int)]
    pool_vectors = _stack_candidate_vectors(pool, lower.size)
    pool_objectives = _stack_candidate_contexts(pool)
    pool_feasible_share = float(np.mean([1.0 if _candidate_is_feasible(candidate) else 0.0 for candidate in pool]))

    wolf_objectives = np.stack([_candidate_objective_context(candidate) for candidate in candidates], axis=0)
    span = np.maximum(upper - lower, _ATTN_EPS)
    norm_pack = (np.asarray(pack_positions, dtype=float) - lower[None, :]) / span[None, :]
    norm_pool = (pool_vectors - lower[None, :]) / span[None, :]

    decision_distance = np.linalg.norm(norm_pack[:, None, :] - norm_pool[None, :, :], axis=2)
    objective_distance = np.linalg.norm(wolf_objectives[:, None, :] - pool_objectives[None, :, :], axis=2)
    score = 0.60 * decision_distance + 0.40 * objective_distance

    if feasible_ratio < 0.45:
        pool_cv = np.asarray([_constraint_violation(candidate, model) for candidate in pool], dtype=float)
        pool_cv = np.where(np.isfinite(pool_cv), np.maximum(pool_cv, 0.0), 1.0)
        span_cv = float(np.ptp(pool_cv))
        if span_cv > _ATTN_EPS:
            score += 0.20 * ((pool_cv - np.min(pool_cv)) / span_cv)[None, :]
    elif sparsity.size == pool_size:
        score -= 0.10 * sparsity[None, :]

    chosen = np.argmin(score, axis=1)
    guides = pool_vectors[chosen]
    return np.clip(guides, lower, upper), activation, pool_feasible_share


# ─────────────────────────────────────────────────────────────────────
# GWO Engine
# ─────────────────────────────────────────────────────────────────────


def _selective_feasibility_repair_restart(
    *,
    positions: np.ndarray,
    candidates: list[Candidate],
    archive: list[Candidate],
    archive_unconstrained: list[Candidate],
    leaders: np.ndarray,
    model: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    fleet_size: int,
    n_waypoints: int,
    repair_rate: float = _REPAIR_RATE,
) -> tuple[np.ndarray, list[Candidate], dict[str, float]]:
    if positions.shape[0] <= 0 or not candidates:
        return positions, candidates, {"attempted": 0.0, "accepted": 0.0, "restart_used": 0.0}

    cv = np.asarray([_constraint_violation(candidate, model) for candidate in candidates], dtype=float)
    cv = np.where(np.isfinite(cv), np.maximum(cv, 0.0), 1.0)
    infeasible_idx = np.where(cv > 0.0)[0]
    if infeasible_idx.size <= 0:
        return positions, candidates, {"attempted": 0.0, "accepted": 0.0, "restart_used": 0.0}

    repair_count = min(
        int(infeasible_idx.size),
        max(1, int(round(float(positions.shape[0]) * float(repair_rate)))),
    )
    chosen_idx = infeasible_idx[np.argsort(-cv[infeasible_idx])[:repair_count]]
    repaired_positions = np.asarray(positions, dtype=float).copy()

    span = upper - lower
    span_safe = np.maximum(span, _ATTN_EPS)
    feasible_archive = [
        candidate for candidate in list(archive) + list(archive_unconstrained) if _candidate_is_feasible(candidate)
    ]
    external_pool = list(archive_unconstrained) + list(archive)
    boundary_pool = [candidate for candidate in external_pool if _constraint_violation(candidate, model) > 0.0]
    if not boundary_pool:
        if external_pool:
            boundary_pool = sorted(
                external_pool,
                key=lambda candidate: _repair_candidate_rank(candidate, model),
            )[: max(4, 2 * repair_count)]
        else:
            boundary_pool = [
                candidates[int(idx)] for idx in np.argsort(cv)[: max(4, min(len(candidates), 2 * repair_count))]
            ]

    boundary_vectors = _stack_candidate_vectors(boundary_pool, positions.shape[1])
    boundary_objectives = _stack_candidate_contexts(boundary_pool)
    feasible_vectors = _stack_candidate_vectors(feasible_archive, positions.shape[1])
    feasible_objectives = _stack_candidate_contexts(feasible_archive)

    restart_pool = np.zeros((0, positions.shape[1]), dtype=float)

    proposal_rows: list[np.ndarray] = []
    proposal_owner: list[int] = []
    restart_used = 0.0

    for idx in chosen_idx:
        current = repaired_positions[int(idx)]
        current_obj = _candidate_objective_context(candidates[int(idx)])
        proposals: list[np.ndarray] = []

        boundary = None
        if boundary_vectors.size > 0:
            decision_distance = np.linalg.norm((boundary_vectors - current[None, :]) / span_safe[None, :], axis=1)
            objective_distance = np.linalg.norm(boundary_objectives - current_obj.reshape(1, -1), axis=1)
            boundary_id = int(np.argmin(0.75 * decision_distance + 0.25 * objective_distance))
            boundary = boundary_vectors[boundary_id].copy()

        feasible_anchor = None
        if feasible_vectors.size > 0:
            if boundary is None:
                feasible_anchor = feasible_vectors[int(np.random.randint(0, feasible_vectors.shape[0]))].copy()
            else:
                decision_distance = np.linalg.norm(
                    (feasible_vectors - boundary.reshape(1, -1)) / span_safe[None, :], axis=1
                )
                objective_distance = np.linalg.norm(feasible_objectives - current_obj.reshape(1, -1), axis=1)
                feasible_id = int(np.argmin(0.60 * decision_distance + 0.40 * objective_distance))
                feasible_anchor = feasible_vectors[feasible_id].copy()

        if boundary is not None:
            boundary_pull = float(np.clip(0.34 + 0.22 * np.tanh(cv[int(idx)]), 0.34, 0.56))
            proposals.append(
                current
                + boundary_pull * (boundary - current)
                + np.random.normal(0.0, 1.0, size=current.shape) * (0.006 * span)
            )
            proposals.append(
                0.60 * current + 0.40 * boundary + np.random.normal(0.0, 1.0, size=current.shape) * (0.004 * span)
            )

        if boundary is not None and feasible_anchor is not None:
            pump_pull = float(np.clip(0.48 + 0.16 * np.tanh(cv[int(idx)]), 0.48, 0.64))
            proposals.append(
                boundary
                + pump_pull * (feasible_anchor - boundary)
                + np.random.normal(0.0, 1.0, size=current.shape) * (0.006 * span)
            )
            proposals.append(
                0.35 * current
                + 0.30 * boundary
                + 0.35 * feasible_anchor
                + np.random.normal(0.0, 1.0, size=current.shape) * (0.004 * span)
            )
            elite_pull = float(np.clip(0.56 + 0.16 * np.tanh(cv[int(idx)]), 0.56, 0.74))
            proposals.append(
                (1.0 - elite_pull) * current
                + elite_pull * feasible_anchor
                + np.random.normal(0.0, 1.0, size=current.shape) * (0.008 * span)
            )
        elif feasible_anchor is not None:
            pull = float(np.clip(0.58 + 0.18 * np.tanh(cv[int(idx)]), 0.58, 0.78))
            proposals.append(
                (1.0 - pull) * current
                + pull * feasible_anchor
                + np.random.normal(0.0, 1.0, size=current.shape) * (0.01 * span)
            )
        else:
            if restart_pool.size == 0:
                restart_pool, _ = _terrain_seed_population(
                    model,
                    lower=lower,
                    upper=upper,
                    pop_size=max(1, repair_count),
                    fleet_size=fleet_size,
                    n_waypoints=n_waypoints,
                    seed_fraction=1.0,
                )
                if restart_pool.size == 0:
                    restart_pool = np.asarray(leaders, dtype=float)
            restart_anchor = restart_pool[int(np.random.randint(0, restart_pool.shape[0]))].copy()
            proposals.append(restart_anchor + np.random.normal(0.0, 1.0, size=current.shape) * (0.005 * span))
            restart_used = 1.0

        for proposal in proposals:
            proposal_rows.append(np.clip(proposal, lower, upper))
            proposal_owner.append(int(idx))

    if not proposal_rows:
        return (
            positions,
            candidates,
            {"attempted": float(chosen_idx.size), "accepted": 0.0, "restart_used": float(restart_used)},
        )

    proposal_matrix = np.stack(proposal_rows, axis=0)
    repaired_candidates = _evaluate_population(
        proposal_matrix,
        model,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
    )
    updated_candidates = list(candidates)
    accepted = 0.0
    best_by_owner: dict[int, tuple[Candidate, np.ndarray]] = {}

    for owner_idx, proposal_vec, candidate in zip(proposal_owner, proposal_matrix, repaired_candidates, strict=False):
        current_best = best_by_owner.get(int(owner_idx))
        if current_best is None:
            best_by_owner[int(owner_idx)] = (candidate, proposal_vec)
            continue
        if _repair_candidate_rank(candidate, model) < _repair_candidate_rank(current_best[0], model):
            best_by_owner[int(owner_idx)] = (candidate, proposal_vec)

    for candidate_idx in chosen_idx:
        owner = int(candidate_idx)
        old_candidate = updated_candidates[owner]
        best_pair = best_by_owner.get(owner)
        if best_pair is None:
            continue
        new_candidate, proposal_vec = best_pair
        old_cv = float(cv[int(candidate_idx)])
        new_cv = float(max(0.0, _constraint_violation(new_candidate, model)))
        old_feasible = old_cv <= 0.0
        new_feasible = new_cv <= 0.0

        accept = False
        if new_feasible and not old_feasible or new_cv + 1e-12 < old_cv:
            accept = True
        elif new_feasible and old_feasible:
            accept = _candidate_objective_sum(new_candidate) + 1e-12 < _candidate_objective_sum(old_candidate)

        if accept:
            repaired_positions[owner] = np.asarray(proposal_vec, dtype=float)
            updated_candidates[owner] = new_candidate
            accepted += 1.0
        else:
            repaired_positions[owner] = np.asarray(old_candidate.vector, dtype=float)

    return (
        repaired_positions,
        updated_candidates,
        {
            "attempted": float(chosen_idx.size),
            "accepted": float(accepted),
            "restart_used": float(restart_used),
        },
    )
