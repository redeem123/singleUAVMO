from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.sac_smopso.initialization import (
    _model_constraint_values,
    _paths_to_spherical_decision,
)
from uav_benchmark.algorithms.sac_smopso.scoring import _reservoir_score
from uav_benchmark.algorithms.shared.fleet_runner import _decision_to_paths_spherical
from uav_benchmark.algorithms.shared.nmopso_engine import NMOPSOEngine, _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.evaluate_path import _bilinear_interpolate, _dist_points_to_segments_2d, _interpolate_path
from uav_benchmark.core.mission_encoding import paths_to_decision


def _violation_value(details: dict[str, Any]) -> float:
    """Scalar violation metric used for repair accept/reject decisions."""
    sep = float(details.get("separationViolation", 0.0))
    col = float(details.get("collisionViolation", 0.0))
    turn = float(details.get("turnViolation", 0.0))
    conf = float(details.get("conflictRate", 0.0))
    return sep + col + 0.5 * turn + 0.5 * conf


def _conflict_repair_step(
    engine: NMOPSOEngine,
    fleet_size: int,
    n_waypoints: int,
    offspring: np.ndarray,
    repair_intensity: float,
) -> np.ndarray:
    """Apply a single Gaussian repair pass to every row of ``offspring``.

    Called from inside the SBX+reservoir injection so the repair step is
    tightly coupled to the operator that produced the offspring — there is
    no separate ``_conflict_repair_step`` generation pass any more.

    The sigma schedule is controlled by the SAC-produced ``repair_intensity``
    in [0, 1]:
      * < 0.05 disables the pass (returns offspring unchanged).
      * Otherwise, adds Gaussian noise of magnitude ~(0.05 + 0.15 * ri) * span
        to every decision variable, then clips to bounds. Offspring that
        improve on their input under ``_violation_value`` are kept; others
        revert to the SBX output. Offspring geometry ranking (via
        ``_reservoir_score``) happens in the caller, so we do not need to
        track per-UAV conflict logs here.

    Returns a ``(N, D)`` array of (possibly repaired) vectors.
    """
    ri = float(np.clip(repair_intensity, 0.0, 1.0))
    if ri < 0.05 or offspring.size == 0 or fleet_size <= 1:
        return offspring
    lower = np.asarray(engine.lower, dtype=float)
    upper = np.asarray(engine.upper, dtype=float)
    span = np.asarray(engine.span, dtype=float)
    sigma = (0.05 + 0.15 * ri) * span
    noise = np.random.normal(0.0, 1.0, size=offspring.shape) * sigma
    trial = np.clip(offspring + noise, lower, upper)
    return trial


def _smooth_path(path_xyz: np.ndarray, passes: int = 1) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    if path.shape[0] <= 2:
        return path
    for _ in range(max(1, int(passes))):
        path[1:-1] = 0.25 * path[:-2] + 0.50 * path[1:-1] + 0.25 * path[2:]
    return path


def _lift_path(path_xyz: np.ndarray, delta_z: float) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    if path.shape[0] <= 2 or delta_z <= 0.0:
        return path
    path[1:-1, 2] += float(delta_z)
    return path


def _collect_obstacle_discs(model: dict[str, Any]) -> np.ndarray:
    discs: list[np.ndarray] = []
    threats = np.asarray(model.get("threats", np.zeros((0, 4))), dtype=float)
    if threats.ndim == 2 and threats.shape[1] >= 4:
        discs.append(threats[:, [0, 1, 3]])
    nofly_center = np.asarray(model.get("nofly_c", np.zeros((0, 2))), dtype=float)
    nofly_radius = np.asarray(model.get("nofly_r", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    if nofly_center.ndim == 1 and nofly_center.size >= 2:
        nofly_center = nofly_center.reshape(1, -1)
    if nofly_center.ndim == 2 and nofly_center.shape[1] >= 2 and nofly_center.shape[0] > 0:
        if nofly_radius.size == 1:
            nofly_radius = np.repeat(nofly_radius, nofly_center.shape[0])
        elif nofly_radius.size < nofly_center.shape[0] and nofly_radius.size > 0:
            nofly_radius = np.pad(nofly_radius, (0, nofly_center.shape[0] - nofly_radius.size), mode="edge")
        elif nofly_radius.size == 0:
            nofly_radius = np.zeros(nofly_center.shape[0], dtype=float)
        discs.append(np.column_stack([nofly_center[:, 0], nofly_center[:, 1], nofly_radius[: nofly_center.shape[0]]]))
    if not discs:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(discs).astype(float)


def _push_path_from_obstacles(
    path_xyz: np.ndarray, model: dict[str, Any], clearance_target: float, intensity: float
) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    if path.shape[0] <= 2:
        return path
    discs = _collect_obstacle_discs(model)
    if discs.size == 0:
        return path
    interior = path[1:-1, :2].copy()
    for idx in range(interior.shape[0]):
        point = interior[idx]
        delta = point.reshape(1, 2) - discs[:, :2]
        distance = np.linalg.norm(delta, axis=1)
        distance = np.where(np.isfinite(distance), distance, np.inf)
        nearest = int(np.argmin(distance))
        required = float(discs[nearest, 2]) + float(clearance_target)
        shortfall = required - float(distance[nearest])
        if shortfall <= 0.0:
            continue
        if float(distance[nearest]) > 1e-9:
            direction = delta[nearest] / float(distance[nearest])
        else:
            direction = np.array([1.0, 0.0], dtype=float)
        interior[idx] = point + direction * shortfall * (0.85 + 0.65 * float(intensity))
    path[1:-1, :2] = interior
    return path


def _ratio_to_index(n_points: int, ratio: float) -> int:
    if n_points <= 1:
        return 0
    scaled = int(round(float(np.clip(ratio, 0.0, 1.0)) * max(0, n_points - 1)))
    return int(np.clip(scaled, 0, n_points - 1))


def _window_indices(n_points: int, center_index: int, span: int) -> np.ndarray:
    if n_points <= 2:
        return np.zeros(0, dtype=int)
    lo = max(1, int(center_index) - int(span))
    hi = min(n_points - 2, int(center_index) + int(span))
    if lo > hi:
        return np.zeros(0, dtype=int)
    return np.arange(lo, hi + 1, dtype=int)


def _gaussian_weights(indices: np.ndarray, center_index: int, span: int) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(0, dtype=float)
    sigma = max(1.0, 0.65 * max(1, int(span)))
    delta = np.asarray(indices, dtype=float) - float(center_index)
    weights = np.exp(-0.5 * (delta / sigma) ** 2)
    peak = float(np.max(weights))
    return weights / peak if peak > 1e-12 else np.ones(indices.size, dtype=float)


def _smooth_window(path_xyz: np.ndarray, start_idx: int, end_idx: int, passes: int = 1) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    if path.shape[0] <= 3:
        return path
    lo = max(1, int(start_idx))
    hi = min(path.shape[0] - 2, int(end_idx))
    if lo > hi:
        return path
    for _ in range(max(1, int(passes))):
        updated = path.copy()
        for idx in range(lo, hi + 1):
            updated[idx] = 0.25 * path[idx - 1] + 0.50 * path[idx] + 0.25 * path[idx + 1]
        path = updated
    return path


def _clip_absolute_path(path_xyz: np.ndarray, model: dict[str, Any], original: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    xmin = float(model["xmin"])
    xmax = float(model["xmax"])
    ymin = float(model["ymin"])
    ymax = float(model["ymax"])
    zmin = float(model["zmin"])
    zmax = float(model["zmax"])
    safe_h = float(model["safeH"]) if "safeH" in model and model["safeH"] is not None else None

    path[:, 0] = np.clip(path[:, 0], xmin, xmax)
    path[:, 1] = np.clip(path[:, 1], ymin, ymax)
    ground = _bilinear_interpolate(np.asarray(model["H"], dtype=float), path[:, 0] - 1.0, path[:, 1] - 1.0)
    z_lo = ground + zmin
    if safe_h is not None:
        z_lo = np.maximum(z_lo, ground + safe_h)
    z_hi = ground + zmax
    path[:, 2] = np.clip(path[:, 2], z_lo, z_hi)
    path[0] = np.asarray(original[0], dtype=float)
    path[-1] = np.asarray(original[-1], dtype=float)
    return path


def _path_clearance_profile(path_xyz: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = np.asarray(path_xyz, dtype=float)
    if path.shape[0] < 2:
        return path.copy(), np.zeros(0, dtype=float)
    step_size = float(model.get("collisionStep", 1.0))
    if step_size <= 0.0:
        step_size = 1.0
    interpolated = _interpolate_path(path, step_size)
    if interpolated.shape[0] < 2:
        return interpolated, np.zeros(0, dtype=float)
    height_map = np.asarray(model["H"], dtype=float)
    x_interp = np.asarray(interpolated[:, 0], dtype=float)
    y_interp = np.asarray(interpolated[:, 1], dtype=float)
    ground_interp = _bilinear_interpolate(height_map, x_interp - 1.0, y_interp - 1.0)
    z_rel = np.asarray(interpolated[:, 2], dtype=float) - ground_interp
    terrain_clearance = np.minimum(z_rel[:-1], z_rel[1:])
    discs = _collect_obstacle_discs(model)
    if discs.ndim != 2 or discs.shape[0] == 0:
        return interpolated, terrain_clearance
    seg_starts = np.column_stack([x_interp[:-1], y_interp[:-1]])
    seg_ends = np.column_stack([x_interp[1:], y_interp[1:]])
    distance_matrix = _dist_points_to_segments_2d(discs[:, :2], seg_starts, seg_ends)
    obstacle_clearance = distance_matrix - discs[:, 2].reshape(1, -1)
    min_obstacle_clearance = np.min(obstacle_clearance, axis=1)
    return interpolated, np.minimum(terrain_clearance, min_obstacle_clearance)


def _path_perpendicular(path_xyz: np.ndarray, center_index: int) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float)
    left = max(0, int(center_index) - 1)
    right = min(path.shape[0] - 1, int(center_index) + 1)
    direction = path[right, :2] - path[left, :2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = path[-1, :2] - path[0, :2]
        norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    direction = direction / norm
    return np.array([-direction[1], direction[0]], dtype=float)


def _localized_clearance_edit(
    path_xyz: np.ndarray,
    model: dict[str, Any],
    *,
    center_ratio: float,
    severity: float,
    span: int,
) -> np.ndarray:
    target = np.asarray(path_xyz, dtype=float).copy()
    center_index = _ratio_to_index(target.shape[0], center_ratio)
    window = _window_indices(target.shape[0], center_index, span)
    if window.size == 0:
        return target
    drone_size = float(model.get("droneSize", 1.0))
    safe_dist = float(model.get("safeDist", 10.0))
    lift = max(1.0, float(severity) * 1.10)
    direction = _path_perpendicular(target, center_index)
    weights = _gaussian_weights(window, center_index, span)
    discs = _collect_obstacle_discs(model)
    if discs.ndim == 2 and discs.shape[0] > 0:
        center_xy = np.asarray(target[center_index, :2], dtype=float)
        delta = center_xy.reshape(1, 2) - discs[:, :2]
        dist = np.linalg.norm(delta, axis=1)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        nearest = int(np.argmin(dist))
        required = float(discs[nearest, 2]) + float(drone_size + 0.45 * safe_dist)
        radial = delta[nearest] / float(dist[nearest]) if float(dist[nearest]) > 1e-9 else direction.copy()
        radial_push = max(0.0, required - float(dist[nearest])) + 0.20 * safe_dist
        target[window, 0] += radial[0] * radial_push * weights
        target[window, 1] += radial[1] * radial_push * weights
    target[window, 2] += lift * weights
    target[window, 0] += direction[0] * (0.30 * safe_dist) * weights
    target[window, 1] += direction[1] * (0.30 * safe_dist) * weights
    if center_ratio > 0.80 and target.shape[0] > 3:
        tail_window = np.arange(max(1, target.shape[0] - 3), target.shape[0] - 1, dtype=int)
        tail_weights = np.linspace(1.0, 0.6, tail_window.size, dtype=float)
        target[tail_window, 2] += (1.50 * lift) * tail_weights
        ground_tail = _bilinear_interpolate(
            np.asarray(model["H"], dtype=float),
            target[tail_window, 0] - 1.0,
            target[tail_window, 1] - 1.0,
        )
        zmax = float(model["zmax"])
        target[tail_window, 2] = np.maximum(target[tail_window, 2], ground_tail + 0.98 * zmax)
    target = _smooth_window(target, int(window[0]), int(window[-1]), passes=2)
    return _clip_absolute_path(target, model, np.asarray(path_xyz, dtype=float))


def _bridge_to_guide_path(
    path_xyz: np.ndarray,
    guide_xyz: np.ndarray,
    model: dict[str, Any],
    *,
    center_ratio: float,
    severity: float,
    span: int,
) -> np.ndarray:
    target = np.asarray(path_xyz, dtype=float).copy()
    guide = np.asarray(guide_xyz, dtype=float)
    if target.shape != guide.shape or target.shape[0] <= 2:
        return target
    center_index = _ratio_to_index(target.shape[0], center_ratio)
    window = _window_indices(target.shape[0], center_index, span)
    if window.size == 0:
        return target
    alpha = float(np.clip(0.25 + 0.35 * severity, 0.20, 0.75))
    target[window] = (1.0 - alpha) * target[window] + alpha * guide[window]
    target = _smooth_window(target, int(window[0]), int(window[-1]), passes=2)
    return _clip_absolute_path(target, model, np.asarray(path_xyz, dtype=float))


def _decode_paths(
    vector: np.ndarray,
    *,
    model: dict[str, Any],
    representation: str,
    fleet_size: int,
    n_waypoints: int,
) -> list[np.ndarray]:
    if str(representation).strip().lower() == "cart":
        from uav_benchmark.core.mission_encoding import decision_to_paths

        return decision_to_paths(vector, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    return _decision_to_paths_spherical(vector, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)


def _encode_paths(
    paths: list[np.ndarray],
    *,
    model: dict[str, Any],
    representation: str,
    fleet_size: int,
    n_waypoints: int,
) -> np.ndarray:
    if str(representation).strip().lower() == "cart":
        return paths_to_decision(paths, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    return _paths_to_spherical_decision(paths, model, fleet_size, n_waypoints)


def _repair_paths(
    candidate: Candidate,
    *,
    model: dict[str, Any],
    representation: str,
    fleet_size: int,
    n_waypoints: int,
) -> list[np.ndarray]:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    raw_paths = details.get("paths")
    if isinstance(raw_paths, list):
        source_paths = raw_paths
    else:
        source_paths = _decode_paths(
            candidate.vector,
            model=model,
            representation=representation,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
        )
    return [np.asarray(path, dtype=float).copy() for path in source_paths]


def _repair_span_from_shortfall(shortfall: float, target: float) -> int:
    severity = float(np.clip(float(shortfall) / max(float(target), 1e-9), 0.0, 1.75))
    scaled = int(np.clip(np.floor(min(0.999, severity) * 4.0), 0, 3))
    return max(1, min(4, 1 + scaled))


def _targeted_geometry_repair(
    engine: NMOPSOEngine,
    *,
    model: dict[str, Any],
    representation: str,
    fleet_size: int,
    n_waypoints: int,
    lower: np.ndarray,
    upper: np.ndarray,
    repair_intensity: float,
    aux_candidates: list[Candidate] | None = None,
) -> dict[str, float]:
    intensity = float(np.clip(repair_intensity, 0.0, 1.0))
    stats = {"effectCount": 0.0, "evalCount": 0.0}
    if intensity < 0.05 or not engine.candidates:
        return stats

    separation_min, drone_size, max_turn_deg = _model_constraint_values(model)
    clearance_target = float(model.get("droneSize", 1.0)) + 0.25 * float(
        model.get("safeDist", model.get("separationMin", 10.0))
    )
    turn_limit = max_turn_deg
    candidate_scores: list[tuple[float, int]] = []
    for idx, candidate in enumerate(engine.candidates):
        details = candidate.details if isinstance(candidate.details, dict) else {}
        min_clearance = float(details.get("minClearance", np.nan))
        collision_flag = float(details.get("collisionViolation", 0.0)) > 0.5
        turn_excess = max(0.0, float(details.get("maxTurnDeg", 0.0)) - turn_limit)
        clearance_shortfall = clearance_target - min_clearance if np.isfinite(min_clearance) else clearance_target
        pressure = max(0.0, clearance_shortfall) + 0.05 * max(0.0, turn_excess)
        if collision_flag or pressure > 0.0:
            candidate_scores.append((pressure, idx))
    if not candidate_scores:
        return stats

    candidate_scores.sort(reverse=True)
    repair_budget = max(1, min(len(candidate_scores), int(round(engine.pop_size * (0.10 + 0.30 * intensity)))))
    target_indices = [idx for _score, idx in candidate_scores[:repair_budget]]
    trial_vectors: list[np.ndarray] = []
    original_candidates: list[Candidate] = []
    original_indices: list[int] = []

    for idx in target_indices:
        candidate = engine.candidates[idx]
        candidate_details = candidate.details if isinstance(candidate.details, dict) else {}
        candidate_min_clearance = float(candidate_details.get("minClearance", np.nan))
        candidate_collision = float(candidate_details.get("collisionViolation", 0.0)) > 0.5
        candidate_turn_excess = max(0.0, float(candidate_details.get("maxTurnDeg", 0.0)) - turn_limit)
        paths = _repair_paths(
            candidate,
            model=model,
            representation=representation,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
        )
        changed = False
        if (
            candidate_collision
            or not np.isfinite(candidate_min_clearance)
            or candidate_min_clearance < clearance_target
        ):
            n_clear_iters = 1 + int(round(3.0 * intensity))
            last_best_uav = 0
            last_best_ratio = 0.5
            last_severity = 0.0
            for _ in range(max(1, n_clear_iters)):
                offender_info: list[tuple[float, int, float]] = []
                for uav_idx, path in enumerate(paths):
                    _interp, clearance = _path_clearance_profile(path, model)
                    if clearance.size == 0:
                        continue
                    local_idx = int(np.argmin(clearance))
                    local_clearance = float(clearance[local_idx])
                    shortfall = max(0.0, clearance_target - local_clearance)
                    if shortfall <= 1e-6:
                        continue
                    offender_info.append((shortfall, uav_idx, float(local_idx / max(1, clearance.size - 1))))
                if not offender_info:
                    break
                offender_info.sort(reverse=True)
                shortfall, best_uav, best_ratio = offender_info[0]
                last_best_uav = best_uav
                last_best_ratio = best_ratio
                last_severity = shortfall
                if shortfall <= 1e-6:
                    break
                max_offenders = max(1, min(len(offender_info), 1 + int(round(2.0 * intensity))))
                for shortfall_i, uav_idx_i, ratio_i in offender_info[:max_offenders]:
                    span_i = _repair_span_from_shortfall(shortfall_i, clearance_target)
                    paths[uav_idx_i] = _localized_clearance_edit(
                        paths[uav_idx_i],
                        model,
                        center_ratio=ratio_i,
                        severity=max(shortfall_i, 1.0 + 8.0 * intensity),
                        span=span_i,
                    )
            if last_severity > 1e-6 and aux_candidates:
                guide_pool = [
                    candidate
                    for candidate in aux_candidates
                    if isinstance(candidate.details, dict)
                    and float(candidate.details.get("minClearance", -np.inf)) > candidate_min_clearance + 1e-6
                    and isinstance(candidate.details.get("paths", None), list)
                ]
                if guide_pool:
                    guide_pool.sort(
                        key=lambda candidate: (
                            -float(candidate.details.get("minClearance", -np.inf)),
                            _reservoir_score(
                                candidate,
                                separation_min=separation_min,
                                drone_size=drone_size,
                                max_turn_deg=max_turn_deg,
                            ),
                        )
                    )
                    guide = guide_pool[0]
                    guide_paths = [np.asarray(path, dtype=float) for path in guide.details.get("paths", [])]
                    if (
                        len(guide_paths) > last_best_uav
                        and guide_paths[last_best_uav].shape == paths[last_best_uav].shape
                    ):
                        paths[last_best_uav] = _bridge_to_guide_path(
                            paths[last_best_uav],
                            guide_paths[last_best_uav],
                            model,
                            center_ratio=last_best_ratio,
                            severity=max(last_severity, 1.0 + 8.0 * intensity),
                            span=_repair_span_from_shortfall(last_severity, clearance_target),
                        )
            changed = True
        if candidate_turn_excess > 0.0:
            for path_idx, path in enumerate(paths):
                paths[path_idx] = _smooth_path(path, passes=1 + int(round(2.0 * intensity)))
            changed = True
        if not changed:
            continue
        vector = _encode_paths(
            paths,
            model=model,
            representation=representation,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
        )
        trial_vectors.append(np.clip(np.asarray(vector, dtype=float).reshape(-1), lower, upper))
        original_candidates.append(candidate)
        original_indices.append(idx)

    if not trial_vectors:
        return stats
    trial_matrix = np.stack(trial_vectors, axis=0)
    trial_candidates = engine._evaluate(trial_matrix)  # noqa: SLF001
    stats["evalCount"] = float(len(trial_candidates))
    effect_count = 0
    for local_idx, repaired in enumerate(trial_candidates):
        idx = original_indices[local_idx]
        current = original_candidates[local_idx]
        repaired_violation = _violation_value(repaired.details if isinstance(repaired.details, dict) else {})
        current_violation = _violation_value(current.details if isinstance(current.details, dict) else {})
        if repaired_violation + 1e-4 < current_violation:
            engine.population[idx] = trial_matrix[local_idx]
            engine.candidates[idx] = repaired
            effect_count += 1
    if effect_count > 0:
        engine.current_obj = _candidate_matrix(engine.candidates)
        engine.update_archive(engine.candidates)
        if engine.velocity.shape == engine.population.shape:
            engine.velocity *= 0.5
    stats["effectCount"] = float(effect_count)
    return stats


# ──────────────────────────────────────────────────────────────────
# CMOSMA-inspired unconstrained reservoir + dual-population SBX step
# ──────────────────────────────────────────────────────────────────
#
# Why this exists:
#   NMOPSO's velocity update collapses into 180° hairpin turns on
#   s_120 fleet=1 and all fleet>=3 problems (see diagnostics
#   2026-04). CMOSMA avoids this by maintaining an *unconstrained*
#   archive (AP) of best-by-objective candidates and mating them
#   with the constrained population via SBX + SOM neighborhoods.
#   We port a lightweight version of that idea: a flat top-K
#   reservoir (no SOM) and a SAC-scheduled SBX injection that pulls
#   parents from (population, reservoir). Cost: one extra action
#   dim (``sbx_weight``) and a small Python-side list.
#
# Key differences from CMOSMA:
#   * The SBX intensity is *learned* per generation by the SAC
#     policy, not by a fixed schedule.
#   * Reservoir ranking uses the pre-infeasibility makespan/energy/
#     risk/turn stored in ``details``; ``candidate.objective`` has
#     already been masked to inf when any constraint is violated,
#     so we cannot rank directly on it.
#   * No SOM neighborhoods — a flat top-K is enough at our pop
#     sizes (48–96) and keeps state tiny.
