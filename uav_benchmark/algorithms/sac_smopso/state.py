from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from uav_benchmark.algorithms.sac_smopso.constants import (
    _ARCHIVE_TOKEN_COUNT,
    _CANDIDATE_TOKEN_DIM,
    _ENVIRONMENT_TOKEN_COUNT,
    _ENVIRONMENT_TOKEN_DIM,
    _GLOBAL_STATE_DIM,
    _INTERACTION_TOKEN_COUNT,
    _INTERACTION_TOKEN_DIM,
    _OBJECTIVE_COUNT,
    _POLICY_MODE_ALIASES,
    _POPULATION_TOKEN_COUNT,
    _STATE_REPRESENTATION_ALIASES,
    _TEMPORAL_WINDOW,
    _TOPOLOGY_TOKEN_COUNT,
    _TOPOLOGY_TOKEN_DIM,
)
from uav_benchmark.algorithms.sac_smopso.controller import TemporalRelationalState
from uav_benchmark.algorithms.sac_smopso.initialization import (
    _model_constraint_values,
    _report_ready_candidate,
    _safe_mean,
)
from uav_benchmark.algorithms.sac_smopso.scoring import _reservoir_score
from uav_benchmark.algorithms.shared.nmopso_engine import (
    NMOPSOEngine,
    _candidate_feasible_flags,
    _candidate_matrix,
    _fixed_hv_reference,
    _hypergrid_cell_id,
    _objective_score,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.metrics import cal_metric


def _spatial_cell_ids(points: np.ndarray, n_bins: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=int)
    matrix = np.asarray(points, dtype=float)
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return np.zeros(matrix.shape[0], dtype=int)
    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    normalized = np.clip((matrix - mins) / span, 0.0, 1.0)
    grid = np.clip((normalized * max(2, int(n_bins))).astype(int), 0, max(2, int(n_bins)) - 1)
    multipliers = np.power(max(2, int(n_bins)), np.arange(grid.shape[1], dtype=int))
    return np.sum(grid * multipliers, axis=1)


def _problem_descriptors(model: dict[str, Any], fleet_size: int, max_turn_deg: float) -> np.ndarray:
    threats = np.asarray(model.get("threats", np.zeros((0, 4))), dtype=float)
    threat_count = int(threats.shape[0]) if threats.ndim == 2 else 0
    nofly_center = np.asarray(model.get("nofly_c", np.zeros((0, 2))), dtype=float)
    if nofly_center.ndim == 1 and nofly_center.size > 0:
        nofly_count = 1
    elif nofly_center.ndim == 2:
        nofly_count = int(nofly_center.shape[0])
    else:
        nofly_count = 0
    map_area = max(
        1.0,
        (float(model.get("xmax", 1.0)) - float(model.get("xmin", 0.0)))
        * (float(model.get("ymax", 1.0)) - float(model.get("ymin", 0.0))),
    )
    density = np.clip((threat_count + nofly_count) / max(1.0, map_area / 2500.0), 0.0, 1.0)
    fleet_scale = np.clip((float(fleet_size) - 1.0) / 7.0, 0.0, 1.0)
    turn_tightness = np.clip((90.0 - float(max_turn_deg)) / 60.0, 0.0, 1.0)
    return np.asarray([fleet_scale, density, turn_tightness], dtype=float)


def _resolve_policy_mode(extra: dict[str, Any]) -> str:
    raw = str(extra.get("sacPolicyMode", extra.get("sac_policy_mode", "online"))).strip().lower()
    return _POLICY_MODE_ALIASES.get(raw, "online")


def _resolve_state_representation(extra: dict[str, Any]) -> tuple[str, str]:
    raw = str(extra.get("stateRepresentation", extra.get("state_representation", "TRFTS"))).strip()
    key = raw.lower().replace(" ", "").replace("_", "-")
    return _STATE_REPRESENTATION_ALIASES.get(key, ("TRFTS", "learned"))


def _map_diagonal(model: dict[str, Any]) -> float:
    dx = float(model.get("xmax", 1.0)) - float(model.get("xmin", 0.0))
    dy = float(model.get("ymax", 1.0)) - float(model.get("ymin", 0.0))
    dz = float(model.get("zmax", 1.0)) - float(model.get("zmin", 0.0))
    return max(np.sqrt(dx * dx + dy * dy + dz * dz), 1.0)


def _build_environment_tokens(model: dict[str, Any], max_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    xmin = float(model.get("xmin", 0.0))
    xmax = float(model.get("xmax", 1.0))
    ymin = float(model.get("ymin", 0.0))
    ymax = float(model.get("ymax", 1.0))
    zmin = float(model.get("zmin", 0.0))
    zmax = float(model.get("zmax", 1.0))
    x_span = max(1e-6, xmax - xmin)
    y_span = max(1e-6, ymax - ymin)
    z_span = max(1e-6, zmax - zmin)
    map_diag = _map_diagonal(model)
    map_center = np.asarray([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=float)

    tokens: list[np.ndarray] = []
    scores: list[float] = []

    threats = np.asarray(model.get("threats", np.zeros((0, 4))), dtype=float)
    if threats.ndim == 2:
        for row in threats:
            if row.size < 4 or not np.all(np.isfinite(row[:4])):
                continue
            center = np.asarray(row[:2], dtype=float)
            token = np.asarray(
                [
                    1.0,
                    0.0,
                    np.clip((float(row[0]) - xmin) / x_span, 0.0, 1.0),
                    np.clip((float(row[1]) - ymin) / y_span, 0.0, 1.0),
                    np.clip((float(row[2]) - zmin) / z_span, 0.0, 1.0),
                    np.clip(float(row[3]) / max(1.0, 0.5 * map_diag), 0.0, 1.0),
                    np.clip(np.linalg.norm(center - map_center) / max(1.0, 0.5 * map_diag), 0.0, 1.0),
                    0.0,
                ],
                dtype=float,
            )
            tokens.append(token)
            scores.append(-float(token[5]))

    nofly_center = np.asarray(model.get("nofly_c", np.zeros((0, 2))), dtype=float)
    nofly_radius = np.asarray(model.get("nofly_r", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    nofly_height = np.asarray(model.get("nofly_h", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    if nofly_center.ndim == 2:
        for index, center in enumerate(nofly_center):
            if center.size < 2 or not np.all(np.isfinite(center[:2])):
                continue
            radius = float(nofly_radius[index]) if index < nofly_radius.size else 0.0
            height = float(nofly_height[index]) if index < nofly_height.size else zmax
            token = np.asarray(
                [
                    0.0,
                    1.0,
                    np.clip((float(center[0]) - xmin) / x_span, 0.0, 1.0),
                    np.clip((float(center[1]) - ymin) / y_span, 0.0, 1.0),
                    np.clip((height - zmin) / z_span, 0.0, 1.0),
                    np.clip(radius / max(1.0, 0.5 * map_diag), 0.0, 1.0),
                    np.clip(np.linalg.norm(center[:2] - map_center) / max(1.0, 0.5 * map_diag), 0.0, 1.0),
                    1.0,
                ],
                dtype=float,
            )
            tokens.append(token)
            scores.append(-float(token[5]))

    if not tokens:
        return _pack_tokens(np.zeros((0, _ENVIRONMENT_TOKEN_DIM), dtype=float), max_tokens, _ENVIRONMENT_TOKEN_DIM)
    matrix = np.stack(tokens, axis=0)
    selected = _select_diverse_indices(matrix, np.asarray(scores, dtype=float), max_tokens=max_tokens)
    return _pack_tokens(matrix[selected], max_tokens, _ENVIRONMENT_TOKEN_DIM)


def _candidate_paths(candidate: Candidate) -> list[np.ndarray]:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    raw_paths = details.get("paths", [])
    paths: list[np.ndarray] = []
    for path in raw_paths:
        arr = np.asarray(path, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] >= 2:
            paths.append(arr[:, :3])
    return paths


def _path_topology_signature(path_xyz: np.ndarray, map_diag: float) -> tuple[float, float, float, float]:
    arr = np.asarray(path_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0
    delta = np.diff(arr[:, :3], axis=0)
    seg_len = np.linalg.norm(delta, axis=1)
    length_norm = float(np.clip(np.sum(seg_len) / max(1e-6, map_diag), 0.0, 2.0))
    delta_xy = np.diff(arr[:, :2], axis=0)
    if delta_xy.shape[0] < 2:
        return length_norm, 0.0, 0.5, 0.0
    heading = np.unwrap(np.arctan2(delta_xy[:, 1], delta_xy[:, 0]))
    turn = np.diff(heading)
    if turn.size == 0:
        return length_norm, 0.0, 0.5, 0.0
    mean_turn = float(np.clip(np.mean(np.abs(turn)) / np.pi, 0.0, 1.0))
    signed_turn = float(np.clip((np.sum(turn) / (np.pi * max(1, turn.size)) + 1.0) * 0.5, 0.0, 1.0))
    turn_std = float(np.clip(np.std(turn) / np.pi, 0.0, 1.0))
    return length_norm, mean_turn, signed_turn, turn_std


def _candidate_overlap_index(paths: list[np.ndarray], separation_min: float) -> float:
    if len(paths) < 2:
        return 0.0
    overlap_values: list[float] = []
    threshold = max(1.0, 1.5 * float(separation_min))
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            length = min(paths[left].shape[0], paths[right].shape[0])
            if length < 2:
                continue
            dist = np.linalg.norm(paths[left][:length, :3] - paths[right][:length, :3], axis=1)
            overlap_values.append(float(np.mean(dist < threshold)))
    return float(np.clip(np.mean(overlap_values), 0.0, 1.0)) if overlap_values else 0.0


def _candidate_topology_vector(
    candidate: Candidate,
    model: dict[str, Any],
    separation_min: float,
    max_turn_deg: float,
    quality: float,
) -> np.ndarray:
    paths = _candidate_paths(candidate)
    map_diag = _map_diagonal(model)
    if paths:
        signatures = np.asarray([_path_topology_signature(path, map_diag) for path in paths], dtype=float)
        length_norm = float(np.clip(np.mean(signatures[:, 0]), 0.0, 1.0))
        mean_turn = float(np.clip(np.mean(signatures[:, 1]), 0.0, 1.0))
        signed_turn = float(np.clip(np.mean(signatures[:, 2]), 0.0, 1.0))
        turn_std = float(np.clip(np.mean(signatures[:, 3]), 0.0, 1.0))
    else:
        length_norm = 0.0
        mean_turn = 0.0
        signed_turn = 0.5
        turn_std = 0.0

    details = candidate.details if isinstance(candidate.details, dict) else {}
    min_clearance = float(details.get("minClearance", np.nan))
    if np.isfinite(min_clearance):
        clearance_pressure = 1.0 - float(np.clip(min_clearance / max(1.0, 3.0 * separation_min), 0.0, 1.0))
    else:
        clearance_pressure = 0.0
    max_turn_used = float(details.get("maxTurnDeg", np.nan))
    turn_violation = float(details.get("turnViolation", 0.0))
    if np.isfinite(max_turn_used) and max_turn_deg > 1e-6:
        turn_saturation = float(np.clip(max_turn_used / max_turn_deg, 0.0, 1.0))
    else:
        turn_saturation = float(np.clip(0.5 + turn_violation, 0.0, 1.0))
    overlap_index = _candidate_overlap_index(paths, separation_min=separation_min)
    return np.asarray(
        [
            np.clip(float(quality), 0.0, 1.0),
            length_norm,
            mean_turn,
            signed_turn,
            turn_std,
            np.clip(clearance_pressure, 0.0, 1.0),
            np.clip(overlap_index, 0.0, 1.0),
            np.clip(turn_saturation, 0.0, 1.0),
        ],
        dtype=float,
    )


def _candidate_extra_features(
    candidate: Candidate,
    separation_min: float,
    max_turn_deg: float,
) -> np.ndarray:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    feasible = np.clip(float(details.get("feasible", 0.0)), 0.0, 1.0)
    conflict = np.clip(float(details.get("conflictRate", 0.0)), 0.0, 1.0)
    violation = np.clip(
        float(details.get("turnViolation", 0.0))
        + float(details.get("separationViolation", 0.0))
        + float(details.get("collisionViolation", 0.0)),
        0.0,
        1.0,
    )
    min_clearance = float(details.get("minClearance", np.nan))
    if np.isfinite(min_clearance):
        clearance_pressure = 1.0 - float(np.clip(min_clearance / max(1.0, 3.0 * separation_min), 0.0, 1.0))
    else:
        clearance_pressure = 0.0
    max_turn_used = float(details.get("maxTurnDeg", np.nan))
    if np.isfinite(max_turn_used) and max_turn_deg > 1e-6:
        turn_saturation = float(np.clip(max_turn_used / max_turn_deg, 0.0, 1.0))
    else:
        turn_saturation = float(np.clip(0.5 + float(details.get("turnViolation", 0.0)), 0.0, 1.0))
    overlap_index = _candidate_overlap_index(_candidate_paths(candidate), separation_min=separation_min)
    return np.asarray(
        [
            feasible,
            conflict,
            violation,
            np.clip(clearance_pressure, 0.0, 1.0),
            np.clip(turn_saturation, 0.0, 1.0),
            np.clip(overlap_index, 0.0, 1.0),
        ],
        dtype=float,
    )


def _select_diverse_indices(features: np.ndarray, scores: np.ndarray, max_tokens: int) -> np.ndarray:
    count = int(features.shape[0])
    if count == 0 or max_tokens <= 0:
        return np.zeros(0, dtype=int)
    if count <= max_tokens:
        return np.arange(count, dtype=int)

    order = np.argsort(np.asarray(scores, dtype=float))
    anchor = max(1, max_tokens // 2)
    selected: list[int] = [int(idx) for idx in order[:anchor].tolist()]
    remaining = [int(idx) for idx in order[anchor:].tolist()]
    feature_matrix = np.asarray(features, dtype=float)
    while remaining and len(selected) < max_tokens:
        chosen_matrix = feature_matrix[np.asarray(selected, dtype=int)]
        best_index = remaining[0]
        best_distance = -np.inf
        for candidate_index in remaining:
            diff = chosen_matrix - feature_matrix[int(candidate_index)].reshape(1, -1)
            min_distance = float(np.min(np.linalg.norm(diff, axis=1))) if diff.size > 0 else 0.0
            if min_distance > best_distance:
                best_distance = min_distance
                best_index = int(candidate_index)
        selected.append(best_index)
        remaining.remove(best_index)
    return np.asarray(selected[:max_tokens], dtype=int)


def _pack_tokens(matrix: np.ndarray, max_tokens: int, token_dim: int) -> tuple[np.ndarray, np.ndarray]:
    packed = np.zeros((max_tokens, token_dim), dtype=float)
    mask = np.zeros(max_tokens, dtype=float)
    if matrix.size == 0:
        return packed, mask
    count = min(max_tokens, matrix.shape[0])
    packed[:count] = matrix[:count]
    mask[:count] = 1.0
    return packed, mask


def _build_candidate_tokens(
    candidates: list[Candidate],
    base_features: np.ndarray,
    separation_min: float,
    max_turn_deg: float,
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        return _pack_tokens(np.zeros((0, _CANDIDATE_TOKEN_DIM), dtype=float), max_tokens, _CANDIDATE_TOKEN_DIM)
    extras = np.stack(
        [
            _candidate_extra_features(
                candidate,
                separation_min=separation_min,
                max_turn_deg=max_turn_deg,
            )
            for candidate in candidates
        ],
        axis=0,
    )
    base = np.asarray(base_features, dtype=float)
    width = min(base.shape[1], _CANDIDATE_TOKEN_DIM - extras.shape[1])
    combined = np.zeros((len(candidates), _CANDIDATE_TOKEN_DIM), dtype=float)
    if width > 0:
        combined[:, :width] = base[:, :width]
    combined[:, width : width + extras.shape[1]] = extras
    scores = _objective_score(_candidate_matrix(candidates))
    selected = _select_diverse_indices(combined, scores, max_tokens=max_tokens)
    return _pack_tokens(combined[selected], max_tokens, _CANDIDATE_TOKEN_DIM)


def _build_topology_tokens(
    archive_candidates: list[Candidate],
    current_candidates: list[Candidate],
    model: dict[str, Any],
    separation_min: float,
    max_turn_deg: float,
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = list(archive_candidates) + list(current_candidates)
    if not candidates:
        return _pack_tokens(np.zeros((0, _TOPOLOGY_TOKEN_DIM), dtype=float), max_tokens, _TOPOLOGY_TOKEN_DIM)
    objective_scores = _objective_score(_candidate_matrix(candidates))
    quality = 1.0 - _normalize_scores(objective_scores)
    topology = np.stack(
        [
            _candidate_topology_vector(
                candidate,
                model=model,
                separation_min=separation_min,
                max_turn_deg=max_turn_deg,
                quality=float(quality[index]),
            )
            for index, candidate in enumerate(candidates)
        ],
        axis=0,
    )
    selected = _select_diverse_indices(topology, objective_scores, max_tokens=max_tokens)
    return _pack_tokens(topology[selected], max_tokens, _TOPOLOGY_TOKEN_DIM)


def _pair_interaction_token(
    path_a: np.ndarray,
    path_b: np.ndarray,
    start_a: np.ndarray,
    start_b: np.ndarray,
    end_a: np.ndarray,
    end_b: np.ndarray,
    separation_min: float,
    map_diag: float,
    quality: float,
    pair_conflict_ratio: float,
) -> np.ndarray:
    length = min(path_a.shape[0], path_b.shape[0])
    if length < 2:
        return np.zeros(_INTERACTION_TOKEN_DIM, dtype=float)
    delta = path_a[:length, :3] - path_b[:length, :3]
    dist = np.linalg.norm(delta, axis=1)
    closest_pressure = 1.0 - float(np.clip(np.min(dist) / max(1.0, 3.0 * separation_min), 0.0, 1.0))
    overlap_ratio = float(np.mean(dist < max(1.0, 1.5 * separation_min)))
    start_sep = float(np.clip(np.linalg.norm(start_a[:3] - start_b[:3]) / map_diag, 0.0, 1.0))
    end_sep = float(np.clip(np.linalg.norm(end_a[:3] - end_b[:3]) / map_diag, 0.0, 1.0))
    heading_a = end_a[:2] - start_a[:2]
    heading_b = end_b[:2] - start_b[:2]
    norm_a = np.linalg.norm(heading_a)
    norm_b = np.linalg.norm(heading_b)
    if norm_a > 1e-9 and norm_b > 1e-9:
        cosine = float(np.clip(np.dot(heading_a, heading_b) / (norm_a * norm_b), -1.0, 1.0))
        heading_divergence = float(np.arccos(cosine) / np.pi)
    else:
        heading_divergence = 0.0
    return np.asarray(
        [
            start_sep,
            end_sep,
            np.clip(closest_pressure, 0.0, 1.0),
            np.clip(overlap_ratio, 0.0, 1.0),
            np.clip(heading_divergence, 0.0, 1.0),
            np.clip(pair_conflict_ratio, 0.0, 1.0),
            np.clip(quality, 0.0, 1.0),
        ],
        dtype=float,
    )


def _build_interaction_tokens(
    archive_candidates: list[Candidate],
    current_candidates: list[Candidate],
    model: dict[str, Any],
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = list(archive_candidates) + list(current_candidates)
    fleet_size = int(model.get("fleetSize", 1))
    if fleet_size <= 1 or not candidates:
        return _pack_tokens(np.zeros((0, _INTERACTION_TOKEN_DIM), dtype=float), max_tokens, _INTERACTION_TOKEN_DIM)

    scores = _objective_score(_candidate_matrix(candidates))
    quality = 1.0 - _normalize_scores(scores)
    map_diag = _map_diagonal(model)
    start = np.asarray(model.get("start", np.zeros((fleet_size, 3))), dtype=float)
    end = np.asarray(model.get("end", np.zeros((fleet_size, 3))), dtype=float)
    if start.ndim == 1:
        start = np.tile(start.reshape(1, -1), (fleet_size, 1))
    if end.ndim == 1:
        end = np.tile(end.reshape(1, -1), (fleet_size, 1))
    interaction_tokens: list[np.ndarray] = []
    interaction_scores: list[float] = []
    selected_candidates = np.argsort(scores)[: min(3, len(candidates))]
    separation_min = float(model.get("separationMin", model.get("safeDist", 10.0)))
    for candidate_index in selected_candidates.tolist():
        candidate = candidates[int(candidate_index)]
        paths = _candidate_paths(candidate)
        if len(paths) < 2:
            continue
        details = candidate.details if isinstance(candidate.details, dict) else {}
        conflict_log = np.asarray(details.get("conflictLog", np.zeros((0, 5), dtype=float)), dtype=float)
        if conflict_log.ndim == 1 and conflict_log.size == 5:
            conflict_log = conflict_log.reshape(1, 5)
        for left in range(len(paths)):
            for right in range(left + 1, len(paths)):
                if conflict_log.ndim == 2 and conflict_log.shape[0] > 0:
                    pair_hits = np.logical_or(
                        np.logical_and(conflict_log[:, 1] == float(left), conflict_log[:, 2] == float(right)),
                        np.logical_and(conflict_log[:, 1] == float(right), conflict_log[:, 2] == float(left)),
                    )
                    pair_conflict_ratio = float(np.mean(pair_hits))
                else:
                    pair_conflict_ratio = 0.0
                token = _pair_interaction_token(
                    path_a=paths[left],
                    path_b=paths[right],
                    start_a=start[left],
                    start_b=start[right],
                    end_a=end[left],
                    end_b=end[right],
                    separation_min=separation_min,
                    map_diag=map_diag,
                    quality=float(quality[int(candidate_index)]),
                    pair_conflict_ratio=pair_conflict_ratio,
                )
                interaction_tokens.append(token)
                interaction_scores.append(-float(token[2] + 0.7 * token[5] + 0.3 * token[3]))
    if not interaction_tokens:
        return _pack_tokens(np.zeros((0, _INTERACTION_TOKEN_DIM), dtype=float), max_tokens, _INTERACTION_TOKEN_DIM)
    matrix = np.stack(interaction_tokens, axis=0)
    selected = _select_diverse_indices(matrix, np.asarray(interaction_scores, dtype=float), max_tokens=max_tokens)
    return _pack_tokens(matrix[selected], max_tokens, _INTERACTION_TOKEN_DIM)


def _build_temporal_window(
    history: deque[np.ndarray],
    current_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((_TEMPORAL_WINDOW, _GLOBAL_STATE_DIM), dtype=float)
    mask = np.zeros(_TEMPORAL_WINDOW, dtype=float)
    sequence = [np.asarray(item, dtype=float).reshape(-1) for item in list(history)[-(_TEMPORAL_WINDOW - 1) :]]
    sequence.append(np.asarray(current_global, dtype=float).reshape(-1))
    clipped = sequence[-_TEMPORAL_WINDOW:]
    for index, token in enumerate(clipped):
        width = min(token.size, _GLOBAL_STATE_DIM)
        tokens[index, :width] = token[:width]
        mask[index] = 1.0
    return tokens, mask


def _archive_snapshot(engine: NMOPSOEngine) -> dict[str, float]:
    archive = list(engine.archive_candidates)
    archive_matrix = _candidate_matrix([_report_ready_candidate(candidate) for candidate in archive])
    finite_archive = (
        archive_matrix[np.all(np.isfinite(archive_matrix), axis=1)] if archive_matrix.size > 0 else archive_matrix
    )
    if engine.hv_ref_point is None and finite_archive.size > 0:
        engine.hv_ref_point = _fixed_hv_reference(finite_archive)

    hv = (
        cal_metric(1, finite_archive, 0, _OBJECTIVE_COUNT, ref_point=engine.hv_ref_point)
        if finite_archive.size > 0
        else 0.0
    )
    diversity = float(np.mean(np.std(finite_archive, axis=0))) if finite_archive.size > 0 else 0.0
    objective_occupancy = (
        len(np.unique(_hypergrid_cell_id(finite_archive, max(4, int(engine.grid_cells)))))
        / max(1, finite_archive.shape[0])
        if finite_archive.size > 0
        else 0.0
    )
    if archive:
        centroids = np.stack([engine._candidate_centroid(candidate) for candidate in archive], axis=0)
        spatial_occupancy = len(np.unique(_spatial_cell_ids(centroids, max(3, int(engine.grid_cells // 2))))) / max(
            1, centroids.shape[0]
        )
    else:
        spatial_occupancy = 0.0

    feasible_archive = _safe_mean(
        [float(getattr(candidate, "details", {}).get("feasible", np.nan)) for candidate in archive],
        default=0.0,
    )
    mean_violation = _safe_mean(
        [
            float(getattr(candidate, "details", {}).get("turnViolation", 0.0))
            + float(getattr(candidate, "details", {}).get("separationViolation", 0.0))
            + float(getattr(candidate, "details", {}).get("collisionViolation", 0.0))
            for candidate in archive
        ],
        default=0.0,
    )
    feasible_ratio = (
        float(np.mean(_candidate_feasible_flags(engine.candidates, engine.current_obj))) if engine.candidates else 0.0
    )
    conflict_rate = _safe_mean(
        [float(getattr(candidate, "details", {}).get("conflictRate", np.nan)) for candidate in engine.candidates],
        default=0.0,
    )
    separation_min, drone_size, max_turn_deg = _model_constraint_values(engine.model)
    geometry_scores = np.asarray(
        [
            _reservoir_score(
                candidate,
                separation_min=separation_min,
                drone_size=drone_size,
                max_turn_deg=max_turn_deg,
            )
            for candidate in engine.candidates
        ],
        dtype=float,
    )
    finite_geometry = geometry_scores[np.isfinite(geometry_scores)]
    best_geometry = float(np.min(finite_geometry)) if finite_geometry.size > 0 else 100.0
    median_geometry = float(np.median(finite_geometry)) if finite_geometry.size > 0 else 100.0
    r2_signal = np.clip(np.tanh(engine.r2_before() / max(1.0, len(archive))), 0.0, 1.0)
    return {
        "hv": float(hv),
        "diversity": float(diversity),
        "archive_fill": np.clip(len(archive) / max(1, int(engine.archive_size)), 0.0, 1.0),
        "objective_occupancy": float(objective_occupancy),
        "spatial_occupancy": float(spatial_occupancy),
        "feasible_archive": np.clip(float(feasible_archive), 0.0, 1.0),
        "mean_violation": np.clip(float(mean_violation), 0.0, 1.0),
        "feasible_ratio": np.clip(float(feasible_ratio), 0.0, 1.0),
        "conflict_rate": max(0.0, float(conflict_rate)),
        "best_geometry": float(best_geometry),
        "median_geometry": float(median_geometry),
        "r2_signal": float(r2_signal),
    }


def _build_controller_state(
    engine: NMOPSOEngine,
    snapshot: dict[str, float],
    generation: int,
    total_generations: int,
    last_hv: float,
    stagnation: int,
    diversity_ref: float,
    problem_descriptor: np.ndarray,
) -> np.ndarray:
    base = engine.state_features(
        generation=int(generation),
        total_generations=int(total_generations),
        last_hv=float(last_hv),
        stagnation=int(stagnation),
        diversity_ref=max(1e-6, float(diversity_ref)),
    )
    archive_features = np.asarray(
        [
            snapshot["archive_fill"],
            snapshot["objective_occupancy"],
            snapshot["spatial_occupancy"],
            snapshot["feasible_archive"],
            snapshot["mean_violation"],
            snapshot["r2_signal"],
        ],
        dtype=float,
    )
    state = np.concatenate(
        [
            base,
            archive_features,
            np.zeros(4, dtype=float),
            np.asarray(problem_descriptor, dtype=float).reshape(-1),
        ],
        axis=0,
    )
    padded = np.zeros(19, dtype=float)
    padded[: min(19, state.size)] = state[: min(19, state.size)]
    return np.clip(np.nan_to_num(padded, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _build_temporal_relational_state(
    engine: NMOPSOEngine,
    model: dict[str, Any],
    snapshot: dict[str, float],
    generation: int,
    total_generations: int,
    last_hv: float,
    stagnation: int,
    diversity_ref: float,
    problem_descriptor: np.ndarray,
    global_history: deque[np.ndarray],
) -> TemporalRelationalState:
    scalar_state = _build_controller_state(
        engine=engine,
        snapshot=snapshot,
        generation=generation,
        total_generations=total_generations,
        last_hv=last_hv,
        stagnation=stagnation,
        diversity_ref=diversity_ref,
        problem_descriptor=problem_descriptor,
    )
    population_tokens, population_mask = _build_candidate_tokens(
        candidates=list(engine.candidates),
        base_features=engine.get_particle_features(),
        separation_min=float(model.get("separationMin", model.get("safeDist", 10.0))),
        max_turn_deg=float(model.get("maxTurnDeg", 75.0)),
        max_tokens=_POPULATION_TOKEN_COUNT,
    )
    archive_tokens, archive_mask = _build_candidate_tokens(
        candidates=list(engine.archive_candidates),
        base_features=engine.get_archive_features(),
        separation_min=float(model.get("separationMin", model.get("safeDist", 10.0))),
        max_turn_deg=float(model.get("maxTurnDeg", 75.0)),
        max_tokens=_ARCHIVE_TOKEN_COUNT,
    )
    topology_tokens, topology_mask = _build_topology_tokens(
        archive_candidates=list(engine.archive_candidates),
        current_candidates=list(engine.candidates),
        model=model,
        separation_min=float(model.get("separationMin", model.get("safeDist", 10.0))),
        max_turn_deg=float(model.get("maxTurnDeg", 75.0)),
        max_tokens=_TOPOLOGY_TOKEN_COUNT,
    )
    interaction_tokens, interaction_mask = _build_interaction_tokens(
        archive_candidates=list(engine.archive_candidates),
        current_candidates=list(engine.candidates),
        model=model,
        max_tokens=_INTERACTION_TOKEN_COUNT,
    )
    environment_tokens, environment_mask = _build_environment_tokens(
        model=model,
        max_tokens=_ENVIRONMENT_TOKEN_COUNT,
    )

    topology_valid = topology_tokens[topology_mask > 0.5]
    interaction_valid = interaction_tokens[interaction_mask > 0.5]
    topology_diversity = float(np.mean(np.std(topology_valid, axis=0))) if topology_valid.size > 0 else 0.0
    clearance_pressure = float(np.mean(topology_valid[:, 5])) if topology_valid.size > 0 else 0.0
    overlap_pressure = float(np.mean(topology_valid[:, 6])) if topology_valid.size > 0 else 0.0
    turn_saturation = float(np.mean(topology_valid[:, 7])) if topology_valid.size > 0 else 0.0
    interaction_pressure = float(np.mean(interaction_valid[:, 2])) if interaction_valid.size > 0 else 0.0
    interaction_count_norm = float(np.clip(np.sum(interaction_mask) / max(1, _INTERACTION_TOKEN_COUNT), 0.0, 1.0))

    global_features = np.zeros(_GLOBAL_STATE_DIM, dtype=float)
    global_features[:19] = scalar_state
    global_features[19] = np.clip(topology_diversity, 0.0, 1.0)
    global_features[20] = np.clip(clearance_pressure, 0.0, 1.0)
    global_features[21] = np.clip(overlap_pressure, 0.0, 1.0)
    global_features[22] = np.clip(interaction_pressure, 0.0, 1.0)
    global_features[23] = np.clip(turn_saturation * (0.6 + 0.4 * interaction_count_norm), 0.0, 1.0)
    temporal_tokens, temporal_mask = _build_temporal_window(global_history, global_features)

    return TemporalRelationalState(
        global_features=np.asarray(global_features, dtype=np.float32),
        population_tokens=np.asarray(population_tokens, dtype=np.float32),
        population_mask=np.asarray(population_mask, dtype=np.float32),
        archive_tokens=np.asarray(archive_tokens, dtype=np.float32),
        archive_mask=np.asarray(archive_mask, dtype=np.float32),
        topology_tokens=np.asarray(topology_tokens, dtype=np.float32),
        topology_mask=np.asarray(topology_mask, dtype=np.float32),
        interaction_tokens=np.asarray(interaction_tokens, dtype=np.float32),
        interaction_mask=np.asarray(interaction_mask, dtype=np.float32),
        environment_tokens=np.asarray(environment_tokens, dtype=np.float32),
        environment_mask=np.asarray(environment_mask, dtype=np.float32),
        temporal_tokens=np.asarray(temporal_tokens, dtype=np.float32),
        temporal_mask=np.asarray(temporal_mask, dtype=np.float32),
    )


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        return vector
    mins = np.min(vector)
    maxs = np.max(vector)
    span = max(maxs - mins, 1e-9)
    return np.clip((vector - mins) / span, 0.0, 1.0)
