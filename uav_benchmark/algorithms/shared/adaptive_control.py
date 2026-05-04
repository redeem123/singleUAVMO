from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from uav_benchmark.algorithms.sac_smopso.controller import (
    ControllerAction,
    ControllerConfig,
    HybridSACController,
    TemporalRelationalState,
    TemporalRelationalStateSpec,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate

AdaptiveRelationalState = TemporalRelationalState
AdaptiveRelationalStateSpec = TemporalRelationalStateSpec
AdaptiveControllerAction = ControllerAction
AdaptiveControllerConfig = ControllerConfig
AdaptiveSACController = HybridSACController

GLOBAL_STATE_DIM = 24
CANDIDATE_TOKEN_DIM = 14
TOPOLOGY_TOKEN_DIM = 8
INTERACTION_TOKEN_DIM = 7
ENVIRONMENT_TOKEN_DIM = 8
POPULATION_TOKEN_COUNT = 12
ARCHIVE_TOKEN_COUNT = 16
TOPOLOGY_TOKEN_COUNT = 8
INTERACTION_TOKEN_COUNT = 12
ENVIRONMENT_TOKEN_COUNT = 16
TEMPORAL_WINDOW = 6


def adaptive_state_spec() -> AdaptiveRelationalStateSpec:
    return AdaptiveRelationalStateSpec(
        global_dim=GLOBAL_STATE_DIM,
        population_dim=CANDIDATE_TOKEN_DIM,
        archive_dim=CANDIDATE_TOKEN_DIM,
        topology_dim=TOPOLOGY_TOKEN_DIM,
        interaction_dim=INTERACTION_TOKEN_DIM,
        environment_dim=ENVIRONMENT_TOKEN_DIM,
        temporal_dim=GLOBAL_STATE_DIM,
    )


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float(default)


def _map_diagonal(model: dict[str, Any]) -> float:
    dx = float(model.get("xmax", 1.0)) - float(model.get("xmin", 0.0))
    dy = float(model.get("ymax", 1.0)) - float(model.get("ymin", 0.0))
    dz = float(model.get("zmax", 1.0)) - float(model.get("zmin", 0.0))
    return max(np.sqrt(dx * dx + dy * dy + dz * dz), 1.0)


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


def _pack_tokens(matrix: np.ndarray, max_tokens: int, token_dim: int) -> tuple[np.ndarray, np.ndarray]:
    packed = np.zeros((max_tokens, token_dim), dtype=float)
    mask = np.zeros(max_tokens, dtype=float)
    if matrix.size == 0:
        return packed, mask
    count = min(max_tokens, matrix.shape[0])
    packed[:count] = matrix[:count]
    mask[:count] = 1.0
    return packed, mask


def _candidate_paths(candidate: Candidate) -> list[np.ndarray]:
    details = candidate.details if isinstance(candidate.details, dict) else {}
    raw_paths = details.get("paths", [])
    paths: list[np.ndarray] = []
    for path in raw_paths:
        arr = np.asarray(path, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] >= 2:
            paths.append(arr[:, :3])
    return paths


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


def _candidate_feature_row(candidate: Candidate, separation_min: float, max_turn_deg: float) -> np.ndarray:
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    details = candidate.details if isinstance(candidate.details, dict) else {}
    row = np.zeros(CANDIDATE_TOKEN_DIM, dtype=float)
    width = min(4, objective.size)
    if width > 0:
        safe_obj = np.asarray(objective[:width], dtype=float)
        if np.any(np.isfinite(safe_obj)):
            finite = safe_obj[np.isfinite(safe_obj)]
            scale = max(1e-6, float(np.max(np.abs(finite)))) if finite.size > 0 else 1.0
            row[:width] = np.clip(np.nan_to_num(safe_obj / scale, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
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
    clearance_pressure = (
        1.0 - float(np.clip(min_clearance / max(1.0, 3.0 * separation_min), 0.0, 1.0))
        if np.isfinite(min_clearance)
        else 0.0
    )
    max_turn_used = float(details.get("maxTurnDeg", np.nan))
    if np.isfinite(max_turn_used) and max_turn_deg > 1e-6:
        turn_saturation = float(np.clip(max_turn_used / max_turn_deg, 0.0, 1.0))
    else:
        turn_saturation = float(np.clip(0.5 + float(details.get("turnViolation", 0.0)), 0.0, 1.0))
    overlap_index = _candidate_overlap_index(_candidate_paths(candidate), separation_min=separation_min)
    row[4:10] = [feasible, conflict, violation, clearance_pressure, turn_saturation, overlap_index]
    row[10] = np.clip(float(details.get("makespan", 0.0)) / 1000.0, 0.0, 1.0)
    row[11] = np.clip(float(details.get("energy", 0.0)) / 1000.0, 0.0, 1.0)
    row[12] = np.clip(float(details.get("risk", 0.0)) / 1000.0, 0.0, 1.0)
    row[13] = np.clip(float(details.get("minSeparation", separation_min)) / max(1.0, 3.0 * separation_min), 0.0, 1.0)
    return np.clip(np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


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


def _candidate_topology_row(
    candidate: Candidate, model: dict[str, Any], separation_min: float, max_turn_deg: float
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
    clearance_pressure = (
        1.0 - float(np.clip(min_clearance / max(1.0, 3.0 * separation_min), 0.0, 1.0))
        if np.isfinite(min_clearance)
        else 0.0
    )
    max_turn_used = float(details.get("maxTurnDeg", np.nan))
    turn_saturation = (
        float(np.clip(max_turn_used / max_turn_deg, 0.0, 1.0))
        if np.isfinite(max_turn_used) and max_turn_deg > 1e-6
        else float(np.clip(0.5 + float(details.get("turnViolation", 0.0)), 0.0, 1.0))
    )
    overlap_index = _candidate_overlap_index(paths, separation_min=separation_min)
    quality = 0.0
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    finite = objective[np.isfinite(objective)]
    if finite.size > 0:
        quality = float(np.clip(1.0 / max(1.0, np.mean(np.abs(finite))), 0.0, 1.0))
    return np.asarray(
        [quality, length_norm, mean_turn, signed_turn, turn_std, clearance_pressure, overlap_index, turn_saturation],
        dtype=float,
    )


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
    threats = np.asarray(model.get("threats", np.zeros((0, 4))), dtype=float)
    if threats.ndim == 2:
        for row in threats:
            if row.size < 4 or not np.all(np.isfinite(row[:4])):
                continue
            center = np.asarray(row[:2], dtype=float)
            tokens.append(
                np.asarray(
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
            )
    nofly_center = np.asarray(model.get("nofly_c", np.zeros((0, 2))), dtype=float)
    nofly_radius = np.asarray(model.get("nofly_r", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    nofly_height = np.asarray(model.get("nofly_h", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    if nofly_center.ndim == 2:
        for index, center in enumerate(nofly_center):
            if center.size < 2 or not np.all(np.isfinite(center[:2])):
                continue
            radius = float(nofly_radius[index]) if index < nofly_radius.size else 0.0
            height = float(nofly_height[index]) if index < nofly_height.size else zmax
            tokens.append(
                np.asarray(
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
            )
    matrix = np.stack(tokens, axis=0) if tokens else np.zeros((0, ENVIRONMENT_TOKEN_DIM), dtype=float)
    return _pack_tokens(matrix[:max_tokens], max_tokens, ENVIRONMENT_TOKEN_DIM)


def _build_temporal_window(history: deque[np.ndarray], current_global: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((TEMPORAL_WINDOW, GLOBAL_STATE_DIM), dtype=float)
    mask = np.zeros(TEMPORAL_WINDOW, dtype=float)
    sequence = [np.asarray(item, dtype=float).reshape(-1) for item in list(history)[-(TEMPORAL_WINDOW - 1) :]]
    sequence.append(np.asarray(current_global, dtype=float).reshape(-1))
    clipped = sequence[-TEMPORAL_WINDOW:]
    for index, token in enumerate(clipped):
        width = min(token.size, GLOBAL_STATE_DIM)
        tokens[index, :width] = token[:width]
        mask[index] = 1.0
    return tokens, mask


def build_adaptive_state(
    candidates: list[Candidate],
    archive_candidates: list[Candidate],
    model: dict[str, Any],
    generation: int,
    total_generations: int,
    last_metrics: dict[str, float],
    algorithm_features: np.ndarray,
    history: deque[np.ndarray],
    state_representation: str = "TRFTS",
) -> AdaptiveRelationalState:
    normalized_representation = str(state_representation).strip().lower()
    use_constraint_pressure_tokens = normalized_representation in {
        "trfts-cp",
        "trfts_cp",
        "constraint-pressure",
        "constraint_pressure",
    }
    progress = float(generation) / max(1.0, float(total_generations))
    feasible_ratio = float(np.clip(last_metrics.get("feasible_ratio", 0.0), 0.0, 1.0))
    conflict_rate = float(np.clip(last_metrics.get("conflict_rate", 0.0), 0.0, 1.0))
    hv_trend = float(np.clip(last_metrics.get("hv_trend", 0.0), 0.0, 1.0))
    diversity = float(np.clip(last_metrics.get("diversity", 0.0), 0.0, 1.0))
    stagnation = float(np.clip(last_metrics.get("stagnation", 0.0), 0.0, 1.0))
    archive_fill = float(np.clip(last_metrics.get("archive_fill", 0.0), 0.0, 1.0))
    objective_occupancy = float(np.clip(last_metrics.get("objective_occupancy", 0.0), 0.0, 1.0))
    spatial_occupancy = float(np.clip(last_metrics.get("spatial_occupancy", 0.0), 0.0, 1.0))
    feasible_archive = float(np.clip(last_metrics.get("feasible_archive", 0.0), 0.0, 1.0))
    mean_violation = float(np.clip(last_metrics.get("mean_violation", 0.0), 0.0, 1.0))
    quality_signal = float(np.clip(last_metrics.get("quality_signal", 0.0), 0.0, 1.0))
    problem_descriptor = _problem_descriptors(
        model=model,
        fleet_size=int(model.get("fleetSize", 1)),
        max_turn_deg=float(model.get("maxTurnDeg", 75.0)),
    )
    algo_vec = np.zeros(6, dtype=float)
    flat_algo = np.asarray(algorithm_features, dtype=float).reshape(-1)
    algo_vec[: min(6, flat_algo.size)] = np.clip(
        np.nan_to_num(flat_algo[:6], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    )

    if normalized_representation == "flat":
        global_features = np.zeros(GLOBAL_STATE_DIM, dtype=float)
        global_features[:12] = [
            progress,
            feasible_ratio,
            conflict_rate,
            hv_trend,
            diversity,
            stagnation,
            archive_fill,
            objective_occupancy,
            spatial_occupancy,
            feasible_archive,
            mean_violation,
            quality_signal,
        ]
        global_features[12:18] = algo_vec
        global_features[18:21] = problem_descriptor
        temporal_tokens, temporal_mask = _build_temporal_window(history, global_features)
        empty14, empty14_mask = _pack_tokens(
            np.zeros((0, CANDIDATE_TOKEN_DIM), dtype=float), POPULATION_TOKEN_COUNT, CANDIDATE_TOKEN_DIM
        )
        empty8, empty8_mask = _pack_tokens(
            np.zeros((0, TOPOLOGY_TOKEN_DIM), dtype=float), TOPOLOGY_TOKEN_COUNT, TOPOLOGY_TOKEN_DIM
        )
        empty7, empty7_mask = _pack_tokens(
            np.zeros((0, INTERACTION_TOKEN_DIM), dtype=float), INTERACTION_TOKEN_COUNT, INTERACTION_TOKEN_DIM
        )
        env, env_mask = _pack_tokens(
            np.zeros((0, ENVIRONMENT_TOKEN_DIM), dtype=float), ENVIRONMENT_TOKEN_COUNT, ENVIRONMENT_TOKEN_DIM
        )
        return AdaptiveRelationalState(
            global_features=np.asarray(global_features, dtype=np.float32),
            population_tokens=np.asarray(empty14, dtype=np.float32),
            population_mask=np.asarray(empty14_mask, dtype=np.float32),
            archive_tokens=np.asarray(empty14, dtype=np.float32),
            archive_mask=np.asarray(empty14_mask, dtype=np.float32),
            topology_tokens=np.asarray(empty8, dtype=np.float32),
            topology_mask=np.asarray(empty8_mask, dtype=np.float32),
            interaction_tokens=np.asarray(empty7, dtype=np.float32),
            interaction_mask=np.asarray(empty7_mask, dtype=np.float32),
            environment_tokens=np.asarray(env, dtype=np.float32),
            environment_mask=np.asarray(env_mask, dtype=np.float32),
            temporal_tokens=np.asarray(temporal_tokens, dtype=np.float32),
            temporal_mask=np.asarray(temporal_mask, dtype=np.float32),
        )

    separation_min = float(model.get("separationMin", model.get("safeDist", 10.0)))
    max_turn_deg = float(model.get("maxTurnDeg", 75.0))
    pop_matrix = (
        np.stack(
            [
                _candidate_feature_row(candidate, separation_min=separation_min, max_turn_deg=max_turn_deg)
                for candidate in candidates
            ],
            axis=0,
        )
        if candidates
        else np.zeros((0, CANDIDATE_TOKEN_DIM), dtype=float)
    )
    archive_matrix = (
        np.stack(
            [
                _candidate_feature_row(candidate, separation_min=separation_min, max_turn_deg=max_turn_deg)
                for candidate in archive_candidates
            ],
            axis=0,
        )
        if archive_candidates
        else np.zeros((0, CANDIDATE_TOKEN_DIM), dtype=float)
    )
    topology_candidates = list(archive_candidates) + list(candidates)
    topology_matrix = (
        np.stack(
            [
                _candidate_topology_row(
                    candidate, model=model, separation_min=separation_min, max_turn_deg=max_turn_deg
                )
                for candidate in topology_candidates
            ],
            axis=0,
        )
        if topology_candidates
        else np.zeros((0, TOPOLOGY_TOKEN_DIM), dtype=float)
    )

    interaction_rows: list[np.ndarray] = []
    map_diag = _map_diagonal(model)
    for candidate in topology_candidates[:3]:
        paths = _candidate_paths(candidate)
        if len(paths) < 2:
            continue
        for left in range(len(paths)):
            for right in range(left + 1, len(paths)):
                length = min(paths[left].shape[0], paths[right].shape[0])
                if length < 2:
                    continue
                dist = np.linalg.norm(paths[left][:length, :3] - paths[right][:length, :3], axis=1)
                start_sep = float(
                    np.clip(np.linalg.norm(paths[left][0, :3] - paths[right][0, :3]) / map_diag, 0.0, 1.0)
                )
                end_sep = float(
                    np.clip(np.linalg.norm(paths[left][-1, :3] - paths[right][-1, :3]) / map_diag, 0.0, 1.0)
                )
                closest_pressure = 1.0 - float(np.clip(np.min(dist) / max(1.0, 3.0 * separation_min), 0.0, 1.0))
                closest_index = int(np.argmin(dist))
                closest_time = float(closest_index / max(1, length - 1))
                overlap_ratio = float(np.mean(dist < max(1.0, 1.5 * separation_min)))
                heading_a = paths[left][-1, :2] - paths[left][0, :2]
                heading_b = paths[right][-1, :2] - paths[right][0, :2]
                norm_a = np.linalg.norm(heading_a)
                norm_b = np.linalg.norm(heading_b)
                if norm_a > 1e-9 and norm_b > 1e-9:
                    cosine = float(np.clip(np.dot(heading_a, heading_b) / (norm_a * norm_b), -1.0, 1.0))
                    heading_div = float(np.arccos(cosine) / np.pi)
                else:
                    heading_div = 0.0
                conflict = (
                    float(np.clip(candidate.details.get("conflictRate", 0.0), 0.0, 1.0))
                    if isinstance(candidate.details, dict)
                    else 0.0
                )
                if use_constraint_pressure_tokens:
                    # Minimal safety relation: how close the pair gets, when
                    # that happens, how long it stays too close, and whether
                    # the routes are convergent or divergent.
                    interaction_rows.append(
                        np.asarray(
                            [
                                closest_pressure,
                                closest_time,
                                overlap_ratio,
                                heading_div,
                                0.0,
                                0.0,
                                0.0,
                            ],
                            dtype=float,
                        )
                    )
                else:
                    interaction_rows.append(
                        np.asarray(
                            [start_sep, end_sep, closest_pressure, overlap_ratio, heading_div, conflict, 0.5],
                            dtype=float,
                        )
                    )
    interaction_matrix = (
        np.stack(interaction_rows, axis=0) if interaction_rows else np.zeros((0, INTERACTION_TOKEN_DIM), dtype=float)
    )
    environment_tokens, environment_mask = _build_environment_tokens(model=model, max_tokens=ENVIRONMENT_TOKEN_COUNT)

    topology_valid = topology_matrix if topology_matrix.size > 0 else np.zeros((0, TOPOLOGY_TOKEN_DIM), dtype=float)
    interaction_valid = (
        interaction_matrix if interaction_matrix.size > 0 else np.zeros((0, INTERACTION_TOKEN_DIM), dtype=float)
    )
    global_features = np.zeros(GLOBAL_STATE_DIM, dtype=float)
    global_features[:12] = [
        progress,
        feasible_ratio,
        conflict_rate,
        hv_trend,
        diversity,
        stagnation,
        archive_fill,
        objective_occupancy,
        spatial_occupancy,
        feasible_archive,
        mean_violation,
        quality_signal,
    ]
    global_features[12:18] = algo_vec
    global_features[18:21] = problem_descriptor
    global_features[21] = float(np.mean(topology_valid[:, 5])) if topology_valid.size > 0 else 0.0
    interaction_pressure_column = 0 if use_constraint_pressure_tokens else 2
    global_features[22] = (
        float(np.mean(interaction_valid[:, interaction_pressure_column])) if interaction_valid.size > 0 else 0.0
    )
    global_features[23] = float(np.mean(topology_valid[:, 7])) if topology_valid.size > 0 else 0.0
    temporal_tokens, temporal_mask = _build_temporal_window(history, global_features)

    pop_tokens, pop_mask = _pack_tokens(
        pop_matrix[:POPULATION_TOKEN_COUNT], POPULATION_TOKEN_COUNT, CANDIDATE_TOKEN_DIM
    )
    arc_tokens, arc_mask = _pack_tokens(archive_matrix[:ARCHIVE_TOKEN_COUNT], ARCHIVE_TOKEN_COUNT, CANDIDATE_TOKEN_DIM)
    top_tokens, top_mask = _pack_tokens(
        topology_matrix[:TOPOLOGY_TOKEN_COUNT], TOPOLOGY_TOKEN_COUNT, TOPOLOGY_TOKEN_DIM
    )
    int_tokens, int_mask = _pack_tokens(
        interaction_matrix[:INTERACTION_TOKEN_COUNT], INTERACTION_TOKEN_COUNT, INTERACTION_TOKEN_DIM
    )
    return AdaptiveRelationalState(
        global_features=np.asarray(global_features, dtype=np.float32),
        population_tokens=np.asarray(pop_tokens, dtype=np.float32),
        population_mask=np.asarray(pop_mask, dtype=np.float32),
        archive_tokens=np.asarray(arc_tokens, dtype=np.float32),
        archive_mask=np.asarray(arc_mask, dtype=np.float32),
        topology_tokens=np.asarray(top_tokens, dtype=np.float32),
        topology_mask=np.asarray(top_mask, dtype=np.float32),
        interaction_tokens=np.asarray(int_tokens, dtype=np.float32),
        interaction_mask=np.asarray(int_mask, dtype=np.float32),
        environment_tokens=np.asarray(environment_tokens, dtype=np.float32),
        environment_mask=np.asarray(environment_mask, dtype=np.float32),
        temporal_tokens=np.asarray(temporal_tokens, dtype=np.float32),
        temporal_mask=np.asarray(temporal_mask, dtype=np.float32),
    )
