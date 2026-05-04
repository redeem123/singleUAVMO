from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.core.evaluate_path import _bilinear_interpolate
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort

_OBJECTIVE_COUNT = 4
_CGPO_CONTROL_KEY = "_cgpoControls"
_CGPO_CANDIDATE_EVAL_KEY = "_cgpoCandidateEvaluations"
_CGPO_PROXY_EVAL_KEY = "_cgpoProjectionProxyEvaluations"


# ---------------------------------------------------------------------------
# Ablation controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _AblationControls:
    """Knobs exposed to lean CGPO ablation studies.

    The controls toggle the three retained mechanisms (CIG / PPF / OVO) at a
    per-edge-family granularity plus trace emission.
    """

    # CIG edge families (published mechanism #1)
    cig_edge_coupling_enabled: bool = True
    cig_terrain_edges_enabled: bool = True
    cig_obstacle_edges_enabled: bool = True
    cig_turn_edges_enabled: bool = True
    cig_smoothing_edges_enabled: bool = True
    cig_pairwise_edges_enabled: bool = True
    # PPF (published mechanism #2)
    ppf_pressure_enabled: bool = True
    # OVO (published mechanism #3)
    ovo_variation_enabled: bool = True
    ovo_coordination_enabled: bool = True
    # Diagnostics
    trace_enabled: bool = True


def _nested(extra: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in extra:
            return extra[key]
    nested = extra.get("cgpo")
    if isinstance(nested, dict):
        for key in keys:
            if key in nested:
                return nested[key]
    return default


def _bool_any(extra: dict[str, Any], keys: tuple[str, ...], default: bool) -> bool:
    raw = _nested(extra, *keys, default=default)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _float_any(extra: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    raw = _nested(extra, *keys, default=default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _controls_from_extra(extra: dict[str, Any]) -> _AblationControls:
    cig_edge_coupling_enabled = _bool_any(
        extra,
        ("cgpoUseCigEdgeCoupling", "cgpoCigEdgeCouplingEnabled", "use_cig_edge_coupling"),
        True,
    )

    return _AblationControls(
        cig_edge_coupling_enabled=cig_edge_coupling_enabled,
        cig_terrain_edges_enabled=_bool_any(
            extra,
            ("cgpoUseCigTerrainEdges", "cgpoCigTerrainEdgesEnabled", "use_cig_terrain_edges"),
            True,
        ),
        cig_obstacle_edges_enabled=_bool_any(
            extra,
            ("cgpoUseCigObstacleEdges", "cgpoCigObstacleEdgesEnabled", "use_cig_obstacle_edges"),
            True,
        ),
        cig_turn_edges_enabled=_bool_any(
            extra,
            ("cgpoUseCigTurnEdges", "cgpoCigTurnEdgesEnabled", "use_cig_turn_edges"),
            True,
        ),
        cig_smoothing_edges_enabled=_bool_any(
            extra,
            ("cgpoUseCigSmoothingEdges", "cgpoCigSmoothingEdgesEnabled", "use_cig_smoothing_edges"),
            cig_edge_coupling_enabled,
        ),
        cig_pairwise_edges_enabled=_bool_any(
            extra,
            ("cgpoUseCigPairwiseEdges", "cgpoCigPairwiseEdgesEnabled", "use_cig_pairwise_edges"),
            cig_edge_coupling_enabled,
        ),
        ppf_pressure_enabled=_bool_any(
            extra,
            ("cgpoUsePpfPressure", "cgpoPpfPressureEnabled", "use_ppf_pressure"),
            True,
        ),
        ovo_variation_enabled=_bool_any(
            extra,
            ("cgpoUseOvoVariation", "cgpoOvoVariationEnabled", "use_ovo_variation"),
            True,
        ),
        ovo_coordination_enabled=_bool_any(
            extra,
            ("cgpoUseOvoCoordination", "cgpoOvoCoordinationEnabled", "use_ovo_coordination"),
            True,
        ),
        trace_enabled=_bool_any(extra, ("cgpoTraceEnabled", "trace_enabled"), True),
    )


def _controls(model: dict[str, Any]) -> _AblationControls:
    control = model.get(_CGPO_CONTROL_KEY)
    if isinstance(control, _AblationControls):
        return control
    return _AblationControls()


def _graph_kwargs(model: dict[str, Any]) -> dict[str, Any]:
    controls = _controls(model)
    return {
        "use_edge_coupling": controls.cig_edge_coupling_enabled,
        "use_terrain_edges": controls.cig_terrain_edges_enabled,
        "use_obstacle_edges": controls.cig_obstacle_edges_enabled,
        "use_turn_edges": controls.cig_turn_edges_enabled,
        "use_smoothing_edges": controls.cig_smoothing_edges_enabled,
        "use_pairwise_edges": controls.cig_pairwise_edges_enabled,
    }


# ---------------------------------------------------------------------------
# Candidate / population helpers
# ---------------------------------------------------------------------------


def _ground(model: dict[str, Any], xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    return _bilinear_interpolate(np.asarray(model["H"], dtype=float), points[:, 0] - 1.0, points[:, 1] - 1.0)


def _path_vector(paths: list[np.ndarray]) -> np.ndarray:
    if not paths:
        return np.zeros(0, dtype=float)
    return np.concatenate([np.asarray(path, dtype=float).reshape(-1) for path in paths])


def _clone_paths(paths: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(path, dtype=float).copy() for path in paths]


def _candidate_from_paths(paths: list[np.ndarray], model: dict[str, Any]) -> Candidate:
    model[_CGPO_CANDIDATE_EVAL_KEY] = int(model.get(_CGPO_CANDIDATE_EVAL_KEY, 0)) + 1
    obj, details = evaluate_mission_details(paths, model)
    details["paths"] = _clone_paths(paths)
    return Candidate(vector=_path_vector(paths), objective=np.asarray(obj, dtype=float), details=details)


def _safe_objective_matrix(candidates: list[Candidate]) -> np.ndarray:
    matrix = _candidate_matrix(candidates)
    if matrix.size == 0:
        return np.zeros((0, _OBJECTIVE_COUNT), dtype=float)
    if matrix.ndim != 2:
        matrix = matrix.reshape(-1, _OBJECTIVE_COUNT)
    finite = np.isfinite(matrix)
    if np.all(finite):
        return matrix
    col_max = np.zeros(matrix.shape[1], dtype=float)
    for col in range(matrix.shape[1]):
        values = matrix[finite[:, col], col]
        col_max[col] = float(np.max(values)) if values.size else 1.0
    penalty = np.sum(~finite, axis=1, keepdims=True).astype(float)
    return np.where(finite, matrix, col_max.reshape(1, -1) + 1_000.0 + penalty)


def _is_feasible(candidate: Candidate, model: dict[str, Any]) -> bool:
    return (
        bool(np.all(np.isfinite(np.asarray(candidate.objective, dtype=float))))
        and float(candidate.details.get("feasible", 0.0)) > 0.5
        and _constraint_pressure(candidate, model) <= 1e-12
    )


def _constraint_pressure(candidate: Candidate, model: dict[str, Any]) -> float:
    """Graded CGPO feasibility pressure used before a feasible archive exists.

    The shared fleet violation helper intentionally stays conservative for
    cross-algorithm compatibility.  CGPO needs a smoother signal: otherwise
    most turn-infeasible paths collapse to the same score and PPF/selection
    cannot preserve candidates that are moving toward feasible geometry.
    """
    details = candidate.details if isinstance(candidate.details, dict) else {}
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)

    pressure = 0.0
    separation_min = float(model.get("separationMin", model.get("safeDist", 10.0)))
    drone_size = float(model.get("droneSize", 1.0))
    max_turn = float(model.get("maxTurnDeg", model.get("maxTurnAngleDeg", 75.0)))

    min_sep = float(details.get("minSeparation", np.nan))
    if float(details.get("separationViolation", 0.0)) > 0.5 or (np.isfinite(min_sep) and min_sep < separation_min):
        if np.isfinite(min_sep):
            pressure += max(0.0, (separation_min - min_sep) / max(separation_min, 1e-9))
        else:
            pressure += 1.0

    min_clearance = float(details.get("minClearance", np.nan))
    if float(details.get("collisionViolation", 0.0)) > 0.5 or (
        np.isfinite(min_clearance) and min_clearance <= drone_size
    ):
        if np.isfinite(min_clearance):
            pressure += max(0.0, (drone_size - min_clearance) / max(drone_size, 1e-9))
        else:
            pressure += 1.0

    observed_turn = float(details.get("maxTurnDeg", np.nan))
    if float(details.get("turnViolation", 0.0)) > 0.5 or (np.isfinite(observed_turn) and observed_turn > max_turn):
        if np.isfinite(observed_turn):
            pressure += max(0.0, (observed_turn - max_turn) / max(max_turn, 1e-9))
        else:
            pressure += 1.0

    if (objective.size == 0 or np.any(~np.isfinite(objective))) and pressure <= 0.0:
        # Preserve the graded geometry signal above; add only a fallback when
        # details are missing or inconclusive.
        pressure += 1.0

    if float(details.get("feasible", 1.0)) <= 0.5 and pressure <= 0.0:
        pressure = 1.0
    return float(max(0.0, pressure))


def _project_fleet(paths: list[np.ndarray], model: dict[str, Any]) -> list[np.ndarray]:
    """Pure domain projection (no repair).  Used to clip random/OVO offspring."""
    out: list[np.ndarray] = []
    xmin, xmax = float(model["xmin"]), float(model["xmax"])
    ymin, ymax = float(model["ymin"]), float(model["ymax"])
    zmin, zmax = float(model["zmin"]), float(model["zmax"])
    for path in paths:
        arr = np.asarray(path, dtype=float).copy()
        if arr.ndim != 2 or arr.shape[1] != 3:
            out.append(arr.reshape(-1, 3))
            continue
        arr[:, 0] = np.clip(arr[:, 0], xmin, xmax)
        arr[:, 1] = np.clip(arr[:, 1], ymin, ymax)
        ground = _ground(model, arr[:, :2])
        arr[:, 2] = np.clip(arr[:, 2], ground + zmin, ground + zmax)
        out.append(arr)
    return out


def _horizontal_turns(path: np.ndarray) -> np.ndarray:
    if path.shape[0] < 3:
        return np.zeros(0, dtype=float)
    xy = np.asarray(path[:, :2], dtype=float)
    keep = np.ones(xy.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9
    xy = xy[keep]
    if xy.shape[0] < 3:
        return np.zeros(0, dtype=float)
    v1 = xy[1:-1] - xy[:-2]
    v2 = xy[2:] - xy[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-9) & (n2 > 1e-9)
    turns = np.zeros(xy.shape[0] - 2, dtype=float)
    if np.any(valid):
        dot = np.sum(v1[valid] * v2[valid], axis=1) / np.maximum(n1[valid] * n2[valid], 1e-9)
        turns[valid] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return turns


def _lift_segment_clearance(
    path: np.ndarray, model: dict[str, Any], target_clearance: float, alpha: float
) -> np.ndarray:
    """Raise waypoint altitudes using sampled segment clearance deficits."""
    lifted = np.asarray(path, dtype=float).copy()
    if lifted.ndim != 2 or lifted.shape[1] != 3 or lifted.shape[0] < 2:
        return lifted.reshape(-1, 3)
    zmin, zmax = float(model["zmin"]), float(model["zmax"])
    if zmax <= zmin:
        return lifted
    n = lifted.shape[0]
    rel_lift = np.zeros(n, dtype=float)
    for seg_idx in range(n - 1):
        start = lifted[seg_idx]
        end = lifted[seg_idx + 1]
        dist = float(np.linalg.norm(end[:3] - start[:3]))
        samples = max(3, min(24, int(np.ceil(dist / 4.0)) + 1))
        t = np.linspace(0.0, 1.0, samples)
        xy = (1.0 - t).reshape(-1, 1) * start[:2] + t.reshape(-1, 1) * end[:2]
        z = (1.0 - t) * start[2] + t * end[2]
        clearance = z - _ground(model, xy)
        deficit = float(max(0.0, target_clearance - float(np.min(clearance))))
        if deficit <= 0.0:
            continue
        rel_lift[seg_idx] = max(rel_lift[seg_idx], deficit)
        rel_lift[seg_idx + 1] = max(rel_lift[seg_idx + 1], deficit)

    if not np.any(rel_lift > 0.0):
        return lifted
    ground = _ground(model, lifted[:, :2])
    rel = lifted[:, 2] - ground
    # Preserve fixed endpoint semantics unless the endpoint itself is below
    # the hard collision floor. Internal waypoints carry segment-clearance
    # adaptation for the evolutionary loop.
    endpoint_floor = max(float(model.get("droneSize", 1.0)) + 1e-3, zmin)
    for idx in range(n):
        if idx in (0, n - 1) and rel[idx] >= endpoint_floor:
            continue
        rel[idx] = min(zmax, max(zmin, rel[idx] + float(np.clip(alpha, 0.0, 1.0)) * rel_lift[idx]))
    lifted[:, 2] = ground + np.clip(rel, zmin, zmax)
    return lifted


def _shape_feasible_geometry(
    paths: list[np.ndarray],
    model: dict[str, Any],
    *,
    strength: float,
    graph_tension: np.ndarray | None = None,
    pairwise: bool = True,
) -> list[np.ndarray]:
    """Constraint-aware CGPO variation shaping.

    This keeps offspring inside a smooth, terrain-relative search manifold so
    CIG/PPF can select toward feasibility with ordinary evolutionary pressure.
    """
    shaped = _project_fleet(paths, model)
    if not shaped:
        return shaped

    xmin, xmax = float(model["xmin"]), float(model["xmax"])
    ymin, ymax = float(model["ymin"]), float(model["ymax"])
    zmin, zmax = float(model["zmin"]), float(model["zmax"])
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    altitude_span = max(zmax - zmin, 1e-9)
    turn_limit = float(model.get("maxTurnDeg", 75.0))
    clearance_target = min(zmax, max(float(model.get("droneSize", 1.0)) + 2.5, zmin + 0.18 * altitude_span))
    alpha = float(np.clip(strength, 0.0, 1.0))

    for uav_idx, path in enumerate(shaped):
        if path.ndim != 2 or path.shape[1] != 3:
            continue
        n = path.shape[0]
        if n <= 2:
            continue

        # Pull severe turn spikes toward their local chord.  This is the same
        # geometric pressure encoded by CIG turn edges, applied as a bounded
        # evolutionary variation step rather than an accept/reject repair.
        target_turn = min(0.48 * turn_limit, turn_limit - 1.0)
        for _ in range(3):
            turns = _horizontal_turns(path)
            if turns.size == 0:
                break
            for local_idx, turn in enumerate(turns, start=1):
                if turn <= target_turn:
                    continue
                midpoint = 0.5 * (path[local_idx - 1, :2] + path[local_idx + 1, :2])
                excess = max(0.0, (float(turn) - target_turn) / max(turn_limit, 1e-9))
                local_alpha = float(np.clip(alpha * (0.25 + 0.85 * excess), 0.08, 0.88))
                path[local_idx, :2] = (1.0 - local_alpha) * path[local_idx, :2] + local_alpha * midpoint

        if graph_tension is not None and uav_idx < graph_tension.shape[0]:
            n_tension = min(n, graph_tension.shape[1])
            if n_tension > 2:
                delta = np.asarray(graph_tension[uav_idx, :n_tension], dtype=float)
                max_xy_step = 0.035 * span
                max_z_step = 0.12 * altitude_span
                delta_xy = np.clip(alpha * 0.10 * delta[1 : n_tension - 1, :2], -max_xy_step, max_xy_step)
                delta_z = np.clip(alpha * 0.08 * delta[1 : n_tension - 1, 2], -max_z_step, max_z_step)
                path[1 : n_tension - 1, :2] += delta_xy
                path[1 : n_tension - 1, 2] += delta_z

        # Mild Laplacian smoothing is part of the lean CGPO variation manifold:
        # it preserves path continuity so selection can compete with CMOSMA on
        # J1/J4 instead of merely finding feasible but jagged routes.
        if n > 3:
            smooth_alpha = float(np.clip(0.08 + 0.22 * alpha, 0.08, 0.30))
            original = path.copy()
            midpoint_xy = 0.5 * (original[:-2, :2] + original[2:, :2])
            midpoint_z = 0.5 * (original[:-2, 2] + original[2:, 2])
            path[1:-1, :2] = (1.0 - smooth_alpha) * original[1:-1, :2] + smooth_alpha * midpoint_xy
            path[1:-1, 2] = (1.0 - 0.5 * smooth_alpha) * original[1:-1, 2] + 0.5 * smooth_alpha * midpoint_z

        ground = _ground(model, path[:, :2])
        rel = path[:, 2] - ground
        target_rel = np.maximum(rel, clearance_target)
        # Keep endpoints fixed by mission definition, but keep internal
        # waypoints terrain-relative so segment interpolation has a chance to
        # remain clear without post-hoc repair.
        path[1:-1, 2] = ground[1:-1] + np.clip(
            (1.0 - alpha) * rel[1:-1] + alpha * target_rel[1:-1],
            zmin,
            zmax,
        )
        shaped[uav_idx] = _lift_segment_clearance(path, model, clearance_target, alpha=max(alpha, 0.65))

    if pairwise and len(shaped) > 1:
        separation = float(model.get("separationMin", model.get("safeDist", 10.0)))
        n_points = max((path.shape[0] for path in shaped), default=0)
        for point_idx in range(1, n_points - 1):
            for u_idx in range(len(shaped) - 1):
                if point_idx >= shaped[u_idx].shape[0] - 1:
                    continue
                for v_idx in range(u_idx + 1, len(shaped)):
                    if point_idx >= shaped[v_idx].shape[0] - 1:
                        continue
                    delta = shaped[u_idx][point_idx, :2] - shaped[v_idx][point_idx, :2]
                    dist = float(np.linalg.norm(delta))
                    if dist >= separation:
                        continue
                    direction = delta / max(dist, 1e-9)
                    if not np.all(np.isfinite(direction)) or float(np.linalg.norm(direction)) <= 1e-9:
                        direction = np.array([1.0, 0.0], dtype=float)
                    shift = alpha * 0.5 * (separation - dist + 0.25) * direction
                    shaped[u_idx][point_idx, :2] += shift
                    shaped[v_idx][point_idx, :2] -= shift

    shaped = _project_fleet(shaped, model)
    for uav_idx, path in enumerate(shaped):
        shaped[uav_idx] = _lift_segment_clearance(path, model, clearance_target, alpha=max(alpha, 0.65))
    return _project_fleet(shaped, model)


def _select_candidates(candidates: list[Candidate], model: dict[str, Any], n_keep: int) -> list[Candidate]:
    """NSGA-II constraint-domination selection with per-objective anchor seeding."""
    if len(candidates) <= n_keep:
        return list(candidates)
    objective = _safe_objective_matrix(candidates)
    cv = np.asarray([_constraint_pressure(candidate, model) for candidate in candidates], dtype=float)
    feasible = np.asarray([_is_feasible(candidate, model) for candidate in candidates], dtype=bool)
    rank_score = np.full(len(candidates), np.inf, dtype=float)
    crowd = np.zeros(len(candidates), dtype=float)

    if np.any(feasible):
        feasible_idx = np.flatnonzero(feasible)
        front_no, _max_front = n_d_sort(objective[feasible_idx], None, feasible_idx.size)
        rank_score[feasible_idx] = front_no
        crowd[feasible_idx] = crowding_distance(objective[feasible_idx], front_no)

    infeasible_idx = np.flatnonzero(~feasible)
    if infeasible_idx.size:
        conflict = np.asarray(
            [float(candidates[i].details.get("conflictRate", 0.0)) for i in infeasible_idx], dtype=float
        )
        collision = np.asarray(
            [float(candidates[i].details.get("collisionViolation", 0.0)) for i in infeasible_idx], dtype=float
        )
        turn = np.asarray([float(candidates[i].details.get("turnViolation", 0.0)) for i in infeasible_idx], dtype=float)
        rank_score[infeasible_idx] = 10_000.0 + cv[infeasible_idx] + 0.25 * conflict + 0.15 * collision + 0.10 * turn

    total_obj = np.sum(objective, axis=1)
    order = np.lexsort((total_obj, -crowd, rank_score, (~feasible).astype(int)))
    anchors: list[int] = []
    anchor_pool = np.flatnonzero(feasible) if np.any(feasible) else np.arange(len(candidates), dtype=int)
    if n_keep >= 8 and anchor_pool.size:
        for objective_idx in range(objective.shape[1]):
            values = objective[anchor_pool, objective_idx]
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            finite_pool = anchor_pool[finite]
            finite_values = values[finite]
            anchors.append(int(finite_pool[int(np.argmin(finite_values))]))

    selected: list[int] = []
    seen: set[int] = set()
    for idx in anchors:
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= n_keep:
            break
    for idx_raw in order:
        idx = int(idx_raw)
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= n_keep:
            break
    return [candidates[int(i)] for i in selected[:n_keep]]


def _update_archives(
    population: list[Candidate],
    feasible_archive: list[Candidate],
    relaxed_archive: list[Candidate],
    unconstrained_archive: list[Candidate],
    model: dict[str, Any],
    archive_size: int,
) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    pool = list(population) + list(feasible_archive) + list(relaxed_archive) + list(unconstrained_archive)
    feasible_pool = [candidate for candidate in pool if _is_feasible(candidate, model)]
    feasible = _select_candidates(feasible_pool, model, archive_size) if feasible_pool else []
    return feasible, [], []


# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------


def _random_paths(model: dict[str, Any], fleet_size: int, n_points: int, rng: np.random.Generator) -> list[np.ndarray]:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    xmin, xmax = float(model["xmin"]), float(model["xmax"])
    ymin, ymax = float(model["ymin"]), float(model["ymax"])
    zmin, zmax = float(model["zmin"]), float(model["zmax"])
    paths: list[np.ndarray] = []
    for uav_idx in range(fleet_size):
        start = starts[uav_idx].reshape(-1)[:3]
        goal = goals[uav_idx].reshape(-1)[:3]
        xy = np.zeros((n_points, 2), dtype=float)
        xy[0] = start[:2]
        xy[-1] = goal[:2]
        if n_points > 2:
            xy[1:-1, 0] = rng.uniform(xmin, xmax, size=n_points - 2)
            xy[1:-1, 1] = rng.uniform(ymin, ymax, size=n_points - 2)
        ground = _ground(model, xy)
        rel = rng.uniform(zmin, zmax, size=n_points)
        rel[0] = min(zmax, max(zmin, float(start[2])))
        rel[-1] = min(zmax, max(zmin, float(goal[2])))
        paths.append(np.column_stack([xy, ground + rel]))
    return _project_fleet(paths, model)


def _smooth_evolutionary_paths(
    model: dict[str, Any],
    fleet_size: int,
    n_points: int,
    rng: np.random.Generator,
    variant: int,
) -> list[np.ndarray]:
    """Sample smooth terrain-relative paths for lean CGPO's initial pool.

    This is normal evolutionary initialisation, not a post-evaluation repair:
    candidates are born in a smoother manifold so crossover/mutation can
    preserve partial feasibility the same way CMOSMA preserves good vectors.
    """
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    xmin, xmax = float(model["xmin"]), float(model["xmax"])
    ymin, ymax = float(model["ymin"]), float(model["ymax"])
    zmin, zmax = float(model["zmin"]), float(model["zmax"])
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    altitude_span = max(zmax - zmin, 1e-9)
    clearance = min(zmax, max(float(model.get("droneSize", 1.0)) + 3.0, zmin + 0.22 * altitude_span))
    t = np.linspace(0.0, 1.0, n_points)
    paths: list[np.ndarray] = []

    for uav_idx in range(fleet_size):
        start = starts[uav_idx].reshape(-1)[:3]
        goal = goals[uav_idx].reshape(-1)[:3]
        chord = goal[:2] - start[:2]
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm <= 1e-9:
            normal = np.array([1.0, 0.0], dtype=float)
        else:
            normal = np.array([-chord[1], chord[0]], dtype=float) / chord_norm
        if variant % 6 == 0:
            # Seed true low-curvature anchors; PPF/environmental selection can
            # then preserve short, low-turn paths instead of rediscovering
            # them from noisy waypoint blends.
            lane = 0.0
            amplitude = 0.0
            phase = 0.0
            freq = 1.0
        else:
            lane = (
                (float(uav_idx) - 0.5 * float(max(0, fleet_size - 1))) * 0.05 * float(model.get("separationMin", 10.0))
            )
            amplitude = rng.uniform(-0.025, 0.025) * span
            phase = rng.uniform(0.0, 2.0 * np.pi)
            freq = 1.0 + float((variant + uav_idx) % 2)

        xy = start[:2].reshape(1, 2) + t.reshape(-1, 1) * chord.reshape(1, 2)
        envelope = np.sin(np.pi * t)
        lateral = lane * envelope + amplitude * envelope * np.sin(freq * np.pi * t + phase)
        xy += lateral.reshape(-1, 1) * normal.reshape(1, 2)
        xy[:, 0] = np.clip(xy[:, 0], xmin, xmax)
        xy[:, 1] = np.clip(xy[:, 1], ymin, ymax)
        xy[0] = start[:2]
        xy[-1] = goal[:2]

        ground = _ground(model, xy)
        rel = np.full(n_points, clearance, dtype=float)
        rel += rng.normal(0.0, 0.025 * altitude_span, size=n_points) * envelope
        rel = np.clip(rel, zmin, zmax)
        rel[0] = min(zmax, max(zmin, float(start[2])))
        rel[-1] = min(zmax, max(zmin, float(goal[2])))
        paths.append(np.column_stack([xy, ground + rel]))

    return _shape_feasible_geometry(paths, model, strength=0.55, pairwise=True)


def _initial_population(
    model: dict[str, Any],
    fleet_size: int,
    n_points: int,
    pop_size: int,
    rng: np.random.Generator,
) -> list[Candidate]:
    """Lean CGPO initialisation with smooth and random evolutionary paths."""
    candidates: list[Candidate] = []

    while len(candidates) < pop_size:
        smooth_ratio = float(np.clip(float(model.get("_cgpoSmoothInitRatio", 0.75)), 0.0, 1.0))
        use_smooth = len(candidates) < int(np.ceil(smooth_ratio * pop_size))
        paths = (
            _smooth_evolutionary_paths(model, fleet_size, n_points, rng, variant=len(candidates))
            if use_smooth
            else _random_paths(model, fleet_size, n_points, rng)
        )
        candidates.append(_candidate_from_paths(paths, model))
    return candidates
