"""Orchestrated Variation Operator (OVO).

Two-parent blend whose blend weights and perturbation magnitude are driven by
the per-waypoint scalar tension supplied by the constraint-interaction graph.
Optionally pushes pairs of UAVs apart at temporally-aligned waypoints when
their inter-vehicle distance falls below the separation threshold.

This module deliberately operates on the *native* waypoint grid rather than a
resampled high-resolution trajectory.  The earlier implementation resampled
each path to ``max(20, 4 * n_points)`` synchronisation points and ran an
``O(n_samples * fleet^2)`` pairwise scan on every offspring, which was both
expensive and largely redundant once the optimiser had any feasible
candidates -- the same conflict signal that the rescan recovered is already
encoded in the CIG pairwise edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.algorithms.cgpo.cig import ConstraintInteractionGraph
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.evaluate_path import _bilinear_interpolate


@dataclass(frozen=True, slots=True)
class OVOTrace:
    perturbation_scale: float
    perturbed_waypoints: int
    coordinated_clusters: int
    parent_blend_entropy: float


def _clone_paths(candidate: Candidate) -> list[np.ndarray]:
    paths = candidate.details.get("paths", []) if isinstance(candidate.details, dict) else []
    return [np.asarray(path, dtype=float).copy() for path in paths]


def _ground(model: dict[str, Any], xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    return _bilinear_interpolate(np.asarray(model["H"], dtype=float), points[:, 0] - 1.0, points[:, 1] - 1.0)


def orchestrated_variation(
    parent_a: Candidate,
    parent_b: Candidate,
    graph_a: ConstraintInteractionGraph,
    graph_b: ConstraintInteractionGraph,
    model: dict[str, Any],
    rng: np.random.Generator,
    exploration_scale: float,
    use_coordination: bool = True,
) -> tuple[list[np.ndarray], OVOTrace]:
    paths_a = _clone_paths(parent_a)
    paths_b = _clone_paths(parent_b)
    if not paths_a:
        return [], OVOTrace(
            perturbation_scale=0.0, perturbed_waypoints=0, coordinated_clusters=0, parent_blend_entropy=0.0
        )
    if len(paths_b) != len(paths_a):
        paths_b = [path.copy() for path in paths_a]

    x_span = max(float(model["xmax"]) - float(model["xmin"]), 1e-9)
    y_span = max(float(model["ymax"]) - float(model["ymin"]), 1e-9)
    z_span = max(float(model["zmax"]) - float(model["zmin"]), 1e-9)
    spatial_span = max(x_span, y_span)
    sigma_xy = float(exploration_scale) * spatial_span
    sigma_z = float(exploration_scale) * z_span

    scalar_parts: list[np.ndarray] = []
    if graph_a.scalar_tension.size:
        scalar_parts.append(np.asarray(graph_a.scalar_tension, dtype=float).reshape(-1))
    if graph_b.scalar_tension.size:
        scalar_parts.append(np.asarray(graph_b.scalar_tension, dtype=float).reshape(-1))
    tension_floor = float(np.quantile(np.concatenate(scalar_parts), 0.40)) if scalar_parts else 0.0

    child: list[np.ndarray] = []
    perturbed = 0
    entropy_total = 0.0
    entropy_count = 0
    for u_idx, path_a in enumerate(paths_a):
        path_b = paths_b[u_idx] if u_idx < len(paths_b) and paths_b[u_idx].shape == path_a.shape else path_a
        out = path_a.copy()
        n = path_a.shape[0]
        for point_idx in range(n):
            if point_idx == 0 or point_idx == n - 1:
                out[point_idx] = path_a[point_idx]
                continue
            tau_a = (
                graph_a.scalar_tension[u_idx, point_idx]
                if u_idx < graph_a.scalar_tension.shape[0] and point_idx < graph_a.scalar_tension.shape[1]
                else 0.0
            )
            tau_b = (
                graph_b.scalar_tension[u_idx, point_idx]
                if u_idx < graph_b.scalar_tension.shape[0] and point_idx < graph_b.scalar_tension.shape[1]
                else 0.0
            )
            inv = np.asarray([1.0 / (1.0 + tau_a), 1.0 / (1.0 + tau_b)], dtype=float)
            weights = inv / max(float(np.sum(inv)), 1e-9)
            entropy_total += -float(np.sum(weights * np.log(np.maximum(weights, 1e-12))))
            entropy_count += 1
            tension_sum = max(0.0, float(tau_a + tau_b))
            tension_gain = float(np.clip(1.0 / (1.0 + 0.35 * tension_sum), 0.25, 1.0))
            blended = weights[0] * path_a[point_idx] + weights[1] * path_b[point_idx]
            if max(tau_a, tau_b) <= tension_floor and rng.random() < 0.45:
                out[point_idx] = blended
                continue
            noise = np.array(
                [
                    rng.normal(0.0, sigma_xy * tension_gain),
                    rng.normal(0.0, sigma_xy * tension_gain),
                    rng.normal(0.0, sigma_z * tension_gain),
                ],
                dtype=float,
            )
            out[point_idx] = blended + noise
            perturbed += 1
        child.append(out)

    coordinated = 0
    if use_coordination and len(child) > 1:
        separation = float(model.get("separationMin", model.get("safeDist", 10.0)))
        # Coordinate at the native waypoint grid only.  This costs
        # ``O(n_points * fleet^2)`` rather than the previous
        # ``O(n_samples * fleet^2)`` resample-then-rescan.  The CIG already
        # encodes the high-resolution conflict signal as pairwise edges, so
        # the coarser scan here is sufficient for the operator to nudge
        # offspring apart at the violating waypoints.
        n_points = max((path.shape[0] for path in child), default=0)
        for point_idx in range(1, n_points - 1):
            for u_idx in range(len(child) - 1):
                if point_idx >= child[u_idx].shape[0] - 1:
                    continue
                for v_idx in range(u_idx + 1, len(child)):
                    if point_idx >= child[v_idx].shape[0] - 1:
                        continue
                    delta_xyz = child[u_idx][point_idx, :3] - child[v_idx][point_idx, :3]
                    dist = float(np.linalg.norm(delta_xyz))
                    if dist >= separation:
                        continue
                    delta_xy = delta_xyz[:2]
                    horiz = float(np.linalg.norm(delta_xy))
                    direction = np.array([1.0, 0.0], dtype=float) if horiz <= 1e-09 else delta_xy / horiz
                    if not np.all(np.isfinite(direction)) or float(np.linalg.norm(direction)) <= 1e-9:
                        direction = np.array([1.0, 0.0], dtype=float)
                    shift = 0.5 * (separation - dist + 1.0) * direction
                    child[u_idx][point_idx, :2] += shift
                    child[v_idx][point_idx, :2] -= shift
                    coordinated += 1

    for path in child:
        path[:, 0] = np.clip(path[:, 0], float(model["xmin"]), float(model["xmax"]))
        path[:, 1] = np.clip(path[:, 1], float(model["ymin"]), float(model["ymax"]))
        ground = _ground(model, path[:, :2])
        path[:, 2] = np.clip(path[:, 2], ground + float(model["zmin"]), ground + float(model["zmax"]))
    return child, OVOTrace(
        perturbation_scale=float(exploration_scale),
        perturbed_waypoints=int(perturbed),
        coordinated_clusters=int(coordinated),
        parent_blend_entropy=float(entropy_total / max(1, entropy_count)),
    )
