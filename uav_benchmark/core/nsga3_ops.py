from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

from uav_benchmark.core.chromosome import Chromosome
from uav_benchmark.core.nsga2_ops import n_d_sort


def uniform_point(n_points: int, objective_count: int, method: str = "NBI") -> tuple[np.ndarray, int]:
    method_key = method.upper()
    if method_key == "NBI":
        return _uniform_point_nbi(n_points, objective_count)
    if method_key == "ILD":
        return _uniform_point_ild(n_points, objective_count)
    if method_key == "GRID":
        return _uniform_point_grid(n_points, objective_count)
    if method_key == "LATIN":
        return _uniform_point_latin(n_points, objective_count)
    return _uniform_point_nbi(n_points, objective_count)


def _uniform_point_nbi(n_points: int, objective_count: int) -> tuple[np.ndarray, int]:
    h1 = 1
    while math.comb(h1 + objective_count, objective_count - 1) <= n_points:
        h1 += 1
    combinations = np.array(list(itertools.combinations(range(1, h1 + objective_count), objective_count - 1)), dtype=int)
    if combinations.size == 0:
        vectors = np.ones((1, objective_count), dtype=float) / objective_count
        return vectors, vectors.shape[0]
    adjustment = np.arange(objective_count - 1, dtype=int)
    vectors = combinations - adjustment - 1
    vectors = np.hstack([vectors, np.full((vectors.shape[0], 1), h1, dtype=int)]) - np.hstack([np.zeros((vectors.shape[0], 1), dtype=int), vectors])
    weights = vectors / h1
    if h1 < objective_count:
        h2 = 0
        while math.comb(h1 + objective_count - 1, objective_count - 1) + math.comb(h2 + objective_count, objective_count - 1) <= n_points:
            h2 += 1
        if h2 > 0:
            combinations_two = np.array(list(itertools.combinations(range(1, h2 + objective_count), objective_count - 1)), dtype=int)
            vectors_two = combinations_two - adjustment - 1
            vectors_two = np.hstack([vectors_two, np.full((vectors_two.shape[0], 1), h2, dtype=int)]) - np.hstack(
                [np.zeros((vectors_two.shape[0], 1), dtype=int), vectors_two]
            )
            weights_two = vectors_two / h2
            weights = np.vstack([weights, weights_two / 2.0 + 1.0 / (2.0 * objective_count)])
    weights = np.maximum(weights, 1e-6)
    return weights.astype(float), weights.shape[0]


def _uniform_point_ild(n_points: int, objective_count: int) -> tuple[np.ndarray, int]:
    identity = objective_count * np.eye(objective_count, dtype=int)
    weights = np.zeros((1, objective_count), dtype=int)
    edge = weights.copy()
    while weights.shape[0] < n_points:
        edge = np.repeat(edge, objective_count, axis=0) + np.tile(identity, (edge.shape[0], 1))
        edge = np.unique(edge, axis=0)
        edge = edge[np.min(edge, axis=1) == 0]
        weights = np.vstack([weights + 1, edge])
    lattice = weights / np.sum(weights, axis=1, keepdims=True)
    lattice = np.maximum(lattice, 1e-6)
    return lattice.astype(float), lattice.shape[0]


def _uniform_point_grid(n_points: int, objective_count: int) -> tuple[np.ndarray, int]:
    gap = np.linspace(0.0, 1.0, int(math.ceil(n_points ** (1.0 / objective_count))))
    mesh = np.meshgrid(*([gap] * objective_count))
    stacked = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    return stacked, stacked.shape[0]


def _uniform_point_latin(n_points: int, objective_count: int) -> tuple[np.ndarray, int]:
    ranks = np.argsort(np.random.rand(n_points, objective_count), axis=0)
    points = (np.random.rand(n_points, objective_count) + ranks) / n_points
    return points, points.shape[0]


def environmental_selection_nsga3(
    population: Sequence[Chromosome],
    n_keep: int,
    reference_points: np.ndarray,
    zmin: np.ndarray,
    use_constraints: bool = False,
) -> list[Chromosome]:
    if not population:
        return []
    objective_matrix = np.array([individual.objs for individual in population], dtype=float)
    constraint_matrix = None
    if use_constraints:
        constraint_matrix = np.array([[individual.cons] for individual in population], dtype=float)
    front_no, max_front = n_d_sort(objective_matrix.copy(), constraint_matrix, n_keep)
    next_mask = front_no < max_front
    last_indices = np.where(front_no == max_front)[0]
    if np.sum(next_mask) < n_keep and last_indices.size > 0:
        choose = _last_selection(objective_matrix[next_mask], objective_matrix[last_indices], int(n_keep - np.sum(next_mask)), reference_points, zmin)
        next_mask[last_indices[choose]] = True
    selected = [population[index] for index in np.where(next_mask)[0]]
    return selected


def _last_selection(pop_obj_first: np.ndarray, pop_obj_last: np.ndarray, k_keep: int, reference_points: np.ndarray, zmin: np.ndarray) -> np.ndarray:
    if k_keep <= 0:
        return np.zeros(pop_obj_last.shape[0], dtype=bool)
    pop_obj = np.vstack([pop_obj_first, pop_obj_last]) - zmin.reshape(1, -1)
    n_total, objective_count = pop_obj.shape
    n_first = pop_obj_first.shape[0]
    n_last = pop_obj_last.shape[0]
    n_ref = reference_points.shape[0]

    weight = np.full((objective_count, objective_count), 1e-6, dtype=float) + np.eye(objective_count, dtype=float)
    extreme = np.zeros(objective_count, dtype=int)
    for objective_index in range(objective_count):
        extreme[objective_index] = int(np.argmin(np.max(pop_obj / weight[objective_index], axis=1)))
    try:
        hyperplane = np.linalg.solve(pop_obj[extreme], np.ones(objective_count))
        axis_intercept = 1.0 / hyperplane
    except np.linalg.LinAlgError:
        axis_intercept = np.max(pop_obj, axis=0)
    axis_intercept[~np.isfinite(axis_intercept)] = np.max(pop_obj, axis=0)[~np.isfinite(axis_intercept)]
    axis_intercept[axis_intercept == 0] = 1.0
    normalized = pop_obj / axis_intercept.reshape(1, -1)

    cosine = 1.0 - _pairwise_cosine_distance(normalized, reference_points)
    distance = np.linalg.norm(normalized, axis=1).reshape(-1, 1) * np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    nearest_ref = np.argmin(distance, axis=1)
    nearest_dist = distance[np.arange(distance.shape[0]), nearest_ref]
    rho = np.bincount(nearest_ref[:n_first], minlength=n_ref)

    choose = np.zeros(n_last, dtype=bool)
    active_ref = np.ones(n_ref, dtype=bool)
    while np.sum(choose) < k_keep:
        available = np.where(active_ref)[0]
        if available.size == 0:
            remaining = np.where(~choose)[0]
            random_pick = np.random.choice(remaining)
            choose[random_pick] = True
            continue
        min_rho = np.min(rho[available])
        candidates = available[rho[available] == min_rho]
        selected_ref = int(np.random.choice(candidates))
        associated = np.where((~choose) & (nearest_ref[n_first:] == selected_ref))[0]
        if associated.size > 0:
            if rho[selected_ref] == 0:
                local_pick = associated[np.argmin(nearest_dist[n_first + associated])]
            else:
                local_pick = int(np.random.choice(associated))
            choose[local_pick] = True
            rho[selected_ref] += 1
        else:
            active_ref[selected_ref] = False
    return choose


def _pairwise_cosine_distance(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    lhs_norm = np.linalg.norm(lhs, axis=1, keepdims=True)
    rhs_norm = np.linalg.norm(rhs, axis=1, keepdims=True)
    lhs_norm[lhs_norm == 0] = 1.0
    rhs_norm[rhs_norm == 0] = 1.0
    similarity = lhs @ rhs.T / (lhs_norm * rhs_norm.T)
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1.0 - similarity
