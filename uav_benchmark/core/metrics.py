from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _head(matrix: np.ndarray) -> np.ndarray:
    return matrix[0] if matrix.shape[0] > 0 else np.array([])


def _tail(matrix: np.ndarray) -> np.ndarray:
    return matrix[1:] if matrix.shape[0] > 1 else np.zeros((0, matrix.shape[1])) if matrix.ndim == 2 else np.array([])


def _insert(point: np.ndarray, objective_index: int, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return point.reshape(1, -1)
    merged = []
    pointer = 0
    while pointer < points.shape[0] and points[pointer, objective_index] < point[objective_index]:
        merged.append(points[pointer])
        pointer += 1
    merged.append(point)
    while pointer < points.shape[0]:
        compare = points[pointer]
        dominated = False
        better_found = False
        for compare_index in range(objective_index, point.shape[0]):
            if point[compare_index] < compare[compare_index]:
                better_found = True
            elif point[compare_index] > compare[compare_index]:
                dominated = True
        if dominated or better_found:
            merged.append(compare)
        pointer += 1
    return np.vstack(merged)


def _add_slice(slice_entry: tuple[float, np.ndarray], slices: list[tuple[float, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    for index, existing in enumerate(slices):
        if np.array_equal(existing[1], slice_entry[1]):
            slices[index] = (existing[0] + slice_entry[0], existing[1])
            return slices
    slices.append(slice_entry)
    return slices


def _slice(points: np.ndarray, objective_index: int, ref_point: np.ndarray) -> list[tuple[float, np.ndarray]]:
    current = _head(points)
    remainder = _tail(points)
    queue = np.zeros((0, points.shape[1]))
    slices: list[tuple[float, np.ndarray]] = []
    while remainder.size > 0:
        queue = _insert(current, objective_index + 1, queue)
        next_point = _head(remainder)
        slices = _add_slice((abs(current[objective_index] - next_point[objective_index]), queue.copy()), slices)
        current = next_point
        remainder = _tail(remainder)
    queue = _insert(current, objective_index + 1, queue)
    slices = _add_slice((abs(current[objective_index] - ref_point[objective_index]), queue), slices)
    return slices


def hypervolume(pop_obj: np.ndarray, max_pop_obj: np.ndarray, sample_num: int | None = None) -> float:
    pop_obj = np.asarray(pop_obj, dtype=float)
    if pop_obj.size == 0:
        return 0.0
    max_pop_obj = np.asarray(max_pop_obj, dtype=float).reshape(-1)
    n_points, n_objectives = pop_obj.shape

    min_values = np.minimum(np.min(pop_obj, axis=0), np.zeros(n_objectives))
    max_values = np.maximum(max_pop_obj, np.zeros(n_objectives))
    denominator = (max_values - min_values) * 1.1
    denominator[denominator == 0] = 1.0
    normalized = (pop_obj - min_values) / denominator
    normalized = normalized[~np.any(normalized > 1.0, axis=1)]
    reference = np.ones(n_objectives, dtype=float)
    if normalized.size == 0:
        return 0.0
    if sample_num is None or sample_num <= 0:
        sample_num = 10_000

    if n_objectives < 4:
        sorted_points = normalized[np.lexsort(np.flipud(normalized.T))]
        slices: list[tuple[float, np.ndarray]] = [(1.0, sorted_points)]
        for objective_index in range(n_objectives - 1):
            next_slices: list[tuple[float, np.ndarray]] = []
            for weight, segment in slices:
                for segment_weight, segment_points in _slice(segment, objective_index, reference):
                    next_slices = _add_slice((segment_weight * weight, segment_points), next_slices)
            slices = next_slices
        score = 0.0
        for weight, segment in slices:
            head = _head(segment)
            score += weight * abs(head[n_objectives - 1] - reference[n_objectives - 1])
        return float(score)

    minimum = np.min(normalized, axis=0)
    maximum = reference
    samples = np.random.uniform(minimum, maximum, size=(sample_num, n_objectives))
    for candidate in normalized:
        dominated = np.all(candidate <= samples, axis=1)
        samples = samples[~dominated]
        if samples.size == 0:
            break
    return float(np.prod(maximum - minimum) * (1.0 - samples.shape[0] / sample_num))


def pure_diversity(pop_obj: np.ndarray, max_points: int = 200) -> float:
    """Compute the pure diversity metric for a set of objective vectors.

    Only finite rows are considered.  Large populations are sub-sampled
    to *max_points* to keep runtime manageable (O(n^3) algorithm).
    """
    pop_obj = np.asarray(pop_obj, dtype=float)
    # Filter out rows containing inf/nan
    finite_mask = np.all(np.isfinite(pop_obj), axis=1)
    pop_obj = pop_obj[finite_mask]
    if pop_obj.shape[0] < 2:
        return 0.0
    # Sub-sample if too large
    if pop_obj.shape[0] > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(pop_obj.shape[0], max_points, replace=False)
        pop_obj = pop_obj[indices]
    n_points = pop_obj.shape[0]
    connect = np.eye(n_points, dtype=bool)
    distance = np.linalg.norm(pop_obj[:, None, :] - pop_obj[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    score = 0.0
    for _ in range(n_points - 1):
        max_inner_iters = n_points * 2  # safety cap
        found = False
        for _inner in range(max_inner_iters):
            nearest = np.argmin(distance, axis=1)
            nearest_distance = distance[np.arange(n_points), nearest]
            source = int(np.argmax(nearest_distance))
            target = int(nearest[source])
            if not np.isfinite(distance[target, source]):
                distance[target, source] = np.inf
            if not np.isfinite(distance[source, target]):
                distance[source, target] = np.inf
            reachable = connect[source, :].copy()
            for _bfs in range(n_points):
                if reachable[target]:
                    break
                expanded = np.any(connect[reachable, :], axis=0)
                if np.array_equal(expanded, reachable):
                    break
                reachable = expanded
            if not reachable[target]:
                found = True
                break
        connect[source, target] = True
        connect[target, source] = True
        distance[source, :] = -np.inf
        score += float(nearest_distance[source])
    return float(score)


def cal_metric(
    metric_index: int,
    pop_obj: np.ndarray,
    problem_index: int,
    objective_count: int,
    hv_samples: int | None = None,
    ref_point: np.ndarray | None = None,
) -> float:
    del problem_index
    pop_obj = np.asarray(pop_obj, dtype=float)
    if pop_obj.size == 0:
        return 0.0
    if metric_index == 1:
        finite = pop_obj[np.all(np.isfinite(pop_obj), axis=1)]
        if finite.size == 0:
            return 0.0
        if ref_point is None:
            max_values = np.max(finite, axis=0)
            ref_point = max_values * 1.1
            ref_point[ref_point <= 0] = 1.0
        return hypervolume(finite, ref_point, hv_samples)
    if pop_obj.shape[1] != objective_count and pop_obj.shape[0] == objective_count:
        pop_obj = pop_obj.T
    # Filter inf rows before the expensive O(n^3) pure_diversity call
    finite = pop_obj[np.all(np.isfinite(pop_obj), axis=1)]
    if finite.shape[0] < 2:
        return 0.0
    return pure_diversity(finite)


@dataclass(slots=True)
class MetricPair:
    hypervolume: float
    pure_diversity: float
