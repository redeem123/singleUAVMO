"""Shared Pareto-utility helpers.

These functions were duplicated across cmosma, mfo_spea2, spea2, gcnmoea and
moead.  Consolidating them here removes ~80 lines of dead duplication and
ensures a single, auditable implementation.
"""

from __future__ import annotations

import copy

import numpy as np

from uav_benchmark.algorithms.shared.pso_types import Candidate


def _sanitize_objectives(pop_obj: np.ndarray) -> np.ndarray:
    """Replace non-finite objective values with large finite penalties."""
    matrix = np.asarray(pop_obj, dtype=float)
    if matrix.size == 0:
        return matrix.reshape(0, 0)
    finite_mask = np.isfinite(matrix)
    if np.all(finite_mask):
        return matrix
    col_max = np.zeros(matrix.shape[1], dtype=float)
    for col in range(matrix.shape[1]):
        col_values = matrix[finite_mask[:, col], col]
        if col_values.size > 0:
            col_max[col] = float(np.max(col_values))
    penalties = np.sum(~finite_mask, axis=1, keepdims=True).astype(float)
    replacement = col_max.reshape(1, -1) + 1e6 + penalties
    return np.where(finite_mask, matrix, replacement)


def _pairwise_distance(pop_obj: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix with inf on the diagonal."""
    if pop_obj.size == 0:
        return np.zeros((0, 0), dtype=float)
    diff = pop_obj[:, np.newaxis, :] - pop_obj[np.newaxis, :, :]
    distance = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distance, np.inf)
    return distance


def _clone_candidate(candidate: Candidate, vector: np.ndarray | None = None) -> Candidate:
    """Deep-copy a Candidate, optionally replacing its decision vector."""
    cloned_details = copy.deepcopy(candidate.details) if isinstance(candidate.details, dict) else {}
    return Candidate(
        vector=np.asarray(vector if vector is not None else candidate.vector, dtype=float).copy(),
        objective=np.asarray(candidate.objective, dtype=float).copy(),
        details=cloned_details,
    )
