"""R2-indicator-based archive management for constrained multi-objective PSO.

Provides R2-contribution archive pruning, which directly maximises archive
quality measured by the unary R2 indicator (Tchebycheff scalarisation).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Weight vector generation (simplex lattice design)
# ---------------------------------------------------------------------------

def uniform_weight_vectors(n_obj: int, n_divisions: int = 15) -> np.ndarray:
    """Generate uniformly distributed weight vectors on the unit simplex.

    Parameters
    ----------
    n_obj : int
        Number of objectives.
    n_divisions : int
        Number of divisions along each axis.  Total vectors ≈ C(n_divisions +
        n_obj - 1, n_obj - 1).

    Returns
    -------
    np.ndarray
        Shape ``(n_vectors, n_obj)`` with each row summing to 1.
    """
    if n_obj == 1:
        return np.ones((1, 1), dtype=float)

    # Recursive lattice generation
    def _lattice(depth: int, remaining: int, prefix: list[float]) -> list[list[float]]:
        if depth == n_obj - 1:
            return [prefix + [remaining / n_divisions]]
        result: list[list[float]] = []
        for i in range(remaining + 1):
            result.extend(_lattice(depth + 1, remaining - i, prefix + [i / n_divisions]))
        return result

    raw = np.asarray(_lattice(0, n_divisions, []), dtype=float)
    # Avoid zero weights (cause division issues in Tchebycheff)
    raw = np.clip(raw, 1e-6, None)
    raw /= raw.sum(axis=1, keepdims=True)
    return raw


# ---------------------------------------------------------------------------
# R2 indicator (Tchebycheff-based)
# ---------------------------------------------------------------------------

def r2_indicator(pop_obj: np.ndarray, weights: np.ndarray,
                 z_ideal: np.ndarray) -> float:
    """Compute the unary R2 indicator.

    Lower is better (measures distance to ideal).

    Parameters
    ----------
    pop_obj : np.ndarray  (N, M)
    weights : np.ndarray  (W, M)
    z_ideal : np.ndarray  (M,)
    """
    if pop_obj.size == 0:
        return float("inf")
    shifted = pop_obj - z_ideal  # (N, M)
    # Tchebycheff: for each weight vector, min over pop of max_j(w_j * |shifted_j|)
    tcheby = np.max(
        weights[np.newaxis, :, :] * np.abs(shifted[:, np.newaxis, :]),
        axis=2,
    )  # (N, W)
    return float(np.mean(np.min(tcheby, axis=0)))


def r2_contribution(pop_obj: np.ndarray, index: int,
                    weights: np.ndarray,
                    z_ideal: np.ndarray) -> float:
    """Marginal R2 contribution of solution *index*.

    Returns a positive value if the solution improves the archive
    (i.e. removing it worsens R2).
    """
    full = r2_indicator(pop_obj, weights, z_ideal)
    reduced = r2_indicator(np.delete(pop_obj, index, axis=0), weights, z_ideal)
    return float(reduced - full)  # positive = solution is valuable


def _all_r2_contributions(pop_obj: np.ndarray, weights: np.ndarray,
                          z_ideal: np.ndarray) -> np.ndarray:
    """Compute R2 contribution for every member.  O(N²·W)."""
    n = pop_obj.shape[0]
    contribs = np.zeros(n, dtype=float)
    for i in range(n):
        contribs[i] = r2_contribution(pop_obj, i, weights, z_ideal)
    return contribs


# ---------------------------------------------------------------------------
# Archive update with infeasible / duplicate handling
# ---------------------------------------------------------------------------

def r2_archive_update(
    archive_obj: np.ndarray,
    archive_vectors: np.ndarray,
    candidate_obj: np.ndarray,
    candidate_vectors: np.ndarray,
    max_size: int,
    weights: np.ndarray,
    z_ideal: np.ndarray,
    eps_rel: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update archive using R2-contribution pruning.

    Handles infeasible (inf/nan) solutions and near-duplicates.

    Parameters
    ----------
    archive_obj, archive_vectors : existing archive objectives and decision
        vectors.
    candidate_obj, candidate_vectors : new candidates to consider.
    max_size : maximum archive size.
    weights : weight vectors for R2 computation.
    z_ideal : current ideal point (will be updated in-place).
    eps_rel : relative epsilon for near-duplicate suppression.

    Returns
    -------
    new_obj, new_vectors, z_ideal : updated archive and ideal point.
    """
    # --- Merge candidates into pool ---
    if archive_obj.size == 0 and candidate_obj.size == 0:
        return archive_obj, archive_vectors, z_ideal

    if archive_obj.size > 0 and candidate_obj.size > 0:
        all_obj = np.vstack([archive_obj, candidate_obj])
        all_vec = np.vstack([archive_vectors, candidate_vectors])
    elif archive_obj.size > 0:
        all_obj = archive_obj.copy()
        all_vec = archive_vectors.copy()
    else:
        all_obj = candidate_obj.copy()
        all_vec = candidate_vectors.copy()

    # --- Separate feasible vs infeasible ---
    feasible_mask = np.all(np.isfinite(all_obj), axis=1)
    feas_obj = all_obj[feasible_mask]
    feas_vec = all_vec[feasible_mask]

    if feas_obj.shape[0] == 0:
        # All infeasible — keep those with smallest constraint violation
        # (sum of abs of inf/nan objectives replaced by large penalty)
        penalties = np.nansum(np.where(np.isfinite(all_obj), all_obj, 1e18), axis=1)
        order = np.argsort(penalties)[:max_size]
        return all_obj[order], all_vec[order], z_ideal

    # --- Update ideal point (monotonically improving) ---
    new_ideal = np.min(feas_obj, axis=0)
    if z_ideal.size == feas_obj.shape[1]:
        z_ideal = np.minimum(z_ideal, new_ideal)
    else:
        z_ideal = new_ideal.copy()

    # --- Near-duplicate suppression ---
    nadir = np.max(feas_obj, axis=0)
    scale = np.linalg.norm(nadir - z_ideal)
    eps = eps_rel * max(scale, 1e-12)

    keep = np.ones(feas_obj.shape[0], dtype=bool)
    for i in range(1, feas_obj.shape[0]):
        if not keep[i]:
            continue
        dists = np.linalg.norm(feas_obj[:i][keep[:i]] - feas_obj[i], axis=1)
        if dists.size > 0 and np.min(dists) < eps:
            keep[i] = False

    feas_obj = feas_obj[keep]
    feas_vec = feas_vec[keep]

    # --- Prune to max_size using R2 contribution ---
    while feas_obj.shape[0] > max_size:
        contribs = _all_r2_contributions(feas_obj, weights, z_ideal)
        worst = int(np.argmin(contribs))
        feas_obj = np.delete(feas_obj, worst, axis=0)
        feas_vec = np.delete(feas_vec, worst, axis=0)

    return feas_obj, feas_vec, z_ideal
