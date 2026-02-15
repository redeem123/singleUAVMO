"""Pareto dominance utilities shared across algorithms."""

from __future__ import annotations

import numpy as np


def dominates(left: np.ndarray, right: np.ndarray) -> bool:
    """Return True if *left* Pareto-dominates *right*.

    A solution dominates another if it is no worse on all objectives,
    strictly better on at least one, and all values are finite.
    """
    return bool(np.all(left <= right) and np.any(left < right) and np.all(np.isfinite(left)))
