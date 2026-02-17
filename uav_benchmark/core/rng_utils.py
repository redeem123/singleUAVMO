"""Shared PRNG utility for reproducible seeding.

All stochastic functions in the benchmark should accept an optional
``rng: np.random.Generator | None`` parameter.  This module provides
``ensure_rng`` to normalize that argument into a proper Generator.
"""
from __future__ import annotations
import numpy as np


def ensure_rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    """Return *rng* if it is already a Generator, else create one.

    When ``rng is None`` a new Generator is seeded from the legacy global
    ``np.random`` state, so run-level ``np.random.seed(...)`` remains
    reproducible for call-sites that do not yet pass an explicit Generator.
    """
    if rng is not None:
        return rng
    seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return np.random.default_rng(seed)
