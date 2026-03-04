from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from uav_benchmark.config import BenchmarkParams


@runtime_checkable
class UAVAlgorithm(Protocol):
    """Protocol for UAV path-planning algorithms."""

    def __call__(self, model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
        """Execute the algorithm.

        Parameters
        ----------
        model : dict[str, Any]
            The problem model (terrain, obstacles, constraints).
        params : BenchmarkParams
            Benchmark configuration (generations, population, fleet size, etc.).

        Returns
        -------
        np.ndarray
            A summary of the run's performance (e.g., hypervolume scores).
        """
        ...
