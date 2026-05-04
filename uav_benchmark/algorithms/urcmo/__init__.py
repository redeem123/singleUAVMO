from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.platemo_bridge import run_platemo_algorithm


def run_urcmo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_platemo_algorithm(model, params, "URCMO")


__all__ = ["run_urcmo"]
