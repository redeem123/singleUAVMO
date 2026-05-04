from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.platemo_bridge import run_reference_legacy_algorithm


def run_emmop(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_reference_legacy_algorithm(model, params, "EMMOP")


__all__ = ["run_emmop"]
