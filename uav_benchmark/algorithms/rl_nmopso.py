from __future__ import annotations

from typing import Any

from uav_benchmark.algorithms.multi_uav import run_multi_rl_nmopso
from uav_benchmark.algorithms.nmopso import run_nmopso
from uav_benchmark.config import BenchmarkParams


def run_rl_nmopso(model: dict[str, Any], params: BenchmarkParams):
    if str(params.mode).lower() == "multi":
        return run_multi_rl_nmopso(model, params)
    # Backward-compatible fallback in single mode.
    return run_nmopso(model, params)

