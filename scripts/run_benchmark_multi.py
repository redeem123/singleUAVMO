from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.benchmark import run_benchmark
from uav_benchmark.config import BenchmarkParams


def main() -> None:
    params = BenchmarkParams(
        generations=300,
        population=80,
        runs=10,
        compute_metrics=True,
        safe_dist=20.0,
        drone_size=1.0,
        results_dir=(PROJECT_ROOT / "results" / "paper_multi"),
        seed=11,
        mode="multi",
        fleet_sizes=(3, 5, 8),
        separation_min=10.0,
        max_turn_deg=75.0,
        scenario_set="paper_medium",
        gpu_mode="auto",
    )
    run_benchmark(PROJECT_ROOT, params)


if __name__ == "__main__":
    main()

