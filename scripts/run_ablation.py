from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.benchmark import run_nmopso_ablation
from uav_benchmark.config import BenchmarkParams


def main() -> None:
    project_root = PROJECT_ROOT
    params = BenchmarkParams(
        generations=200,
        population=80,
        runs=6,
        compute_metrics=False,
        use_parallel=False,
        parallel_mode="none",
        safe_dist=20.0,
        drone_size=1.0,
        results_dir=project_root / "results" / "NMOPSO_ABLATION",
    )
    params.extra["ablationStudy"] = True
    run_nmopso_ablation(project_root, params)


if __name__ == "__main__":
    main()
