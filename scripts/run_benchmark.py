from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.analysis.generate_research_plots import generate_research_plots
from uav_benchmark.benchmark import run_benchmark
from uav_benchmark.config import BenchmarkParams


def main() -> None:
    project_root = PROJECT_ROOT
    params = BenchmarkParams(
        generations=500,
        population=100,
        runs=14,
        compute_metrics=False,
        use_parallel=False,
        parallel_mode="none",
        safe_dist=20.0,
        drone_size=1.0,
        results_dir=project_root / "results",
    )
    params.extra["ctm"] = {
        "scheduler": "performance",
        "lambda0": 0.0,
        "stepSize": 0.08,
        "stepSizeDown": 0.06,
        "progressDrift": 0.01,
        "feasHigh": 0.80,
        "feasLow": 0.35,
        "divThreshold": 0.08,
        "otInterval": 5,
        "transferFraction": 0.50,
        "minArchiveForOT": 30,
        "archiveMaxSize": 500,
        "archiveInjectCount": 16,
        "beta0": 1.2,
        "betaMin": 0.1,
        "betaMax": 8.0,
        "betaEta": 0.10,
        "useCounterfactual": True,
    }
    run_benchmark(project_root, params)
    generate_research_plots(project_root, params.results_dir)


if __name__ == "__main__":
    main()
