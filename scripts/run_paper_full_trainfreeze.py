from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_benchmark.analysis.benchmark_report import ReportConfig, generate_benchmark_report
from uav_benchmark.benchmark import run_benchmark
from uav_benchmark.config import BenchmarkParams


def _base_params(results_dir: Path) -> BenchmarkParams:
    return BenchmarkParams(
        generations=300,
        population=80,
        runs=10,
        compute_metrics=True,
        safe_dist=20.0,
        drone_size=1.0,
        results_dir=results_dir,
        seed=11,
        mode="multi",
        fleet_sizes=(3, 5, 8),
        separation_min=10.0,
        max_turn_deg=75.0,
        scenario_set="paper_medium",
        gpu_mode="auto",
    )


def main() -> None:
    results_dir = (PROJECT_ROOT / "results" / "paper_full_trainfreeze").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Best first-problem setting found during freeze-eval tuning sweep.
    tuned_rl_extra = {
        "metricInterval": 20,
        "nRep": 100,
        "nGrid": 9,
        "kappa": 2.4,
        "velocityClampRatio": 0.23,
        "mutationProb": 0.11,
        "c1": 1.5,
        "c2": 1.5,
        "w": 0.82,
        "rlControllerBackend": "auto",
        "rlUseGpuPolicy": True,
        "rlRewardNStep": 5,
        "rlRewardGamma": 0.93,
        "rlPhaseGating": True,
        "rlRewardHvWeight": 0.76,
        "rlRewardFeasibleWeight": 0.10,
        "rlRewardDiversityWeight": 0.12,
        "rlRewardConflictWeight": 0.02,
        "rlRewardHvScale": 0.009,
        "rlEliteRefine": True,
        "rlEliteRefineTopK": 4,
        "rlEliteRefineIters": 3,
        "rlEliteRefineSigmaStart": 0.08,
        "rlEliteRefineSigmaEnd": 0.015,
        "rlGpuHiddenDim": 384,
        "rlGpuBatchSize": 2048,
        "rlGpuTrainSteps": 18,
        "maxWorkers": 12,
        "resumeExistingRuns": True,
    }

    # Phase 1: baseline algorithms only.
    baseline_params = _base_params(results_dir)
    baseline_params.extra.update(
        {
            "algorithms": ["NMOPSO", "MOPSO", "NSGA-II", "NSGA-III"],
            "maxWorkers": 12,
        }
    )
    run_benchmark(PROJECT_ROOT, baseline_params)

    # Phase 2: RL warmstart training (checkpoint accumulation).
    warmstart_params = _base_params(results_dir)
    warmstart_params.extra.update(tuned_rl_extra)
    warmstart_params.extra.update(
        {
            "algorithms": ["RL-NMOPSO"],
            "rlPolicyMode": "warmstart",
        }
    )
    run_benchmark(PROJECT_ROOT, warmstart_params)

    # Phase 3: RL freeze evaluation (fair comparison run).
    freeze_params = _base_params(results_dir)
    freeze_params.extra.update(tuned_rl_extra)
    freeze_params.extra.update(
        {
            "algorithms": ["RL-NMOPSO"],
            "rlPolicyMode": "freeze",
        }
    )
    run_benchmark(PROJECT_ROOT, freeze_params)

    report = generate_benchmark_report(
        ReportConfig(
            project_root=PROJECT_ROOT,
            results_dir=results_dir,
            output_dir=results_dir / "metrics",
            baseline_algorithm="NMOPSO",
            max_runs=10,
            seed=0,
        )
    )
    print(f"summary_json={report['summary_json']}")
    print(f"summary_csv={report['summary_csv']}")
    print(f"pairwise_csv={report['pairwise_csv']}")
    print(f"win_tie_loss_csv={report.get('win_tie_loss_csv')}")


if __name__ == "__main__":
    main()
