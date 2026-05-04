from pathlib import Path

import numpy as np

from scripts.summary_helpers import case_mat_files, objective_metric_values

RESULTS_ROOT = Path("results/rebenchmark_ablation")
CASES = [
    "full",
    "pure_gwo",
    "loo_diversity",
    "loo_step_limiter",
    "loo_pressure",
    "loo_step_div_driver",
]


def main() -> None:
    print(f"{'Case':15s} | {'UAVs':4s} | {'HV Mean':10s} | {'IG+ Mean':10s}")
    print("-" * 50)

    for case in CASES:
        case_dir = RESULTS_ROOT / case
        for uav_count in [1, 2, 3]:
            hvs, igs = objective_metric_values(case_mat_files(case_dir, "final_popobj.mat", uav_count))

            if hvs:
                print(f"{case:15s} | {uav_count:4d} | {np.mean(hvs):.4f}     | {np.mean(igs):.4f}")
            else:
                print(f"{case:15s} | {uav_count:4d} | No data")


if __name__ == "__main__":
    main()
