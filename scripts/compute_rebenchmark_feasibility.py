from pathlib import Path

import numpy as np

from scripts.summary_helpers import case_mat_files, feasibility_values

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
    print(f"{'Case':15s} | {'UAVs':4s} | {'Feas Ratio':10s} | {'Total Feas':10s}")
    print("-" * 55)

    for case in CASES:
        case_dir = RESULTS_ROOT / case
        for uav_count in [1, 2, 3]:
            ratios, total_feas = feasibility_values(case_mat_files(case_dir, "run_stats.mat", uav_count))

            if ratios:
                print(f"{case:15s} | {uav_count:4d} | {np.mean(ratios):.4f}     | {np.mean(total_feas):.1f}")
            else:
                print(f"{case:15s} | {uav_count:4d} | No data")


if __name__ == "__main__":
    main()
