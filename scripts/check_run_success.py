from pathlib import Path

import numpy as np

from scripts.summary_helpers import case_mat_files, run_success_flags

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
    print(f"{'Case':15s} | {'UAVs':4s} | {'Run Success Rate':18s}")
    print("-" * 45)

    for case in CASES:
        case_dir = RESULTS_ROOT / case
        for uav_count in [1, 2, 3]:
            success_flags = run_success_flags(case_mat_files(case_dir, "run_stats.mat", uav_count))

            if success_flags:
                print(
                    f"{case:15s} | {uav_count:4d} | {np.mean(success_flags) * 100:6.1f}% ({int(np.sum(success_flags))}/{len(success_flags)})"
                )


if __name__ == "__main__":
    main()
