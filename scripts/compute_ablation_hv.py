from pathlib import Path

import numpy as np

from scripts.summary_helpers import case_mat_files, objective_metric_values

RESULTS_ROOT = Path("results/mogwo_ablation_fast")
CASES = ["full", "pure_gwo", "loo_diversity", "loo_step_limiter", "loo_pressure"]


def main() -> None:
    print(f"{'Case':20s} | {'HV Mean':10s} | {'IG+ Mean':10s}")
    print("-" * 46)

    for case in CASES:
        case_dir = RESULTS_ROOT / case
        hvs, igs = objective_metric_values(case_mat_files(case_dir, "final_popobj.mat"))

        if hvs:
            print(f"{case:20s} | {np.mean(hvs):.4f}     | {np.mean(igs):.4f}")
        else:
            print(f"{case:20s} | No data")


if __name__ == "__main__":
    main()
