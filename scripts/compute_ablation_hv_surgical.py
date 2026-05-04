from pathlib import Path

import numpy as np

from scripts.summary_helpers import case_mat_files, objective_metric_values

RESULTS_ROOT = Path("results/mogwo_ablation_surgical")
CASES = ["full", "loo_step_div_driver"]


def main() -> None:
    print(f"{'Case':25s} | {'HV Mean':10s} | {'IG+ Mean':10s}")
    print("-" * 55)

    for case in CASES:
        case_dir = RESULTS_ROOT / case
        hvs, igs = objective_metric_values(case_mat_files(case_dir, "final_popobj.mat"))
        if hvs:
            print(f"{case:25s} | {np.mean(hvs):.4f}     | {np.mean(igs):.4f}")
        else:
            print(f"{case:25s} | No data")


if __name__ == "__main__":
    main()
