import json
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path("results/mogwo_ablation_fast")
CASES = ["full", "loo_diversity", "loo_step_limiter", "loo_pressure"]


def main() -> None:
    print(f"{'Case':20s} | {'HV Mean':10s} | {'IG+ Mean':10s}")
    print("-" * 46)

    for case in CASES:
        case_dir = RESULTS_ROOT / case
        hvs = []
        igs = []
        artifact_files = list(case_dir.glob("**/artifacts.json"))

        for art in artifact_files:
            try:
                with art.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                metrics = data.get("metrics", {})
                hv = metrics.get("hypervolume")
                ig = metrics.get("invertedGenerationalDistancePlus")
                if hv is not None:
                    hvs.append(float(hv))
                if ig is not None:
                    igs.append(float(ig))
            except Exception:
                continue

        if hvs:
            print(f"{case:20s} | {np.mean(hvs):.4f}     | {np.mean(igs):.4f}")
        else:
            print(f"{case:20s} | No data")


if __name__ == "__main__":
    main()
