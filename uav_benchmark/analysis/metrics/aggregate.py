from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_benchmark.io.matlab import load_mat


def aggregate_results(results_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for folder in sorted(results_dir.glob("Population_*")):
        if not folder.is_dir():
            continue
        metric_file = folder / "final_hv.mat"
        if not metric_file.exists():
            continue
        data = load_mat(metric_file)
        if "bestScores" not in data:
            continue
        best_scores = np.asarray(data["bestScores"], dtype=float)
        if best_scores.ndim != 2 or best_scores.shape[1] < 2:
            continue
        hv = best_scores[:, 0]
        pd = best_scores[:, 1]
        rows.append(
            {
                "problem": folder.name.replace("Population_", ""),
                "mean_hv": float(np.mean(hv)),
                "std_hv": float(np.std(hv)),
                "mean_pd": float(np.mean(pd)),
                "std_pd": float(np.std(pd)),
            }
        )
    return rows
