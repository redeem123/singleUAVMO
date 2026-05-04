from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from uav_benchmark.core.metrics import cal_metric


def _iter_runs(root: Path):
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        case = case_dir.name
        for pop_path in sorted(case_dir.glob("**/final_popobj.mat")):
            stats_path = pop_path.parent / "run_stats.mat"
            if not stats_path.exists():
                continue
            yield case, pop_path, stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rebenchmark ablation cases.")
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--hv-samples", type=int, default=2000)
    args = parser.parse_args()

    case_rows: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    scenario_rows: dict[tuple[str, str], list[tuple[float, float, float, float, float]]] = defaultdict(list)

    for case, pop_path, stats_path in _iter_runs(args.results_root):
        pop = loadmat(str(pop_path))
        stats = loadmat(str(stats_path))
        obj = np.asarray(pop["PopObj"], dtype=float)
        problem_idx = int(pop["problemIndex"][0, 0])
        objective_count = int(pop["M"][0, 0])
        hv = cal_metric(1, obj, problem_idx, objective_count, hv_samples=args.hv_samples)
        pd = cal_metric(2, obj, problem_idx, objective_count)
        feasible_count = float(stats["feasibleCount"][0, 0])
        solution_count = float(stats["solutionCount"][0, 0])
        feasible_ratio = feasible_count / solution_count if solution_count > 0 else 0.0
        success = 1.0 if feasible_count > 0 else 0.0
        runtime_sec = float(stats["runtimeSec"][0, 0])
        problem_name = pop_path.parent.parent.name
        row = (hv, pd, feasible_ratio, success, runtime_sec)
        case_rows[case].append(row)
        scenario_rows[(case, problem_name)].append(row)

    print("CASE\tRUNS\tHV\tPD\tFEAS\tSUCCESS\tRUNTIME")
    summary: dict[str, np.ndarray] = {}
    for case in sorted(case_rows):
        arr = np.asarray(case_rows[case], dtype=float)
        summary[case] = np.mean(arr, axis=0)
        print(
            f"{case}\t{arr.shape[0]}"
            f"\t{summary[case][0]:.6f}"
            f"\t{summary[case][1]:.6f}"
            f"\t{summary[case][2]:.4f}"
            f"\t{summary[case][3]:.4f}"
            f"\t{summary[case][4]:.2f}"
        )

    print("\nTOP_HV_BY_SCENARIO")
    scenarios = sorted({problem_name for _case, problem_name in scenario_rows})
    for scenario in scenarios:
        scored: list[tuple[float, float, float, str]] = []
        for case in sorted(case_rows):
            arr = np.asarray(scenario_rows[(case, scenario)], dtype=float)
            scored.append((float(np.mean(arr[:, 0])), float(np.mean(arr[:, 2])), float(np.mean(arr[:, 4])), case))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        print(scenario)
        for hv, feas, runtime, case in scored[:5]:
            print(f"  {case}\tHV={hv:.6f}\tFEAS={feas:.4f}\tRT={runtime:.2f}")

    print("\nAVG_RANK\tCASE\tHV_R\tPD_R\tFEAS_R\tSUCCESS_R\tRT_R\tAVG")
    cases = sorted(summary)
    matrix = np.asarray([summary[case] for case in cases], dtype=float)
    ranks = np.zeros((len(cases), 5), dtype=float)
    for column in range(5):
        order = np.argsort(-matrix[:, column]) if column < 4 else np.argsort(matrix[:, column])
        rank = np.empty(len(cases), dtype=float)
        rank[order] = np.arange(1, len(cases) + 1)
        ranks[:, column] = rank
    for idx in np.argsort(np.mean(ranks, axis=1)):
        print(
            f"{cases[idx]}"
            f"\t{int(ranks[idx, 0])}"
            f"\t{int(ranks[idx, 1])}"
            f"\t{int(ranks[idx, 2])}"
            f"\t{int(ranks[idx, 3])}"
            f"\t{int(ranks[idx, 4])}"
            f"\t{np.mean(ranks[idx]):.2f}"
        )


if __name__ == "__main__":
    main()
