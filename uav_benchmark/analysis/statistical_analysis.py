from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uav_benchmark.analysis.compute_metrics import MetricConfig, _load_run_popobj, _sanitize_popobj, _build_ref_points
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import load_mat, save_mat


@dataclass(slots=True)
class StatisticalRow:
    problem: str
    mean_hv: float
    std_hv: float
    mean_obj: np.ndarray
    std_obj: np.ndarray


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float)
    means = np.zeros(values.shape[1], dtype=float)
    stds = np.zeros(values.shape[1], dtype=float)
    for index in range(values.shape[1]):
        column = values[:, index]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        means[index] = float(np.mean(column))
        stds[index] = float(np.std(column))
    return means, stds


def statistical_analysis(results_dir: Path, config: MetricConfig | None = None) -> dict[str, list[StatisticalRow]]:
    cfg = config or MetricConfig()
    np.random.seed(cfg.seed)
    ref_points = _build_ref_points(results_dir)
    report: dict[str, list[StatisticalRow]] = {}
    for algorithm_dir in sorted(results_dir.iterdir()):
        if not algorithm_dir.is_dir() or algorithm_dir.name.startswith(".") or algorithm_dir.name == "Plots":
            continue
        if cfg.target_algorithms and algorithm_dir.name not in cfg.target_algorithms:
            continue
        rows: list[StatisticalRow] = []
        for problem_dir in sorted(algorithm_dir.iterdir()):
            if not problem_dir.is_dir() or problem_dir.name.startswith("."):
                continue
            if cfg.target_problems and problem_dir.name not in cfg.target_problems:
                continue
            run_dirs = [entry for entry in sorted(problem_dir.glob("Run_*")) if entry.is_dir()]
            if cfg.max_runs > 0:
                run_dirs = run_dirs[: cfg.max_runs]
            hv_scores = []
            mission_conflicts = []
            objective_means = []
            problem_index = 3
            for run_dir in run_dirs:
                pop_obj, objective_count, problem_index = _load_run_popobj(run_dir)
                pop_obj = _sanitize_popobj(pop_obj, objective_count)
                if pop_obj.size == 0:
                    continue
                if cfg.max_points > 0 and pop_obj.shape[0] > cfg.max_points:
                    picks = np.random.permutation(pop_obj.shape[0])[: cfg.max_points]
                    pop_obj = pop_obj[picks]
                reference = ref_points.get(problem_dir.name)
                hv_scores.append(cal_metric(1, pop_obj, problem_index, objective_count or pop_obj.shape[1], cfg.hv_samples, reference))
                mission_file = run_dir / "mission_stats.mat"
                conflict = 0.0
                if mission_file.exists():
                    payload = load_mat(mission_file)
                    if "conflictRate" in payload:
                        arr = np.asarray(payload["conflictRate"], dtype=float).reshape(-1)
                        arr = arr[np.isfinite(arr)]
                        if arr.size > 0:
                            conflict = float(np.mean(arr))
                mission_conflicts.append(conflict)
                objective_means.append(np.mean(pop_obj, axis=0))
            if not hv_scores:
                continue
            hv_array = np.asarray(hv_scores, dtype=float).reshape(-1, 1)
            objective_array = np.asarray(objective_means, dtype=float)
            hv_mean, hv_std = _mean_std(hv_array)
            obj_mean, obj_std = _mean_std(objective_array)
            rows.append(
                StatisticalRow(
                    problem=problem_dir.name,
                    mean_hv=float(hv_mean[0]),
                    std_hv=float(hv_std[0]),
                    mean_obj=obj_mean,
                    std_obj=obj_std,
                )
            )
            conflict_array = np.asarray(mission_conflicts, dtype=float).reshape(-1, 1)
            if conflict_array.shape[0] != hv_array.shape[0]:
                conflict_array = np.zeros_like(hv_array)
            save_mat(problem_dir / "final_hv.mat", {"bestScores": np.column_stack([hv_array, conflict_array])})
        report[algorithm_dir.name] = rows
    return report
