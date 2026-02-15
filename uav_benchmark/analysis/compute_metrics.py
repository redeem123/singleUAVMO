from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import load_mat, save_mat
from uav_benchmark.io.results import collect_run_dirs


@dataclass(slots=True)
class MetricConfig:
    hv_samples: int = 2000
    max_points: int = 100
    target_algorithms: tuple[str, ...] = ()
    target_problems: tuple[str, ...] = ()
    max_runs: int = 0
    seed: int = 0


def _sanitize_popobj(pop_obj: np.ndarray, objective_count: int | None) -> np.ndarray:
    if pop_obj.size == 0:
        return pop_obj
    if objective_count is not None and pop_obj.ndim == 2 and pop_obj.shape[1] != objective_count and pop_obj.shape[0] == objective_count:
        pop_obj = pop_obj.T
    pop_obj = pop_obj[np.all(np.isfinite(pop_obj), axis=1)]
    return pop_obj


def _align_mask(mask: np.ndarray, size: int) -> np.ndarray:
    aligned = np.zeros(size, dtype=bool)
    if size <= 0 or mask.size == 0:
        return aligned
    use = min(size, mask.size)
    aligned[:use] = mask[:use]
    return aligned


def _load_feasible_mask(run_dir: Path, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=bool)
    fallback = np.ones(count, dtype=bool)
    mission_file = run_dir / "mission_stats.mat"
    if not mission_file.exists():
        return fallback
    payload = load_mat(mission_file)
    if "feasible" in payload:
        feasible = np.asarray(payload["feasible"], dtype=float).reshape(-1)
        mask = _align_mask(feasible > 0.5, count)
    else:
        mask = fallback.copy()
        if "turnViolation" in payload:
            turn_violation = np.asarray(payload["turnViolation"], dtype=float).reshape(-1)
            mask &= ~_align_mask(turn_violation > 0.5, count)
        if "separationViolation" in payload:
            separation_violation = np.asarray(payload["separationViolation"], dtype=float).reshape(-1)
            mask &= ~_align_mask(separation_violation > 0.5, count)
    return mask


def _load_run_popobj(run_dir: Path) -> tuple[np.ndarray, int | None, int]:
    pop_file = run_dir / "final_popobj.mat"
    pop_obj = np.zeros((0, 4), dtype=float)
    objective_count: int | None = None
    problem_index = 3
    if pop_file.exists():
        data = load_mat(pop_file)
        if "PopObj" in data:
            pop_obj = np.asarray(data["PopObj"], dtype=float)
        if "M" in data:
            objective_count = int(np.asarray(data["M"]).reshape(-1)[0])
        if "problemIndex" in data:
            problem_index = int(np.asarray(data["problemIndex"]).reshape(-1)[0])
    if pop_obj.size > 0 and objective_count is None:
        objective_count = pop_obj.shape[1]
    if pop_obj.size == 0:
        bp_files = sorted(run_dir.glob("bp_*.mat"))
        rows = []
        for bp_file in bp_files:
            bp_data = load_mat(bp_file)
            if "dt_sv" in bp_data and isinstance(bp_data["dt_sv"], dict) and "objs" in bp_data["dt_sv"]:
                rows.append(np.asarray(bp_data["dt_sv"]["objs"], dtype=float).reshape(-1))
        if rows:
            pop_obj = np.vstack(rows)
            objective_count = pop_obj.shape[1]
    return pop_obj, objective_count, problem_index


def _build_ref_points(results_dir: Path) -> dict[str, np.ndarray]:
    ref_points: dict[str, np.ndarray] = {}
    for algorithm_dir in sorted(results_dir.iterdir()):
        if not algorithm_dir.is_dir() or algorithm_dir.name.startswith(".") or algorithm_dir.name == "Plots":
            continue
        for run_entry in collect_run_dirs(algorithm_dir):
            pop_obj, objective_count, _ = _load_run_popobj(run_entry.run_dir)
            pop_obj = _sanitize_popobj(pop_obj, objective_count)
            if pop_obj.size == 0:
                continue
            feasible_mask = _load_feasible_mask(run_entry.run_dir, pop_obj.shape[0])
            pop_obj = pop_obj[feasible_mask] if feasible_mask.size == pop_obj.shape[0] else pop_obj
            if pop_obj.size == 0:
                continue
            max_values = np.max(pop_obj, axis=0)
            if run_entry.problem_name in ref_points:
                ref_points[run_entry.problem_name] = np.maximum(ref_points[run_entry.problem_name], max_values)
            else:
                ref_points[run_entry.problem_name] = max_values
    for problem_name, reference in list(ref_points.items()):
        ref = reference * 1.1
        ref[ref <= 0] = 1
        ref_points[problem_name] = ref
    return ref_points


def _load_mission_conflict(run_dir: Path) -> float:
    mission_file = run_dir / "mission_stats.mat"
    if not mission_file.exists():
        return 0.0
    payload = load_mat(mission_file)
    if "conflictRate" not in payload:
        return 0.0
    values = np.asarray(payload["conflictRate"], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def compute_metrics(results_dir: Path, config: MetricConfig | None = None) -> None:
    cfg = config or MetricConfig()
    np.random.seed(cfg.seed)
    ref_points = _build_ref_points(results_dir)
    for algorithm_dir in sorted(results_dir.iterdir()):
        if not algorithm_dir.is_dir() or algorithm_dir.name.startswith(".") or algorithm_dir.name == "Plots":
            continue
        if cfg.target_algorithms and algorithm_dir.name not in cfg.target_algorithms:
            continue
        problem_dirs = [item for item in sorted(algorithm_dir.iterdir()) if item.is_dir() and not item.name.startswith(".")]
        for problem_dir in problem_dirs:
            problem_name = problem_dir.name
            if cfg.target_problems and problem_name not in cfg.target_problems:
                continue
            run_dirs = [item for item in sorted(problem_dir.glob("Run_*")) if item.is_dir()]
            if cfg.max_runs > 0:
                run_dirs = run_dirs[: cfg.max_runs]
            if not run_dirs:
                continue
            best_scores = []
            for run_dir in run_dirs:
                pop_obj, objective_count, problem_index = _load_run_popobj(run_dir)
                pop_obj = _sanitize_popobj(pop_obj, objective_count)
                if pop_obj.size == 0:
                    continue
                feasible_mask = _load_feasible_mask(run_dir, pop_obj.shape[0])
                pop_obj = pop_obj[feasible_mask] if feasible_mask.size == pop_obj.shape[0] else pop_obj
                if pop_obj.size == 0:
                    continue
                if cfg.max_points > 0 and pop_obj.shape[0] > cfg.max_points:
                    picks = np.random.permutation(pop_obj.shape[0])[: cfg.max_points]
                    pop_obj = pop_obj[picks]
                reference = ref_points.get(problem_name)
                hv = cal_metric(1, pop_obj, problem_index, objective_count or pop_obj.shape[1], cfg.hv_samples, reference)
                mission_conflict = _load_mission_conflict(run_dir)
                best_scores.append([hv, mission_conflict])
            if best_scores:
                save_mat(problem_dir / "final_hv.mat", {"bestScores": np.asarray(best_scores, dtype=float)})
