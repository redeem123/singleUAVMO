from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from scipy.io import loadmat

from uav_benchmark.core.metrics import cal_metric


def matches_uav_count(path: Path, uav_count: int) -> bool:
    parent_name = path.parent.parent.name
    if int(uav_count) == 1:
        return "_uav" not in parent_name
    return f"_uav{int(uav_count)}" in parent_name


def case_mat_files(case_dir: Path, filename: str, uav_count: int | None = None) -> Iterable[Path]:
    for path in case_dir.glob(f"**/{filename}"):
        if uav_count is None or matches_uav_count(path, uav_count):
            yield path


def objective_metric_values(paths: Iterable[Path]) -> tuple[list[float], list[float]]:
    hvs: list[float] = []
    igs: list[float] = []
    for path in paths:
        try:
            data = loadmat(str(path))
            obj = data["PopObj"]
            if obj.size <= 0:
                continue
            prob_idx = int(data["problemIndex"][0, 0])
            objective_count = int(data["M"][0, 0])
            hvs.append(cal_metric(1, obj, prob_idx, objective_count))
            igs.append(cal_metric(2, obj, prob_idx, objective_count))
        except Exception:
            continue
    return hvs, igs


def feasibility_values(paths: Iterable[Path]) -> tuple[list[float], list[float]]:
    ratios: list[float] = []
    totals: list[float] = []
    for path in paths:
        try:
            data = loadmat(str(path))
            feasible = float(data["feasibleCount"][0, 0])
            total = float(data["solutionCount"][0, 0])
            if total > 0:
                ratios.append(feasible / total)
                totals.append(feasible)
        except Exception:
            continue
    return ratios, totals


def run_success_flags(paths: Iterable[Path]) -> list[float]:
    flags: list[float] = []
    for path in paths:
        try:
            data = loadmat(str(path))
            feasible = float(data["feasibleCount"][0, 0])
            flags.append(1.0 if feasible > 0 else 0.0)
        except Exception:
            continue
    return flags
