from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunDirectory:
    run_dir: Path
    problem_name: str


def save_run_summary_json(
    path: Path,
    params: Any,
    stats: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save a standardized JSON summary of a benchmark run."""
    summary = {
        "metadata": {
            "algorithm": getattr(params, "algorithm", "unknown"),
            "problem": getattr(params, "problem_name", "unknown"),
            "fleet_size": getattr(params, "fleet_size", 1),
            "generations": getattr(params, "generations", 0),
            "population": getattr(params, "population", 0),
            "seed": getattr(params, "seed", None),
        },
        "statistics": stats,
        "metrics": metrics or {},
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def collect_run_dirs(base_dir: Path) -> list[RunDirectory]:
    run_dirs: list[RunDirectory] = []
    if not base_dir.exists():
        return run_dirs
    for level_one in sorted(base_dir.iterdir()):
        if not level_one.is_dir() or level_one.name.startswith("."):
            continue
        direct_runs = sorted(level_one.glob("Run_*"))
        if direct_runs:
            for run_dir in direct_runs:
                if run_dir.is_dir():
                    run_dirs.append(RunDirectory(run_dir=run_dir, problem_name=level_one.name))
            continue
        for level_two in sorted(level_one.iterdir()):
            if not level_two.is_dir() or level_two.name.startswith("."):
                continue
            for run_dir in sorted(level_two.glob("Run_*")):
                if run_dir.is_dir():
                    run_dirs.append(RunDirectory(run_dir=run_dir, problem_name=level_two.name))
    return run_dirs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
