from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunDirectory:
    run_dir: Path
    problem_name: str


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
