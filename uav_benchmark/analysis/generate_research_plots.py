from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def generate_research_plots(project_root: Path, results_dir: Path | None = None) -> None:
    project_root = project_root.resolve()
    results_dir = (results_dir.resolve() if results_dir is not None else (project_root / "results").resolve())
    matlab_driver = (project_root / "uav_benchmark" / "analysis" / "matlab" / "generate_research_plots_cli.m").resolve()

    if not matlab_driver.exists():
        raise FileNotFoundError(f"MATLAB plot driver not found: {matlab_driver}")

    matlab_bin = shutil.which("matlab")
    if matlab_bin is None:
        raise RuntimeError("MATLAB executable not found in PATH.")

    matlab_cmd = (
        "addpath('uav_benchmark/analysis/matlab'); "
        "pr = getenv('UAV_PROJECT_ROOT'); "
        "rd = getenv('UAV_RESULTS_DIR'); "
        "generate_research_plots_cli(pr, rd);"
    )
    env = os.environ.copy()
    # MATLAB should run with its own runtime paths, not Python package shims.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["UAV_PROJECT_ROOT"] = str(project_root)
    env["UAV_RESULTS_DIR"] = str(results_dir)
    # Keep MATLAB invocation conservative when called from Python subprocesses.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    subprocess.run([matlab_bin, "-batch", matlab_cmd], check=True, env=env, cwd=str(project_root))
