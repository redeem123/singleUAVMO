from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _quote_matlab_string(value: str) -> str:
    return value.replace("'", "''")


def generate_research_plots(project_root: Path, results_dir: Path | None = None) -> None:
    project_root = project_root.resolve()
    results_dir = (results_dir.resolve() if results_dir is not None else (project_root / "results").resolve())
    matlab_root = (project_root / "matlabimplementation").resolve()
    matlab_driver = matlab_root / "analysis" / "generate_research_plots_cli.m"

    if not matlab_driver.exists():
        raise FileNotFoundError(f"MATLAB plot driver not found: {matlab_driver}")

    matlab_bin = shutil.which("matlab")
    if matlab_bin is None:
        raise RuntimeError("MATLAB executable not found in PATH.")

    matlab_cmd = (
        f"cd('{_quote_matlab_string(str(matlab_root))}'); "
        "run('startup.m'); "
        f"generate_research_plots_cli('{_quote_matlab_string(str(project_root))}', "
        f"'{_quote_matlab_string(str(results_dir))}');"
    )
    env = os.environ.copy()
    # MATLAB should run with its own runtime paths, not Python package shims.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Keep MATLAB invocation conservative when called from Python subprocesses.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    subprocess.run([matlab_bin, "-batch", matlab_cmd], check=True, env=env)
