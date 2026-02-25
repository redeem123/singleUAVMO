from __future__ import annotations

from pathlib import Path


def generate_fleet_plots(project_root: Path, results_dir: Path) -> list[Path]:
    """Generate fleet plots as MATLAB .fig files.

    This API is kept for backward compatibility, but plotting is delegated
    to the research plotter so we no longer emit PNG artifacts.
    """
    from uav_benchmark.analysis.plotting.research import generate_research_plots

    project_root = project_root.resolve()
    results_dir = results_dir.resolve()
    generate_research_plots(project_root=project_root, results_dir=results_dir)

    plots_dir = results_dir / "Plots"
    return sorted(plots_dir.glob("*.fig")) if plots_dir.exists() else []
