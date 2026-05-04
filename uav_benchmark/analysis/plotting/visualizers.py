from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from uav_benchmark.analysis.metrics.feasibility import finite_rows_mask, load_mission_feasible_mask, normalize_pop_obj
from uav_benchmark.analysis.plotting.helpers import require_matplotlib
from uav_benchmark.io.matlab import load_mat


def _save_matplotlib_fig(figure, out_path: Path) -> None:
    """Persist a matplotlib figure object with .fig extension."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(figure, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _bp_files_by_index(run_dir: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in sorted(run_dir.glob("bp_*.mat")):
        stem = path.stem
        if "_" not in stem:
            continue
        try:
            idx = int(stem.split("_", 1)[1])
        except ValueError:
            continue
        indexed[idx] = path
    return indexed


def _feasible_indices(run_dir: Path, available_indices: set[int]) -> list[int]:
    final_popobj = run_dir / "final_popobj.mat"
    if not final_popobj.exists():
        return []
    try:
        data = load_mat(final_popobj)
    except (OSError, RuntimeError, ValueError):
        return []
    if "PopObj" not in data:
        return []
    pop = normalize_pop_obj(np.asarray(data["PopObj"], dtype=float), objective_count=4)
    finite_rows = finite_rows_mask(pop)
    feasible_mask = load_mission_feasible_mask(run_dir, count=pop.shape[0], base_mask=finite_rows)

    feasible = [row + 1 for row, ok in enumerate(feasible_mask) if ok and (row + 1) in available_indices]
    return feasible


def path_visualizer(
    project_root: Path,
    problem_name: str,
    run_num: int,
    algorithm: str = "NSGA-II",
    show: bool = False,
    path_index: int | None = None,
    feasible_only: bool = True,
    display_lift: float = 0.0,
) -> Path:
    plt = require_matplotlib()
    terrain_file = project_root / "problems" / f"terrainStruct_{problem_name}.mat"
    if not terrain_file.exists():
        raise FileNotFoundError(f"Terrain file not found: {terrain_file}")
    terrain_data = load_mat(terrain_file)
    terrain = terrain_data["terrainStruct"]
    if not isinstance(terrain, dict):
        raise RuntimeError("terrainStruct parsing failed")
    run_dir = project_root / "results" / algorithm / problem_name / f"Run_{run_num}"
    bp_map = _bp_files_by_index(run_dir)
    if not bp_map:
        raise FileNotFoundError(f"No path files found: {run_dir}")
    available_indices = set(bp_map.keys())
    selected_index: int
    if path_index is not None:
        if path_index not in bp_map:
            raise FileNotFoundError(f"Path index bp_{path_index}.mat not found in {run_dir}")
        selected_index = path_index
    else:
        candidates = sorted(available_indices)
        if feasible_only:
            feasible_candidates = _feasible_indices(run_dir, available_indices)
            if not feasible_candidates:
                raise RuntimeError(
                    f"No feasible paths found in {run_dir} "
                    "(finite objectives + mission violation masks). Re-run with feasible_only=False."
                )
            candidates = feasible_candidates
        selected_index = int(np.random.choice(candidates))
    selected = bp_map[selected_index]
    solution_data = load_mat(selected)["dt_sv"]
    path = np.asarray(solution_data["path"], dtype=float)
    x_grid = np.asarray(terrain["X"], dtype=float).reshape(-1)
    y_grid = np.asarray(terrain["Y"], dtype=float).reshape(-1)
    z_grid = np.asarray(terrain["H"], dtype=float)
    # Auto-lift: nudge path just above the terrain surface so it isn't
    # swallowed by surface triangles, but tall buildings still occlude it.
    z_range = float(np.nanmax(z_grid) - np.nanmin(z_grid))
    auto_lift = max(z_range * 0.02, 1.0)  # at least 1 unit
    z_draw = path[:, 2] + max(0.0, float(display_lift)) + auto_lift

    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection="3d")
    axes.computed_zorder = False
    xv, yv = np.meshgrid(x_grid, y_grid)
    # Terrain: semi-transparent so buildings remain visible even when
    # the path is forced on top (matplotlib 3D can't do true depth
    # testing for lines vs surfaces).
    axes.plot_surface(xv, yv, z_grid, cmap="YlGnBu", alpha=0.70, linewidth=0, zorder=1)
    # Ground shadow: dashed projection on the floor for spatial context.
    z_floor = np.full_like(z_draw, float(np.nanmin(z_grid)))
    axes.plot(path[:, 0], path[:, 1], z_floor, color="gray", linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)
    # Path itself – always on top of surface.
    axes.plot(path[:, 0], path[:, 1], z_draw, color="#ff0000", linewidth=2.8, alpha=1.0, zorder=10)
    axes.scatter(path[0, 0], path[0, 1], z_draw[0], color="green", s=60, zorder=11)
    axes.scatter(path[-1, 0], path[-1, 1], z_draw[-1], color="magenta", s=60, zorder=11)
    axes.set_title(f"3D Path: {problem_name} / Run {run_num} / bp_{selected_index}")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Altitude")
    figure.tight_layout()
    out_dir = project_root / "results" / "Plots" / "Paths"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"Path_{problem_name}_run{run_num}.fig"
    _save_matplotlib_fig(figure, out_path)
    if show:
        # Keep the figure interactive so users can rotate/zoom in 3D.
        plt.show(block=True)
    plt.close(figure)
    return out_path


def peak_visualizer(project_root: Path, output_dir: Path | None = None) -> list[Path]:
    plt = require_matplotlib()
    problems_dir = project_root / "problems"
    out_dir = output_dir or (project_root / "results" / "Plots" / "Peaks")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for terrain_file in sorted(problems_dir.glob("*.mat")):
        data = load_mat(terrain_file)
        terrain = data.get("terrainStruct")
        if not isinstance(terrain, dict):
            continue
        x_grid = np.asarray(terrain["X"], dtype=float).reshape(-1)
        y_grid = np.asarray(terrain["Y"], dtype=float).reshape(-1)
        z_grid = np.asarray(terrain["H"], dtype=float)
        xv, yv = np.meshgrid(x_grid, y_grid)
        figure = plt.figure(figsize=(10, 6))
        axes = figure.add_subplot(111, projection="3d")
        axes.plot_surface(xv, yv, z_grid, cmap="terrain", linewidth=0)
        axes.view_init(elev=30, azim=-10)
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Height")
        figure.tight_layout()
        name = terrain_file.stem.replace("terrainStruct_", "")
        out_path = out_dir / f"{name}.fig"
        _save_matplotlib_fig(figure, out_path)
        plt.close(figure)
        outputs.append(out_path)
    return outputs
