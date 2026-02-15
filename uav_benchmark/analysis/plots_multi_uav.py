from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_benchmark.io.matlab import load_mat, load_terrain_struct


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc
    return plt


def _require_scipy_loadmat():
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for multi-UAV 3D path plotting") from exc
    return loadmat


def _normalize_pop_obj(pop_obj: np.ndarray) -> np.ndarray:
    pop = np.asarray(pop_obj, dtype=float)
    if pop.ndim == 1:
        pop = pop.reshape(1, -1)
    if pop.ndim == 2 and pop.shape[1] != 4 and pop.shape[0] == 4:
        pop = pop.T
    return pop


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _best_feasible_row(run_dir: Path) -> tuple[int, float] | None:
    pop_file = run_dir / "final_popobj.mat"
    if not pop_file.exists():
        return None
    payload = load_mat(pop_file)
    if "PopObj" not in payload:
        return None
    pop_obj = _normalize_pop_obj(np.asarray(payload["PopObj"], dtype=float))
    finite = np.all(np.isfinite(pop_obj), axis=1)
    if not np.any(finite):
        return None
    feasible_rows = np.where(finite)[0]
    scores = np.sum(pop_obj[feasible_rows], axis=1)
    local_best = int(np.argmin(scores))
    return int(feasible_rows[local_best]), float(scores[local_best])


def _load_fleet_paths(run_dir: Path) -> np.ndarray | None:
    fleet_file = run_dir / "fleet_paths.mat"
    if not fleet_file.exists():
        return None
    loadmat = _require_scipy_loadmat()
    payload = loadmat(str(fleet_file), squeeze_me=True, struct_as_record=False)
    if "fleetPaths" not in payload:
        return None
    arr = np.asarray(payload["fleetPaths"], dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        return None
    return arr


def _select_representative_mission(problem_dir: Path) -> tuple[np.ndarray, str, int] | None:
    best_score = np.inf
    best_mission: np.ndarray | None = None
    best_run = ""
    best_row = -1
    for run_dir in sorted(problem_dir.glob("Run_*")):
        row_info = _best_feasible_row(run_dir)
        if row_info is None:
            continue
        row_idx, score = row_info
        fleet_paths = _load_fleet_paths(run_dir)
        if fleet_paths is None:
            continue
        if row_idx < 0 or row_idx >= fleet_paths.shape[0]:
            continue
        if score < best_score:
            best_score = score
            best_mission = np.asarray(fleet_paths[row_idx], dtype=float)
            best_run = run_dir.name
            best_row = row_idx
    if best_mission is None:
        return None
    return best_mission, best_run, best_row


def _plot_mission_3d(plt, terrain: dict, mission_paths: np.ndarray, title: str, output_path: Path) -> None:
    height_map = np.asarray(terrain["H"], dtype=float)
    x_grid = np.asarray(terrain.get("X", np.arange(1, height_map.shape[1] + 1)), dtype=float).reshape(-1)
    y_grid = np.asarray(terrain.get("Y", np.arange(1, height_map.shape[0] + 1)), dtype=float).reshape(-1)
    if x_grid.size != height_map.shape[1]:
        x_grid = np.linspace(1, height_map.shape[1], height_map.shape[1], dtype=float)
    if y_grid.size != height_map.shape[0]:
        y_grid = np.linspace(1, height_map.shape[0], height_map.shape[0], dtype=float)

    xv, yv = np.meshgrid(x_grid, y_grid)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    ax.plot_surface(xv, yv, height_map, cmap="terrain", alpha=0.68, linewidth=0, zorder=1)

    z_range = float(np.nanmax(height_map) - np.nanmin(height_map))
    lift = max(1.0, 0.02 * z_range)
    palette = plt.get_cmap("tab10")
    for uav_idx in range(mission_paths.shape[0]):
        path_xyz = np.asarray(mission_paths[uav_idx], dtype=float)
        if path_xyz.ndim != 2 or path_xyz.shape[1] != 3:
            continue
        color = palette(uav_idx % 10)
        ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2] + lift, color=color, linewidth=2.2, zorder=10)
        ax.scatter(path_xyz[0, 0], path_xyz[0, 1], path_xyz[0, 2] + lift, color=color, marker="o", s=36, zorder=11)
        ax.scatter(path_xyz[-1, 0], path_xyz[-1, 1], path_xyz[-1, 2] + lift, color=color, marker="^", s=40, zorder=11)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Altitude")
    ax.view_init(elev=31, azim=-53)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _collect_metric(results_dir: Path, metric: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for algo_dir in sorted(results_dir.iterdir()):
        if not algo_dir.is_dir() or algo_dir.name.startswith(".") or algo_dir.name == "Plots":
            continue
        algo_name = algo_dir.name
        values.setdefault(algo_name, [])
        for problem_dir in sorted(algo_dir.iterdir()):
            if not problem_dir.is_dir():
                continue
            for run_dir in sorted(problem_dir.glob("Run_*")):
                if not run_dir.is_dir():
                    continue
                if metric in {"conflictRate", "minSeparation", "makespan", "energy", "risk"}:
                    mission_file = run_dir / "mission_stats.mat"
                    if not mission_file.exists():
                        continue
                    payload = load_mat(mission_file)
                    if metric not in payload:
                        continue
                    arr = np.asarray(payload[metric], dtype=float).reshape(-1)
                    arr = arr[np.isfinite(arr)]
                    values[algo_name].extend(arr.tolist())
                elif metric == "hv":
                    pop_file = run_dir / "final_popobj.mat"
                    if not pop_file.exists():
                        continue
                    payload = load_mat(pop_file)
                    if "PopObj" not in payload:
                        continue
                    pop_obj = np.asarray(payload["PopObj"], dtype=float)
                    if pop_obj.ndim == 2 and pop_obj.shape[1] != 4 and pop_obj.shape[0] == 4:
                        pop_obj = pop_obj.T
                    finite = pop_obj[np.all(np.isfinite(pop_obj), axis=1)]
                    if finite.size == 0:
                        continue
                    # use inverse sum as a monotonic scalar proxy for plotting only
                    score = float(np.mean(1.0 / np.maximum(1e-9, np.sum(finite, axis=1))))
                    values[algo_name].append(score)
    return values


def _boxplot(plt, metric_values: dict[str, list[float]], title: str, ylabel: str, output_path: Path) -> None:
    labels = []
    data = []
    for label, values in metric_values.items():
        if not values:
            continue
        labels.append(label)
        data.append(values)
    if not data:
        return
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate_multi_uav_plots(project_root: Path, results_dir: Path) -> list[Path]:
    plt = _require_matplotlib()
    project_root = project_root.resolve()
    results_dir = results_dir.resolve()
    out_dir = results_dir / "plots_multi_uav"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    metrics = [
        ("hv", "Mission Quality Proxy", "Proxy Score"),
        ("makespan", "Mission Makespan", "Distance/Time Units"),
        ("energy", "Mission Energy", "Energy Surrogate"),
        ("conflictRate", "Conflict Rate", "Violation Ratio"),
        ("minSeparation", "Minimum Separation", "Distance Units"),
    ]
    for metric, title, ylabel in metrics:
        values = _collect_metric(results_dir, metric)
        output = out_dir / f"{metric}_boxplot.png"
        _boxplot(plt, values, title, ylabel, output)
        if output.exists():
            outputs.append(output)

    for algo_dir in sorted(results_dir.iterdir()):
        if (
            not algo_dir.is_dir()
            or algo_dir.name.startswith(".")
            or algo_dir.name in {"Plots", "metrics", "plots_multi_uav"}
        ):
            continue
        for problem_dir in sorted(algo_dir.iterdir()):
            if not problem_dir.is_dir():
                continue
            selected = _select_representative_mission(problem_dir)
            if selected is None:
                continue
            mission_paths, run_name, row_idx = selected
            terrain_file = project_root / "problems" / f"terrainStruct_{problem_dir.name}.mat"
            if not terrain_file.exists():
                continue
            try:
                terrain = load_terrain_struct(terrain_file)
            except Exception:
                continue
            output = out_dir / f"path3d_{_safe_name(algo_dir.name)}_{_safe_name(problem_dir.name)}.png"
            title = f"{algo_dir.name} | {problem_dir.name} | {run_name} | sol {row_idx + 1}"
            _plot_mission_3d(plt, terrain, mission_paths, title, output)
            if output.exists():
                outputs.append(output)

    summary_file = out_dir / "plot_index.txt"
    summary_file.write_text("\n".join(str(path) for path in outputs), encoding="utf-8")
    outputs.append(summary_file)
    _ = project_root
    return outputs
