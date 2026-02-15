from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.analysis.compute_metrics import _build_ref_points
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import load_mat, load_terrain_struct

try:  # Optional dependency gate for statistical tests.
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover - optional dependency
    mannwhitneyu = None


@dataclass(slots=True)
class ReportConfig:
    project_root: Path
    results_dir: Path
    output_dir: Path | None = None
    hv_samples: int = 2000
    max_runs: int = 0
    baseline_algorithm: str = "NMOPSO"
    seed: int = 0


@dataclass(slots=True)
class RunRecord:
    algorithm: str
    problem: str
    run_id: int
    run_dir: Path
    run_success: float
    feasible_ratio: float
    hv: float
    igd_plus: float
    runtime_sec: float
    min_clearance: float
    max_turn_deg: float
    turn_exceed_rate: float
    mission_conflict_rate: float
    mission_min_separation: float
    mission_makespan: float
    mission_energy: float
    feasible_obj: np.ndarray


def _finite_array(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else float("nan")


def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _iqr(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    q1 = float(np.percentile(finite, 25))
    q3 = float(np.percentile(finite, 75))
    return q3 - q1


def _load_popobj_raw(run_dir: Path) -> np.ndarray:
    pop_file = run_dir / "final_popobj.mat"
    pop_obj = np.zeros((0, 4), dtype=float)
    if pop_file.exists():
        data = load_mat(pop_file)
        if "PopObj" in data:
            pop_obj = np.asarray(data["PopObj"], dtype=float)
    if pop_obj.ndim == 1 and pop_obj.size > 0:
        pop_obj = pop_obj.reshape(1, -1)
    if pop_obj.ndim == 2 and pop_obj.shape[1] != 4 and pop_obj.shape[0] == 4:
        pop_obj = pop_obj.T
    if pop_obj.size > 0:
        return pop_obj

    rows: list[np.ndarray] = []
    for bp_file in sorted(run_dir.glob("bp_*.mat")):
        bp_data = load_mat(bp_file)
        dt = bp_data.get("dt_sv")
        if isinstance(dt, dict) and "objs" in dt:
            rows.append(np.asarray(dt["objs"], dtype=float).reshape(-1))
    if rows:
        return np.vstack(rows)
    return np.zeros((0, 4), dtype=float)


def _align_mask(mask: np.ndarray, size: int) -> np.ndarray:
    aligned = np.zeros(size, dtype=bool)
    if size <= 0:
        return aligned
    if mask.size == 0:
        return aligned
    use = min(size, mask.size)
    aligned[:use] = mask[:use]
    return aligned


def _load_feasible_mask(run_dir: Path, pop_obj: np.ndarray) -> np.ndarray:
    count = int(pop_obj.shape[0]) if pop_obj.ndim == 2 else 0
    finite_mask = np.all(np.isfinite(pop_obj), axis=1) if count > 0 else np.zeros(0, dtype=bool)
    if count == 0:
        return finite_mask
    mission_file = run_dir / "mission_stats.mat"
    if not mission_file.exists():
        return finite_mask
    payload = load_mat(mission_file)
    if "feasible" in payload:
        feasible = np.asarray(payload["feasible"], dtype=float).reshape(-1)
        feasible_mask = _align_mask(feasible > 0.5, count)
    else:
        feasible_mask = finite_mask.copy()
        if "turnViolation" in payload:
            turn_violation = np.asarray(payload["turnViolation"], dtype=float).reshape(-1)
            feasible_mask &= ~_align_mask(turn_violation > 0.5, count)
        if "separationViolation" in payload:
            separation_violation = np.asarray(payload["separationViolation"], dtype=float).reshape(-1)
            feasible_mask &= ~_align_mask(separation_violation > 0.5, count)
    return feasible_mask & finite_mask


def _extract_obstacles(model: dict[str, Any]) -> np.ndarray:
    obstacles: list[np.ndarray] = []
    if "threats" in model and model["threats"] is not None:
        threat_array = np.asarray(model["threats"], dtype=float)
        if threat_array.ndim == 2 and threat_array.shape[1] >= 4:
            obstacles.append(threat_array[:, :4])
    if "nofly_c" in model and model["nofly_c"] is not None and "nofly_r" in model and model["nofly_r"] is not None:
        centers = np.asarray(model["nofly_c"], dtype=float)
        if centers.ndim == 1:
            centers = centers.reshape(1, -1)
        if centers.shape[1] >= 2:
            centers = centers[:, :2]
            radii = np.asarray(model["nofly_r"], dtype=float).reshape(-1)
            if radii.size == 1:
                radii = np.repeat(radii, centers.shape[0])
            elif radii.size < centers.shape[0]:
                radii = np.pad(radii, (0, centers.shape[0] - radii.size), mode="edge")
            nofly = np.column_stack(
                [centers[:, 0], centers[:, 1], np.zeros(centers.shape[0]), radii[: centers.shape[0]]]
            )
            obstacles.append(nofly)
    return np.vstack(obstacles) if obstacles else np.zeros((0, 4), dtype=float)


def _interpolate_path(path_xyz: np.ndarray, step_size: float) -> np.ndarray:
    if path_xyz.shape[0] < 2:
        return path_xyz.copy()
    diffs = np.diff(path_xyz, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    steps_per_seg = np.maximum(1, np.ceil(distances / step_size).astype(int))
    total_points = 1 + int(np.sum(steps_per_seg))
    result = np.empty((total_points, 3), dtype=float)
    result[0] = path_xyz[0]
    cursor = 1
    for seg_idx in range(path_xyz.shape[0] - 1):
        n_steps = steps_per_seg[seg_idx]
        t = np.arange(1, n_steps + 1, dtype=float) / n_steps
        result[cursor : cursor + n_steps] = (
            (1.0 - t[:, np.newaxis]) * path_xyz[seg_idx] + t[:, np.newaxis] * path_xyz[seg_idx + 1]
        )
        cursor += n_steps
    return result[:cursor]


def _dist_points_to_segments_2d(centers: np.ndarray, seg_starts: np.ndarray, seg_ends: np.ndarray) -> np.ndarray:
    seg_dirs = seg_ends - seg_starts
    seg_len_sq = np.sum(seg_dirs**2, axis=1, keepdims=True)
    seg_len_sq = np.maximum(seg_len_sq, 1e-30)
    diff = centers[np.newaxis, :, :] - seg_starts[:, np.newaxis, :]
    t = np.sum(diff * seg_dirs[:, np.newaxis, :], axis=2) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj = seg_starts[:, np.newaxis, :] + t[:, :, np.newaxis] * seg_dirs[:, np.newaxis, :]
    return np.linalg.norm(centers[np.newaxis, :, :] - proj, axis=2)


def _path_min_clearance(path_xyz: np.ndarray, model: dict[str, Any]) -> float:
    path_xyz = np.asarray(path_xyz, dtype=float)
    if path_xyz.ndim != 2 or path_xyz.shape[1] != 3 or path_xyz.shape[0] < 2:
        return float("nan")
    height_map = np.asarray(model["H"], dtype=float)
    xmax = int(float(model["xmax"]))
    ymax = int(float(model["ymax"]))
    drone_size = float(model.get("droneSize", model.get("drone_size", 1.0)))
    _ = drone_size  # Explicitly keep for traceability to evaluator assumptions.
    step_size = float(model.get("collisionStep", 1.0))
    if step_size <= 0:
        step_size = 1.0
    interpolated = _interpolate_path(path_xyz, step_size)
    x_interp = interpolated[:, 0]
    y_interp = interpolated[:, 1]
    z_interp_abs = interpolated[:, 2]
    x_index_interp = np.clip(np.rint(x_interp).astype(int), 1, xmax) - 1
    y_index_interp = np.clip(np.rint(y_interp).astype(int), 1, ymax) - 1
    ground_interp = height_map[y_index_interp, x_index_interp]
    z_interp_rel = z_interp_abs - ground_interp
    if interpolated.shape[0] < 2:
        return float(np.min(z_interp_rel)) if z_interp_rel.size else float("nan")
    terrain_clearance = np.minimum(z_interp_rel[:-1], z_interp_rel[1:])
    obstacles = _extract_obstacles(model)
    if obstacles.shape[0] == 0:
        min_clearance = terrain_clearance
    else:
        seg_starts = np.column_stack([x_interp[:-1], y_interp[:-1]])
        seg_ends = np.column_stack([x_interp[1:], y_interp[1:]])
        obs_centers = obstacles[:, :2]
        obs_radii = obstacles[:, 3]
        dist_matrix = _dist_points_to_segments_2d(obs_centers, seg_starts, seg_ends)
        obs_clearance = dist_matrix - obs_radii[np.newaxis, :]
        min_obs_clearance = np.min(obs_clearance, axis=1)
        min_clearance = np.minimum(terrain_clearance, min_obs_clearance)
    return float(np.min(min_clearance)) if min_clearance.size else float("nan")


def _path_max_turn_deg(path_xyz: np.ndarray) -> float:
    path_xyz = np.asarray(path_xyz, dtype=float)
    if path_xyz.ndim != 2 or path_xyz.shape[1] != 3 or path_xyz.shape[0] < 3:
        return 0.0
    v1 = path_xyz[1:-1] - path_xyz[:-2]
    v2 = path_xyz[2:] - path_xyz[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 0) & (n2 > 0)
    if not np.any(valid):
        return 0.0
    cross_norms = np.linalg.norm(np.cross(v1[valid], v2[valid]), axis=1)
    dots = np.sum(v1[valid] * v2[valid], axis=1)
    angles = np.arctan2(cross_norms, dots)
    return float(np.degrees(np.max(np.abs(angles))))


def _load_path(run_dir: Path, path_index: int) -> np.ndarray | None:
    path_file = run_dir / f"bp_{path_index}.mat"
    if not path_file.exists():
        return None
    data = load_mat(path_file)
    dt = data.get("dt_sv")
    if not isinstance(dt, dict) or "path" not in dt:
        return None
    path = np.asarray(dt["path"], dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        return None
    return path


def _load_runtime(run_dir: Path) -> float:
    stats_file = run_dir / "run_stats.mat"
    if not stats_file.exists():
        return float("nan")
    data = load_mat(stats_file)
    if "runtimeSec" not in data:
        return float("nan")
    return float(np.asarray(data["runtimeSec"]).reshape(-1)[0])


def _load_mission_metric(run_dir: Path, key: str) -> float:
    mission_file = run_dir / "mission_stats.mat"
    if not mission_file.exists():
        return float("nan")
    payload = load_mat(mission_file)
    if key not in payload:
        return float("nan")
    values = np.asarray(payload[key], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if key == "minSeparation":
        return float(np.min(values))
    return float(np.mean(values))


def _non_dominated(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    unique = np.unique(points, axis=0)
    n_points = unique.shape[0]
    keep = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not keep[i]:
            continue
        for j in range(n_points):
            if i == j or not keep[j]:
                continue
            if np.all(unique[j] <= unique[i]) and np.any(unique[j] < unique[i]):
                keep[i] = False
                break
    return unique[keep]


def _igd_plus(pop_obj: np.ndarray, reference_front: np.ndarray) -> float:
    if pop_obj.size == 0 or reference_front.size == 0:
        return float("nan")
    ideal = np.min(reference_front, axis=0)
    nadir = np.max(reference_front, axis=0)
    span = np.maximum(nadir - ideal, 1e-12)
    normalized_pop = (pop_obj - ideal) / span
    normalized_ref = (reference_front - ideal) / span
    # IGD+ distance for minimization: d+(r, P) = min_p ||max(p-r, 0)||2
    delta = np.maximum(normalized_pop[np.newaxis, :, :] - normalized_ref[:, np.newaxis, :], 0.0)
    distance = np.linalg.norm(delta, axis=2)
    return float(np.mean(np.min(distance, axis=1)))


def _cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    wins = 0
    losses = 0
    for value_left in left:
        wins += int(np.sum(value_left > right))
        losses += int(np.sum(value_left < right))
    total = left.size * right.size
    return float((wins - losses) / total) if total > 0 else float("nan")


def _holm_correct(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row["problem"]), str(row["metric"]))
        grouped.setdefault(key, []).append(index)
    for indices in grouped.values():
        valid = [(idx, float(rows[idx]["p_value"])) for idx in indices if np.isfinite(float(rows[idx]["p_value"]))]
        if not valid:
            continue
        valid.sort(key=lambda item: item[1])
        m = len(valid)
        adjusted_ordered: list[float] = []
        for rank, (_idx, p_value) in enumerate(valid):
            adjusted_ordered.append(min(1.0, (m - rank) * p_value))
        # Enforce monotonicity.
        for rank in range(1, len(adjusted_ordered)):
            adjusted_ordered[rank] = max(adjusted_ordered[rank], adjusted_ordered[rank - 1])
        for (idx, _), adjusted in zip(valid, adjusted_ordered):
            rows[idx]["p_holm"] = float(adjusted)


def _summarize_group(records: list[RunRecord]) -> dict[str, Any]:
    hv_values = np.asarray([record.hv for record in records], dtype=float)
    igd_values = np.asarray([record.igd_plus for record in records], dtype=float)
    runtime_values = np.asarray([record.runtime_sec for record in records], dtype=float)
    feasible_values = np.asarray([record.feasible_ratio for record in records], dtype=float)
    success_values = np.asarray([record.run_success for record in records], dtype=float)
    min_clearance_values = np.asarray([record.min_clearance for record in records], dtype=float)
    max_turn_values = np.asarray([record.max_turn_deg for record in records], dtype=float)
    turn_exceed_values = np.asarray([record.turn_exceed_rate for record in records], dtype=float)
    mission_conflict_values = np.asarray([record.mission_conflict_rate for record in records], dtype=float)
    mission_min_sep_values = np.asarray([record.mission_min_separation for record in records], dtype=float)
    mission_makespan_values = np.asarray([record.mission_makespan for record in records], dtype=float)
    mission_energy_values = np.asarray([record.mission_energy for record in records], dtype=float)

    feasible_obj_rows = [record.feasible_obj for record in records if record.feasible_obj.size > 0]
    all_feasible = np.vstack(feasible_obj_rows) if feasible_obj_rows else np.zeros((0, 4), dtype=float)

    summary: dict[str, Any] = {
        "algorithm": records[0].algorithm,
        "problem": records[0].problem,
        "runs": int(len(records)),
        "fr_run_success": _mean(success_values),
        "feasible_ratio_mean": _mean(feasible_values),
        "feasible_ratio_std": _std(feasible_values),
        "hv_mean": _mean(hv_values),
        "hv_std": _std(hv_values),
        "hv_median": _median(hv_values),
        "hv_iqr": _iqr(hv_values),
        "igd_plus_mean": _mean(igd_values),
        "igd_plus_std": _std(igd_values),
        "igd_plus_median": _median(igd_values),
        "igd_plus_iqr": _iqr(igd_values),
        "runtime_sec_mean": _mean(runtime_values),
        "runtime_sec_std": _std(runtime_values),
        "runtime_sec_median": _median(runtime_values),
        "runtime_sec_iqr": _iqr(runtime_values),
        "min_clearance_min": float(np.min(min_clearance_values[np.isfinite(min_clearance_values)]))
        if np.any(np.isfinite(min_clearance_values))
        else float("nan"),
        "min_clearance_median": _median(min_clearance_values),
        "max_turn_deg_max": float(np.max(max_turn_values[np.isfinite(max_turn_values)]))
        if np.any(np.isfinite(max_turn_values))
        else float("nan"),
        "max_turn_deg_median": _median(max_turn_values),
        "turn_exceed_rate_mean": _mean(turn_exceed_values),
        "mission_conflict_rate_mean": _mean(mission_conflict_values),
        "mission_conflict_rate_std": _std(mission_conflict_values),
        "mission_min_separation_min": float(np.min(mission_min_sep_values[np.isfinite(mission_min_sep_values)]))
        if np.any(np.isfinite(mission_min_sep_values))
        else float("nan"),
        "mission_makespan_mean": _mean(mission_makespan_values),
        "mission_makespan_std": _std(mission_makespan_values),
        "mission_energy_mean": _mean(mission_energy_values),
        "mission_energy_std": _std(mission_energy_values),
    }

    for objective_index in range(4):
        key_median = f"j{objective_index + 1}_median_feasible"
        key_best = f"j{objective_index + 1}_best_feasible"
        if all_feasible.size == 0:
            summary[key_median] = float("nan")
            summary[key_best] = float("nan")
        else:
            summary[key_median] = float(np.median(all_feasible[:, objective_index]))
            summary[key_best] = float(np.min(all_feasible[:, objective_index]))
    return summary


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def generate_benchmark_report(config: ReportConfig) -> dict[str, Any]:
    np.random.seed(config.seed)
    results_dir = config.results_dir.resolve()
    project_root = config.project_root.resolve()
    output_dir = (config.output_dir or (results_dir / "metrics")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_points = _build_ref_points(results_dir)
    problem_points: dict[str, list[np.ndarray]] = {}
    snapshots: list[tuple[str, str, int, Path, np.ndarray, np.ndarray]] = []

    for algorithm_dir in sorted(results_dir.iterdir()):
        if not algorithm_dir.is_dir() or algorithm_dir.name.startswith(".") or algorithm_dir.name == "Plots":
            continue
        algorithm_name = algorithm_dir.name
        for problem_dir in sorted(algorithm_dir.iterdir()):
            if not problem_dir.is_dir() or problem_dir.name.startswith("."):
                continue
            run_dirs = [entry for entry in sorted(problem_dir.glob("Run_*")) if entry.is_dir()]
            if config.max_runs > 0:
                run_dirs = run_dirs[: config.max_runs]
            for run_dir in run_dirs:
                run_id = int(run_dir.name.split("_", 1)[1]) if "_" in run_dir.name else 0
                pop_obj = _load_popobj_raw(run_dir)
                feasible_mask = _load_feasible_mask(run_dir, pop_obj)
                feasible_obj = pop_obj[feasible_mask] if pop_obj.size else np.zeros((0, 4), dtype=float)
                if feasible_obj.size > 0:
                    problem_points.setdefault(problem_dir.name, []).append(feasible_obj)
                snapshots.append((algorithm_name, problem_dir.name, run_id, run_dir, pop_obj, feasible_mask))

    reference_fronts: dict[str, np.ndarray] = {}
    for problem_name, stacks in problem_points.items():
        merged = np.vstack(stacks) if stacks else np.zeros((0, 4), dtype=float)
        reference_fronts[problem_name] = _non_dominated(merged) if merged.size else np.zeros((0, 4), dtype=float)

    terrain_cache: dict[str, dict[str, Any]] = {}
    records: list[RunRecord] = []
    for algorithm_name, problem_name, run_id, run_dir, pop_obj, feasible_mask in snapshots:
        model = terrain_cache.get(problem_name)
        if model is None:
            terrain_file = project_root / "problems" / f"terrainStruct_{problem_name}.mat"
            if terrain_file.exists():
                model = load_terrain_struct(terrain_file)
            else:
                model = {}
            terrain_cache[problem_name] = model

        n_total = int(pop_obj.shape[0]) if pop_obj.ndim == 2 else 0
        feasible_obj = pop_obj[feasible_mask] if pop_obj.size else np.zeros((0, 4), dtype=float)
        feasible_count = int(feasible_obj.shape[0])
        feasible_ratio = float(feasible_count / n_total) if n_total > 0 else 0.0
        run_success = 1.0 if feasible_count > 0 else 0.0

        reference_point = ref_points.get(problem_name)
        objective_count = feasible_obj.shape[1] if feasible_obj.size else 4
        hv_value = cal_metric(1, feasible_obj, 0, objective_count, config.hv_samples, reference_point) if feasible_obj.size else 0.0
        igd_value = _igd_plus(feasible_obj, reference_fronts.get(problem_name, np.zeros((0, 4), dtype=float)))

        runtime_sec = _load_runtime(run_dir)
        mission_conflict_rate = _load_mission_metric(run_dir, "conflictRate")
        mission_min_separation = _load_mission_metric(run_dir, "minSeparation")
        mission_makespan = _load_mission_metric(run_dir, "makespan")
        mission_energy = _load_mission_metric(run_dir, "energy")
        mission_max_turn = _load_mission_metric(run_dir, "maxTurnDeg")
        mission_turn_violation = _load_mission_metric(run_dir, "turnViolation")

        min_clearances: list[float] = []
        max_turns: list[float] = []
        turn_limit_deg = float(model.get("maxTurnDeg", model.get("maxTurnAngleDeg", 75.0))) if model else 75.0
        for index, is_feasible in enumerate(feasible_mask, start=1):
            if not bool(is_feasible):
                continue
            path_xyz = _load_path(run_dir, index)
            if path_xyz is None:
                continue
            if model:
                min_clearances.append(_path_min_clearance(path_xyz, model))
            max_turns.append(_path_max_turn_deg(path_xyz))
        min_clearance = float(np.min(min_clearances)) if min_clearances else float("nan")
        max_turn_deg = float(np.max(max_turns)) if max_turns else float("nan")
        turn_exceed_rate = (
            float(np.mean(np.asarray(max_turns, dtype=float) > turn_limit_deg)) if max_turns else float("nan")
        )
        if np.isfinite(mission_max_turn):
            max_turn_deg = float(mission_max_turn)
        if np.isfinite(mission_turn_violation):
            turn_exceed_rate = float(mission_turn_violation)

        records.append(
            RunRecord(
                algorithm=algorithm_name,
                problem=problem_name,
                run_id=run_id,
                run_dir=run_dir,
                run_success=run_success,
                feasible_ratio=feasible_ratio,
                hv=hv_value,
                igd_plus=igd_value,
                runtime_sec=runtime_sec,
                min_clearance=min_clearance,
                max_turn_deg=max_turn_deg,
                turn_exceed_rate=turn_exceed_rate,
                mission_conflict_rate=mission_conflict_rate,
                mission_min_separation=mission_min_separation,
                mission_makespan=mission_makespan,
                mission_energy=mission_energy,
                feasible_obj=feasible_obj,
            )
        )

    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.algorithm, record.problem), []).append(record)

    summary_rows = [_summarize_group(group_records) for _, group_records in sorted(grouped.items())]

    pairwise_rows: list[dict[str, Any]] = []
    metrics = (
        "hv",
        "igd_plus",
        "feasible_ratio",
        "runtime_sec",
        "mission_conflict_rate",
        "mission_makespan",
        "mission_energy",
    )
    for problem_name in sorted({record.problem for record in records}):
        baseline = [record for record in records if record.problem == problem_name and record.algorithm == config.baseline_algorithm]
        if not baseline:
            continue
        competitors = sorted({record.algorithm for record in records if record.problem == problem_name and record.algorithm != config.baseline_algorithm})
        for metric in metrics:
            baseline_values = _finite_array([float(getattr(record, metric)) for record in baseline])
            for algorithm_name in competitors:
                competitor = [record for record in records if record.problem == problem_name and record.algorithm == algorithm_name]
                competitor_values = _finite_array([float(getattr(record, metric)) for record in competitor])
                p_value = float("nan")
                if mannwhitneyu is not None and baseline_values.size > 0 and competitor_values.size > 0:
                    p_value = float(mannwhitneyu(baseline_values, competitor_values, alternative="two-sided").pvalue)
                pairwise_rows.append(
                    {
                        "problem": problem_name,
                        "metric": metric,
                        "baseline": config.baseline_algorithm,
                        "algorithm": algorithm_name,
                        "n_baseline": int(baseline_values.size),
                        "n_algorithm": int(competitor_values.size),
                        "baseline_mean": _mean(baseline_values),
                        "algorithm_mean": _mean(competitor_values),
                        "p_value": p_value,
                        "p_holm": float("nan"),
                        "cliffs_delta": _cliffs_delta(baseline_values, competitor_values),
                    }
                )
    _holm_correct(pairwise_rows)

    # Scenario-level win/tie/loss against baseline by HV (higher better).
    wtl_rows: list[dict[str, Any]] = []
    for problem_name in sorted({record.problem for record in records}):
        baseline = [record for record in records if record.problem == problem_name and record.algorithm == config.baseline_algorithm]
        if not baseline:
            continue
        baseline_values = _finite_array([record.hv for record in baseline])
        baseline_median = _median(baseline_values)
        for algorithm_name in sorted({record.algorithm for record in records if record.problem == problem_name and record.algorithm != config.baseline_algorithm}):
            comp = [record for record in records if record.problem == problem_name and record.algorithm == algorithm_name]
            comp_values = _finite_array([record.hv for record in comp])
            comp_median = _median(comp_values)
            if not np.isfinite(comp_median) or not np.isfinite(baseline_median):
                outcome = "tie"
            elif comp_median > baseline_median + 1e-9:
                outcome = "win"
            elif comp_median < baseline_median - 1e-9:
                outcome = "loss"
            else:
                outcome = "tie"
            wtl_rows.append(
                {
                    "problem": problem_name,
                    "baseline": config.baseline_algorithm,
                    "algorithm": algorithm_name,
                    "baseline_hv_median": baseline_median,
                    "algorithm_hv_median": comp_median,
                    "outcome": outcome,
                }
            )

    summary_csv = output_dir / "benchmark_metrics_summary.csv"
    pairwise_csv = output_dir / "pairwise_stats.csv"
    wtl_csv = output_dir / "win_tie_loss.csv"
    summary_json = output_dir / "benchmark_metrics_summary.json"

    if summary_rows:
        summary_fields = list(summary_rows[0].keys())
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    if pairwise_rows:
        pairwise_fields = list(pairwise_rows[0].keys())
        with pairwise_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
            writer.writeheader()
            for row in pairwise_rows:
                writer.writerow(row)
    if wtl_rows:
        wtl_fields = list(wtl_rows[0].keys())
        with wtl_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=wtl_fields)
            writer.writeheader()
            for row in wtl_rows:
                writer.writerow(row)

    payload = {
        "config": {key: _to_serializable(value) for key, value in asdict(config).items()},
        "summary": [{key: _to_serializable(value) for key, value in row.items()} for row in summary_rows],
        "pairwise": [{key: _to_serializable(value) for key, value in row.items()} for row in pairwise_rows],
        "win_tie_loss": [{key: _to_serializable(value) for key, value in row.items()} for row in wtl_rows],
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "summary_csv": summary_csv,
        "pairwise_csv": pairwise_csv if pairwise_rows else None,
        "win_tie_loss_csv": wtl_csv if wtl_rows else None,
        "summary_json": summary_json,
        "summary_rows": len(summary_rows),
        "pairwise_rows": len(pairwise_rows),
    }
