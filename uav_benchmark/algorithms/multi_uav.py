from __future__ import annotations

from collections import deque
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import uav_benchmark.algorithms.nmopso_utils as nmopso_utils
from uav_benchmark.algorithms.rl_controller import LinUCBController, RLAction, TorchBanditController
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.mission_encoding import decision_size, decision_to_paths
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort, tournament_selection
from uav_benchmark.core.nsga3_ops import environmental_selection_nsga3, uniform_point
from uav_benchmark.io.matlab import load_mat, save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.utils.gpu import resolve_gpu


@dataclass(slots=True)
class Candidate:
    vector: np.ndarray
    objective: np.ndarray
    details: dict[str, Any]


@dataclass(slots=True)
class _NSGA3Candidate:
    objs: np.ndarray
    cons: float
    index: int


def _build_bounds(model: dict[str, Any], fleet_size: int, n_waypoints: int) -> tuple[np.ndarray, np.ndarray]:
    lower_single = np.array([float(model["xmin"]), float(model["ymin"]), float(model["zmin"])], dtype=float)
    upper_single = np.array([float(model["xmax"]), float(model["ymax"]), float(model["zmax"])], dtype=float)
    total = decision_size(fleet_size, n_waypoints)
    lower = np.tile(lower_single, total // 3)
    upper = np.tile(upper_single, total // 3)
    return lower, upper


def _build_navigation_bounds(
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    max_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    lower = np.zeros((fleet_size, n_waypoints, 3), dtype=float)
    upper = np.zeros((fleet_size, n_waypoints, 3), dtype=float)
    for uav_idx in range(fleet_size):
        start = starts[uav_idx].reshape(-1)[:3]
        goal = goals[uav_idx].reshape(-1)[:3]
        path_diag = float(np.linalg.norm(goal - start))
        r_max = max(1e-3, 3.0 * path_diag / max(1, n_waypoints))
        r_min = max(1e-4, r_max / 9.0)
        lower[uav_idx, :, 0] = r_min
        upper[uav_idx, :, 0] = r_max
        lower[uav_idx, :, 1] = -max_angle_rad
        upper[uav_idx, :, 1] = max_angle_rad
        lower[uav_idx, :, 2] = -max_angle_rad
        upper[uav_idx, :, 2] = max_angle_rad
    return lower.reshape(-1), upper.reshape(-1)


def _decision_to_paths_spherical(
    vector: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
) -> list[np.ndarray]:
    starts = np.asarray(model["starts"], dtype=float)
    goals = np.asarray(model["goals"], dtype=float)
    block = np.asarray(vector, dtype=float).reshape(fleet_size, n_waypoints, 3)
    paths: list[np.ndarray] = []
    for uav_idx in range(fleet_size):
        local_model = dict(model)
        local_model["start"] = starts[uav_idx].reshape(-1)[:3]
        local_model["end"] = goals[uav_idx].reshape(-1)[:3]
        position = {
            "r": block[uav_idx, :, 0],
            "phi": block[uav_idx, :, 1],
            "psi": block[uav_idx, :, 2],
        }
        cart = nmopso_utils.position_to_cart(position, local_model, "SC")
        paths.append(nmopso_utils.cart_to_absolute_path(cart, local_model))
    return paths


def _evaluate_population(
    population: np.ndarray,
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    representation: str = "cart",
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for idx in range(population.shape[0]):
        vector = np.asarray(population[idx], dtype=float).copy()
        if representation == "SC":
            paths = _decision_to_paths_spherical(vector, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        else:
            paths = decision_to_paths(vector, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        objective, details = evaluate_mission_details(paths, model)
        details["paths"] = paths
        candidates.append(Candidate(vector=vector, objective=objective, details=details))
    return candidates


def _candidate_matrix(candidates: list[Candidate]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 4), dtype=float)
    return np.array([candidate.objective for candidate in candidates], dtype=float)


def _finite_min(matrix: np.ndarray) -> np.ndarray:
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return np.min(matrix, axis=0) if matrix.size else np.zeros(4, dtype=float)
    return np.min(finite, axis=0)


def _archive_front(candidates: list[Candidate], max_size: int) -> list[Candidate]:
    if not candidates:
        return []
    matrix = _candidate_matrix(candidates)
    front_no, _ = n_d_sort(matrix.copy(), None, min(max_size, matrix.shape[0]))
    archive = [candidate for candidate, front in zip(candidates, front_no) if front == 1]
    if len(archive) > max_size:
        archive_matrix = _candidate_matrix(archive)
        fronts = np.ones(archive_matrix.shape[0], dtype=float)
        crowd = crowding_distance(archive_matrix, fronts)
        crowd = np.asarray(crowd, dtype=float)
        crowd[~np.isfinite(crowd)] = -np.inf
        order = np.argsort(-crowd)
        if order.size < max_size:
            pick = np.random.choice(len(archive), size=max_size, replace=False)
            archive = [archive[int(i)] for i in pick]
        else:
            archive = [archive[int(i)] for i in order[:max_size]]
    return archive


def _grid_cell_id(matrix: np.ndarray, n_grid: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=int)
    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    span = np.maximum(maximum - minimum, 1e-12)
    normalized = (matrix - minimum) / span
    bins = np.floor(normalized * n_grid).astype(int)
    bins = np.clip(bins, 0, max(0, n_grid - 1))
    factors = (n_grid ** np.arange(matrix.shape[1], dtype=int)).reshape(1, -1)
    return np.sum(bins * factors, axis=1).astype(int)


def _hypergrid_cell_id(matrix: np.ndarray, n_grid: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0, dtype=int)
    n_grid = max(2, int(n_grid))
    lower = np.min(matrix, axis=0)
    upper = np.max(matrix, axis=0)
    eps = (upper - lower) / (2.0 * max(1, n_grid - 1))
    gl = lower - eps
    gu = upper + eps
    denom = np.maximum(gu - gl, 1e-12)
    coords = np.rint((n_grid - 1) * (matrix - gl) / denom).astype(int)
    coords = np.clip(coords, 0, n_grid - 1)
    factors = (n_grid ** np.arange(matrix.shape[1], dtype=int)).reshape(1, -1)
    return np.sum(coords * factors, axis=1).astype(int)


def _hypergrid_occupied_count(matrix: np.ndarray, n_grid: int) -> int:
    if matrix.size == 0:
        return 0
    return int(np.unique(_hypergrid_cell_id(matrix, n_grid)).size)


def _sample_hypergrid_leaders(
    matrix: np.ndarray,
    n_pick: int,
    n_grid: int,
    kappa: float,
) -> tuple[np.ndarray, int]:
    if matrix.size == 0:
        return np.zeros(0, dtype=int), 0
    cell_id = _hypergrid_cell_id(matrix, n_grid)
    unique_cells, inverse = np.unique(cell_id, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    gamma = np.exp(-float(kappa) * counts)
    if np.sum(gamma) <= 0 or not np.all(np.isfinite(gamma)):
        probability = np.ones_like(gamma, dtype=float) / max(1, gamma.size)
    else:
        probability = gamma / np.sum(gamma)
    picks = np.zeros(int(max(0, n_pick)), dtype=int)
    for idx in range(picks.size):
        chosen_cell = int(unique_cells[int(np.random.choice(unique_cells.size, p=probability))])
        members = np.where(cell_id == chosen_cell)[0]
        picks[idx] = int(np.random.choice(members)) if members.size else int(np.random.randint(0, matrix.shape[0]))
    return picks, int(unique_cells.size)


def _leader_index(archive: list[Candidate], leader_bias: float, use_grid: bool = False, n_grid: int = 8) -> int:
    if not archive:
        return 0
    bias = float(np.clip(leader_bias, 0.0, 1.0))
    scores = np.array([np.sum(candidate.objective) for candidate in archive], dtype=float)
    scores[~np.isfinite(scores)] = np.inf
    if not use_grid:
        if np.random.rand() > bias:
            return int(np.random.randint(0, len(archive)))
        return int(np.argmin(scores))

    archive_matrix = _candidate_matrix(archive)
    finite_idx = np.where(np.all(np.isfinite(archive_matrix), axis=1))[0]
    if finite_idx.size == 0:
        return int(np.random.randint(0, len(archive)))
    finite_obj = archive_matrix[finite_idx]
    cell_id = _grid_cell_id(finite_obj, max(2, int(n_grid)))
    unique_cells, inverse = np.unique(cell_id, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    counts[counts <= 0] = 1.0
    sparse_weights = 1.0 / counts
    sparse_weights = sparse_weights / np.sum(sparse_weights)
    chosen_cell = int(unique_cells[int(np.random.choice(len(unique_cells), p=sparse_weights))])
    members = np.where(cell_id == chosen_cell)[0]
    if members.size == 0:
        members = np.arange(finite_idx.size, dtype=int)

    if np.random.rand() <= bias:
        local_scores = scores[finite_idx[members]]
        return int(finite_idx[members[int(np.argmin(local_scores))]])
    return int(finite_idx[int(np.random.choice(members))])


def _gpu_velocity_update(
    population: np.ndarray,
    velocity: np.ndarray,
    pbest: np.ndarray,
    leaders: np.ndarray,
    inertia: float,
    c1: float,
    c2: float,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_limit: np.ndarray | None,
    gpu_mode: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    info = resolve_gpu(gpu_mode)
    if str(gpu_mode).lower() == "force" and not info.enabled:
        raise RuntimeError(f"GPU mode is 'force' but no GPU backend is available ({info.reason}).")
    if info.enabled and info.backend == "cupy":
        import cupy as cp  # type: ignore

        x = cp.asarray(population)
        v = cp.asarray(velocity)
        pb = cp.asarray(pbest)
        gd = cp.asarray(leaders)
        vlim = cp.asarray(velocity_limit) if velocity_limit is not None else None
        r1 = cp.random.random(x.shape)
        r2 = cp.random.random(x.shape)
        v = inertia * v + c1 * r1 * (pb - x) + c2 * r2 * (gd - x)
        if vlim is not None:
            v = cp.clip(v, -vlim, vlim)
        x = cp.clip(x + v, cp.asarray(lower), cp.asarray(upper))
        return cp.asnumpy(x), cp.asnumpy(v), f"{info.backend}:{info.device}"
    if info.enabled and info.backend == "torch":
        import torch  # type: ignore

        device = torch.device(info.device)
        x = torch.tensor(population, dtype=torch.float32, device=device)
        v = torch.tensor(velocity, dtype=torch.float32, device=device)
        pb = torch.tensor(pbest, dtype=torch.float32, device=device)
        gd = torch.tensor(leaders, dtype=torch.float32, device=device)
        lo = torch.tensor(lower, dtype=torch.float32, device=device)
        hi = torch.tensor(upper, dtype=torch.float32, device=device)
        vlim = torch.tensor(velocity_limit, dtype=torch.float32, device=device) if velocity_limit is not None else None
        r1 = torch.rand_like(x)
        r2 = torch.rand_like(x)
        v = inertia * v + c1 * r1 * (pb - x) + c2 * r2 * (gd - x)
        if vlim is not None:
            v = torch.clamp(v, -vlim, vlim)
        x = torch.clamp(x + v, lo, hi)
        return x.detach().cpu().numpy(), v.detach().cpu().numpy(), f"{info.backend}:{info.device}"

    r1 = np.random.rand(*population.shape)
    r2 = np.random.rand(*population.shape)
    velocity = inertia * velocity + c1 * r1 * (pbest - population) + c2 * r2 * (leaders - population)
    if velocity_limit is not None:
        velocity = np.clip(velocity, -velocity_limit, velocity_limit)
    population = np.clip(population + velocity, lower, upper)
    return population, velocity, "numpy:cpu"


def _torch_device_peak_bytes(device_tag: str) -> float:
    try:
        import torch  # type: ignore
    except Exception:
        return 0.0
    tag = str(device_tag).lower()
    try:
        if tag.endswith("mps"):
            current = float(torch.mps.current_allocated_memory())
            driver = float(torch.mps.driver_allocated_memory())
            return max(current, driver)
        if "cuda" in tag and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated())
    except Exception:
        return 0.0
    return 0.0


def _fixed_hv_reference(matrix: np.ndarray) -> np.ndarray | None:
    if matrix.size == 0:
        return None
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        return None
    reference = np.max(finite, axis=0) * 1.1
    reference = np.asarray(reference, dtype=float)
    reference[reference <= 0] = 1.0
    return reference


def _finite_mean(values: list[float], default: float = 0.0) -> float:
    vector = np.asarray(values, dtype=float).reshape(-1)
    vector = vector[np.isfinite(vector)]
    if vector.size == 0:
        return float(default)
    return float(np.mean(vector))


def _resume_run_scores(
    run_dir: Path,
    problem_index: int,
    objective_count: int,
    compute_metrics: bool,
) -> np.ndarray | None:
    final_path = run_dir / "final_popobj.mat"
    if not final_path.exists():
        return None
    if not compute_metrics:
        return np.zeros(2, dtype=float)
    try:
        payload = load_mat(final_path)
        pop_obj = np.asarray(payload.get("PopObj", np.zeros((0, objective_count), dtype=float)), dtype=float)
        pop_obj = np.squeeze(pop_obj)
        if pop_obj.ndim == 1 and pop_obj.size > 0:
            pop_obj = pop_obj.reshape(1, -1)
        if pop_obj.ndim != 2:
            return None
        return np.array(
            [
                cal_metric(1, pop_obj, problem_index, objective_count),
                cal_metric(2, pop_obj, problem_index, objective_count),
            ],
            dtype=float,
        )
    except Exception:
        return None


def _parse_action_indices(raw: Any, fallback: tuple[int, ...], n_actions: int) -> np.ndarray:
    values: list[int] = []
    if raw is None:
        values = [int(item) for item in fallback]
    elif isinstance(raw, str):
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, np.ndarray)):
        values = [int(item) for item in raw]
    else:
        values = [int(item) for item in fallback]
    picks = np.asarray(values, dtype=int).reshape(-1)
    picks = np.unique(picks[(picks >= 0) & (picks < n_actions)])
    if picks.size == 0:
        return np.arange(n_actions, dtype=int)
    return picks


def _phase_allowed_actions(
    generation: int,
    generations: int,
    n_actions: int,
    phase_enabled: bool,
    phase_early_end: float,
    phase_mid_end: float,
    early_actions: np.ndarray,
    mid_actions: np.ndarray,
    late_actions: np.ndarray,
) -> np.ndarray | None:
    if not phase_enabled:
        return None
    progress = float(generation / max(1, generations))
    if progress <= phase_early_end:
        return early_actions
    if progress <= phase_mid_end:
        return mid_actions
    return late_actions


def _save_multi_artifacts(
    run_dir: Path,
    final_candidates: list[Candidate],
    problem_index: int,
    objective_count: int,
    runtime_sec: float,
    gpu_backend: str,
    gpu_peak_bytes: float,
    rl_trace: dict[str, np.ndarray] | None,
    gpu_update_time_sec: float = 0.0,
    rl_controller_time_sec: float = 0.0,
    rl_policy_backend: str = "none",
    rl_policy_gpu_peak_bytes: float = 0.0,
    rl_policy_loss_ema: float = 0.0,
    rl_metadata: dict[str, Any] | None = None,
) -> None:
    final_objectives = _candidate_matrix(final_candidates)
    save_run_popobj(run_dir / "final_popobj.mat", final_objectives, problem_index, objective_count)
    all_fleet_paths: list[np.ndarray] = []
    all_conflicts: list[np.ndarray] = []
    feasible_count = int(np.sum(np.all(np.isfinite(final_objectives), axis=1)))
    for idx, candidate in enumerate(final_candidates, start=1):
        paths = candidate.details.get("paths", [])
        if paths:
            first_path = np.asarray(paths[0], dtype=float)
            save_bp(run_dir / f"bp_{idx}.mat", first_path, candidate.objective)
            fleet_tensor = np.stack([np.asarray(path, dtype=float) for path in paths], axis=0)
            all_fleet_paths.append(fleet_tensor)
        conflict = np.asarray(candidate.details.get("conflictLog", np.zeros((0, 5), dtype=float)), dtype=float)
        if conflict.size > 0:
            all_conflicts.append(conflict)

    mission_payload = {
        "feasible": np.array([candidate.details.get("feasible", np.nan) for candidate in final_candidates], dtype=float),
        "conflictRate": np.array([candidate.details.get("conflictRate", np.nan) for candidate in final_candidates], dtype=float),
        "minSeparation": np.array([candidate.details.get("minSeparation", np.nan) for candidate in final_candidates], dtype=float),
        "makespan": np.array([candidate.details.get("makespan", np.nan) for candidate in final_candidates], dtype=float),
        "energy": np.array([candidate.details.get("energy", np.nan) for candidate in final_candidates], dtype=float),
        "risk": np.array([candidate.details.get("risk", np.nan) for candidate in final_candidates], dtype=float),
        "maxTurnDeg": np.array([candidate.details.get("maxTurnDeg", np.nan) for candidate in final_candidates], dtype=float),
        "turnPenalty": np.array([candidate.details.get("turnPenalty", np.nan) for candidate in final_candidates], dtype=float),
        "turnViolation": np.array([candidate.details.get("turnViolation", np.nan) for candidate in final_candidates], dtype=float),
        "separationViolation": np.array(
            [candidate.details.get("separationViolation", np.nan) for candidate in final_candidates],
            dtype=float,
        ),
    }
    save_mat(run_dir / "mission_stats.mat", mission_payload)
    if all_fleet_paths:
        save_mat(run_dir / "fleet_paths.mat", {"fleetPaths": np.array(all_fleet_paths, dtype=object)})
    else:
        save_mat(run_dir / "fleet_paths.mat", {"fleetPaths": np.array([], dtype=object)})
    if all_conflicts:
        save_mat(run_dir / "conflict_log.mat", {"conflicts": np.vstack(all_conflicts)})
    else:
        save_mat(run_dir / "conflict_log.mat", {"conflicts": np.zeros((0, 5), dtype=float)})

    stats_payload: dict[str, Any] = {
        "runtimeSec": float(runtime_sec),
        "feasibleCount": feasible_count,
        "solutionCount": int(final_objectives.shape[0]),
        "gpuBackend": gpu_backend,
        "gpuMemPeakBytes": float(gpu_peak_bytes),
        "gpuUpdateTimeSec": float(gpu_update_time_sec),
        "rlControllerTimeSec": float(rl_controller_time_sec),
        "rlPolicyBackend": rl_policy_backend,
        "rlPolicyGpuMemPeakBytes": float(rl_policy_gpu_peak_bytes),
        "rlPolicyLossEma": float(rl_policy_loss_ema),
    }
    if rl_trace is not None:
        stats_payload["rlActions"] = rl_trace.get("action", np.zeros(0, dtype=float))
        stats_payload["rlRewards"] = rl_trace.get("reward", np.zeros(0, dtype=float))
        stats_payload["rlFeatures"] = rl_trace.get("feature", np.zeros((0, 6), dtype=float))
        if "rewardImmediate" in rl_trace:
            stats_payload["rlImmediateRewards"] = rl_trace["rewardImmediate"]
        if "rewardUpdate" in rl_trace:
            stats_payload["rlUpdateRewards"] = rl_trace["rewardUpdate"]
        if "actionMaskSize" in rl_trace:
            stats_payload["rlActionMaskSize"] = rl_trace["actionMaskSize"]
    if rl_metadata:
        stats_payload.update(rl_metadata)
    save_mat(run_dir / "run_stats.mat", stats_payload)


def _elite_refine_candidates(
    archive: list[Candidate],
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    representation: str,
    lower: np.ndarray,
    upper: np.ndarray,
    span: np.ndarray,
    sigma: float,
    top_k: int,
    iters: int,
) -> list[Candidate]:
    if not archive or top_k <= 0 or iters <= 0 or sigma <= 0.0:
        return []
    ranked = sorted(range(len(archive)), key=lambda idx: float(np.sum(archive[idx].objective)))
    picks = ranked[: min(len(ranked), int(top_k))]
    if not picks:
        return []
    trial_vectors: list[np.ndarray] = []
    for _ in range(int(iters)):
        for idx in picks:
            base = archive[idx].vector
            noise = np.random.normal(0.0, 1.0, size=base.shape) * float(sigma) * span
            trial = np.clip(base + noise, lower, upper)
            trial_vectors.append(np.asarray(trial, dtype=float))
    if not trial_vectors:
        return []
    trial_population = np.stack(trial_vectors, axis=0)
    return _evaluate_population(
        trial_population,
        model=model,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
        representation=representation,
    )


def _run_multi_pso(
    model: dict[str, Any],
    params: BenchmarkParams,
    label: str,
    use_rl: bool,
) -> np.ndarray:
    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    fleet_size = int(params.fleet_size or model.get("fleetSize", 1))
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)

    paper_nmopso_global = bool(params.extra.get("nmopsoPaperMode", True)) and label in {"NMOPSO", "RL-NMOPSO"}
    representation = "SC" if paper_nmopso_global else "cart"
    if representation == "SC":
        max_angle_rad = float(np.deg2rad(params.max_turn_deg))
        lower, upper = _build_navigation_bounds(
            model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            max_angle_rad=max_angle_rad,
        )
    else:
        lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dimensions = int(lower.size)
    span = np.maximum(upper - lower, 1e-9)
    archive_size = int(params.extra.get("nRep", params.population))
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in range(1, params.runs + 1):
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue
        run_start = time.perf_counter()
        population = np.random.uniform(lower, upper, size=(params.population, dimensions))
        velocity = np.zeros_like(population)
        candidates = _evaluate_population(
            population,
            model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            representation=representation,
        )
        pbest = population.copy()
        pbest_obj = _candidate_matrix(candidates).copy()
        archive = _archive_front(candidates, max_size=archive_size)

        controller: Any | None = None
        rl_policy_backend = "none"
        rl_policy_gpu_peak_bytes = 0.0
        rl_policy_loss_ema = 0.0
        rl_policy_loaded = False
        rl_policy_saved = False
        rl_policy_frozen = False
        rl_policy_checkpoint = ""
        rl_policy_mode = "train"
        if use_rl:
            if paper_nmopso_global:
                paper_actions = (
                    # Expanded action set for phase-aware RL control.
                    RLAction(1.10, 1.9, 1.1, 1.25, 1.20, 1.25, 1.30, 1.25),
                    RLAction(1.08, 1.8, 1.2, 1.20, 1.15, 1.20, 1.20, 1.20),
                    RLAction(1.06, 1.7, 1.3, 1.15, 1.10, 1.15, 1.10, 1.15),
                    RLAction(1.04, 1.6, 1.4, 1.10, 1.10, 1.10, 1.05, 1.10),
                    RLAction(1.02, 1.4, 1.6, 1.10, 1.00, 1.05, 1.00, 1.05),
                    RLAction(1.00, 1.7, 1.3, 1.00, 1.05, 1.00, 1.05, 1.00),
                    RLAction(1.00, 1.5, 1.5, 1.00, 1.00, 1.00, 1.00, 1.00),
                    RLAction(0.99, 1.3, 1.7, 0.98, 0.95, 0.98, 0.95, 0.95),
                    RLAction(0.98, 1.6, 1.4, 0.95, 1.00, 0.95, 1.00, 0.95),
                    RLAction(0.97, 1.4, 1.6, 0.92, 0.95, 0.92, 0.95, 0.90),
                    RLAction(0.96, 1.2, 1.8, 0.90, 0.90, 0.90, 0.90, 0.85),
                    RLAction(0.95, 1.1, 1.9, 0.88, 0.85, 0.88, 0.85, 0.82),
                    RLAction(0.94, 1.2, 1.8, 0.85, 0.85, 0.85, 0.82, 0.80),
                    RLAction(0.93, 1.0, 2.0, 0.82, 0.80, 0.82, 0.80, 0.78),
                    RLAction(0.92, 1.3, 1.7, 0.80, 0.90, 0.80, 0.85, 0.76),
                    RLAction(0.90, 1.0, 2.0, 0.75, 0.80, 0.75, 0.78, 0.72),
                )
                backend_choice = str(params.extra.get("rlControllerBackend", "auto")).strip().lower()
                prefer_gpu_policy = bool(params.extra.get("rlUseGpuPolicy", True))
                use_gpu_policy = False
                policy_device = "cpu"
                if prefer_gpu_policy and backend_choice != "linucb":
                    gpu_info = resolve_gpu(params.gpu_mode)
                    if gpu_info.enabled and gpu_info.backend == "torch":
                        use_gpu_policy = True
                        policy_device = "cuda:0" if "cuda" in gpu_info.device else ("mps" if "mps" in gpu_info.device else "cpu")
                if use_gpu_policy:
                    try:
                        controller = TorchBanditController(
                            n_features=6,
                            actions=paper_actions,
                            device=policy_device,
                            warmup_steps=int(params.extra.get("rlWarmupSteps", 30)),
                            hidden_dim=int(params.extra.get("rlGpuHiddenDim", 384)),
                            lr=float(params.extra.get("rlGpuLr", 8e-4)),
                            batch_size=int(params.extra.get("rlGpuBatchSize", 2048)),
                            train_steps=int(params.extra.get("rlGpuTrainSteps", 16)),
                            min_train_size=int(params.extra.get("rlGpuMinTrainSize", 128)),
                            replay_capacity=int(params.extra.get("rlGpuReplayCapacity", 65536)),
                            epsilon_start=float(params.extra.get("rlEpsilonStart", 0.18)),
                            epsilon_end=float(params.extra.get("rlEpsilonEnd", 0.01)),
                            epsilon_decay_steps=int(params.extra.get("rlEpsilonDecaySteps", 4000)),
                            seed=int(params.seed) if params.seed is not None else 0,
                        )
                        rl_policy_backend = controller.device_tag
                    except Exception:
                        controller = LinUCBController(
                            n_features=6,
                            alpha=0.20,
                            warmup_steps=30,
                            actions=paper_actions,
                        )
                        rl_policy_backend = "linucb:cpu"
                else:
                    controller = LinUCBController(
                        n_features=6,
                        alpha=0.20,
                        warmup_steps=30,
                        actions=paper_actions,
                    )
                    rl_policy_backend = "linucb:cpu"
            else:
                controller = LinUCBController(n_features=6, alpha=1.0, warmup_steps=20)
                rl_policy_backend = "linucb:cpu"

        last_hv = 0.0
        stagnation = 0
        rl_actions = []
        rl_rewards = []
        rl_rewards_immediate = []
        rl_rewards_update = []
        rl_mask_sizes = []
        rl_features = []
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        gpu_backend = "numpy:cpu"
        gpu_peak_bytes = 0.0
        gpu_update_time_sec = 0.0
        rl_controller_time_sec = 0.0
        rl_policy_save_resolved = False

        rl_reward_n_step = int(params.extra.get("rlRewardNStep", 5 if paper_nmopso_global else 1))
        rl_reward_n_step = max(1, rl_reward_n_step)
        rl_reward_gamma = float(np.clip(params.extra.get("rlRewardGamma", 0.90), 0.0, 1.0))
        rl_hv_scale = float(max(1e-6, params.extra.get("rlRewardHvScale", 0.01)))
        rl_reward_hv_w = float(params.extra.get("rlRewardHvWeight", 0.70))
        rl_reward_feas_w = float(params.extra.get("rlRewardFeasibleWeight", 0.15))
        rl_reward_div_w = float(params.extra.get("rlRewardDiversityWeight", 0.10))
        rl_reward_conflict_w = float(params.extra.get("rlRewardConflictWeight", 0.05))
        rl_elite_refine = bool(params.extra.get("rlEliteRefine", use_rl and paper_nmopso_global))
        rl_elite_refine_top_k = int(max(0, params.extra.get("rlEliteRefineTopK", 3)))
        rl_elite_refine_iters = int(max(0, params.extra.get("rlEliteRefineIters", 2)))
        rl_elite_sigma_start = float(max(0.0, params.extra.get("rlEliteRefineSigmaStart", 0.06)))
        rl_elite_sigma_end = float(max(0.0, params.extra.get("rlEliteRefineSigmaEnd", 0.015)))
        rl_elite_trials_total = 0

        if controller is not None:
            raw_mode = str(params.extra.get("rlPolicyMode", "train")).strip().lower()
            rl_policy_mode = raw_mode if raw_mode in {"train", "warmstart", "freeze"} else "train"
            rl_policy_load = rl_policy_mode in {"warmstart", "freeze"}
            rl_policy_save = rl_policy_mode in {"train", "warmstart"}
            if "rlPolicyLoad" in params.extra:
                rl_policy_load = bool(params.extra.get("rlPolicyLoad"))
            if "rlPolicySave" in params.extra:
                rl_policy_save = bool(params.extra.get("rlPolicySave"))
            rl_policy_frozen = bool(params.extra.get("rlPolicyFreeze", rl_policy_mode == "freeze"))

            checkpoint_raw = str(params.extra.get("rlPolicyCheckpointPath", "")).strip()
            if checkpoint_raw:
                rl_policy_checkpoint = str(Path(checkpoint_raw).expanduser().resolve())
            else:
                suffix = ".pt" if isinstance(controller, TorchBanditController) else ".npz"
                rl_policy_checkpoint = str((results_path / "_rl_policy" / f"{params.problem_name}_uav{fleet_size}{suffix}").resolve())

            if rl_policy_load and rl_policy_checkpoint:
                try:
                    control_t0 = time.perf_counter()
                    rl_policy_loaded = bool(controller.load(rl_policy_checkpoint, freeze=rl_policy_frozen))
                    rl_controller_time_sec += float(time.perf_counter() - control_t0)
                except Exception:
                    rl_policy_loaded = False

            if rl_policy_frozen and not rl_policy_loaded:
                rl_policy_frozen = False
            if rl_policy_frozen and hasattr(controller, "set_frozen"):
                controller.set_frozen(True)
            rl_policy_save_resolved = bool(rl_policy_save and (not rl_policy_frozen))

        is_nmopso_family = label in {"NMOPSO", "RL-NMOPSO"}
        paper_nmopso = paper_nmopso_global
        inertia = float(params.extra.get("w", 1.0 if is_nmopso_family else 0.7))
        inertia_damp = float(params.extra.get("wdamp", 0.98 if paper_nmopso else (0.995 if is_nmopso_family else 1.0)))
        inertia_min = float(params.extra.get("w_min", 0.40 if is_nmopso_family else 0.30))
        c1 = float(params.extra.get("c1", 1.5))
        c2 = float(params.extra.get("c2", 1.5))
        mutation_prob = float(params.extra.get("mutationProb", 0.15 if is_nmopso_family else 0.1))
        leader_bias = 0.5
        use_grid_leader = bool(params.extra.get("nmopsoGridLeader", is_nmopso_family))
        grid_cells = int(params.extra.get("nGrid", 7 if paper_nmopso else 8))
        grid_kappa = float(params.extra.get("kappa", 2.0))
        velocity_limit_ratio = float(params.extra.get("velocityClampRatio", 0.22 if is_nmopso_family else 0.30))
        velocity_limit_base = velocity_limit_ratio * span
        archive_init = _candidate_matrix(archive)
        archive_init = archive_init[np.all(np.isfinite(archive_init), axis=1)] if archive_init.size else archive_init
        delta_cells = float(max(1, _hypergrid_occupied_count(archive_init, grid_cells))) if paper_nmopso else 1.0
        diversity_ref = float(np.mean(np.std(archive_init, axis=0))) if archive_init.size else 1.0
        diversity_ref = max(diversity_ref, 1e-9)
        rl_div_scale = float(max(1e-9, params.extra.get("rlRewardDivScale", diversity_ref)))
        hv_ref_point = _fixed_hv_reference(archive_init)

        phase_enabled = bool(params.extra.get("rlPhaseGating", paper_nmopso_global))
        phase_early_end = float(np.clip(params.extra.get("rlPhaseEarlyEnd", 0.33), 0.0, 1.0))
        phase_mid_end = float(np.clip(params.extra.get("rlPhaseMidEnd", 0.75), phase_early_end, 1.0))
        n_actions = len(controller.actions) if controller is not None and hasattr(controller, "actions") else 0
        early_actions = _parse_action_indices(params.extra.get("rlPhaseEarlyActions"), tuple(range(0, 7)), n_actions) if n_actions else np.zeros(0, dtype=int)
        mid_actions = _parse_action_indices(params.extra.get("rlPhaseMidActions"), tuple(range(3, 13)), n_actions) if n_actions else np.zeros(0, dtype=int)
        late_actions = _parse_action_indices(params.extra.get("rlPhaseLateActions"), tuple(range(8, max(8, n_actions))), n_actions) if n_actions else np.zeros(0, dtype=int)

        pending_actions: deque[int] = deque()
        pending_features: deque[np.ndarray] = deque()
        pending_rewards: deque[float] = deque()

        for generation in range(1, params.generations + 1):
            if is_nmopso_family:
                inertia = max(inertia_min, inertia * inertia_damp)
            feasible_ratio = float(np.mean(np.all(np.isfinite(_candidate_matrix(candidates)), axis=1)))
            conflict_rate = _finite_mean(
                [float(candidate.details.get("conflictRate", np.nan)) for candidate in candidates],
                default=0.0,
            )
            archive_matrix = _candidate_matrix(archive)
            finite_archive = archive_matrix[np.all(np.isfinite(archive_matrix), axis=1)]
            if finite_archive.size > 0:
                if hv_ref_point is None:
                    hv_ref_point = _fixed_hv_reference(finite_archive)
                hv_now = cal_metric(
                    1,
                    finite_archive,
                    params.problem_index,
                    objective_count,
                    ref_point=hv_ref_point if hv_ref_point is not None else None,
                )
                diversity = float(np.mean(np.std(finite_archive, axis=0)))
            else:
                hv_now = 0.0
                diversity = 0.0
            hv_slope = hv_now - last_hv
            if hv_slope <= 1e-8:
                stagnation += 1
            else:
                stagnation = 0
            features = np.array(
                [
                    generation / max(1, params.generations),
                    np.clip(feasible_ratio, 0.0, 1.0),
                    np.clip(max(0.0, conflict_rate) / 0.02, 0.0, 1.0),
                    0.5 * (np.tanh(hv_slope / 0.01) + 1.0),
                    np.clip(np.log1p(max(0.0, diversity)) / np.log1p(3.0 * diversity_ref), 0.0, 1.0),
                    min(1.0, stagnation / max(1, params.generations)),
                ],
                dtype=float,
            )

            action_idx = -1
            kappa_scale = 1.0
            delta_scale = 1.0
            velocity_scale = 1.0
            region_scale = 1.0
            allowed_actions: np.ndarray | None = None
            if controller is not None:
                allowed_actions = _phase_allowed_actions(
                    generation=generation,
                    generations=params.generations,
                    n_actions=n_actions,
                    phase_enabled=phase_enabled,
                    phase_early_end=phase_early_end,
                    phase_mid_end=phase_mid_end,
                    early_actions=early_actions,
                    mid_actions=mid_actions,
                    late_actions=late_actions,
                )
                control_t0 = time.perf_counter()
                action_idx, action, _ = controller.select_action(features, allowed_actions=allowed_actions)
                rl_controller_time_sec += float(time.perf_counter() - control_t0)
                if isinstance(controller, TorchBanditController):
                    rl_policy_gpu_peak_bytes = max(rl_policy_gpu_peak_bytes, _torch_device_peak_bytes(controller.device_tag))
                c1 = action.c1
                c2 = action.c2
                if paper_nmopso:
                    inertia = float(np.clip(inertia * action.inertia, inertia_min, 1.20))
                    delta_scale = float(np.clip(action.mutation_prob, 0.65, 1.35))
                    kappa_scale = float(np.clip(action.kappa_scale, 0.70, 1.35)) * float(
                        np.clip(action.leader_bias, 0.75, 1.30)
                    )
                    velocity_scale = float(np.clip(action.velocity_scale, 0.70, 1.40))
                    region_scale = float(np.clip(action.region_scale, 0.60, 1.40))
                else:
                    inertia = action.inertia
                    mutation_prob = action.mutation_prob
                    leader_bias = action.leader_bias
                rl_actions.append(float(action_idx))
                rl_features.append(features.copy())
                if n_actions > 0:
                    rl_mask_sizes.append(float(len(allowed_actions) if allowed_actions is not None else n_actions))
                else:
                    rl_mask_sizes.append(0.0)

            occupied_cells = max(1, _hypergrid_occupied_count(finite_archive, grid_cells)) if paper_nmopso else 1
            if archive:
                if paper_nmopso and finite_archive.size > 0:
                    finite_idx = np.where(np.all(np.isfinite(_candidate_matrix(archive)), axis=1))[0]
                    picks, occupied_cells = _sample_hypergrid_leaders(
                        _candidate_matrix(archive)[finite_idx],
                        params.population,
                        grid_cells,
                        grid_kappa * kappa_scale,
                    )
                    leader_vectors = np.stack([archive[int(finite_idx[int(k)])].vector for k in picks], axis=0)
                else:
                    leader_vectors = np.stack(
                        [
                            archive[_leader_index(archive, leader_bias, use_grid=use_grid_leader, n_grid=grid_cells)].vector
                            for _ in range(params.population)
                        ],
                        axis=0,
                    )
            else:
                leader_vectors = pbest.copy()

            velocity_limit = velocity_limit_base * velocity_scale
            gpu_t0 = time.perf_counter()
            population, velocity, gpu_backend = _gpu_velocity_update(
                population=population,
                velocity=velocity,
                pbest=pbest,
                leaders=leader_vectors,
                inertia=inertia,
                c1=c1,
                c2=c2,
                lower=lower,
                upper=upper,
                velocity_limit=velocity_limit,
                gpu_mode=params.gpu_mode,
            )
            gpu_update_time_sec += float(time.perf_counter() - gpu_t0)
            if gpu_backend.startswith("torch:"):
                try:
                    import torch  # type: ignore

                    if gpu_backend == "torch:mps":
                        current = float(torch.mps.current_allocated_memory())
                        driver = float(torch.mps.driver_allocated_memory())
                        gpu_peak_bytes = max(gpu_peak_bytes, current, driver)
                    elif gpu_backend == "torch:cuda:0" or gpu_backend.startswith("torch:cuda"):
                        current = float(torch.cuda.max_memory_allocated())
                        gpu_peak_bytes = max(gpu_peak_bytes, current)
                except Exception:
                    pass
            elif gpu_backend.startswith("cupy:"):
                try:
                    import cupy as cp  # type: ignore

                    gpu_peak_bytes = max(gpu_peak_bytes, float(cp.get_default_memory_pool().used_bytes()))
                except Exception:
                    pass

            if paper_nmopso:
                gain = float(np.tanh((delta_cells * delta_scale * region_scale) / max(1.0, float(occupied_cells))))
                particle_idx = int(np.random.randint(0, params.population))
                component_idx = int(np.random.randint(0, dimensions))
                nij = float(np.random.normal(0.0, 1.0))
                population[particle_idx, component_idx] = np.clip(
                    population[particle_idx, component_idx] + nij * gain * pbest[particle_idx, component_idx],
                    lower[component_idx],
                    upper[component_idx],
                )
            else:
                mutation_mask = np.random.rand(params.population) < mutation_prob
                if np.any(mutation_mask):
                    generation_progress = generation / max(1, params.generations)
                    sigma_scale = (
                        0.12 * (1.0 - generation_progress) + 0.02
                        if is_nmopso_family
                        else 0.05
                    )
                    mutation_sigma = sigma_scale * span
                    noise = np.random.normal(0.0, 1.0, size=(int(np.sum(mutation_mask)), dimensions)) * mutation_sigma
                    population[mutation_mask] = np.clip(population[mutation_mask] + noise, lower, upper)

            candidates = _evaluate_population(
                population,
                model,
                fleet_size=fleet_size,
                n_waypoints=n_waypoints,
                representation=representation,
            )
            current_obj = _candidate_matrix(candidates)
            pbest_matrix = np.asarray(pbest_obj, dtype=float)
            better = np.logical_and(
                np.all(current_obj <= pbest_matrix, axis=1),
                np.any(current_obj < pbest_matrix, axis=1),
            )
            ties = np.logical_and(
                np.all(current_obj == pbest_matrix, axis=1),
                np.random.rand(params.population) < 0.5,
            )
            replace = np.logical_or(better, ties)
            if np.any(replace):
                pbest[replace] = population[replace]
                pbest_obj[replace] = current_obj[replace]

            archive = _archive_front(archive + candidates, max_size=archive_size)
            if rl_elite_refine and use_rl and archive:
                progress = generation / max(1, params.generations)
                sigma = (1.0 - progress) * rl_elite_sigma_start + progress * rl_elite_sigma_end
                refined = _elite_refine_candidates(
                    archive=archive,
                    model=model,
                    fleet_size=fleet_size,
                    n_waypoints=n_waypoints,
                    representation=representation,
                    lower=lower,
                    upper=upper,
                    span=span,
                    sigma=float(sigma),
                    top_k=rl_elite_refine_top_k,
                    iters=rl_elite_refine_iters,
                )
                if refined:
                    rl_elite_trials_total += len(refined)
                    archive = _archive_front(archive + refined, max_size=archive_size)
            archive_matrix = _candidate_matrix(archive)
            finite_archive = archive_matrix[np.all(np.isfinite(archive_matrix), axis=1)]
            if finite_archive.size:
                if hv_ref_point is None:
                    hv_ref_point = _fixed_hv_reference(finite_archive)
                hv_after = cal_metric(
                    1,
                    finite_archive,
                    params.problem_index,
                    objective_count,
                    ref_point=hv_ref_point if hv_ref_point is not None else None,
                )
            else:
                hv_after = 0.0
            feasible_after = float(np.mean(np.all(np.isfinite(current_obj), axis=1)))
            conflict_after = _finite_mean(
                [float(candidate.details.get("conflictRate", np.nan)) for candidate in candidates],
                default=0.0,
            )
            diversity_after = float(np.mean(np.std(finite_archive, axis=0))) if finite_archive.size > 0 else 0.0

            if controller is not None and action_idx >= 0:
                delta_hv = float(np.tanh((hv_after - hv_now) / rl_hv_scale))
                delta_feasible = float(np.clip(feasible_after - feasible_ratio, -1.0, 1.0))
                delta_diversity = float(np.tanh((diversity_after - diversity) / rl_div_scale))
                conflict_penalty = float(np.clip(max(0.0, conflict_after) / 0.02, 0.0, 1.0))
                reward = float(
                    np.clip(
                        rl_reward_hv_w * delta_hv
                        + rl_reward_feas_w * delta_feasible
                        + rl_reward_div_w * delta_diversity
                        - rl_reward_conflict_w * conflict_penalty,
                        -1.0,
                        1.0,
                    )
                )
                rl_rewards_immediate.append(float(reward))
                rl_rewards.append(float(reward))
                pending_actions.append(int(action_idx))
                pending_features.append(features.copy())
                pending_rewards.append(float(reward))

                while len(pending_actions) >= rl_reward_n_step:
                    reward_window = np.asarray(list(pending_rewards)[:rl_reward_n_step], dtype=float)
                    discounts = np.power(rl_reward_gamma, np.arange(reward_window.size, dtype=float))
                    delayed_reward = float(np.clip(np.sum(discounts * reward_window), -1.0, 1.0))
                    update_action = int(pending_actions.popleft())
                    update_features = pending_features.popleft()
                    pending_rewards.popleft()
                    control_t0 = time.perf_counter()
                    controller.update(update_action, update_features, delayed_reward)
                    rl_controller_time_sec += float(time.perf_counter() - control_t0)
                    rl_rewards_update.append(delayed_reward)
                if isinstance(controller, TorchBanditController):
                    rl_policy_gpu_peak_bytes = max(rl_policy_gpu_peak_bytes, _torch_device_peak_bytes(controller.device_tag))
                    rl_policy_loss_ema = controller.loss_ema

            if params.compute_metrics:
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = hv_after
                    hv_history[generation - 1, 1] = cal_metric(2, finite_archive, params.problem_index, objective_count) if finite_archive.size else 0.0
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]
            last_hv = hv_after

        if controller is not None:
            while len(pending_actions) > 0:
                reward_window = np.asarray(list(pending_rewards), dtype=float)
                discounts = np.power(rl_reward_gamma, np.arange(reward_window.size, dtype=float))
                delayed_reward = float(np.clip(np.sum(discounts * reward_window), -1.0, 1.0))
                update_action = int(pending_actions.popleft())
                update_features = pending_features.popleft()
                pending_rewards.popleft()
                control_t0 = time.perf_counter()
                controller.update(update_action, update_features, delayed_reward)
                rl_controller_time_sec += float(time.perf_counter() - control_t0)
                rl_rewards_update.append(delayed_reward)
            if isinstance(controller, TorchBanditController):
                rl_policy_gpu_peak_bytes = max(rl_policy_gpu_peak_bytes, _torch_device_peak_bytes(controller.device_tag))
                rl_policy_loss_ema = controller.loss_ema
            if rl_policy_checkpoint and rl_policy_save_resolved:
                try:
                    control_t0 = time.perf_counter()
                    controller.save(rl_policy_checkpoint)
                    rl_controller_time_sec += float(time.perf_counter() - control_t0)
                    rl_policy_saved = True
                except Exception:
                    rl_policy_saved = False

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        final_candidates = archive if archive else candidates
        rl_trace = None
        if controller is not None:
            rl_trace = {
                "action": np.asarray(rl_actions, dtype=float),
                "reward": np.asarray(rl_rewards, dtype=float),
                "feature": np.asarray(rl_features, dtype=float),
                "rewardImmediate": np.asarray(rl_rewards_immediate, dtype=float),
                "rewardUpdate": np.asarray(rl_rewards_update, dtype=float),
                "actionMaskSize": np.asarray(rl_mask_sizes, dtype=float),
            }
        rl_metadata = None
        if controller is not None:
            rl_metadata = {
                "rlPolicyMode": rl_policy_mode,
                "rlPolicyCheckpointPath": rl_policy_checkpoint,
                "rlPolicyLoaded": float(1.0 if rl_policy_loaded else 0.0),
                "rlPolicySaved": float(1.0 if rl_policy_saved else 0.0),
                "rlPolicyFrozen": float(1.0 if rl_policy_frozen else 0.0),
                "rlRewardNStep": float(rl_reward_n_step),
                "rlRewardGamma": float(rl_reward_gamma),
                "rlPhaseGating": float(1.0 if phase_enabled else 0.0),
                "rlRewardHvScale": float(rl_hv_scale),
                "rlRewardDivScale": float(rl_div_scale),
                "rlRewardHvWeight": float(rl_reward_hv_w),
                "rlRewardFeasibleWeight": float(rl_reward_feas_w),
                "rlRewardDiversityWeight": float(rl_reward_div_w),
                "rlRewardConflictWeight": float(rl_reward_conflict_w),
                "rlEliteRefine": float(1.0 if rl_elite_refine else 0.0),
                "rlEliteRefineTopK": float(rl_elite_refine_top_k),
                "rlEliteRefineIters": float(rl_elite_refine_iters),
                "rlEliteRefineSigmaStart": float(rl_elite_sigma_start),
                "rlEliteRefineSigmaEnd": float(rl_elite_sigma_end),
                "rlEliteRefineTrials": float(rl_elite_trials_total),
            }
        _save_multi_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend=gpu_backend,
            gpu_peak_bytes=gpu_peak_bytes,
            rl_trace=rl_trace,
            gpu_update_time_sec=gpu_update_time_sec,
            rl_controller_time_sec=rl_controller_time_sec,
            rl_policy_backend=rl_policy_backend,
            rl_policy_gpu_peak_bytes=rl_policy_gpu_peak_bytes,
            rl_policy_loss_ema=rl_policy_loss_ema,
            rl_metadata=rl_metadata,
        )

        if params.compute_metrics:
            final_matrix = _candidate_matrix(final_candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_matrix, params.problem_index, objective_count),
                    cal_metric(2, final_matrix, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def _sbx_mutation(parents: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    n_parents, n_dims = parents.shape
    if n_parents % 2 == 1:
        parents = np.vstack([parents, parents[np.random.randint(0, n_parents)]])
        n_parents += 1
    half = n_parents // 2
    p1 = parents[:half]
    p2 = parents[half:]
    dis_c = 20.0
    dis_m = 20.0
    pro_m = 1.0 / max(1, n_dims)

    mu = np.random.rand(*p1.shape)
    beta = np.where(
        mu <= 0.5,
        (2.0 * mu) ** (1.0 / (dis_c + 1.0)),
        (2.0 - 2.0 * mu) ** (-1.0 / (dis_c + 1.0)),
    )
    beta *= np.where(np.random.rand(*beta.shape) < 0.5, 1.0, -1.0)
    c1 = (p1 + p2) * 0.5 + beta * (p1 - p2) * 0.5
    c2 = (p1 + p2) * 0.5 - beta * (p1 - p2) * 0.5
    offspring = np.vstack([c1, c2])
    mutation_mask = np.random.rand(*offspring.shape) < pro_m
    mutation = np.random.normal(0.0, 1.0, size=offspring.shape) * 0.05 * (upper - lower)
    offspring = np.where(mutation_mask, offspring + mutation, offspring)
    return np.clip(offspring, lower, upper)


def _run_multi_nsga2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    fleet_size = int(params.fleet_size or model.get("fleetSize", 1))
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)
    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dimensions = int(lower.size)
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in range(1, params.runs + 1):
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue
        run_start = time.perf_counter()
        population = np.random.uniform(lower, upper, size=(params.population, dimensions))
        candidates = _evaluate_population(population, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            obj = _candidate_matrix(candidates)
            front_no, _ = n_d_sort(obj.copy(), None, params.population)
            crowd = crowding_distance(obj, front_no)
            mating = tournament_selection(2, params.population, front_no, -crowd)
            offspring = _sbx_mutation(population[mating], lower, upper)
            off_candidates = _evaluate_population(offspring, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

            merged_vectors = np.vstack([population, offspring])
            merged_candidates = candidates + off_candidates
            merged_obj = _candidate_matrix(merged_candidates)
            merged_front, _ = n_d_sort(merged_obj.copy(), None, params.population)
            merged_crowd = crowding_distance(merged_obj, merged_front)

            selected = []
            for front in np.unique(merged_front[np.isfinite(merged_front)]):
                idx = np.where(merged_front == front)[0]
                if len(selected) + len(idx) <= params.population:
                    selected.extend(idx.tolist())
                else:
                    order = idx[np.argsort(-merged_crowd[idx])]
                    need = params.population - len(selected)
                    selected.extend(order[:need].tolist())
                    break
            selected = np.asarray(selected, dtype=int)
            population = merged_vectors[selected]
            candidates = [merged_candidates[int(i)] for i in selected]

            if params.compute_metrics:
                final_obj = _candidate_matrix(candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, final_obj, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, final_obj, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        _save_multi_artifacts(
            run_dir=run_dir,
            final_candidates=candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            rl_trace=None,
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_obj, params.problem_index, objective_count),
                    cal_metric(2, final_obj, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def _run_multi_nsga3(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    fleet_size = int(params.fleet_size or model.get("fleetSize", 1))
    model["fleetSize"] = float(fleet_size)
    model["separationMin"] = float(params.separation_min)
    model["maxTurnDeg"] = float(params.max_turn_deg)
    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    dimensions = int(lower.size)
    metric_interval = int(params.extra.get("metricInterval", 20))

    reference_method = str(params.extra.get("refPointMethod", "")).strip() or "NBI"
    reference_points, adjusted_population = uniform_point(params.population, objective_count, reference_method)
    population_size = int(adjusted_population)

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in range(1, params.runs + 1):
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue
        run_start = time.perf_counter()
        population = np.random.uniform(lower, upper, size=(population_size, dimensions))
        candidates = _evaluate_population(population, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        obj = _candidate_matrix(candidates)
        zmin = _finite_min(obj)
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            constraints = np.zeros(population_size, dtype=float)
            mating = tournament_selection(2, population_size, constraints)
            offspring = _sbx_mutation(population[mating], lower, upper)
            off_candidates = _evaluate_population(offspring, model, fleet_size=fleet_size, n_waypoints=n_waypoints)

            merged_vectors = np.vstack([population, offspring])
            merged_candidates = candidates + off_candidates
            merged_obj = _candidate_matrix(merged_candidates)
            if merged_obj.size > 0:
                zmin = np.minimum(zmin, _finite_min(merged_obj))

            wrapped = [
                _NSGA3Candidate(objs=merged_obj[idx], cons=0.0, index=idx)
                for idx in range(merged_obj.shape[0])
            ]
            selected_wrapped = environmental_selection_nsga3(
                wrapped,
                population_size,
                reference_points,
                zmin,
                use_constraints=False,
            )
            selected = np.asarray([item.index for item in selected_wrapped], dtype=int)
            if selected.size < population_size:
                remainder = np.setdiff1d(np.arange(merged_vectors.shape[0], dtype=int), selected, assume_unique=False)
                if remainder.size > 0:
                    need = population_size - selected.size
                    fill = remainder[:need]
                    selected = np.hstack([selected, fill])
            elif selected.size > population_size:
                selected = selected[:population_size]

            population = merged_vectors[selected]
            candidates = [merged_candidates[int(idx)] for idx in selected]

            if params.compute_metrics:
                final_obj = _candidate_matrix(candidates)
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, final_obj, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, final_obj, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        _save_multi_artifacts(
            run_dir=run_dir,
            final_candidates=candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend="numpy:cpu",
            gpu_peak_bytes=0.0,
            rl_trace=None,
        )

        if params.compute_metrics:
            final_obj = _candidate_matrix(candidates)
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, final_obj, params.problem_index, objective_count),
                    cal_metric(2, final_obj, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_multi_nmopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_multi_pso(model=model, params=params, label="NMOPSO", use_rl=False)


def run_multi_mopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_multi_pso(model=model, params=params, label="MOPSO", use_rl=False)


def run_multi_rl_nmopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_multi_pso(model=model, params=params, label="RL-NMOPSO", use_rl=True)


def run_multi_nsga2(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_multi_nsga2(model=model, params=params)


def run_multi_nsga3(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_multi_nsga3(model=model, params=params)
