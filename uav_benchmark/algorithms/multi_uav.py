"""Multi-UAV path-planning experiment runner (Layer 3).

Orchestrates the NMOPSO engine (Layer 1) and RL adaptive controller (Layer 2).
Also provides NSGA-II and NSGA-III runners for comparison.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import uav_benchmark.algorithms.nmopso_utils as nmopso_utils
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.mission_encoding import decision_size, decision_to_paths
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort, tournament_selection
from uav_benchmark.core.nsga3_ops import environmental_selection_nsga3, uniform_point
from uav_benchmark.io.matlab import load_mat, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.problem_generation.multi_uav_assignments import sample_homogeneous_assignments
from uav_benchmark.utils.gpu import resolve_gpu
from uav_benchmark.algorithms.pso_types import Candidate
import uav_benchmark.algorithms.rl_defaults as RLD


@dataclass(slots=True)
class _NSGA3Candidate:
    objs: np.ndarray
    cons: float
    index: int


# ═══════════════════════════════════════════════════════════════════
# Shared helpers (non-duplicated from engine)
# ═══════════════════════════════════════════════════════════════════

def _build_bounds(model: dict[str, Any], fleet_size: int, n_waypoints: int) -> tuple[np.ndarray, np.ndarray]:
    lower_single = np.array([float(model["xmin"]), float(model["ymin"]), float(model["zmin"])], dtype=float)
    upper_single = np.array([float(model["xmax"]), float(model["ymax"]), float(model["zmax"])], dtype=float)
    total = decision_size(fleet_size, n_waypoints)
    lower = np.tile(lower_single, total // 3)
    upper = np.tile(upper_single, total // 3)
    return lower, upper


def _ensure_multi_endpoints(
    model: dict[str, Any],
    fleet_size: int,
    seed: int,
    separation_min: float,
) -> tuple[dict[str, Any], int]:
    """Ensure starts/goals are available for the requested fleet size."""
    normalized = dict(model)
    starts_raw = normalized.get("starts")
    goals_raw = normalized.get("goals")

    starts: np.ndarray | None = None
    goals: np.ndarray | None = None
    if starts_raw is not None and goals_raw is not None:
        starts = np.asarray(starts_raw, dtype=float)
        goals = np.asarray(goals_raw, dtype=float)
        if starts.ndim == 1:
            starts = starts.reshape(1, -1)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        starts = starts[:, :3]
        goals = goals[:, :3]

    if starts is None or goals is None or starts.shape[0] < fleet_size or goals.shape[0] < fleet_size:
        assignment = sample_homogeneous_assignments(
            terrain=normalized,
            fleet_size=int(fleet_size),
            seed=int(seed),
            separation_min=float(separation_min),
            mission_prefix="runtime",
        )
        starts = np.asarray(assignment.starts, dtype=float)
        goals = np.asarray(assignment.goals, dtype=float)
    else:
        starts = starts[:fleet_size]
        goals = goals[:fleet_size]

    normalized["starts"] = starts
    normalized["goals"] = goals
    normalized["fleetSize"] = float(fleet_size)
    normalized["separationMin"] = float(separation_min)
    return normalized, int(fleet_size)


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


def _resume_run_scores(
    run_dir: Path,
    problem_index: int,
    objective_count: int,
    compute_metrics: bool,
) -> np.ndarray | None:
    popobj_path = run_dir / "final_popobj.mat"
    if not popobj_path.exists():
        return None
    try:
        data = load_mat(popobj_path)
        matrix_raw = data.get("PopObj")
        if matrix_raw is None:
            matrix_raw = data.get("final_popobj", np.zeros((0, 0)))
        matrix = np.asarray(matrix_raw, dtype=float)
        if matrix.size == 0:
            return None
        if compute_metrics:
            hv = cal_metric(1, matrix, problem_index, objective_count)
            spacing = cal_metric(2, matrix, problem_index, objective_count)
            return np.array([hv, spacing], dtype=float)
        return np.zeros(2, dtype=float)
    except Exception:
        return None


def _torch_device_peak_bytes(device_tag: str) -> float:
    if "cuda" not in device_tag:
        return 0.0
    try:
        import torch
        device = device_tag.split(":")[-1] if ":" in device_tag else "cuda:0"
        try:
            return float(torch.cuda.max_memory_allocated(device))
        except Exception:
            return 0.0
    except ImportError:
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# Artifact saving
# ═══════════════════════════════════════════════════════════════════

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
    from uav_benchmark.algorithms.nmopso_engine import _candidate_matrix
    ensure_dir(run_dir)

    # Final objective matrix
    final_matrix = _candidate_matrix(final_candidates)
    save_run_popobj(run_dir / "final_popobj.mat", final_matrix, problem_index, objective_count)

    # Run stats
    run_stats: dict[str, Any] = {
        "runtimeSec": float(runtime_sec),
        "gpuBackend": str(gpu_backend),
        "gpuPeakBytes": float(gpu_peak_bytes),
        "gpuUpdateTimeSec": float(gpu_update_time_sec),
        "rlControllerTimeSec": float(rl_controller_time_sec),
        "rlPolicyBackend": str(rl_policy_backend),
        "rlPolicyGpuPeakBytes": float(rl_policy_gpu_peak_bytes),
        "rlPolicyLossEma": float(rl_policy_loss_ema),
    }
    save_mat(run_dir / "run_stats.mat", run_stats)

    # RL trace
    if rl_trace is not None:
        for key, arr in rl_trace.items():
            save_mat(run_dir / f"rl_{key}.mat", {f"rl_{key}": np.asarray(arr, dtype=float)})
    if rl_metadata is not None:
        save_mat(run_dir / "rl_metadata.mat", rl_metadata)

    # Mission stats (fleet paths, conflict logs)
    fleet_paths: list[list[np.ndarray]] = []
    conflict_values: list[float] = []
    for c in final_candidates:
        paths = c.details.get("paths", [])
        fleet_paths.append(paths)
        conflict_values.append(float(c.details.get("conflictRate", np.nan)))

    try:
        save_mat(run_dir / "mission_stats.mat", {
            "conflictMean": float(np.nanmean(conflict_values)) if conflict_values else 0.0,
            "conflictStd": float(np.nanstd(conflict_values)) if conflict_values else 0.0,
            "nSolutions": float(len(final_candidates)),
        })
    except Exception:
        pass

    # Best paths (for visualization)
    if fleet_paths and final_matrix.size > 0:
        try:
            best_idx = int(np.argmin(np.sum(np.where(np.isfinite(final_matrix), final_matrix, 1e9), axis=1)))
            best_paths = fleet_paths[best_idx]
            if best_paths:
                save_mat(run_dir / "fleet_paths.mat", {
                    f"uav{i + 1}": np.asarray(p, dtype=float) for i, p in enumerate(best_paths)
                })
        except Exception:
            pass

    # Conflict log
    if conflict_values:
        save_mat(run_dir / "conflict_log.mat", {"conflicts": np.array(conflict_values, dtype=float)})


# ═══════════════════════════════════════════════════════════════════
# NMOPSO / RL-NMOPSO runner
# ═══════════════════════════════════════════════════════════════════

def _run_multi_pso(
    model: dict[str, Any],
    params: BenchmarkParams,
    label: str,
    use_rl: bool,
) -> np.ndarray:
    """Run the NMOPSO / RL-NMOPSO algorithm.

    Layer 3 (experiment runner) of the 3-layer architecture.
    PSO mechanics → NMOPSOEngine (Layer 1).
    RL adaptive control → RLPSOAdapter (Layer 2).
    """
    from uav_benchmark.algorithms.nmopso_engine import NMOPSOEngine, _candidate_matrix, _fixed_hv_reference
    from uav_benchmark.algorithms.rl_config import parse_rl_config
    from uav_benchmark.algorithms.rl_pso_adapter import RLPSOAdapter

    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 0
    model, fleet_size = _ensure_multi_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["maxTurnDeg"] = float(params.max_turn_deg)

    is_nmopso_family = label in {"NMOPSO", "RL-NMOPSO"}
    paper_nmopso = bool(params.extra.get("nmopsoPaperMode", True)) and is_nmopso_family

    # Representation selection
    representation = "SC" if paper_nmopso else "cart"
    if label == "RL-NMOPSO":
        repr_override = str(params.extra.get("rlRepresentation", "")).strip().lower()
        if repr_override in {"sc", "spherical"}:
            representation = "SC"
        elif repr_override in {"cart", "cartesian"}:
            representation = "cart"

    paper_mutation_mode = bool(paper_nmopso and representation == "SC")
    if representation == "SC":
        max_angle_rad = float(np.deg2rad(params.max_turn_deg))
        lower, upper = _build_navigation_bounds(
            model, fleet_size=fleet_size, n_waypoints=n_waypoints, max_angle_rad=max_angle_rad,
        )
    else:
        lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)

    archive_size = int(params.extra.get("nRep", params.population * (8 if use_rl else 1)))
    metric_interval = int(params.extra.get("metricInterval", RLD.DEFAULT_METRIC_INTERVAL))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    def _evaluate_for_engine(vectors: np.ndarray) -> list[Candidate]:
        return _evaluate_population(
            vectors, model=model, fleet_size=fleet_size,
            n_waypoints=n_waypoints, representation=representation,
        )

    # ── Build PSO engine (Layer 1) ─────────────────────────────────
    engine = NMOPSOEngine(
        model=model,
        pop_size=params.population,
        lower=lower, upper=upper,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
        representation=representation,
        objective_count=objective_count,
        archive_size=archive_size,
        use_r2_archive=bool(params.extra.get("useR2Archive", use_rl)),
        paper_nmopso=paper_mutation_mode,
        is_nmopso_family=is_nmopso_family,
        grid_cells=int(params.extra.get("nGrid", RLD.DEFAULT_GRID_CELLS if paper_mutation_mode else RLD.DEFAULT_GRID_CELLS_GENERIC)),
        grid_kappa=float(params.extra.get("kappa", RLD.DEFAULT_GRID_KAPPA)),
        use_grid_leader=bool(params.extra.get("nmopsoGridLeader", is_nmopso_family)),
        velocity_clamp_ratio=float(params.extra.get("velocityClampRatio", RLD.DEFAULT_VELOCITY_CLAMP_RATIO if is_nmopso_family else RLD.DEFAULT_VELOCITY_CLAMP_RATIO_MOPSO)),
        gpu_mode=params.gpu_mode,
        evaluate_fn=_evaluate_for_engine,
    )

    # ── RL controller setup (Layer 2) ─────────────────────────────
    controller: Any = None
    rl_policy_backend = "none"
    rl_policy_gpu_peak_bytes = 0.0
    rl_policy_loss_ema = 0.0
    rl_policy_checkpoint = ""
    rl_policy_mode = "train"
    rl_policy_loaded = False
    rl_policy_saved = False
    rl_policy_frozen = False

    if use_rl:
        cfg = parse_rl_config(params.extra, use_rl=True)
        rl_policy_mode = cfg.policy_mode

        # Choose controller backend
        use_gpu_policy = False
        policy_device = "cpu"
        backend_choice = cfg.controller_backend
        if backend_choice in {"auto", "unified"} and cfg.use_gpu_policy:
            gpu_info = resolve_gpu(params.gpu_mode)
            if gpu_info.enabled and gpu_info.backend == "torch":
                use_gpu_policy = True
                policy_device = (
                    "cuda:0" if "cuda" in gpu_info.device
                    else ("mps" if "mps" in gpu_info.device else "cpu")
                )

        if use_gpu_policy or backend_choice == "unified":
            try:
                from uav_benchmark.algorithms.rl_controller import UnifiedController
                controller = UnifiedController(
                    device=policy_device,
                    hidden_dim=cfg.hidden_dim,
                    lr=cfg.lr,
                    warmup_steps=cfg.warmup_steps,
                    attention_mode=cfg.attention_mode,
                    attention_key_dim=cfg.attention_key_dim,
                    attention_lr=cfg.attention_lr,
                    attention_batch_size=cfg.attention_batch_size,
                    attention_train_steps=cfg.attention_train_steps,
                    attention_min_train_size=cfg.attention_min_train_size,
                    attention_replay_capacity=cfg.attention_replay_capacity,
                    seed=seed_value,
                )
                rl_policy_backend = controller.device_tag
            except Exception:
                from uav_benchmark.algorithms.rl_controller import FallbackController
                controller = FallbackController(warmup_steps=cfg.warmup_steps, seed=seed_value)
                rl_policy_backend = controller.device_tag
        else:
            from uav_benchmark.algorithms.rl_controller import FallbackController
            controller = FallbackController(warmup_steps=cfg.warmup_steps, seed=seed_value)
            rl_policy_backend = controller.device_tag

        # Checkpoint path
        checkpoint_raw = cfg.checkpoint_path
        if checkpoint_raw:
            rl_policy_checkpoint = str(Path(checkpoint_raw).expanduser().resolve())
        else:
            from uav_benchmark.algorithms.rl_controller import UnifiedController as _UC
            suffix = ".pt" if isinstance(controller, _UC) else ".npz"
            rl_policy_checkpoint = str(
                (results_path / "_rl_policy" / f"{params.problem_name}_uav{fleet_size}{suffix}").resolve()
            )

        # Load / freeze policy
        rl_policy_load = rl_policy_mode in {"warmstart", "freeze"}
        rl_policy_save = rl_policy_mode in {"train", "warmstart"}
        rl_policy_frozen = rl_policy_mode == "freeze"

        if rl_policy_load and rl_policy_checkpoint:
            try:
                rl_policy_loaded = bool(controller.load(rl_policy_checkpoint, freeze=rl_policy_frozen))
            except Exception:
                rl_policy_loaded = False
        if rl_policy_frozen:
            controller.set_frozen(True)

    # ── PSO hyperparameter defaults ────────────────────────────────
    inertia = float(params.extra.get("w", RLD.DEFAULT_INERTIA if is_nmopso_family else RLD.DEFAULT_INERTIA_MOPSO))
    inertia_damp = float(params.extra.get("wdamp", RLD.DEFAULT_INERTIA_DAMP if paper_mutation_mode else (RLD.DEFAULT_INERTIA_DAMP_NMOPSO if is_nmopso_family else 1.0)))
    inertia_min = float(params.extra.get("w_min", RLD.DEFAULT_INERTIA_MIN if is_nmopso_family else RLD.DEFAULT_INERTIA_MIN_MOPSO))
    c1 = float(params.extra.get("c1", RLD.DEFAULT_C1))
    c2 = float(params.extra.get("c2", RLD.DEFAULT_C2))
    mutation_prob = float(params.extra.get("mutationProb", RLD.DEFAULT_MUTATION_PROB if is_nmopso_family else RLD.DEFAULT_MUTATION_PROB_MOPSO))

    # ── Run loop ──────────────────────────────────────────────────
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in range(1, params.runs + 1):
        run_dir = results_path / f"Run_{run_idx}"
        if resume_existing_runs:
            resume_scores = _resume_run_scores(
                run_dir=run_dir, problem_index=params.problem_index,
                objective_count=objective_count, compute_metrics=params.compute_metrics,
            )
            if resume_scores is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resume_scores
                continue
        run_start = time.perf_counter()

        # Reset engine for new run
        engine.reset()

        # RL adapter for this run
        adapter: RLPSOAdapter | None = None
        if controller is not None:
            archive_init = engine._finite_archive_matrix()
            diversity_ref = max(float(np.mean(np.std(archive_init, axis=0))) if archive_init.size else 1.0, 1e-9)

            cfg = parse_rl_config(params.extra, use_rl=True)
            adapter = RLPSOAdapter(
                controller=controller,
                engine=engine,
                total_generations=params.generations,
                hv_scale=cfg.reward_hv_scale,
                div_scale=diversity_ref,
                reward_hv_w=cfg.reward_hv_weight,
                reward_feas_w=cfg.reward_feasible_weight,
                reward_div_w=cfg.reward_diversity_weight,
                reward_aux_cost_w=cfg.reward_aux_cost_weight,
                aux_eval_budget_factor=cfg.aux_eval_budget_factor,
                aux_eval_budget_start_factor=cfg.aux_eval_budget_start_factor,
                aux_eval_budget_end_factor=cfg.aux_eval_budget_end_factor,
                operator_trigger_prob_start=cfg.operator_trigger_prob_start,
                operator_trigger_prob_end=cfg.operator_trigger_prob_end,
                operator_stagnation_boost=cfg.operator_stagnation_boost,
                operator_stagnation_threshold=cfg.operator_stagnation_threshold,
                surrogate_prefilter_enabled=cfg.surrogate_prefilter_enabled,
                surrogate_prefilter_ratio=cfg.surrogate_prefilter_ratio,
                surrogate_prefilter_min_candidates=cfg.surrogate_prefilter_min_candidates,
                surrogate_prefilter_k=cfg.surrogate_prefilter_k,
                attention_enabled=cfg.attention_enabled,
                attention_temperature=cfg.attention_temperature,
                seed=seed_value + run_idx,
            )

        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        # ── Generation loop ────────────────────────────────────────
        for generation in range(1, params.generations + 1):
            if is_nmopso_family:
                inertia = max(inertia_min, inertia * inertia_damp)

            # Pre-step measurements (for reward)
            finite_archive_pre = engine._finite_archive_matrix()
            hv_before = cal_metric(1, finite_archive_pre, 0, objective_count, ref_point=engine.hv_ref_point) if finite_archive_pre.size > 0 else 0.0
            feasible_before = float(np.mean(np.all(np.isfinite(engine.current_obj), axis=1))) if engine.current_obj.size > 0 else 0.0
            diversity_before = float(np.mean(np.std(finite_archive_pre, axis=0))) if finite_archive_pre.size > 0 else 0.0

            # 1. RL observes and acts (or use defaults)
            gen_inertia = inertia
            gen_c1, gen_c2 = c1, c2
            gen_mutation_prob = mutation_prob
            action = None

            if adapter is not None:
                action = adapter.observe_and_act(
                    generation=generation,
                    inertia=inertia,
                    inertia_min=inertia_min,
                    diversity_ref=max(diversity_ref, 1e-9),
                )
                gen_inertia = action.inertia
                gen_c1, gen_c2 = action.c1, action.c2
                gen_mutation_prob = action.mutation_prob
                if paper_mutation_mode:
                    inertia = gen_inertia  # persist

            # 2. PSO step
            result = engine.step(
                inertia=gen_inertia,
                c1=gen_c1, c2=gen_c2,
                mutation_prob=gen_mutation_prob,
                attention_weights=action.attention_weights if action is not None else None,
            )

            # 3. Execute operator
            if adapter is not None and action is not None:
                adapter.execute_operator(action, generation)

            # 4. Post-step: reward computation
            if adapter is not None:
                adapter.post_step(
                    hv_before=hv_before,
                    hv_after=result.hv,
                    feasible_before=feasible_before,
                    feasible_after=result.feasible_ratio,
                    diversity_before=diversity_before,
                    diversity_after=result.diversity,
                )
                if hasattr(controller, 'loss_ema'):
                    rl_policy_loss_ema = controller.loss_ema
                    rl_policy_gpu_peak_bytes = max(rl_policy_gpu_peak_bytes, _torch_device_peak_bytes(controller.device_tag))

            if params.compute_metrics:
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = result.hv
                    hv_history[generation - 1, 1] = cal_metric(2, engine._finite_archive_matrix(), params.problem_index, objective_count) if engine._finite_archive_matrix().size else 0.0
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        # ── Post-run: save artifacts ──────────────────────────────
        if adapter is not None:
            adapter.flush_pending()
            if rl_policy_checkpoint and not rl_policy_frozen:
                try:
                    controller.save(rl_policy_checkpoint)
                    rl_policy_saved = True
                except Exception:
                    rl_policy_saved = False

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        final_candidates = engine.archive if engine.archive else engine.candidates

        rl_trace = adapter.rl_trace() if adapter is not None else None
        rl_metadata = None
        if adapter is not None:
            rl_metadata = adapter.rl_metadata()
            rl_metadata["rlPolicyMode"] = rl_policy_mode
            rl_metadata["rlPolicyCheckpointPath"] = rl_policy_checkpoint
            rl_metadata["rlPolicyLoaded"] = float(1.0 if rl_policy_loaded else 0.0)
            rl_metadata["rlPolicySaved"] = float(1.0 if rl_policy_saved else 0.0)
            rl_metadata["rlPolicyFrozen"] = float(1.0 if rl_policy_frozen else 0.0)
            rl_metadata["rlRepresentation"] = representation
            rl_metadata["rlAttentionEnabled"] = float(1.0 if cfg.attention_enabled else 0.0)
            rl_metadata["rlAttentionTemperature"] = float(cfg.attention_temperature)
            rl_metadata["rlAttentionMode"] = str(cfg.attention_mode)

        _save_multi_artifacts(
            run_dir=run_dir,
            final_candidates=final_candidates,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend=engine.gpu_backend,
            gpu_peak_bytes=engine.gpu_peak_bytes,
            rl_trace=rl_trace,
            gpu_update_time_sec=engine.gpu_update_time_sec,
            rl_controller_time_sec=adapter.rl_controller_time_sec if adapter else 0.0,
            rl_policy_backend=rl_policy_backend,
            rl_policy_gpu_peak_bytes=rl_policy_gpu_peak_bytes,
            rl_policy_loss_ema=rl_policy_loss_ema,
            rl_metadata=rl_metadata,
        )

        if params.compute_metrics:
            final_matrix = _candidate_matrix(final_candidates)
            run_scores[run_idx - 1] = np.array([
                cal_metric(1, final_matrix, params.problem_index, objective_count),
                cal_metric(2, final_matrix, params.problem_index, objective_count),
            ], dtype=float)

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


# ═══════════════════════════════════════════════════════════════════
# NSGA-II runner
# ═══════════════════════════════════════════════════════════════════

def _sbx_mutation(parents: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    n_parents, n_dims = parents.shape
    if n_parents % 2 == 1:
        parents = np.vstack([parents, parents[np.random.randint(0, n_parents)]])
        n_parents += 1
    half = n_parents // 2
    p1 = parents[:half]
    p2 = parents[half:]
    dis_c = 20.0
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
    from uav_benchmark.algorithms.nmopso_engine import _candidate_matrix, _finite_min

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
                run_dir=run_dir, problem_index=params.problem_index,
                objective_count=objective_count, compute_metrics=params.compute_metrics,
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
            mating = tournament_selection(params.population, front_no, -crowd, k_tournament=2)
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
            run_scores[run_idx - 1] = np.array([
                cal_metric(1, final_obj, params.problem_index, objective_count),
                cal_metric(2, final_obj, params.problem_index, objective_count),
            ], dtype=float)

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


# ═══════════════════════════════════════════════════════════════════
# NSGA-III runner
# ═══════════════════════════════════════════════════════════════════

def _run_multi_nsga3(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    from uav_benchmark.algorithms.nmopso_engine import _candidate_matrix, _finite_min

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
                run_dir=run_dir, problem_index=params.problem_index,
                objective_count=objective_count, compute_metrics=params.compute_metrics,
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
            mating = tournament_selection(population_size, constraints, k_tournament=2)
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
                wrapped, population_size, reference_points, zmin, use_constraints=False,
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
            run_scores[run_idx - 1] = np.array([
                cal_metric(1, final_obj, params.problem_index, objective_count),
                cal_metric(2, final_obj, params.problem_index, objective_count),
            ], dtype=float)

    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

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
