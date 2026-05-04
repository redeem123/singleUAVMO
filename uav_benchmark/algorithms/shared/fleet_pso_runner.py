from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared import pso_defaults as PSD
from uav_benchmark.algorithms.shared.fleet_artifacts import _save_fleet_artifacts
from uav_benchmark.algorithms.shared.fleet_common import (
    _build_bounds,
    _build_navigation_bounds,
    _ensure_fleet_endpoints,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _should_write_final_hv,
    _torch_device_peak_bytes,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir
from uav_benchmark.utils.gpu import resolve_gpu
from uav_benchmark.utils.random import seed_everything

LOGGER = logging.getLogger(__name__)


def _run_fleet_pso(
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
    from uav_benchmark.algorithms.shared.nmopso_engine import NMOPSOEngine, _candidate_matrix

    if use_rl:
        from uav_benchmark.algorithms.rl_config import parse_rl_config  # type: ignore[import-not-found]
        from uav_benchmark.algorithms.rl_pso_adapter import RLPSOAdapter  # type: ignore[import-not-found]

    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 42
    model, fleet_size = _ensure_fleet_endpoints(
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
            model,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
            max_angle_rad=max_angle_rad,
        )
    else:
        lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)

    archive_size = int(params.extra.get("nRep", params.population * (8 if use_rl else 1)))
    metric_interval = int(params.extra.get("metricInterval", PSD.DEFAULT_METRIC_INTERVAL))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    def _evaluate_for_engine(vectors: np.ndarray) -> list[Candidate]:
        return _evaluate_population(
            vectors, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints, representation=representation
        )

    initial_population = None

    # ── Build PSO engine (Layer 1) ─────────────────────────────────
    engine = NMOPSOEngine(
        model=model,
        pop_size=params.population,
        lower=lower,
        upper=upper,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
        representation=representation,
        objective_count=objective_count,
        archive_size=archive_size,
        use_r2_archive=bool(params.extra.get("useR2Archive", use_rl)),
        paper_nmopso=paper_mutation_mode,
        is_nmopso_family=is_nmopso_family,
        grid_cells=int(
            params.extra.get("nGrid", PSD.DEFAULT_GRID_CELLS if paper_mutation_mode else PSD.DEFAULT_GRID_CELLS_GENERIC)
        ),
        grid_kappa=float(params.extra.get("kappa", PSD.DEFAULT_GRID_KAPPA)),
        use_grid_leader=bool(params.extra.get("nmopsoGridLeader", is_nmopso_family)),
        velocity_clamp_ratio=float(
            params.extra.get(
                "velocityClampRatio",
                PSD.DEFAULT_VELOCITY_CLAMP_RATIO if is_nmopso_family else PSD.DEFAULT_VELOCITY_CLAMP_RATIO_MOPSO,
            )
        ),
        gpu_mode=params.gpu_mode,
        evaluate_fn=_evaluate_for_engine,
        initial_population=initial_population,
    )

    # ── RL controller setup (Layer 2) ─────────────────────────────
    controller: Any = None
    cfg = parse_rl_config(params.extra, use_rl=True) if use_rl else None
    rl_policy_backend = "none"
    rl_policy_checkpoint = ""
    rl_policy_mode = cfg.policy_mode if cfg is not None else "train"
    rl_policy_online = rl_policy_mode == "online"
    rl_policy_load = rl_policy_mode in {"warmstart", "freeze"}
    rl_policy_save = rl_policy_mode in {"train", "warmstart"}
    rl_policy_loaded = False
    rl_policy_frozen = rl_policy_mode == "freeze"

    def _build_controller(controller_seed: int) -> tuple[Any, str]:
        assert cfg is not None
        use_gpu_policy = False
        policy_device = "cpu"
        backend_choice = cfg.controller_backend
        if backend_choice in {"auto", "unified"} and cfg.use_gpu_policy:
            gpu_info = resolve_gpu(params.gpu_mode)
            if gpu_info.enabled and gpu_info.backend == "torch":
                use_gpu_policy = True
                policy_device = (
                    "cuda:0" if "cuda" in gpu_info.device else ("mps" if "mps" in gpu_info.device else "cpu")
                )
        if use_gpu_policy or backend_choice == "unified":
            try:
                from uav_benchmark.algorithms.rl_controller import UnifiedController  # type: ignore[import-not-found]

                controller_instance = UnifiedController(
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
                    seed=controller_seed,
                )
                return controller_instance, controller_instance.device_tag
            except (ImportError, RuntimeError, ValueError) as exc:
                LOGGER.warning("Falling back to lightweight RL controller: %s", exc)
                from uav_benchmark.algorithms.rl_controller import FallbackController  # type: ignore[import-not-found]

                controller_instance = FallbackController(warmup_steps=cfg.warmup_steps, seed=controller_seed)
                return controller_instance, controller_instance.device_tag
        from uav_benchmark.algorithms.rl_controller import FallbackController  # type: ignore[import-not-found]

        controller_instance = FallbackController(warmup_steps=cfg.warmup_steps, seed=controller_seed)
        return controller_instance, controller_instance.device_tag

    if use_rl and cfg is not None and not rl_policy_online:
        controller, rl_policy_backend = _build_controller(seed_value)

        checkpoint_raw = cfg.checkpoint_path
        if checkpoint_raw:
            rl_policy_checkpoint = str(Path(checkpoint_raw).expanduser().resolve())
        else:
            from uav_benchmark.algorithms.rl_controller import (  # type: ignore[import-not-found]
                UnifiedController as _UC,
            )

            suffix = ".pt" if isinstance(controller, _UC) else ".npz"
            rl_policy_checkpoint = str(
                (results_path / "_rl_policy" / f"{params.problem_name}_uav{fleet_size}{suffix}").resolve()
            )

        if rl_policy_load and rl_policy_checkpoint:
            try:
                rl_policy_loaded = bool(controller.load(rl_policy_checkpoint, freeze=rl_policy_frozen))
            except (OSError, RuntimeError, ValueError) as exc:
                LOGGER.warning("Could not load RL policy checkpoint %s: %s", rl_policy_checkpoint, exc)
                rl_policy_loaded = False
        if rl_policy_frozen:
            controller.set_frozen(True)

    # ── PSO hyperparameter defaults ────────────────────────────────
    inertia = float(params.extra.get("w", PSD.DEFAULT_INERTIA if is_nmopso_family else PSD.DEFAULT_INERTIA_MOPSO))
    inertia_damp = float(
        params.extra.get(
            "wdamp",
            PSD.DEFAULT_INERTIA_DAMP
            if paper_mutation_mode
            else (PSD.DEFAULT_INERTIA_DAMP_NMOPSO if is_nmopso_family else 1.0),
        )
    )
    inertia_min = float(
        params.extra.get("w_min", PSD.DEFAULT_INERTIA_MIN if is_nmopso_family else PSD.DEFAULT_INERTIA_MIN_MOPSO)
    )
    c1 = float(params.extra.get("c1", PSD.DEFAULT_C1))
    c2 = float(params.extra.get("c2", PSD.DEFAULT_C2))
    mutation_prob = float(
        params.extra.get(
            "mutationProb", PSD.DEFAULT_MUTATION_PROB if is_nmopso_family else PSD.DEFAULT_MUTATION_PROB_MOPSO
        )
    )

    # ── Run loop ──────────────────────────────────────────────────
    run_indices = _resolve_run_indices(params)
    resume_existing_runs = bool(params.extra.get("resumeExistingRuns", True))
    for run_idx in run_indices:
        run_controller = controller
        run_rl_policy_backend = rl_policy_backend
        run_rl_policy_checkpoint = rl_policy_checkpoint
        run_rl_policy_loaded = rl_policy_loaded
        run_rl_policy_saved = False
        run_rl_policy_frozen = rl_policy_frozen
        run_rl_policy_gpu_peak_bytes = 0.0
        run_rl_policy_loss_ema = 0.0
        if use_rl and rl_policy_online and cfg is not None:
            run_controller, run_rl_policy_backend = _build_controller(seed_value + run_idx)
            run_rl_policy_checkpoint = ""
            run_rl_policy_loaded = False
            run_rl_policy_saved = False
            run_rl_policy_frozen = False

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

        # Seed all randomness for this run so that repeated invocations with
        # the same params.seed produce identical outputs. Without this, both
        # NMOPSO and SAC-SMOPSO inherit whatever np.random state the caller
        # happens to leave behind — which leaks across diagnostic runs and
        # makes baselines non-reproducible.
        seed_everything(seed_value + run_idx)

        # Reset engine for new run
        engine.reset()

        # RL adapter for this run
        adapter: RLPSOAdapter | None = None
        if run_controller is not None and cfg is not None:
            archive_init = engine._finite_archive_matrix()
            diversity_ref = max(float(np.mean(np.std(archive_init, axis=0))) if archive_init.size else 1.0, 1e-9)

            adapter = RLPSOAdapter(
                controller=run_controller,
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

        hv_history = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )

        # ── Generation loop ────────────────────────────────────────
        for generation in range(1, params.generations + 1):
            if is_nmopso_family:
                inertia = max(inertia_min, inertia * inertia_damp)

            # Pre-step measurements (for reward)
            finite_archive_pre = engine._finite_archive_matrix()
            hv_before = (
                cal_metric(1, finite_archive_pre, 0, objective_count, ref_point=engine.hv_ref_point)
                if finite_archive_pre.size > 0
                else 0.0
            )
            feasible_before = (
                float(np.mean(np.all(np.isfinite(engine.current_obj), axis=1))) if engine.current_obj.size > 0 else 0.0
            )
            diversity_before = (
                float(np.mean(np.std(finite_archive_pre, axis=0))) if finite_archive_pre.size > 0 else 0.0
            )

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
                c1=gen_c1,
                c2=gen_c2,
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
                controller_for_metrics = run_controller
                if controller_for_metrics is not None and hasattr(controller_for_metrics, "loss_ema"):
                    run_rl_policy_loss_ema = controller_for_metrics.loss_ema
                    run_rl_policy_gpu_peak_bytes = max(
                        run_rl_policy_gpu_peak_bytes, _torch_device_peak_bytes(controller_for_metrics.device_tag)
                    )

            if params.compute_metrics:
                if generation == 1 or generation == params.generations or generation % metric_interval == 0:
                    hv_history[generation - 1, 0] = result.hv
                    hv_history[generation - 1, 1] = (
                        cal_metric(2, engine._finite_archive_matrix(), params.problem_index, objective_count)
                        if engine._finite_archive_matrix().size
                        else 0.0
                    )
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        # ── Post-run: save artifacts ──────────────────────────────
        if adapter is not None and cfg is not None:
            adapter.flush_pending()
            if run_rl_policy_checkpoint and not run_rl_policy_frozen and rl_policy_save and run_controller is not None:
                try:
                    run_controller.save(run_rl_policy_checkpoint)
                    run_rl_policy_saved = True
                except (OSError, RuntimeError, ValueError) as exc:
                    LOGGER.warning("Could not save RL policy checkpoint %s: %s", run_rl_policy_checkpoint, exc)
                    run_rl_policy_saved = False

        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        final_candidates = engine.archive if engine.archive else engine.candidates

        rl_trace = adapter.rl_trace() if adapter is not None else None
        rl_metadata = None
        if adapter is not None and cfg is not None:
            rl_metadata = adapter.rl_metadata()
            rl_metadata["rlPolicyMode"] = rl_policy_mode
            rl_metadata["rlPolicyOnline"] = float(1.0 if rl_policy_online else 0.0)
            rl_metadata["rlPolicyCheckpointPath"] = run_rl_policy_checkpoint
            rl_metadata["rlPolicyLoaded"] = float(1.0 if run_rl_policy_loaded else 0.0)
            rl_metadata["rlPolicySaved"] = float(1.0 if run_rl_policy_saved else 0.0)
            rl_metadata["rlPolicyFrozen"] = float(1.0 if run_rl_policy_frozen else 0.0)
            rl_metadata["rlRepresentation"] = representation
            rl_metadata["rlAttentionEnabled"] = float(1.0 if cfg.attention_enabled else 0.0)
            rl_metadata["rlAttentionTemperature"] = float(cfg.attention_temperature)
            rl_metadata["rlAttentionMode"] = str(cfg.attention_mode)

        _save_fleet_artifacts(
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
            rl_policy_backend=run_rl_policy_backend,
            rl_policy_gpu_peak_bytes=run_rl_policy_gpu_peak_bytes,
            rl_policy_loss_ema=run_rl_policy_loss_ema,
            rl_metadata=rl_metadata,
            run_metadata={
                "algorithmName": str(params.algorithm or label),
                "optimizerBackend": f"{label} native Python fleet optimizer",
                "pythonProblemEvaluation": True,
                "benchmarkObjectiveDuringSearch": True,
                "nativePopulationLoop": True,
                "nativeGenerationLoop": True,
                "representation": str(representation),
            },
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

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_fleet_nmopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_pso(model=model, params=params, label="NMOPSO", use_rl=False)


def run_fleet_mopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_pso(model=model, params=params, label="MOPSO", use_rl=False)


def run_fleet_rl_nmopso(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return _run_fleet_pso(model=model, params=params, label="RL-NMOPSO", use_rl=True)
