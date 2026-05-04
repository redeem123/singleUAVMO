"""MOGWO family runner with an exact three-component architecture.

Components:
- ``repair_restart`` — feedback-tracked relaxed-boundary archive that keeps
  near-feasible elites alive and injects them into leader and guide selection.
- ``topology_relay`` — topology-assisted third-leader relay drawn from an
  assisting pool rather than continuous leader reweighting.
- ``dual_archive_explorer`` — competitive dual archives with adaptive
  archive-guided Cauchy-SBX exploration.

Variants:
- ``full`` / ``a2`` — all three components enabled.
- ``no_attention`` — full variant with adaptive attention removed.
- ``standard_gwo`` — plain GWO baseline with all three components disabled.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from uav_benchmark.algorithms.mogwo.archive import (
    _archive_objective_context,
    _attention_context_from_candidates,
    _attention_leader_context,
    _objective_diversity_level,
    _select_leaders,
    _update_archive,
)
from uav_benchmark.algorithms.mogwo.components import (
    _adaptive_archive_explorer,
    _adaptive_explorer_ratio,
    _feedback_relaxation_threshold,
    _relaxed_archive_infusion_ratio,
    _selective_feasibility_repair_restart,
    _terrain_seed_population,
    _topology_relay_guides,
    _update_relaxed_constraint_archive,
)
from uav_benchmark.algorithms.mogwo.constants import (
    _ATTN_STEP_MAX,
    _ATTN_STEP_MIN,
    _ATTN_TAU_OBJ,
)
from uav_benchmark.algorithms.mogwo.engine import QGWO_Engine
from uav_benchmark.algorithms.shared.fleet_runner import (
    _build_bounds,
    _ensure_fleet_endpoints,
    _evaluate_population,
    _resolve_run_indices,
    _resume_run_scores,
    _save_fleet_artifacts,
    _should_write_final_hv,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import n_d_sort
from uav_benchmark.io.matlab import save_mat
from uav_benchmark.io.results import ensure_dir

__all__ = [
    "QGWO_Engine",
    "_build_bounds",
    "_feedback_relaxation_threshold",
    "_selective_feasibility_repair_restart",
    "_terrain_seed_population",
    "_update_archive",
    "_update_relaxed_constraint_archive",
    "run_fleet_mogwo",
    "run_fleet_mogwo_no_attention",
    "run_fleet_mogwo_standard_gwo",
]


def _resolve_variant(raw: Any) -> str:
    key = str(raw).strip().lower()
    if key in {"", "full", "a2", "a2mogwo", "a2-mogwo"}:
        return "full"
    if key in {"no_attention", "no-attention", "noattention", "loo_attention", "loo-attention"}:
        return "no_attention"
    if key in {"standard_gwo", "standard-gwo", "gwo", "standard"}:
        return "standard_gwo"
    return "full"


def _algorithm_name_for_variant(variant: str) -> str:
    if variant == "no_attention":
        return "MOGWO-NO-ATTENTION"
    if variant == "standard_gwo":
        return "MOGWO-STANDARD-GWO"
    return "MOGWO"


def _apply_variant(params: BenchmarkParams, *, variant: str | None = None) -> BenchmarkParams:
    merged_extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    if variant is not None:
        merged_extra["mogwoVariant"] = variant
    return replace(params, extra=merged_extra)


def _attention_inputs(
    candidates: list[Candidate],
    population: int,
    enabled: bool,
) -> tuple[np.ndarray, float]:
    if enabled:
        return _attention_context_from_candidates(candidates)
    return np.ones((population, 4), dtype=float), 1.0


def _explorer_offspring_count(population: int, ratio: float) -> int:
    if ratio <= 1e-12:
        return 0
    if population <= 1:
        return population
    return int(np.clip(round(population * ratio), 1, population - 1))


def _combined_archive_for_guidance(
    *,
    archive: list[Candidate],
    archive_unconstrained: list[Candidate],
    archive_relaxed: list[Candidate],
    archive_size: int,
    relaxation_infusion: float,
    use_constraint_relaxation: bool,
    use_dual_archive_explorer: bool,
) -> list[Candidate]:
    combined_archive = list(archive)
    if use_constraint_relaxation and archive_relaxed:
        relaxed_limit = int(
            np.clip(
                round(archive_size * relaxation_infusion),
                1,
                len(archive_relaxed),
            )
        )
        combined_archive.extend(archive_relaxed[:relaxed_limit])
    if use_dual_archive_explorer:
        combined_archive.extend(archive_unconstrained[: max(1, len(archive_unconstrained) // 3)])
    return combined_archive


def _refresh_engine_guidance(
    *,
    engine: QGWO_Engine,
    candidates: list[Candidate],
    wolf_objectives: np.ndarray,
    feasible_ratio: float,
    attention_context_enabled: bool,
    archive: list[Candidate],
    archive_unconstrained: list[Candidate],
    archive_relaxed: list[Candidate],
    archive_size: int,
    grid_divisions: int,
    use_advanced_archive: bool,
    use_mean_selection: bool,
    use_constraint_relaxation: bool,
    use_dual_archive_explorer: bool,
    relaxation_eps: float,
    relaxation_infusion: float,
    archive_diversity: float,
    model: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    if not archive:
        return

    combined_archive = _combined_archive_for_guidance(
        archive=archive,
        archive_unconstrained=archive_unconstrained,
        archive_relaxed=archive_relaxed,
        archive_size=archive_size,
        relaxation_infusion=relaxation_infusion,
        use_constraint_relaxation=use_constraint_relaxation,
        use_dual_archive_explorer=use_dual_archive_explorer,
    )
    selected_leaders, selected_indices, selected_occ = _select_leaders(
        combined_archive,
        grid_divisions,
        use_advanced_archive=use_advanced_archive,
        use_mean_selection=use_mean_selection,
        model=model,
        relaxation_eps=relaxation_eps if use_constraint_relaxation else 0.0,
    )
    engine.leaders = selected_leaders
    if not attention_context_enabled:
        return

    leader_obj_ctx = _attention_leader_context(combined_archive, selected_indices)
    relay_guides, relay_activation, relay_pool_feasible_share = _topology_relay_guides(
        pack_positions=engine.positions,
        candidates=candidates,
        archive=archive,
        archive_unconstrained=archive_unconstrained,
        relaxation_archive=archive_relaxed,
        model=model,
        lower=lower,
        upper=upper,
        feasible_ratio=feasible_ratio,
        diversity_level=archive_diversity,
        relaxation_eps=relaxation_eps if use_constraint_relaxation else 0.0,
    )
    engine.set_attention_context(
        wolf_objectives=wolf_objectives,
        feasibility_pressure=float(np.clip(1.0 - feasible_ratio, 0.0, 1.0)),
        leader_objectives=leader_obj_ctx,
        diversity_level=archive_diversity,
        leader_occupancy=selected_occ,
        relay_guides=relay_guides,
        relay_activation=relay_activation,
        relay_pool_feasible_share=relay_pool_feasible_share,
    )


def run_fleet_mogwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    """MOGWO family runner with attention fusion and objective-grid archive."""
    objective_count = 4
    model = dict(model)
    n_waypoints = int(model.get("n", 10))
    requested_fleet = max(1, int(params.fleet_size or model.get("fleetSize", 1)))
    seed_value = int(params.seed) if params.seed is not None else 0

    model, fleet_size = _ensure_fleet_endpoints(
        model=model,
        fleet_size=requested_fleet,
        seed=seed_value + requested_fleet,
        separation_min=float(params.separation_min),
    )
    model["maxTurnDeg"] = float(params.max_turn_deg)
    model["is_rl"] = False
    model["hardCollisionConstraint"] = False

    lower, upper = _build_bounds(model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    variant = _resolve_variant(params.extra.get("mogwoVariant", "full"))
    algorithm_name = _algorithm_name_for_variant(variant)

    # ── Exact 3-component flags (ablation control) ──────────────────
    is_standard = variant == "standard_gwo"
    default_component = not is_standard

    use_repair_restart = bool(
        params.extra.get(
            "mogwoUseRepairRestart",
            params.extra.get("mogwoUseTerrainSeeding", default_component),
        )
    )
    use_constraint_relaxation = bool(use_repair_restart)
    use_adaptive_attention = bool(
        params.extra.get(
            "mogwoUseAdaptiveAttention",
            params.extra.get("mogwoUseAttention", variant == "full"),
        )
    )
    use_dual_archive_explorer = bool(
        params.extra.get(
            "mogwoUseDualArchiveExplorer",
            params.extra.get("mogwoUseFeasibilityRecomb", default_component),
        )
    )
    use_advanced_archive = bool(params.extra.get("mogwoUseAdvancedArchive", not is_standard))
    use_mean_selection = bool(params.extra.get("mogwoUseMeanSelection", is_standard))

    if variant == "no_attention":
        use_adaptive_attention = False
    if is_standard:
        use_repair_restart = False
        use_constraint_relaxation = False
        use_adaptive_attention = False
        use_dual_archive_explorer = False

    gpu_backend = "numpy:cpu"

    archive_size = int(params.extra.get("nRep", params.population))
    grid_divisions = int(params.extra.get("nGrid", 10))
    metric_interval = int(params.extra.get("metricInterval", 20))

    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

    run_indices = _resolve_run_indices(params)
    resume_existing = bool(params.extra.get("resumeExistingRuns", True))

    for run_idx in run_indices:
        run_start = time.perf_counter()
        run_dir = results_path / f"Run_{run_idx}"

        if resume_existing:
            resumed = _resume_run_scores(
                run_dir=run_dir,
                problem_index=params.problem_index,
                objective_count=objective_count,
                compute_metrics=params.compute_metrics,
            )
            if resumed is not None:
                if params.compute_metrics:
                    run_scores[run_idx - 1] = resumed
                continue

        engine = QGWO_Engine(
            lower,
            upper,
            params.population,
            use_attention=use_adaptive_attention,
            use_feasibility_pressure=use_adaptive_attention,
            use_diversity_feedback=use_adaptive_attention,
            use_step_limiter=use_adaptive_attention,
            use_attention_guard=use_adaptive_attention,
            attn_tau_obj=float(params.extra.get("mogwoAttnTauObj", _ATTN_TAU_OBJ)),
            attn_step_min=float(params.extra.get("mogwoAttnStepMin", _ATTN_STEP_MIN)),
            attn_step_max=float(params.extra.get("mogwoAttnStepMax", _ATTN_STEP_MAX)),
            # Surgical Flags
            use_attn_feas_boost=bool(params.extra.get("mogwoUseAttnFeasBoost", use_adaptive_attention)),
            use_attn_div_boost=bool(params.extra.get("mogwoUseAttnDivBoost", use_adaptive_attention)),
            use_step_feas_driver=bool(params.extra.get("mogwoUseStepFeasDriver", use_adaptive_attention)),
            use_step_div_driver=bool(params.extra.get("mogwoUseStepDivDriver", use_adaptive_attention)),
        )
        attention_context_enabled = bool(use_adaptive_attention and hasattr(engine, "set_attention_context"))
        hv_hist = (
            np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
        )
        attention_entropy_sum = 0.0
        attention_lambda_feas_sum = 0.0
        attention_lambda_div_sum = 0.0
        attention_diversity_sum = 0.0
        attention_step_scale_sum = 0.0
        attention_guard_active_sum = 0.0
        attention_stage_activation_sum = 0.0
        relay_pool_feasible_share_sum = 0.0
        attention_steps = 0
        explorer_ratio_sum = 0.0
        explorer_steps = 0
        repair_attempt_sum = 0.0
        repair_accept_sum = 0.0
        repair_restart_sum = 0.0
        relaxation_eps_sum = 0.0
        relaxation_infusion_sum = 0.0
        relaxation_steps = 0

        # Initial evaluation
        init_cands = _evaluate_population(engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        init_obj_ctx, init_feasible_ratio = _attention_inputs(
            init_cands,
            params.population,
            attention_context_enabled,
        )
        current_feasible_ratio = float(np.clip(init_feasible_ratio, 0.0, 1.0))
        previous_feasible_ratio = current_feasible_ratio

        archive: list[Candidate] = []
        archive_unconstrained: list[Candidate] = []
        archive_relaxed: list[Candidate] = []

        # Bootstrap archive
        archive = _update_archive(
            [],
            init_cands,
            model,
            archive_size,
            grid_divisions,
            use_constraints=True,
        )
        if use_dual_archive_explorer:
            archive_unconstrained = _update_archive(
                [],
                init_cands,
                model,
                archive_size,
                grid_divisions,
                use_constraints=False,
            )
        relaxation_eps = 0.0
        relaxation_infusion = 0.0
        if use_constraint_relaxation:
            relaxation_eps = _feedback_relaxation_threshold(
                candidates=init_cands,
                archive_unconstrained=archive_unconstrained,
                model=model,
                feasible_ratio=current_feasible_ratio,
                previous_feasible_ratio=previous_feasible_ratio,
                generation=0,
                max_generations=params.generations,
            )
            archive_relaxed = _update_relaxed_constraint_archive(
                [],
                init_cands,
                archive,
                archive_unconstrained,
                model,
                archive_size,
                grid_divisions,
                relaxation_eps,
            )
            relaxation_infusion = _relaxed_archive_infusion_ratio(current_feasible_ratio, relaxation_eps)
            relaxation_eps_sum += float(relaxation_eps)
            relaxation_infusion_sum += float(relaxation_infusion)
            relaxation_steps += 1

        # Set initial leaders from archive
        archive_diversity = _objective_diversity_level(_archive_objective_context(archive))
        _refresh_engine_guidance(
            engine=engine,
            candidates=init_cands,
            wolf_objectives=init_obj_ctx,
            feasible_ratio=current_feasible_ratio,
            attention_context_enabled=attention_context_enabled,
            archive=archive,
            archive_unconstrained=archive_unconstrained,
            archive_relaxed=archive_relaxed,
            archive_size=archive_size,
            grid_divisions=grid_divisions,
            use_advanced_archive=use_advanced_archive,
            use_mean_selection=use_mean_selection,
            use_constraint_relaxation=use_constraint_relaxation,
            use_dual_archive_explorer=use_dual_archive_explorer,
            relaxation_eps=relaxation_eps,
            relaxation_infusion=relaxation_infusion,
            archive_diversity=archive_diversity,
            model=model,
            lower=lower,
            upper=upper,
        )

        # ── Generation Loop ───────────────────────────────────────────
        for gen in range(1, params.generations + 1):
            pack_before_step = engine.positions.copy()
            gwo_positions = engine.step(gen, params.generations)

            if use_dual_archive_explorer:
                explorer_ratio = _adaptive_explorer_ratio(current_feasible_ratio, archive_diversity)
                explorer_count = _explorer_offspring_count(params.population, explorer_ratio)
                exploit_count = max(0, params.population - explorer_count)
                explorer_positions = _adaptive_archive_explorer(
                    pack_positions=pack_before_step,
                    leaders=engine.leaders,
                    convergence_archive=archive,
                    diversity_archive=archive_unconstrained,
                    relaxation_archive=archive_relaxed,
                    lower=lower,
                    upper=upper,
                    offspring_count=explorer_count,
                    feasible_ratio=current_feasible_ratio,
                    diversity_level=archive_diversity,
                    relaxation_share=relaxation_infusion if use_constraint_relaxation else 0.0,
                )
                new_positions = np.zeros_like(engine.positions)
                if exploit_count > 0:
                    new_positions[:exploit_count] = gwo_positions[:exploit_count]
                if explorer_count > 0:
                    new_positions[exploit_count:] = explorer_positions[:explorer_count]
                explorer_ratio_sum += float(explorer_count / max(1, params.population))
                explorer_steps += 1
            else:
                new_positions = gwo_positions.copy()

            # Evaluate new population
            new_cands = _evaluate_population(new_positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            new_obj_ctx, new_feasible_ratio = _attention_inputs(
                new_cands,
                params.population,
                attention_context_enabled,
            )
            current_feasible_ratio = float(np.clip(new_feasible_ratio, 0.0, 1.0))
            # The evaluated blended population is the next pack state.
            engine.positions = np.clip(new_positions.copy(), lower, upper)

            if use_adaptive_attention:
                stats = getattr(engine, "last_attention_stats", None)
                if isinstance(stats, dict):
                    attention_entropy_sum += float(stats.get("entropy_mean", 0.0))
                    attention_lambda_feas_sum += float(stats.get("lambda_feasibility", stats.get("lambda_safe", 0.0)))
                    attention_lambda_div_sum += float(stats.get("lambda_diversity", 0.0))
                    attention_diversity_sum += float(stats.get("diversity_level", 0.0))
                    attention_step_scale_sum += float(stats.get("step_scale", 1.0))
                    attention_guard_active_sum += float(stats.get("attention_guard_active", 0.0))
                    attention_stage_activation_sum += float(stats.get("stage_activation", 0.0))
                    relay_pool_feasible_share_sum += float(stats.get("relay_pool_feasible_share", 0.0))
                    attention_steps += 1

            # Archive update
            archive = _update_archive(
                archive,
                new_cands,
                model,
                archive_size,
                grid_divisions,
                use_constraints=True,
            )
            if use_dual_archive_explorer:
                archive_unconstrained = _update_archive(
                    archive_unconstrained,
                    new_cands,
                    model,
                    archive_size,
                    grid_divisions,
                    use_constraints=False,
                )
            else:
                archive_unconstrained = []

            if use_constraint_relaxation:
                relaxation_eps = _feedback_relaxation_threshold(
                    candidates=new_cands,
                    archive_unconstrained=archive_unconstrained,
                    model=model,
                    feasible_ratio=current_feasible_ratio,
                    previous_feasible_ratio=previous_feasible_ratio,
                    generation=gen,
                    max_generations=params.generations,
                )
                archive_relaxed = _update_relaxed_constraint_archive(
                    archive_relaxed,
                    new_cands,
                    archive,
                    archive_unconstrained,
                    model,
                    archive_size,
                    grid_divisions,
                    relaxation_eps,
                )
                relaxation_infusion = _relaxed_archive_infusion_ratio(current_feasible_ratio, relaxation_eps)
                relaxation_eps_sum += float(relaxation_eps)
                relaxation_infusion_sum += float(relaxation_infusion)
                relaxation_steps += 1
            else:
                relaxation_eps = 0.0
                relaxation_infusion = 0.0
                archive_relaxed = []

            archive_diversity = _objective_diversity_level(_archive_objective_context(archive))
            _refresh_engine_guidance(
                engine=engine,
                candidates=new_cands,
                wolf_objectives=new_obj_ctx,
                feasible_ratio=current_feasible_ratio,
                attention_context_enabled=attention_context_enabled,
                archive=archive,
                archive_unconstrained=archive_unconstrained,
                archive_relaxed=archive_relaxed,
                archive_size=archive_size,
                grid_divisions=grid_divisions,
                use_advanced_archive=use_advanced_archive,
                use_mean_selection=use_mean_selection,
                use_constraint_relaxation=use_constraint_relaxation,
                use_dual_archive_explorer=use_dual_archive_explorer,
                relaxation_eps=relaxation_eps,
                relaxation_infusion=relaxation_infusion,
                archive_diversity=archive_diversity,
                model=model,
                lower=lower,
                upper=upper,
            )
            previous_feasible_ratio = current_feasible_ratio

            # Metrics
            if params.compute_metrics and hv_hist.shape[0] > 0:
                if gen == 1 or gen == params.generations or gen % metric_interval == 0:
                    if archive:
                        arc_obj = np.stack([c.objective for c in archive])
                        hv_hist[gen - 1, 0] = cal_metric(1, arc_obj, params.problem_index, objective_count)
                        hv_hist[gen - 1, 1] = cal_metric(2, arc_obj, params.problem_index, objective_count)
                elif gen > 1:
                    hv_hist[gen - 1] = hv_hist[gen - 2]

        # ── Finalize ──────────────────────────────────────────────────
        ensure_dir(run_dir)
        if params.compute_metrics and hv_hist.shape[0] > 0:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_hist})

        if not archive:
            # Pathological fallback
            last_cands = _evaluate_population(engine.positions, model, fleet_size=fleet_size, n_waypoints=n_waypoints)
            last_obj = np.stack([c.objective for c in last_cands]) if last_cands else np.zeros((0, objective_count))
            if last_obj.size > 0:
                fronts, _ = n_d_sort(last_obj.copy(), None, last_obj.shape[0])
                selected = np.where(fronts == 1)[0]
                if selected.size == 0:
                    selected = np.arange(min(archive_size, last_obj.shape[0]), dtype=int)
                archive = [last_cands[i] for i in selected[:archive_size]]
            else:
                archive = []

        _save_fleet_artifacts(
            run_dir=run_dir,
            final_candidates=archive,
            problem_index=params.problem_index,
            objective_count=objective_count,
            runtime_sec=float(time.perf_counter() - run_start),
            gpu_backend=gpu_backend,
            gpu_peak_bytes=0.0,
            run_metadata={
                "algorithmName": algorithm_name,
                "representation": "cart",
                "mogwoVariant": str(variant),
                "requestedPopulation": float(params.population),
                "effectivePopulation": float(params.population),
                "archiveSize": float(archive_size),
                "mogwoAttentionEntropyMean": float(attention_entropy_sum / max(1, attention_steps)),
                "mogwoLambdaFeasibilityMean": float(attention_lambda_feas_sum / max(1, attention_steps)),
                "mogwoLambdaDiversityMean": float(attention_lambda_div_sum / max(1, attention_steps)),
                "mogwoDiversityLevelMean": float(attention_diversity_sum / max(1, attention_steps)),
                "mogwoStepScaleMean": float(attention_step_scale_sum / max(1, attention_steps)),
                "mogwoAttentionGuardActiveMean": float(attention_guard_active_sum / max(1, attention_steps)),
                "mogwoAttentionStageActivationMean": float(attention_stage_activation_sum / max(1, attention_steps)),
                "mogwoRelayActivationMean": float(attention_stage_activation_sum / max(1, attention_steps)),
                "mogwoRelayPoolFeasibleShareMean": float(relay_pool_feasible_share_sum / max(1, attention_steps)),
                "mogwoRepairAttemptSum": float(repair_attempt_sum),
                "mogwoRepairAcceptSum": float(repair_accept_sum),
                "mogwoRepairRestartSum": float(repair_restart_sum),
                "mogwoRelaxationEpsilonMean": float(relaxation_eps_sum / max(1, relaxation_steps)),
                "mogwoRelaxedArchiveInfusionMean": float(relaxation_infusion_sum / max(1, relaxation_steps)),
                "mogwoExplorerRatioMean": float(explorer_ratio_sum / max(1, explorer_steps)),
                "mogwoComponentRepairRestart": float(1.0 if use_constraint_relaxation else 0.0),
                "mogwoComponentFeedbackRelaxation": float(1.0 if use_constraint_relaxation else 0.0),
                "mogwoComponentBoundaryPump": 0.0,
                "mogwoComponentTopologyRelay": float(1.0 if use_adaptive_attention else 0.0),
                "mogwoComponentAdaptiveAttention": float(1.0 if use_adaptive_attention else 0.0),
                "mogwoComponentDualArchiveExplorer": float(1.0 if use_dual_archive_explorer else 0.0),
                # Backward-compatible fields used by older summaries.
                "mogwoTerrainSeedFraction": 0.0,
                "mogwoTerrainReseedEvents": float(repair_restart_sum),
                "mogwoComponentTerrainSeeding": 0.0,
                "mogwoUseRepairRestart": float(1.0 if use_constraint_relaxation else 0.0),
                "mogwoUseDiversityFeedback": float(1.0 if use_adaptive_attention else 0.0),
                "mogwoUseStepLimiter": float(1.0 if use_adaptive_attention else 0.0),
                "mogwoUseFeasibilityRecomb": float(1.0 if use_dual_archive_explorer else 0.0),
                "mogwoUseAttentionGuard": float(1.0 if use_adaptive_attention else 0.0),
                "mogwoUseFeasibilityPressure": float(1.0 if use_adaptive_attention else 0.0),
                "mogwoUseAdvancedArchive": float(1.0 if use_advanced_archive else 0.0),
                "mogwoUseMeanSelection": float(1.0 if use_mean_selection else 0.0),
                # Surgical Flags
                "mogwoUseAttnFeasBoost": float(
                    1.0 if bool(params.extra.get("mogwoUseAttnFeasBoost", use_adaptive_attention)) else 0.0
                ),
                "mogwoUseAttnDivBoost": float(
                    1.0 if bool(params.extra.get("mogwoUseAttnDivBoost", use_adaptive_attention)) else 0.0
                ),
                "mogwoUseStepFeasDriver": float(
                    1.0 if bool(params.extra.get("mogwoUseStepFeasDriver", use_adaptive_attention)) else 0.0
                ),
                "mogwoUseStepDivDriver": float(
                    1.0 if bool(params.extra.get("mogwoUseStepDivDriver", use_adaptive_attention)) else 0.0
                ),
                # Backward-compatible key for previous analysis tables.
                "mogwoLambdaSafeMean": float(attention_lambda_feas_sum / max(1, attention_steps)),
            },
        )

        if params.compute_metrics:
            arc_obj = np.stack([c.objective for c in archive])
            run_scores[run_idx - 1] = np.array(
                [
                    cal_metric(1, arc_obj, params.problem_index, objective_count),
                    cal_metric(2, arc_obj, params.problem_index, objective_count),
                ],
                dtype=float,
            )

    if params.compute_metrics and _should_write_final_hv(params):
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores


def run_fleet_mogwo_no_attention(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_mogwo(model, _apply_variant(params, variant="no_attention"))


def run_fleet_mogwo_standard_gwo(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    return run_fleet_mogwo(model, _apply_variant(params, variant="standard_gwo"))
