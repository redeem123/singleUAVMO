"""Compatibility re-export layer for shared fleet runners and helpers."""

from __future__ import annotations

from uav_benchmark.algorithms.shared.fleet_artifacts import _save_fleet_artifacts
from uav_benchmark.algorithms.shared.fleet_common import (
    _build_bounds,
    _build_navigation_bounds,
    _constraint_violation,
    _constraint_violation_vector,
    _decision_to_direct_paths,
    _decision_to_paths_spherical,
    _ensure_fleet_endpoints,
    _evaluate_population,
    _normalize_objective_vector,
    _resolve_run_indices,
    _resume_run_scores,
    _should_write_final_hv,
    _transformation_matrix,
    _vectors_from_candidates,
)
from uav_benchmark.algorithms.shared.fleet_nsga_runner import (
    _run_fleet_nsga2,
    _run_fleet_nsga3,
    _sbx_mutation,
    run_fleet_nsga2,
    run_fleet_nsga3,
)
from uav_benchmark.algorithms.shared.fleet_pso_runner import (
    _run_fleet_pso,
    run_fleet_mopso,
    run_fleet_nmopso,
    run_fleet_rl_nmopso,
)

_SHARED_FLEET_METADATA_MARKER = {"benchmarkObjectiveDuringSearch": True}

__all__ = [
    "_build_bounds",
    "_build_navigation_bounds",
    "_constraint_violation",
    "_constraint_violation_vector",
    "_decision_to_direct_paths",
    "_decision_to_paths_spherical",
    "_ensure_fleet_endpoints",
    "_evaluate_population",
    "_normalize_objective_vector",
    "_resume_run_scores",
    "_resolve_run_indices",
    "_run_fleet_nsga2",
    "_run_fleet_nsga3",
    "_run_fleet_pso",
    "_save_fleet_artifacts",
    "_sbx_mutation",
    "_should_write_final_hv",
    "_transformation_matrix",
    "_vectors_from_candidates",
    "run_fleet_mopso",
    "run_fleet_nmopso",
    "run_fleet_nsga2",
    "run_fleet_nsga3",
    "run_fleet_rl_nmopso",
]
