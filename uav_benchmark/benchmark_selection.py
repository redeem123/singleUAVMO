from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from uav_benchmark.algorithms import resolve_algorithm_profile
from uav_benchmark.config import BenchmarkParams, _parse_bool

_ADAPTIVE_VARIANT_ALGORITHMS = {"SAC-SMOPSO", "RA-SMPSO", "RA-NSGA-II"}
_CGPO_VARIANT_ALGORITHM = "CGPO"
_STATE_MODE_KEYS = ("stateRepresentationModes", "state_representation_modes")
_STATE_VALUE_KEYS = ("stateRepresentation", "state_representation")
_POLICY_MODE_KEYS = ("policyModes", "policy_modes")
_POLICY_VALUE_KEYS = ("sacPolicyMode", "sac_policy_mode", "policyMode", "policy_mode")
_CGPO_VARIANT_KEYS = ("cgpoAblationVariants", "cgpo_ablation_variants")
_CGPO_VARIANT_VALUE_KEYS = ("cgpoAblationVariant", "cgpo_ablation_variant")
_VARIANT_EXTRA_DROP_KEYS = _STATE_MODE_KEYS + _POLICY_MODE_KEYS + ("policyMode", "policy_mode")
_TORCH_ACCELERATED_ALGORITHMS = {"SAC-SMOPSO", "RA-SMPSO", "RA-NSGA-II"}
_ALGORITHM_PROFILE_KEYS = ("algorithmProfiles", "algorithm_profiles", "algorithmSets", "algorithm_sets")
_ALGORITHM_PROFILE_VALUE_KEYS = ("algorithmProfile", "algorithm_profile", "algorithmSet", "algorithm_set")
_PAPER_MEDIUM_BASE_PROBLEM_NAMES = (
    "c_100",
    "c_150",
    "c_100_20_nofly",
    "c_70_40_nofly",
    "m_100",
    "m_200",
    "m_100_30c_nofly",
    "m_200_20c_nofly",
    "s_120",
    "s_180",
    "s_110_20_nofly",
    "s_80_40_nofly",
)

_ALGORITHM_SEED_OFFSET: dict[str, int] = {
    "NMOPSO": 11,
    "MOPSO": 23,
    "SMPSO": 29,
    "SAC-SMOPSO": 31,
    "RA-SMPSO": 33,
    "RA-NSGA-II": 35,
    "NSGA-II": 37,
    "NSGA-III": 41,
    "MOEAD": 43,
    "SPEA2": 47,
    "MFO-SPEA2": 61,
    "GCNMOEA": 71,
    "CMOSMA": 59,
    "MO-MFEA": 53,
    "MO-MFEA-II": 67,
    "MOGWO": 83,
    "MOGWO-NO-ATTENTION": 89,
    "MOGWO-STANDARD-GWO": 101,
    "CGPO": 109,
    "L-SHADE-CDP": 97,
    "CCEA-ADVS": 199,
    "TSKAC-NSGA-II": 107,
    "DTAPP-IICR": 197,
    "CMOEA-CD": 113,
    "APSEA": 127,
    "C-TSEA": 131,
    "ToP": 137,
    "CMOCSO": 139,
    "C-TAEA": 149,
    "Two_Arch2": 151,
    "CMOEA-MS": 157,
    "CMOEA-MSG": 163,
    "CCMO": 167,
    "URCMO": 173,
    "MOEA-2DE": 181,
    "MOEAD-AWA-ASTAR": 187,
    "EMMOP": 191,
    "SEM-4D": 193,
}

_ALGORITHM_NAME_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NMOPSO", ("nmopso",)),
    ("MOPSO", ("mopso",)),
    ("SMPSO", ("smpso", "sm-pso", "sm_pso")),
    ("SAC-SMOPSO", ("sac-smopso", "sac_smopso", "sacsmopso", "sac-sm-pso")),
    ("RA-SMPSO", ("ra-smpso", "ra_smpso", "ra-sm-pso")),
    ("RA-NSGA-II", ("ra-nsga-ii", "ra_nsga2", "ra-nsga2")),
    ("NSGA-II", ("nsga-ii", "nsga2", "nsga_ii")),
    ("NSGA-III", ("nsga-iii", "nsga3", "nsga_iii")),
    ("MOEAD", ("moead", "moea/d", "moea-d", "moea_d")),
    ("SPEA2", ("spea2", "spea-2", "spea_2")),
    ("MFO-SPEA2", ("mfo-spea2", "mfospea2", "mfo_spea2", "mfo-spea-2")),
    ("GCNMOEA", ("gcnmoea", "gcn-moea", "gcn_moea")),
    ("CMOSMA", ("cmosma", "cmo-sma", "cmo_sma")),
    ("MO-MFEA", ("mo-mfea", "momfea")),
    ("MO-MFEA-II", ("mo-mfea-ii", "momfea2", "momfea-ii")),
    ("MOGWO", ("mogwo", "a2mogwo", "a2-mogwo")),
    (
        "MOGWO-NO-ATTENTION",
        (
            "mogwo-no-attention",
            "mogwo_no_attention",
            "a2mogwo-no-attention",
            "a2mogwo_no_attention",
            "a2-mogwo-no-attention",
            "a2mogwo-noattention",
        ),
    ),
    (
        "MOGWO-STANDARD-GWO",
        (
            "mogwo-standard-gwo",
            "mogwo_standard_gwo",
            "a2mogwo-standard-gwo",
            "a2mogwo_standard_gwo",
            "a2-mogwo-standard-gwo",
        ),
    ),
    (
        "L-SHADE-CDP",
        (
            "l-shade-cdp",
            "lshade-cdp",
            "lshade_cdp",
            "apex-shade",
            "apexshade",
            "apex_shade",
        ),
    ),
    ("CMOEA-CD", ("cmoea-cd", "cmoeacd", "cmoea_cd")),
    ("APSEA", ("apsea",)),
    ("C-TSEA", ("c-tsea", "ctsea", "c_tsea")),
    ("ToP", ("top", "to-p", "two-phase", "two_phase")),
    ("CMOCSO", ("cmocso", "cmo-cso", "cmo_cso")),
    ("C-TAEA", ("c-taea", "ctaea", "c_taea")),
    ("Two_Arch2", ("two-arch2", "two_arch2", "twoarch2", "two-archive2", "two_archive2")),
    ("CMOEA-MS", ("cmoea-ms", "cmoeams", "cmoea_ms")),
    ("CMOEA-MSG", ("cmoea-msg", "cmoeamsg", "cmoea_msg")),
    ("CCMO", ("ccmo",)),
    ("URCMO", ("urcmo", "ur-cmo", "ur_cmo")),
    ("MOEA-2DE", ("moea-2de", "moea_2de", "moea2de", "dimension-exploration-discrepancy-evolution")),
    (
        "MOEAD-AWA-ASTAR",
        (
            "moead-awa-astar",
            "moead_awa_astar",
            "moeadawa-astar",
            "moeadawa_astar",
            "moea/d-awa-a*",
            "moea/d-awa-astar",
            "heuristic-driven-moead-awa",
            "hde-moead-awa",
        ),
    ),
    ("EMMOP", ("emmop", "e-mmop", "e_mmop")),
    (
        "SEM-4D",
        (
            "sem-4d",
            "sem4d",
            "sem_4d",
            "shielded-evolutionary-multitasking",
            "shielded_evolutionary_multitasking",
            "safety-shielded-evolutionary-multitasking",
        ),
    ),
    (
        "CGPO",
        (
            "cgpo",
            "constraint-graph-projection-optimizer",
            "constraint_graph_projection_optimizer",
            "constraint-graph-policy-optimizer",
            "constraint_graph_policy_optimizer",
        ),
    ),
    (
        "TSKAC-NSGA-II",
        (
            "tskac-nsga-ii",
            "tskac_nsga_ii",
            "tskacnsga2",
            "tskac-nsga2",
            "tskac-nsgaii",
            "tskacnsgaii",
        ),
    ),
    (
        "CCEA-ADVS",
        (
            "ccea-advs",
            "ccea_advs",
            "cceaadvs",
            "cooperative-coevolution-advs",
            "adaptive-decision-variable-selection",
        ),
    ),
    (
        "DTAPP-IICR",
        (
            "dtapp-iicr",
            "dtapp_iicr",
            "dtappiicr",
            "delivery-time-aware-prioritized-planning",
            "incremental-iterative-conflict-resolution",
        ),
    ),
)
_ALGORITHM_NAME_ALIASES: dict[str, str] = {
    alias: canonical for canonical, aliases in _ALGORITHM_NAME_VARIANTS for alias in aliases
}
_ALGORITHM_ORDER: tuple[str, ...] = tuple(canonical for canonical, _aliases in _ALGORITHM_NAME_VARIANTS)
# CGPO ablation variants.
#
# Each entry isolates a single conceptual mechanism of the retained
# three-mechanism algorithm (CIG + PPF + OVO).  All variants run on the lean
# CGPO loop.
_CGPO_ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "full": {
        "cgpoUseCigEdgeCoupling": True,
        "cgpoUseCigTerrainEdges": True,
        "cgpoUseCigObstacleEdges": True,
        "cgpoUseCigTurnEdges": True,
        "cgpoUseCigSmoothingEdges": True,
        "cgpoUseCigPairwiseEdges": True,
        "cgpoUsePpfPressure": True,
        "cgpoUseOvoVariation": True,
        "cgpoUseOvoCoordination": True,
    },
    "random_only": {
        # Pure NSGA-II baseline on top of the same population manager: no CIG,
        # no PPF, no OVO.  Establishes the floor that CGPO has to beat.
        "cgpoUseCigEdgeCoupling": False,
        "cgpoUseCigTerrainEdges": False,
        "cgpoUseCigObstacleEdges": False,
        "cgpoUseCigTurnEdges": False,
        "cgpoUseCigSmoothingEdges": False,
        "cgpoUseCigPairwiseEdges": False,
        "cgpoUsePpfPressure": False,
        "cgpoUseOvoVariation": False,
        "cgpoUseOvoCoordination": False,
    },
    "no_cig_edge_coupling": {
        # Keep CIG node features as variation inputs but drop the typed edges.
        "cgpoUseCigEdgeCoupling": False,
    },
    "no_ppf_pressure": {
        # Uniform parent sampling (no graph-aware boundary stratum).
        "cgpoUsePpfPressure": False,
    },
    "no_ovo_variation": {
        # Replace OVO with plain Gaussian mutation; isolates the variation gain.
        "cgpoUseOvoVariation": False,
        "cgpoUseOvoCoordination": False,
    },
    "no_ovo_fleet_coordination": {
        # Keep OVO blend + perturbation but drop the waypoint-aligned
        # fleet-coordination push; isolates the multi-UAV coupling gain.
        "cgpoUseOvoCoordination": False,
    },
}


def _fleet_from_problem_name(problem_name: str) -> int | None:
    match = re.search(r"_uav(\d+)$", problem_name)
    if not match:
        return None
    return int(match.group(1))


def _base_problem_name(problem_name: str) -> str:
    return re.sub(r"_uav\d+$", "", problem_name)


def _normalize_algorithm_name(name: str) -> str:
    key = str(name).strip().lower()
    return _ALGORITHM_NAME_ALIASES.get(key, str(name).strip())


def _dedupe_tokens(tokens: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tokens))


def _split_extra_tokens(raw: Any, *, allow_scalar: bool = False) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return _dedupe_tokens([item.strip() for item in raw.split(",") if item.strip()])
    if isinstance(raw, (list, tuple)):
        return _dedupe_tokens([str(item).strip() for item in raw if str(item).strip()])
    if not allow_scalar:
        return ()
    token = str(raw).strip()
    if not token:
        return ()
    return (token,)


def _string_token(raw: Any) -> str | None:
    if raw is None:
        return None
    token = str(raw).strip()
    return token or None


def _requested_algorithms(extra: dict[str, Any]) -> tuple[str, ...]:
    requested: list[str] = []
    profile_tokens = _requested_mode_values(
        extra,
        plural_keys=_ALGORITHM_PROFILE_KEYS,
        singular_keys=_ALGORITHM_PROFILE_VALUE_KEYS,
    )
    for profile in profile_tokens:
        requested.extend(resolve_algorithm_profile(profile))
    tokens = _split_extra_tokens(extra.get("algorithms"))
    if tokens:
        requested.extend(_normalize_algorithm_name(token) for token in tokens)
    return _dedupe_tokens(requested)


def _requested_problem_names(extra: dict[str, Any]) -> tuple[str, ...]:
    names = _requested_mode_values(
        extra,
        plural_keys=("problemNames", "problem_names", "problems"),
        singular_keys=("problemName", "problem_name", "problem"),
    )
    return tuple(_base_problem_name(name) if _fleet_from_problem_name(name) == 1 else name for name in names)


def _allow_experimental_algorithms(extra: dict[str, Any]) -> bool:
    raw = extra.get("allowExperimentalAlgorithms", extra.get("allow_experimental_algorithms", False))
    return _parse_bool(raw)


def _requested_mode_values(
    extra: dict[str, Any], plural_keys: tuple[str, ...], singular_keys: tuple[str, ...]
) -> tuple[str, ...]:
    for key in plural_keys:
        tokens = _split_extra_tokens(extra.get(key), allow_scalar=True)
        if tokens:
            return tokens
    for key in singular_keys:
        token = _string_token(extra.get(key))
        if token:
            return (token,)
    return ()


def _variant_algorithm_label(base_algorithm: str, state_representation: str | None, policy_mode: str | None) -> str:
    label = base_algorithm
    if state_representation:
        label = f"{label}__{state_representation}"
    if policy_mode:
        label = f"{label}__{policy_mode}"
    return label


def _normalize_cgpo_variant(raw: str) -> str:
    return str(raw).strip().lower().replace("-", "_")


def _requested_cgpo_variants(extra: dict[str, Any]) -> tuple[str, ...]:
    variants = _requested_mode_values(
        extra,
        plural_keys=_CGPO_VARIANT_KEYS,
        singular_keys=_CGPO_VARIANT_VALUE_KEYS,
    )
    if not variants:
        return ()
    normalized: list[str] = []
    for variant in variants:
        token = _normalize_cgpo_variant(variant)
        if token == "all":
            normalized.extend(_CGPO_ABLATION_VARIANTS)
        else:
            normalized.append(token)
    normalized = list(dict.fromkeys(normalized))
    unknown = [item for item in normalized if item not in _CGPO_ABLATION_VARIANTS]
    if unknown:
        raise ValueError(
            "Unknown CGPO ablation variant(s): "
            + ", ".join(unknown)
            + ". Available variants: "
            + ", ".join(_CGPO_ABLATION_VARIANTS)
        )
    return tuple(normalized)


def _cgpo_variant_tasks(base_algorithm: str, run_params: BenchmarkParams) -> list[tuple[str, str, BenchmarkParams]]:
    extra = dict(run_params.extra) if isinstance(run_params.extra, dict) else {}
    variants = _requested_cgpo_variants(extra)
    if not variants:
        return [(base_algorithm, base_algorithm, run_params)]
    variant_extra_base = dict(extra)
    for key in _CGPO_VARIANT_KEYS + _CGPO_VARIANT_VALUE_KEYS:
        variant_extra_base.pop(key, None)

    out: list[tuple[str, str, BenchmarkParams]] = []
    for variant in variants:
        variant_extra = dict(variant_extra_base)
        variant_extra.update(_CGPO_ABLATION_VARIANTS[variant])
        variant_extra["cgpoAblationVariant"] = variant
        variant_label = f"{base_algorithm}_{variant}"
        variant_params = replace(run_params, extra=variant_extra, algorithm=variant_label)
        out.append((variant_label, base_algorithm, variant_params))
    return out


def _variant_tasks_for_algorithm(
    base_algorithm: str,
    run_params: BenchmarkParams,
) -> list[tuple[str, str, BenchmarkParams]]:
    if base_algorithm == _CGPO_VARIANT_ALGORITHM:
        return _cgpo_variant_tasks(base_algorithm, run_params)
    if base_algorithm not in _ADAPTIVE_VARIANT_ALGORITHMS:
        return [(base_algorithm, base_algorithm, run_params)]

    extra = dict(run_params.extra) if isinstance(run_params.extra, dict) else {}
    state_modes = _requested_mode_values(
        extra,
        plural_keys=_STATE_MODE_KEYS,
        singular_keys=_STATE_VALUE_KEYS,
    )
    policy_modes = _requested_mode_values(
        extra,
        plural_keys=_POLICY_MODE_KEYS,
        singular_keys=_POLICY_VALUE_KEYS,
    )
    state_values = state_modes if state_modes else (None,)
    policy_values = policy_modes if policy_modes else (None,)
    variant_extra_base = dict(extra)
    for key in _VARIANT_EXTRA_DROP_KEYS:
        variant_extra_base.pop(key, None)

    variants: list[tuple[str, str, BenchmarkParams]] = []
    for state_representation in state_values:
        for policy_mode in policy_values:
            variant_extra = dict(variant_extra_base)
            if state_representation is not None:
                variant_extra["stateRepresentation"] = state_representation
            if policy_mode is not None:
                variant_extra["sacPolicyMode"] = policy_mode
            variant_label = _variant_algorithm_label(base_algorithm, state_representation, policy_mode)
            variant_params = replace(run_params, extra=variant_extra, algorithm=variant_label)
            variants.append((variant_label, base_algorithm, variant_params))
    return variants
