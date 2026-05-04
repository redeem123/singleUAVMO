from __future__ import annotations

from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uav_benchmark.config import BenchmarkParams

AlgorithmRunner = Callable[[dict[str, Any], "BenchmarkParams"], Any]
_RUNNER_CACHE: dict[str, AlgorithmRunner] = {}


def _load_runner(ref: str) -> AlgorithmRunner:
    try:
        return _RUNNER_CACHE[ref]
    except KeyError:
        module_name, attr_name = ref.split(":", 1)
        runner = getattr(import_module(module_name), attr_name)
        _RUNNER_CACHE[ref] = runner
        return runner


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    name: str
    runner_ref: str
    availability: str
    summary: str

    @property
    def runner(self) -> AlgorithmRunner:
        return _load_runner(self.runner_ref)


@dataclass(frozen=True, slots=True)
class AlgorithmProfileSpec:
    name: str
    algorithms: tuple[str, ...]
    summary: str
    requires_experimental: bool = False


# Curated catalog: benchmark-safe algorithms are enabled by default, while
# experimental ones stay importable but require an explicit opt-in.
ALGORITHM_SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        "NMOPSO", "uav_benchmark.algorithms.nmopso:run_nmopso", "benchmark", "Repository baseline multi-objective PSO."
    ),
    AlgorithmSpec(
        "MOPSO", "uav_benchmark.algorithms.mopso:run_mopso", "benchmark", "Classic multi-objective PSO baseline."
    ),
    AlgorithmSpec(
        "SMPSO",
        "uav_benchmark.algorithms.smpso:run_smpso",
        "benchmark",
        "Standard SMPSO baseline adapted from PlatEMO.",
    ),
    AlgorithmSpec(
        "NSGA-II",
        "uav_benchmark.algorithms.nsga2:run_nsga2",
        "benchmark",
        "Canonical elitist multi-objective EA baseline.",
    ),
    AlgorithmSpec(
        "NSGA-III",
        "uav_benchmark.algorithms.nsga3:run_nsga3",
        "benchmark",
        "Reference-point many-objective EA baseline.",
    ),
    AlgorithmSpec(
        "MOEAD",
        "uav_benchmark.algorithms.moead:run_moead",
        "benchmark",
        "MOEA/D decomposition baseline adapted from PlatEMO.",
    ),
    AlgorithmSpec(
        "SPEA2", "uav_benchmark.algorithms.spea2:run_spea2", "benchmark", "Canonical SPEA2 strength-based EA baseline."
    ),
    AlgorithmSpec(
        "MFO-SPEA2",
        "uav_benchmark.algorithms.mfo_spea2:run_mfo_spea2",
        "benchmark",
        "Literature multitask SPEA2 variant.",
    ),
    AlgorithmSpec(
        "GCNMOEA",
        "uav_benchmark.algorithms.gcnmoea:run_gcnmoea",
        "benchmark",
        "Graph-based literature MOEA adaptation.",
    ),
    AlgorithmSpec(
        "CMOSMA", "uav_benchmark.algorithms.cmosma:run_cmosma", "benchmark", "Literature swarm-based MOEA adaptation."
    ),
    AlgorithmSpec(
        "MO-MFEA", "uav_benchmark.algorithms.momfea:run_momfea", "benchmark", "Multifactorial evolution baseline."
    ),
    AlgorithmSpec(
        "MO-MFEA-II",
        "uav_benchmark.algorithms.momfea:run_momfea2",
        "benchmark",
        "Adaptive multifactorial evolution baseline.",
    ),
    AlgorithmSpec(
        "L-SHADE-CDP",
        "uav_benchmark.algorithms.apex_shade:run_fleet_lshade_cdp",
        "benchmark",
        "Curated constrained DE baseline.",
    ),
    AlgorithmSpec(
        "CMOEA-CD",
        "uav_benchmark.algorithms.cmoea_cd:run_cmoeacd",
        "benchmark",
        "MATLAB PlatEMO CMOEA-CD optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "APSEA",
        "uav_benchmark.algorithms.apsea:run_apsea",
        "benchmark",
        "MATLAB PlatEMO APSEA optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "C-TSEA",
        "uav_benchmark.algorithms.c_tsea:run_ctsea",
        "benchmark",
        "MATLAB PlatEMO C-TSEA optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "ToP",
        "uav_benchmark.algorithms.top:run_top",
        "benchmark",
        "MATLAB PlatEMO ToP optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "CMOCSO",
        "uav_benchmark.algorithms.cmocso:run_cmocso",
        "benchmark",
        "MATLAB PlatEMO CMOCSO optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "C-TAEA",
        "uav_benchmark.algorithms.c_taea:run_ctaea",
        "benchmark",
        "MATLAB PlatEMO C-TAEA optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "Two_Arch2",
        "uav_benchmark.algorithms.two_arch2:run_two_arch2",
        "benchmark",
        "MATLAB PlatEMO Two_Arch2 optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "CMOEA-MS",
        "uav_benchmark.algorithms.cmoea_ms:run_cmoea_ms",
        "benchmark",
        "MATLAB PlatEMO CMOEA-MS optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "CMOEA-MSG",
        "uav_benchmark.algorithms.cmoea_msg:run_cmoea_msg",
        "benchmark",
        "MATLAB PlatEMO CMOEA-MSG optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "CCMO",
        "uav_benchmark.algorithms.ccmo:run_ccmo",
        "benchmark",
        "MATLAB PlatEMO CCMO optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "URCMO",
        "uav_benchmark.algorithms.urcmo:run_urcmo",
        "benchmark",
        "MATLAB PlatEMO URCMO optimizer with Python UAV evaluation.",
    ),
    AlgorithmSpec(
        "MOEA-2DE",
        "uav_benchmark.algorithms.moea_2de:run_moea_2de",
        "experimental",
        "Official MATLAB MOEA-2DE operators with batched path evaluation routed through the Python UAV benchmark evaluator.",
    ),
    AlgorithmSpec(
        "MOEAD-AWA-ASTAR",
        "uav_benchmark.algorithms.moead_awa_astar:run_moead_awa_astar",
        "experimental",
        "Official MATLAB MOEA/D-AWA with A*-guided operators and batched Python UAV benchmark evaluation.",
    ),
    AlgorithmSpec(
        "EMMOP",
        "uav_benchmark.algorithms.emmop:run_emmop",
        "experimental",
        "Official MATLAB EMMOP mechanics with benchmark four-objective evaluation and DQN state shim.",
    ),
    AlgorithmSpec(
        "SEM-4D",
        "uav_benchmark.algorithms.sem4d:run_sem4d",
        "experimental",
        "Shielded evolutionary multitasking planner with execution-time 4D conflict, dynamic-obstacle, no-fly, energy, and motion checks.",
    ),
    AlgorithmSpec(
        "SAC-SMOPSO",
        "uav_benchmark.algorithms.sac_smopso:run_sac_smopso",
        "experimental",
        "Legacy RL-controlled research optimizer.",
    ),
    AlgorithmSpec(
        "RA-SMPSO",
        "uav_benchmark.algorithms.ra_smpso:run_ra_smpso",
        "experimental",
        "Legacy SAC-adaptive SMPSO research variant.",
    ),
    AlgorithmSpec(
        "RA-NSGA-II",
        "uav_benchmark.algorithms.ra_nsga2:run_ra_nsga2",
        "experimental",
        "Legacy SAC-adaptive NSGA-II research variant.",
    ),
    AlgorithmSpec(
        "MOGWO",
        "uav_benchmark.algorithms.mogwo:run_fleet_mogwo",
        "experimental",
        "Custom MOGWO research family variant.",
    ),
    AlgorithmSpec(
        "MOGWO-NO-ATTENTION",
        "uav_benchmark.algorithms.mogwo:run_fleet_mogwo_no_attention",
        "experimental",
        "Custom MOGWO ablation variant without adaptive attention.",
    ),
    AlgorithmSpec(
        "MOGWO-STANDARD-GWO",
        "uav_benchmark.algorithms.mogwo:run_fleet_mogwo_standard_gwo",
        "experimental",
        "Custom MOGWO family baseline with auxiliary components disabled.",
    ),
    AlgorithmSpec(
        "CGPO",
        "uav_benchmark.algorithms.cgpo:run_cgpo",
        "experimental",
        "CGPO: lean three-mechanism graph-aware MOEA (CIG + PPF + OVO).",
    ),
    AlgorithmSpec(
        "CCEA-ADVS",
        "uav_benchmark.algorithms.ccea_advs:run_ccea_advs",
        "experimental",
        "Clean-room cooperative co-evolutionary UAV planner with adaptive decision-variable selection.",
    ),
    AlgorithmSpec(
        "TSKAC-NSGA-II",
        "uav_benchmark.algorithms.tskac_nsga2:run_tskac_nsga2",
        "experimental",
        "Knowledge-assisted coevolutionary research variant.",
    ),
    AlgorithmSpec(
        "DTAPP-IICR",
        "uav_benchmark.algorithms.dtapp_iicr:run_dtapp_iicr",
        "experimental",
        "Official C++ DTAPP-IICR MAPF-style preflight planner with Python benchmark re-scoring.",
    ),
)

_SPEC_BY_NAME: dict[str, AlgorithmSpec] = {spec.name: spec for spec in ALGORITHM_SPECS}


class _LazyAlgorithmRegistry(MutableMapping[str, AlgorithmRunner]):
    def __init__(self, availability: str | None = None) -> None:
        self._availability = availability
        self._overrides: dict[str, AlgorithmRunner] = {}
        self._removed: set[str] = set()

    def _base_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in ALGORITHM_SPECS
            if self._availability is None or spec.availability == self._availability
        )

    def __getitem__(self, key: str) -> AlgorithmRunner:
        if key in self._overrides:
            return self._overrides[key]
        if key in self._removed:
            raise KeyError(key)
        spec = _SPEC_BY_NAME.get(key)
        if spec is None or (self._availability is not None and spec.availability != self._availability):
            raise KeyError(key)
        return spec.runner

    def __setitem__(self, key: str, value: AlgorithmRunner) -> None:
        self._removed.discard(key)
        self._overrides[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._overrides:
            del self._overrides[key]
            return
        if key in self._base_names() and key not in self._removed:
            self._removed.add(key)
            return
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yielded: set[str] = set()
        for name in self._base_names():
            if name in self._removed:
                continue
            yielded.add(name)
            yield name
        for name in self._overrides:
            if name not in yielded:
                yield name

    def __len__(self) -> int:
        return sum(1 for _name in self)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._overrides:
            return True
        return key not in self._removed and key in self._base_names()


REGISTRY: MutableMapping[str, AlgorithmRunner] = _LazyAlgorithmRegistry("benchmark")
EXPERIMENTAL_REGISTRY: MutableMapping[str, AlgorithmRunner] = _LazyAlgorithmRegistry("experimental")
ALL_REGISTRY: MutableMapping[str, AlgorithmRunner] = _LazyAlgorithmRegistry()
ALGORITHM_AVAILABILITY: dict[str, str] = {spec.name: spec.availability for spec in ALGORITHM_SPECS}
ALGORITHM_SUMMARIES: dict[str, str] = {spec.name: spec.summary for spec in ALGORITHM_SPECS}
ALGORITHM_PROFILES: tuple[AlgorithmProfileSpec, ...] = (
    AlgorithmProfileSpec(
        "benchmark-core",
        ("NSGA-II", "NSGA-III", "SMPSO", "SPEA2", "MOEAD", "NMOPSO", "L-SHADE-CDP"),
        "Small curated baseline benchmark set.",
    ),
    AlgorithmProfileSpec(
        "benchmark-extended",
        (
            "NMOPSO",
            "MOPSO",
            "SMPSO",
            "NSGA-II",
            "NSGA-III",
            "MOEAD",
            "SPEA2",
            "MFO-SPEA2",
            "GCNMOEA",
            "CMOSMA",
            "MO-MFEA",
            "MO-MFEA-II",
            "L-SHADE-CDP",
            "CMOEA-CD",
            "APSEA",
            "C-TSEA",
            "ToP",
            "CMOCSO",
            "C-TAEA",
            "Two_Arch2",
            "CMOEA-MS",
            "CMOEA-MSG",
            "CCMO",
            "URCMO",
        ),
        "Expanded literature-style benchmark set without experimental research methods.",
    ),
    AlgorithmProfileSpec(
        "constraint-cmoea",
        (
            "CMOEA-CD",
            "APSEA",
            "C-TSEA",
            "ToP",
            "CMOCSO",
            "C-TAEA",
            "Two_Arch2",
            "CMOEA-MS",
            "CMOEA-MSG",
            "CCMO",
            "URCMO",
        ),
        "Constrained multi-objective PlatEMO baselines routed through the UAV Python evaluator.",
    ),
    AlgorithmProfileSpec(
        "cgpo-swec",
        (
            "CGPO",
            "NSGA-II",
            "MOEAD",
            "NMOPSO",
            "CCEA-ADVS",
            "TSKAC-NSGA-II",
            "DTAPP-IICR",
        ),
        "Active online CGPO/SWEC paper comparison set.",
        requires_experimental=True,
    ),
    AlgorithmProfileSpec(
        "state-representation",
        ("SAC-SMOPSO", "RA-SMPSO", "RA-NSGA-II"),
        "Legacy SAC/relational-state research track.",
        requires_experimental=True,
    ),
    AlgorithmProfileSpec(
        "mogwo-family",
        ("MOGWO", "MOGWO-NO-ATTENTION", "MOGWO-STANDARD-GWO"),
        "Custom MOGWO family and ablation variants.",
        requires_experimental=True,
    ),
    AlgorithmProfileSpec(
        "uav-reference-code",
        ("MOEA-2DE", "EMMOP"),
        "Online UAV-paper reference-code comparators wired to the benchmark objective vector where feasible.",
        requires_experimental=True,
    ),
    AlgorithmProfileSpec(
        "sem4d",
        ("SEM-4D", "MO-MFEA-II", "NSGA-II", "NMOPSO", "CGPO"),
        "SEM-4D paper track with multitasking, baseline, and lean CGPO comparators.",
        requires_experimental=True,
    ),
    AlgorithmProfileSpec(
        "experimental-all",
        tuple(spec.name for spec in ALGORITHM_SPECS if spec.availability == "experimental"),
        "All research-only experimental algorithms.",
        requires_experimental=True,
    ),
)
_ALGORITHM_PROFILE_MAP: dict[str, tuple[str, ...]] = {spec.name.lower(): spec.algorithms for spec in ALGORITHM_PROFILES}


def algorithm_specs(availability: str | None = None) -> tuple[AlgorithmSpec, ...]:
    if availability is None:
        return ALGORITHM_SPECS
    return tuple(spec for spec in ALGORITHM_SPECS if spec.availability == availability)


def algorithm_names(availability: str | None = None) -> tuple[str, ...]:
    return tuple(spec.name for spec in algorithm_specs(availability))


def algorithm_profile_specs() -> tuple[AlgorithmProfileSpec, ...]:
    return ALGORITHM_PROFILES


def algorithm_profile_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in ALGORITHM_PROFILES)


def resolve_algorithm_profile(name: str) -> tuple[str, ...]:
    key = str(name).strip().lower()
    try:
        return _ALGORITHM_PROFILE_MAP[key]
    except KeyError as exc:
        raise ValueError(f"Unknown algorithm profile: {name}") from exc


_EXPORT_RUNNER_REFS: dict[str, str] = {spec.runner_ref.split(":", 1)[1]: spec.runner_ref for spec in ALGORITHM_SPECS}
_EXPORT_RUNNER_REFS["run_fleet_apex_shade"] = "uav_benchmark.algorithms.apex_shade:run_fleet_apex_shade"


def __getattr__(name: str) -> Any:
    if name in _EXPORT_RUNNER_REFS:
        return _load_runner(_EXPORT_RUNNER_REFS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AlgorithmSpec",
    "AlgorithmProfileSpec",
    "ALGORITHM_SPECS",
    "ALGORITHM_AVAILABILITY",
    "ALGORITHM_SUMMARIES",
    "ALGORITHM_PROFILES",
    "REGISTRY",
    "EXPERIMENTAL_REGISTRY",
    "ALL_REGISTRY",
    "algorithm_specs",
    "algorithm_names",
    "algorithm_profile_specs",
    "algorithm_profile_names",
    "resolve_algorithm_profile",
    "run_momfea",
    "run_momfea2",
    "run_mopso",
    "run_smpso",
    "run_sac_smopso",
    "run_ra_smpso",
    "run_ra_nsga2",
    "run_nmopso",
    "run_nsga2",
    "run_nsga3",
    "run_moead",
    "run_spea2",
    "run_mfo_spea2",
    "run_gcnmoea",
    "run_cmosma",
    "run_fleet_mogwo",
    "run_fleet_mogwo_no_attention",
    "run_fleet_mogwo_standard_gwo",
    "run_cgpo",
    "run_ccea_advs",
    "run_fleet_lshade_cdp",
    "run_fleet_apex_shade",
    "run_tskac_nsga2",
    "run_cmoeacd",
    "run_apsea",
    "run_ctsea",
    "run_top",
    "run_cmocso",
    "run_ctaea",
    "run_two_arch2",
    "run_cmoea_ms",
    "run_cmoea_msg",
    "run_ccmo",
    "run_urcmo",
    "run_moea_2de",
    "run_moead_awa_astar",
    "run_emmop",
    "run_sem4d",
    "run_dtapp_iicr",
]
