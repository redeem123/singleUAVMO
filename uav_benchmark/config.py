from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _lookup(mapping: dict, *keys: str, default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _parse_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off", ""}:
            return False
    return bool(raw)


def _require_int(raw: Any, field_name: str, *, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_float(raw: Any, field_name: str, *, minimum: float | None = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _int_tuple(raw: Any, field_name: str, *, minimum: int | None = 1) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [raw]
    return tuple(_require_int(item, field_name, minimum=minimum) for item in values)


def _run_index_tuple(raw: Any) -> tuple[int, ...] | None:
    values = [item for item in _int_tuple(raw, "run_indices", minimum=None) if item >= 1]
    if not values:
        return None
    return tuple(dict.fromkeys(values))


def _normalized_mapping(mapping: dict) -> dict:
    normalized = dict(mapping)
    aliases = (
        ("problems", "problemNames"),
        ("output_dir", "resultsDir"),
        ("metrics", "computeMetrics"),
    )
    for source, target in aliases:
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    nested_extra = normalized.get("extra")
    if isinstance(nested_extra, dict):
        merged = dict(nested_extra)
        merged.update(normalized)
        normalized = merged
    return normalized


@dataclass(slots=True)
class BenchmarkParams:
    generations: int = 500
    population: int = 100
    runs: int = 14
    compute_metrics: bool = False
    use_parallel: bool = False
    parallel_mode: str = "none"
    safe_dist: float = 20.0
    drone_size: float = 1.0
    results_dir: Path = Path("results")
    problem_name: str = ""
    problem_index: int = 0
    seed: int | None = None
    algorithm: str = ""
    mode: str = "fleet"
    fleet_size: int = 1
    fleet_sizes: tuple[int, ...] = ()
    separation_min: float = 10.0
    max_turn_deg: float = 75.0
    evaluation_budget: int = 0
    scenario_set: str = "paper_medium"
    gpu_mode: str = "auto"
    run_indices: tuple[int, ...] | None = None
    write_final_hv: bool = True
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.generations = _require_int(self.generations, "generations", minimum=1)
        self.population = _require_int(self.population, "population", minimum=1)
        self.runs = _require_int(self.runs, "runs", minimum=1)
        self.safe_dist = _require_float(self.safe_dist, "safe_dist", minimum=0.0)
        self.drone_size = _require_float(self.drone_size, "drone_size", minimum=0.0)
        self.problem_index = _require_int(self.problem_index, "problem_index", minimum=0)
        self.fleet_size = _require_int(self.fleet_size, "fleet_size", minimum=1)
        self.fleet_sizes = _int_tuple(self.fleet_sizes, "fleet_sizes", minimum=1)
        self.separation_min = _require_float(self.separation_min, "separation_min", minimum=0.0)
        self.max_turn_deg = _require_float(self.max_turn_deg, "max_turn_deg", minimum=0.0)
        self.evaluation_budget = _require_int(self.evaluation_budget, "evaluation_budget", minimum=0)
        self.compute_metrics = _parse_bool(self.compute_metrics, self.compute_metrics)
        self.use_parallel = _parse_bool(self.use_parallel, self.use_parallel)
        self.write_final_hv = _parse_bool(self.write_final_hv, self.write_final_hv)
        self.results_dir = Path(self.results_dir)
        self.problem_name = str(self.problem_name)
        self.algorithm = str(self.algorithm)
        self.parallel_mode = str(self.parallel_mode)
        self.mode = str(self.mode).strip().lower() or "fleet"
        self.scenario_set = str(self.scenario_set)
        self.gpu_mode = str(self.gpu_mode)
        if self.seed is not None:
            self.seed = _require_int(self.seed, "seed")
        if self.run_indices is not None:
            self.run_indices = _run_index_tuple(self.run_indices)
        if not isinstance(self.extra, dict):
            raise ValueError(f"extra must be a mapping, got {type(self.extra).__name__}")

    @classmethod
    def from_mapping(cls, mapping: dict) -> BenchmarkParams:
        mapping = _normalized_mapping(mapping)
        defaults = cls()
        seed = _lookup(mapping, "seed", default=defaults.seed)
        run_indices: tuple[int, ...] | None = defaults.run_indices
        raw_mode = str(_lookup(mapping, "mode", default=defaults.mode)).strip().lower()
        mode = "fleet" if raw_mode in {"fleet"} else raw_mode if raw_mode else defaults.mode
        fleet_sizes = defaults.fleet_sizes
        if "fleetSizes" in mapping or "fleet_sizes" in mapping:
            fleet_sizes = _int_tuple(_lookup(mapping, "fleetSizes", "fleet_sizes", default=()), "fleet_sizes")
        if "runIndices" in mapping or "run_indices" in mapping:
            run_indices = _run_index_tuple(_lookup(mapping, "runIndices", "run_indices", default=None))
        return cls(
            generations=_lookup(mapping, "Generations", "generations", default=defaults.generations),
            population=_lookup(mapping, "pop", "population", default=defaults.population),
            runs=_lookup(mapping, "Runs", "runs", default=defaults.runs),
            compute_metrics=_parse_bool(
                _lookup(mapping, "computeMetrics", "compute_metrics", default=defaults.compute_metrics),
                defaults.compute_metrics,
            ),
            use_parallel=_parse_bool(
                _lookup(mapping, "useParallel", "use_parallel", default=defaults.use_parallel),
                defaults.use_parallel,
            ),
            parallel_mode=str(_lookup(mapping, "parallelMode", "parallel_mode", default=defaults.parallel_mode)),
            safe_dist=_lookup(mapping, "safeDist", "safe_dist", default=defaults.safe_dist),
            drone_size=_lookup(mapping, "droneSize", "drone_size", default=defaults.drone_size),
            results_dir=Path(_lookup(mapping, "resultsDir", "results_dir", default=str(defaults.results_dir))),
            problem_name=str(_lookup(mapping, "problemName", "problem_name", default=defaults.problem_name)),
            problem_index=_lookup(mapping, "problemIndex", "problem_index", default=defaults.problem_index),
            seed=None if seed is None else seed,
            algorithm=str(_lookup(mapping, "algorithm", default=defaults.algorithm)),
            mode=mode,
            fleet_size=_lookup(mapping, "fleetSize", "fleet_size", default=defaults.fleet_size),
            fleet_sizes=fleet_sizes,
            separation_min=_lookup(mapping, "separationMin", "separation_min", default=defaults.separation_min),
            max_turn_deg=_lookup(mapping, "maxTurnDeg", "max_turn_deg", default=defaults.max_turn_deg),
            evaluation_budget=_lookup(
                mapping, "evaluationBudget", "evaluation_budget", default=defaults.evaluation_budget
            ),
            scenario_set=str(_lookup(mapping, "scenarioSet", "scenario_set", default=defaults.scenario_set)),
            gpu_mode=str(_lookup(mapping, "gpuMode", "gpu_mode", default=defaults.gpu_mode)),
            run_indices=run_indices,
            write_final_hv=_parse_bool(
                _lookup(mapping, "writeFinalHv", "write_final_hv", default=defaults.write_final_hv),
                defaults.write_final_hv,
            ),
            extra=dict(mapping),
        )
