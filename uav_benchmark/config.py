from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    mode: str = "single"
    fleet_size: int = 1
    fleet_sizes: tuple[int, ...] = ()
    separation_min: float = 10.0
    max_turn_deg: float = 75.0
    evaluation_budget: int = 0
    scenario_set: str = "paper_medium"
    gpu_mode: str = "auto"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict) -> "BenchmarkParams":
        params = cls()
        params.generations = int(mapping.get("Generations", mapping.get("generations", params.generations)))
        params.population = int(mapping.get("pop", mapping.get("population", params.population)))
        params.runs = int(mapping.get("Runs", mapping.get("runs", params.runs)))
        params.compute_metrics = bool(mapping.get("computeMetrics", mapping.get("compute_metrics", params.compute_metrics)))
        params.use_parallel = bool(mapping.get("useParallel", mapping.get("use_parallel", params.use_parallel)))
        params.parallel_mode = str(mapping.get("parallelMode", mapping.get("parallel_mode", params.parallel_mode)))
        params.safe_dist = float(mapping.get("safeDist", mapping.get("safe_dist", params.safe_dist)))
        params.drone_size = float(mapping.get("droneSize", mapping.get("drone_size", params.drone_size)))
        params.results_dir = Path(mapping.get("resultsDir", mapping.get("results_dir", str(params.results_dir))))
        params.problem_name = str(mapping.get("problemName", mapping.get("problem_name", params.problem_name)))
        params.problem_index = int(mapping.get("problemIndex", mapping.get("problem_index", params.problem_index)))
        if "seed" in mapping and mapping["seed"] is not None:
            params.seed = int(mapping["seed"])
        params.mode = str(mapping.get("mode", params.mode))
        params.fleet_size = int(mapping.get("fleetSize", mapping.get("fleet_size", params.fleet_size)))
        if "fleetSizes" in mapping or "fleet_sizes" in mapping:
            raw = mapping.get("fleetSizes", mapping.get("fleet_sizes", ()))
            if isinstance(raw, str):
                params.fleet_sizes = tuple(
                    int(item.strip()) for item in raw.split(",") if item.strip()
                )
            elif isinstance(raw, (list, tuple)):
                params.fleet_sizes = tuple(int(item) for item in raw)
        params.separation_min = float(mapping.get("separationMin", mapping.get("separation_min", params.separation_min)))
        params.max_turn_deg = float(mapping.get("maxTurnDeg", mapping.get("max_turn_deg", params.max_turn_deg)))
        params.evaluation_budget = int(mapping.get("evaluationBudget", mapping.get("evaluation_budget", params.evaluation_budget)))
        params.scenario_set = str(mapping.get("scenarioSet", mapping.get("scenario_set", params.scenario_set)))
        params.gpu_mode = str(mapping.get("gpuMode", mapping.get("gpu_mode", params.gpu_mode)))
        params.extra = dict(mapping)
        return params
