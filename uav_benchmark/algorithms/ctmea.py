from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.chromosome import Chromosome
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.nsga2_ops import environmental_selection, f_operator, tournament_selection
from uav_benchmark.io.matlab import save_bp, save_mat, save_run_popobj
from uav_benchmark.io.results import ensure_dir


@dataclass(slots=True)
class TCOTOptions:
    lambda0: float = 0.0
    step_size: float = 0.08
    step_size_down: float = 0.06
    progress_drift: float = 0.01
    feas_high: float = 0.8
    feas_low: float = 0.35
    div_threshold: float = 0.08
    ot_interval: int = 5
    transfer_fraction: float = 0.5
    min_archive_for_ot: int = 30
    archive_max_size: int = 500
    archive_inject_count: int = 16
    use_counterfactual: bool = True
    beta0: float = 1.2
    beta_min: float = 0.1
    beta_max: float = 8.0
    beta_eta: float = 0.1
    enable_ot_in_feasibility: bool = False
    feasibility_phase_ratio: float = 0.35
    feasibility_lambda_cap: float = 0.75


def _parse_options(params: BenchmarkParams) -> TCOTOptions:
    options = TCOTOptions()
    ctm = params.extra.get("ctm", {})
    if isinstance(ctm, dict):
        options.lambda0 = float(ctm.get("lambda0", options.lambda0))
        options.step_size = float(ctm.get("stepSize", options.step_size))
        options.step_size_down = float(ctm.get("stepSizeDown", options.step_size_down))
        options.progress_drift = float(ctm.get("progressDrift", options.progress_drift))
        options.feas_high = float(ctm.get("feasHigh", options.feas_high))
        options.feas_low = float(ctm.get("feasLow", options.feas_low))
        options.div_threshold = float(ctm.get("divThreshold", options.div_threshold))
        options.ot_interval = int(ctm.get("otInterval", options.ot_interval))
        options.transfer_fraction = float(ctm.get("transferFraction", options.transfer_fraction))
        options.min_archive_for_ot = int(ctm.get("minArchiveForOT", options.min_archive_for_ot))
        options.archive_max_size = int(ctm.get("archiveMaxSize", options.archive_max_size))
        options.archive_inject_count = int(ctm.get("archiveInjectCount", options.archive_inject_count))
        options.use_counterfactual = bool(ctm.get("useCounterfactual", options.use_counterfactual))
        options.beta0 = float(ctm.get("beta0", options.beta0))
        options.beta_min = float(ctm.get("betaMin", options.beta_min))
        options.beta_max = float(ctm.get("betaMax", options.beta_max))
        options.beta_eta = float(ctm.get("betaEta", options.beta_eta))
    return options


def _build_easy_model(model: dict[str, Any], safe_dist_scale: float = 0.5, nofly_scale: float = 0.8) -> dict[str, Any]:
    easy = dict(model)
    if "safeDist" in easy and easy["safeDist"] is not None:
        easy["safeDist"] = max(1.0, float(easy["safeDist"]) * safe_dist_scale)
    if "nofly_r" in easy and easy["nofly_r"] is not None:
        easy["nofly_r"] = np.asarray(easy["nofly_r"], dtype=float) * nofly_scale
    return easy


def _interpolate_models(easy: dict[str, Any], hard: dict[str, Any], lam: float) -> dict[str, Any]:
    blended = dict(hard)
    if "safeDist" in hard and "safeDist" in easy:
        blended["safeDist"] = (1.0 - lam) * float(easy["safeDist"]) + lam * float(hard["safeDist"])
    if "nofly_r" in hard and "nofly_r" in easy and hard["nofly_r"] is not None and easy["nofly_r"] is not None:
        blended["nofly_r"] = (1.0 - lam) * np.asarray(easy["nofly_r"], dtype=float) + lam * np.asarray(hard["nofly_r"], dtype=float)
    return blended


def _population_diversity(objectives: np.ndarray) -> float:
    if objectives.shape[0] <= 1:
        return 0.0
    finite = objectives[np.all(np.isfinite(objectives), axis=1)]
    if finite.shape[0] <= 1:
        return 0.0
    centered = finite - np.mean(finite, axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    if covariance.ndim == 0:
        return float(abs(covariance))
    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size == 0:
        return 0.0
    return float(np.mean(np.sqrt(eigvals)))


def _schedule_lambda(lambda_value: float, feasible_ratio: float, diversity: float, options: TCOTOptions) -> float:
    if feasible_ratio >= options.feas_high and diversity >= options.div_threshold:
        lambda_value = min(1.0, lambda_value + options.step_size + options.progress_drift)
    elif feasible_ratio <= options.feas_low:
        lambda_value = max(0.0, lambda_value - options.step_size_down)
    else:
        lambda_value = min(1.0, lambda_value + options.progress_drift)
    return float(np.clip(lambda_value, 0.0, 1.0))


def _archive_add(archive: list[Chromosome], population: list[Chromosome], max_size: int) -> list[Chromosome]:
    merged = archive + [item.copy() for item in population]
    if len(merged) <= max_size:
        return merged
    objective_matrix = np.array([item.objs for item in merged], dtype=float)
    score = np.sum(objective_matrix, axis=1)
    order = np.argsort(score)
    return [merged[index] for index in order[:max_size]]


def _sample_transfer(archive: list[Chromosome], count: int) -> list[Chromosome]:
    if not archive:
        return []
    picks = np.random.choice(len(archive), size=min(count, len(archive)), replace=False)
    return [archive[int(idx)].copy() for idx in picks]


def run_ctmea(model: dict[str, Any], params: BenchmarkParams) -> np.ndarray:
    objective_count = 4
    model = dict(model)
    model["n"] = 10
    options = _parse_options(params)
    easy_model = _build_easy_model(model)
    results_path = params.results_dir / params.problem_name
    ensure_dir(results_path)
    run_scores = np.zeros((params.runs, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)
    boundary = np.array(
        [
            [float(model["xmax"]), float(model["ymax"]), float(model["zmax"])],
            [float(model["xmin"]), float(model["ymin"]), float(model["zmin"])],
        ],
        dtype=float,
    )

    for run_index in range(1, params.runs + 1):
        lam = options.lambda0
        beta = options.beta0
        current_model = _interpolate_models(easy_model, model, lam)
        population = []
        for _ in range(params.population):
            particle = Chromosome.new(current_model)
            particle.initialize(current_model)
            particle.evaluate(current_model)
            population.append(particle)
        population, front_no, crowding = environmental_selection(population, params.population, objective_count)
        archive: list[Chromosome] = _archive_add([], population, options.archive_max_size)
        hv_history = np.zeros((params.generations, 2), dtype=float) if params.compute_metrics else np.zeros((0, 2), dtype=float)

        for generation in range(1, params.generations + 1):
            phase_progress = generation / max(1, params.generations)
            in_feasibility = phase_progress <= options.feasibility_phase_ratio
            current_model = _interpolate_models(easy_model, model, lam)

            for item in population:
                item.evaluate(current_model)
            population, front_no, crowding = environmental_selection(population, params.population, objective_count)

            mating_pool = tournament_selection(2, params.population, front_no, -crowding)
            offspring = f_operator(population, mating_pool, boundary, current_model)

            transfer_offspring: list[Chromosome] = []
            use_ot = (
                generation % max(1, options.ot_interval) == 0
                and len(archive) >= options.min_archive_for_ot
                and (not in_feasibility or options.enable_ot_in_feasibility)
            )
            if use_ot:
                n_transfer = max(1, int(params.population * options.transfer_fraction))
                transfer_offspring = _sample_transfer(archive, n_transfer)
                if transfer_offspring and options.use_counterfactual:
                    beta = float(np.clip(beta * (1.0 + options.beta_eta), options.beta_min, options.beta_max))

            merged = population + offspring + transfer_offspring
            population, front_no, crowding = environmental_selection(merged, params.population, objective_count)
            archive = _archive_add(archive, population, options.archive_max_size)

            objective_matrix = np.array([item.objs for item in population], dtype=float)
            feasible_ratio = float(np.mean(np.all(np.isfinite(objective_matrix), axis=1)))
            diversity = _population_diversity(objective_matrix)
            lam = _schedule_lambda(lam, feasible_ratio, diversity, options)
            if in_feasibility:
                lam = min(lam, options.feasibility_lambda_cap)

            if params.compute_metrics:
                if generation == 1 or generation == params.generations or generation % 50 == 0:
                    hv_history[generation - 1, 0] = cal_metric(1, objective_matrix, params.problem_index, objective_count)
                    hv_history[generation - 1, 1] = cal_metric(2, objective_matrix, params.problem_index, objective_count)
                elif generation > 1:
                    hv_history[generation - 1] = hv_history[generation - 2]

        run_dir = results_path / f"Run_{run_index}"
        ensure_dir(run_dir)
        if params.compute_metrics:
            save_mat(run_dir / "gen_hv.mat", {"gen_hv": hv_history})
        objective_matrix = np.array([item.objs for item in population], dtype=float)
        save_run_popobj(run_dir / "final_popobj.mat", objective_matrix, params.problem_index, objective_count)
        for solution_index, solution in enumerate(population, start=1):
            save_bp(run_dir / f"bp_{solution_index}.mat", solution.path, solution.objs)

        if params.compute_metrics:
            run_scores[run_index - 1] = np.array(
                [
                    cal_metric(1, objective_matrix, params.problem_index, objective_count),
                    cal_metric(2, objective_matrix, params.problem_index, objective_count),
                ],
                dtype=float,
            )
    if params.compute_metrics:
        save_mat(results_path / "final_hv.mat", {"bestScores": run_scores})
    return run_scores
