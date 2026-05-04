from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.sem4d.core import (
    _evaluate_vectors_for_task,
    _extra_float,
    _extra_int,
    _SEM4DIndividual,
    _ShieldConfig,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.nsga2_ops import crowding_distance, n_d_sort, tournament_selection
from uav_benchmark.io.matlab import save_mat


def _ranking_matrix(objectives: np.ndarray) -> np.ndarray:
    matrix = np.asarray(objectives, dtype=float)
    if matrix.size == 0:
        return matrix.reshape(0, 4)
    fallback = np.ones(matrix.shape[1], dtype=float)
    for column in range(matrix.shape[1]):
        values = matrix[np.isfinite(matrix[:, column]), column]
        if values.size > 0:
            fallback[column] = float(np.max(values))
    penalties = np.sum(~np.isfinite(matrix), axis=1, keepdims=True).astype(float)
    return np.where(np.isfinite(matrix), matrix, fallback.reshape(1, -1) + 1.0 + penalties)


def _select_individuals(individuals: list[_SEM4DIndividual], n_keep: int) -> list[_SEM4DIndividual]:
    if len(individuals) <= n_keep:
        return individuals
    obj = _ranking_matrix(np.asarray([item.objective for item in individuals], dtype=float))
    front_no, max_front = n_d_sort(obj.copy(), None, n_keep)
    crowd = crowding_distance(obj, front_no)
    selected: list[int] = []
    for front in np.unique(front_no[np.isfinite(front_no)]):
        members = np.where(front_no == front)[0]
        if len(selected) + members.size <= n_keep:
            selected.extend(int(item) for item in members)
            continue
        order = members[np.argsort(-crowd[members])]
        need = n_keep - len(selected)
        selected.extend(int(item) for item in order[:need])
        break
    if len(selected) < n_keep:
        remaining = [idx for idx in range(len(individuals)) if idx not in set(selected)]
        selected.extend(remaining[: n_keep - len(selected)])
    return [individuals[index] for index in selected[:n_keep]]


def _select_candidates(candidates: list[Candidate], n_keep: int) -> list[Candidate]:
    if len(candidates) <= n_keep:
        return candidates
    wrapped = [
        _SEM4DIndividual(
            vector=np.asarray(candidate.vector, dtype=float),
            objective=np.asarray(candidate.objective, dtype=float),
            task_id=0,
            candidate=candidate,
        )
        for candidate in candidates
    ]
    return [item.candidate for item in _select_individuals(wrapped, n_keep)]


def _evaluate_initial_population(
    vectors: np.ndarray,
    task_ids: np.ndarray,
    model: dict[str, Any],
    aux_model: dict[str, Any],
    params: BenchmarkParams,
    config: _ShieldConfig,
    fleet_size: int,
    n_waypoints: int,
) -> list[_SEM4DIndividual]:
    ordered: list[_SEM4DIndividual | None] = [None] * int(vectors.shape[0])
    for task_id in np.unique(task_ids):
        mask = task_ids == int(task_id)
        candidates = _evaluate_vectors_for_task(
            vectors[mask],
            int(task_id),
            model=model,
            aux_model=aux_model,
            params=params,
            config=config,
            fleet_size=fleet_size,
            n_waypoints=n_waypoints,
        )
        mask_indices = np.where(mask)[0]
        for source_index, candidate in zip(mask_indices, candidates, strict=False):
            ordered[int(source_index)] = _SEM4DIndividual(
                vector=np.asarray(candidate.vector, dtype=float),
                objective=np.asarray(candidate.objective, dtype=float),
                task_id=int(task_ids[source_index]),
                candidate=candidate,
            )
    return [item for item in ordered if item is not None]


def _make_offspring(
    population: list[_SEM4DIndividual],
    lower: np.ndarray,
    upper: np.ndarray,
    fleet_size: int,
    n_waypoints: int,
    task_ids: np.ndarray,
    params: BenchmarkParams,
) -> tuple[np.ndarray, np.ndarray]:
    pop_size = len(population)
    if pop_size == 0:
        return np.zeros((0, lower.size), dtype=float), np.zeros(0, dtype=int)
    obj = _ranking_matrix(np.asarray([item.objective for item in population], dtype=float))
    front_no, _ = n_d_sort(obj.copy(), None, pop_size)
    crowd = crowding_distance(obj, front_no)
    mating = tournament_selection(2, pop_size, front_no, -crowd)
    crossover_rate = float(np.clip(_extra_float(params, "sem4dCrossoverRate", 0.90), 0.0, 1.0))
    mutation_std = max(0.0, _extra_float(params, "sem4dMutationStd", 0.055))
    rmp = float(np.clip(_extra_float(params, "sem4dRMP", 0.65), 0.0, 1.0))
    uav_transfer_rate = float(np.clip(_extra_float(params, "sem4dUavTransferRate", 0.25), 0.0, 1.0))
    children = np.zeros((pop_size, lower.size), dtype=float)
    child_tasks = np.zeros(pop_size, dtype=int)
    span = upper - lower
    for child_index in range(pop_size):
        parent_a = population[int(mating[child_index])]
        parent_b = population[int(mating[(child_index + np.random.randint(1, pop_size + 1)) % pop_size])]
        cross_task = parent_a.task_id != parent_b.task_id
        allow_transfer = (not cross_task) or np.random.rand() < rmp
        if allow_transfer and np.random.rand() < crossover_rate:
            alpha = np.random.rand(lower.size)
            child = alpha * parent_a.vector + (1.0 - alpha) * parent_b.vector
            child_task = int(np.random.choice([parent_a.task_id, parent_b.task_id]))
        else:
            child = parent_a.vector.copy()
            child_task = int(parent_a.task_id)
        mutation_mask = np.random.rand(lower.size) < (1.0 / max(1, lower.size))
        gaussian = np.random.randn(lower.size) * mutation_std * span
        child = np.where(mutation_mask, child + gaussian, child)
        if fleet_size > 1 and np.random.rand() < uav_transfer_rate:
            block = child.reshape(fleet_size, n_waypoints, 3)
            source, target = np.random.choice(fleet_size, size=2, replace=False)
            block[target] = 0.65 * block[target] + 0.35 * block[source]
            child = block.reshape(-1)
        children[child_index] = np.clip(child, lower, upper)
        child_tasks[child_index] = (
            child_task if child_task in set(int(item) for item in task_ids) else int(np.random.choice(task_ids))
        )
    return children, child_tasks


def _task_ids_for_fleet(fleet_size: int, params: BenchmarkParams) -> np.ndarray:
    max_aux = max(0, _extra_int(params, "sem4dMaxAuxTasks", fleet_size))
    max_aux = min(int(fleet_size), max_aux)
    return np.asarray([0, *range(1, max_aux + 1)], dtype=int)


def _save_shield_artifact(run_dir: Path, final_candidates: list[Candidate]) -> None:
    def _values(key: str, default: float = 0.0) -> np.ndarray:
        return np.asarray(
            [
                float(candidate.details.get(key, default)) if isinstance(candidate.details, dict) else default
                for candidate in final_candidates
            ],
            dtype=float,
        )

    save_mat(
        run_dir / "sem4d_shield.mat",
        {
            "correctionNorm": _values("shieldCorrectionNorm"),
            "terrainCorrections": _values("shieldTerrainCorrections"),
            "interUavCorrections": _values("shieldInterUavCorrections"),
            "dynamicObstacleCorrections": _values("shieldDynamicObstacleCorrections"),
            "noFlyCorrections": _values("shieldNoFlyCorrections"),
            "energyCorrections": _values("shieldEnergyCorrections"),
            "motionCorrections": _values("shieldMotionCorrections"),
            "energyViolation": _values("shieldEnergyViolation"),
            "terrainRisk": _values("shieldTerrainRisk"),
            "dynamicRisk": _values("shieldDynamicRisk"),
            "noFlyRisk": _values("shieldNoFlyRisk"),
            "postShieldDynamicObstacleViolation": _values("postShieldDynamicObstacleViolation"),
            "postShieldDynamicObstacleViolationScore": _values("postShieldDynamicObstacleViolationScore"),
            "postShieldDynamicObstacleMargin": _values("postShieldDynamicObstacleMargin", np.nan),
            "postShieldEnergyViolation": _values("postShieldEnergyViolation"),
            "postShieldEnergyViolationScore": _values("postShieldEnergyViolationScore"),
            "postShieldEnergy": _values("postShieldEnergy", np.nan),
            "preShieldConflictRate": _values("preShieldConflictRate", np.nan),
            "postShieldConflictRate": _values("conflictRate", np.nan),
            "fairness": _values("fairness", np.nan),
            "sem4dTravelTime": _values("sem4dTravelTime", np.nan),
            "sem4dEnergy": _values("sem4dEnergy", np.nan),
            "sem4dRisk": _values("sem4dRisk", np.nan),
            "sem4dSmoothnessFairness": _values("sem4dSmoothnessFairness", np.nan),
        },
    )


def _target_evaluate_population(
    population: list[_SEM4DIndividual],
    model: dict[str, Any],
    aux_model: dict[str, Any],
    params: BenchmarkParams,
    config: _ShieldConfig,
    fleet_size: int,
    n_waypoints: int,
) -> list[Candidate]:
    if not population:
        return []
    vectors = np.stack([item.vector for item in population], axis=0)
    del aux_model
    return _evaluate_vectors_for_task(
        vectors,
        0,
        model=model,
        aux_model=model,
        params=params,
        config=config,
        fleet_size=fleet_size,
        n_waypoints=n_waypoints,
    )
