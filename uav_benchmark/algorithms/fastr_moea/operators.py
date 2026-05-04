from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from uav_benchmark.algorithms.apex_shade import SHADEMemory
from uav_benchmark.algorithms.shared.fleet_runner import _constraint_violation
from uav_benchmark.algorithms.shared.nmopso_engine import _candidate_matrix
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.core.evaluate_path import evaluate_path_details
from uav_benchmark.core.mission_encoding import decision_to_paths, paths_to_decision

_OP_DE = "de"
_OP_BRIDGE = "bridge"
_OP_REPAIR = "repair"
_ALL_OPERATORS = (_OP_DE, _OP_BRIDGE, _OP_REPAIR)


@dataclass(slots=True)
class _OperatorChoice:
    name: str
    parent_idx: int
    stage: str
    f_value: float | None = None
    cr_value: float | None = None


@dataclass(slots=True)
class _StageSplit:
    explore: np.ndarray
    exploit: np.ndarray
    repair: np.ndarray
    explore_share: float
    exploit_share: float
    repair_share: float


class _OperatorBandit:
    """Small UCB selector for the operator pool."""

    def __init__(self, exploration: float = 0.30) -> None:
        self._counts = {name: 0 for name in _ALL_OPERATORS}
        self._values = {name: 0.0 for name in _ALL_OPERATORS}
        self._exploration = float(exploration)

    def select(self, allowed: tuple[str, ...], bias: dict[str, float]) -> str:
        if not allowed:
            return _OP_DE
        untried = [name for name in allowed if self._counts.get(name, 0) <= 0]
        if untried:
            return max(untried, key=lambda name: float(bias.get(name, 0.0)))
        total = max(1, sum(self._counts.values()))
        best_name = allowed[0]
        best_score = -float("inf")
        for name in allowed:
            count = max(1, self._counts.get(name, 0))
            mean_reward = float(self._values.get(name, 0.0))
            bonus = self._exploration * math.sqrt(math.log(total + 1.0) / count)
            score = mean_reward + bonus + float(bias.get(name, 0.0))
            if score > best_score:
                best_name = name
                best_score = score
        return best_name

    def update(self, name: str, reward: float) -> None:
        if name not in self._counts:
            return
        self._counts[name] += 1
        count = self._counts[name]
        current = self._values[name]
        self._values[name] = current + (float(reward) - current) / float(count)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def values(self) -> dict[str, float]:
        return dict(self._values)


def _strict_objective_sum(candidate: Candidate) -> float:
    objective = np.asarray(candidate.objective, dtype=float).reshape(-1)
    if objective.size == 0 or np.any(~np.isfinite(objective)):
        return float("inf")
    return float(np.sum(objective))


def _candidate_sort_key(candidate: Candidate, model: dict[str, Any]) -> tuple[int, float, float]:
    violation = float(max(0.0, _constraint_violation(candidate, model)))
    feasible_rank = 0 if violation <= 0.0 and np.all(np.isfinite(candidate.objective)) else 1
    return (feasible_rank, violation, _strict_objective_sum(candidate))


def _dominates(left: Candidate, right: Candidate) -> bool:
    left_obj = np.asarray(left.objective, dtype=float).reshape(-1)
    right_obj = np.asarray(right.objective, dtype=float).reshape(-1)
    if left_obj.size == 0 or right_obj.size == 0:
        return False
    if np.any(~np.isfinite(left_obj)) or np.any(~np.isfinite(right_obj)):
        return False
    return bool(np.all(left_obj <= right_obj) and np.any(left_obj < right_obj))


def _operator_reward(child: Candidate, parent: Candidate, model: dict[str, Any]) -> float:
    child_cv = float(max(0.0, _constraint_violation(child, model)))
    parent_cv = float(max(0.0, _constraint_violation(parent, model)))
    child_feasible = child_cv <= 0.0 and np.all(np.isfinite(child.objective))
    parent_feasible = parent_cv <= 0.0 and np.all(np.isfinite(parent.objective))

    reward = 0.0
    if child_feasible and not parent_feasible:
        reward += 1.0
    elif parent_feasible and not child_feasible:
        reward -= 0.6

    reward += 0.35 * float(np.clip(parent_cv - child_cv, -1.0, 1.0))

    parent_sum = _strict_objective_sum(parent)
    child_sum = _strict_objective_sum(child)
    if np.isfinite(parent_sum) and np.isfinite(child_sum):
        scale = max(1e-9, abs(parent_sum))
        reward += 0.65 * ((parent_sum - child_sum) / scale)

    if _dominates(child, parent):
        reward += 0.30
    elif _dominates(parent, child):
        reward -= 0.20
    return float(np.clip(reward, -1.5, 1.5))


def _objective_diversity(candidates: list[Candidate]) -> float:
    matrix = _candidate_matrix(candidates)
    if matrix.size == 0:
        return 0.0
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.shape[0] < 2:
        return 0.0
    lo = np.min(finite, axis=0)
    hi = np.max(finite, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normalized = (finite - lo.reshape(1, -1)) / span.reshape(1, -1)
    score = float(np.mean(np.std(normalized, axis=0)))
    return float(np.clip(score, 0.0, 1.0))


def _strict_feasible_ratio(candidates: list[Candidate]) -> float:
    if not candidates:
        return 0.0
    values = [float(candidate.details.get("feasible", 0.0)) > 0.5 for candidate in candidates]
    return float(np.mean(np.asarray(values, dtype=float)))


def _stage_partition(
    candidates: list[Candidate],
    model: dict[str, Any],
    progress: float,
    feasible_ratio: float,
    diversity: float,
) -> _StageSplit:
    n_points = len(candidates)
    if n_points <= 0:
        empty = np.zeros(0, dtype=int)
        return _StageSplit(empty, empty, empty, 0.0, 0.0, 0.0)

    progress = float(np.clip(progress, 0.0, 1.0))
    pressure = 1.0 - float(np.clip(feasible_ratio, 0.0, 1.0))
    low_div = 1.0 - float(np.clip(diversity, 0.0, 1.0))

    repair_share = np.clip(0.12 + 0.36 * pressure, 0.10, 0.45)
    explore_share = np.clip(0.18 + 0.22 * (1.0 - progress) + 0.16 * low_div, 0.15, 0.50)
    exploit_share = max(0.20, 1.0 - repair_share - explore_share)

    total = repair_share + explore_share + exploit_share
    repair_share = float(repair_share / total)
    explore_share = float(explore_share / total)
    exploit_share = float(exploit_share / total)

    repair_count = max(1, int(round(n_points * repair_share)))
    explore_count = max(1, int(round(n_points * explore_share)))
    exploit_count = max(1, n_points - repair_count - explore_count)

    while repair_count + explore_count + exploit_count > n_points:
        if explore_count >= max(repair_count, exploit_count) and explore_count > 1:
            explore_count -= 1
        elif repair_count >= exploit_count and repair_count > 1:
            repair_count -= 1
        elif exploit_count > 1:
            exploit_count -= 1
        else:
            break
    while repair_count + explore_count + exploit_count < n_points:
        explore_count += 1

    indices = list(range(n_points))
    best_order = sorted(indices, key=lambda idx: _candidate_sort_key(candidates[idx], model))
    worst_order = sorted(
        indices,
        key=lambda idx: (
            0 if _constraint_violation(candidates[idx], model) > 0.0 else 1,
            -float(_constraint_violation(candidates[idx], model)),
            -_strict_objective_sum(candidates[idx]),
        ),
    )

    exploit = best_order[:exploit_count]
    repair = [idx for idx in worst_order if idx not in exploit][:repair_count]
    explore = [idx for idx in indices if idx not in exploit and idx not in repair]

    if len(explore) < explore_count:
        refill = [idx for idx in best_order if idx not in exploit and idx not in repair and idx not in explore]
        explore.extend(refill[: explore_count - len(explore)])

    return _StageSplit(
        explore=np.asarray(explore, dtype=int),
        exploit=np.asarray(exploit, dtype=int),
        repair=np.asarray(repair, dtype=int),
        explore_share=float(len(explore) / max(1, n_points)),
        exploit_share=float(len(exploit) / max(1, n_points)),
        repair_share=float(len(repair) / max(1, n_points)),
    )


def _build_relaxed_model(model: dict[str, Any], params: BenchmarkParams, progress: float) -> dict[str, Any]:
    """Auxiliary task: same objectives, easier feasibility early, strict late."""
    aux = dict(model)
    progress = float(np.clip(progress, 0.0, 1.0))

    sep_scale_start = float(params.extra.get("fastrAuxSepScaleStart", 0.58))
    safe_scale_start = float(params.extra.get("fastrAuxSafeScaleStart", 0.65))
    nofly_scale_start = float(params.extra.get("fastrAuxNoFlyScaleStart", 0.78))
    hard_collision_progress = float(params.extra.get("fastrAuxHardCollisionProgress", 0.60))

    sep_scale = sep_scale_start + (1.0 - sep_scale_start) * (progress**0.85)
    safe_scale = safe_scale_start + (1.0 - safe_scale_start) * (progress**0.90)
    nofly_scale = nofly_scale_start + (1.0 - nofly_scale_start) * (progress**1.05)

    if "separationMin" in aux and aux["separationMin"] is not None:
        aux["separationMin"] = max(1.0, float(aux["separationMin"]) * sep_scale)
    else:
        aux["separationMin"] = max(1.0, float(params.separation_min) * sep_scale)

    if "safeDist" in aux and aux["safeDist"] is not None:
        aux["safeDist"] = max(1.0, float(aux["safeDist"]) * safe_scale)
    else:
        aux["safeDist"] = max(1.0, float(params.safe_dist) * safe_scale)

    if "nofly_r" in aux and aux["nofly_r"] is not None:
        aux["nofly_r"] = np.asarray(aux["nofly_r"], dtype=float) * nofly_scale

    aux["hardCollisionConstraint"] = bool(progress >= hard_collision_progress)
    aux["turnSpikePenaltyWeight"] = float(model.get("turnSpikePenaltyWeight", 1.0)) * (0.60 + 0.40 * progress)
    return aux


def _path_perpendicular(path_xyz: np.ndarray) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float)
    direction = path[-1, :2] - path[0, :2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    direction = direction / norm
    return np.array([-direction[1], direction[0]], dtype=float)


def _smooth_path(path_xyz: np.ndarray, passes: int = 1) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    if path.shape[0] <= 3:
        return path
    for _ in range(max(1, int(passes))):
        path[1:-1] = 0.25 * path[:-2] + 0.5 * path[1:-1] + 0.25 * path[2:]
    return path


def _lift_path(path_xyz: np.ndarray, delta_z: float) -> np.ndarray:
    path = np.asarray(path_xyz, dtype=float).copy()
    if path.shape[0] <= 2 or delta_z <= 0.0:
        return path
    path[1:-1, 2] += float(delta_z)
    return path


def _path_issue_key(path_xyz: np.ndarray, model: dict[str, Any]) -> tuple[int, float, float]:
    objective, details = evaluate_path_details(np.asarray(path_xyz, dtype=float), model)
    collision = 1 if float(details.get("collisionViolation", 0.0)) > 0.5 else 0
    max_turn = float(details.get("maxTurnDeg", 0.0))
    limit = float(model.get("maxTurnDeg", 75.0))
    excess_turn = max(0.0, max_turn - limit)
    objective_sum = float(np.sum(objective)) if np.all(np.isfinite(objective)) else float("inf")
    return (collision, excess_turn, objective_sum)


def _select_partner_vector(
    population: np.ndarray,
    candidates: list[Candidate],
    strict_archive: list[Candidate],
    relaxed_archive: list[Candidate],
    pressure: float,
    exclude_idx: int,
) -> np.ndarray:
    choose_relaxed = relaxed_archive and np.random.rand() < (0.22 + 0.40 * pressure)
    if choose_relaxed:
        return relaxed_archive[int(np.random.randint(0, len(relaxed_archive)))].vector.copy()
    if strict_archive and np.random.rand() < 0.60:
        top_k = max(1, min(5, len(strict_archive)))
        return strict_archive[int(np.random.randint(0, top_k))].vector.copy()
    other_ids = [idx for idx in range(population.shape[0]) if idx != int(exclude_idx)]
    if not other_ids:
        return population[int(exclude_idx)].copy()
    pick = int(other_ids[int(np.random.randint(0, len(other_ids)))])
    if 0 <= pick < len(candidates):
        return candidates[pick].vector.copy()
    return population[pick].copy()


def _select_transfer_vectors(
    relaxed_archive: list[Candidate],
    strict_archive: list[Candidate],
    count: int,
) -> np.ndarray:
    if count <= 0 or not relaxed_archive:
        return np.zeros((0, 0), dtype=float)
    selected: list[np.ndarray] = []
    strict_vectors = [np.asarray(candidate.vector, dtype=float) for candidate in strict_archive]
    for candidate in sorted(relaxed_archive, key=_strict_objective_sum):
        vector = np.asarray(candidate.vector, dtype=float)
        duplicate = any(np.allclose(vector, strict_vector, atol=1e-8, rtol=0.0) for strict_vector in strict_vectors)
        if duplicate:
            continue
        if any(np.allclose(vector, keep, atol=1e-8, rtol=0.0) for keep in selected):
            continue
        selected.append(vector.copy())
        if len(selected) >= count:
            break
    if not selected:
        return np.zeros((0, 0), dtype=float)
    return np.stack(selected, axis=0)


def _de_child(
    parent_idx: int,
    population: np.ndarray,
    candidates: list[Candidate],
    strict_archive: list[Candidate],
    relaxed_archive: list[Candidate],
    shade: SHADEMemory,
    lower: np.ndarray,
    upper: np.ndarray,
    model: dict[str, Any],
    pressure: float,
) -> tuple[np.ndarray, float, float]:
    parent = np.asarray(population[int(parent_idx)], dtype=float)
    sampled_f, sampled_cr = shade.sample(1)
    f_value = float(sampled_f[0])
    cr_value = float(sampled_cr[0])

    donor_candidates = list(candidates) + list(strict_archive)
    if relaxed_archive:
        donor_candidates += list(relaxed_archive)
    donor_vectors = [np.asarray(candidate.vector, dtype=float) for candidate in donor_candidates]
    donor_vectors = [vector for vector in donor_vectors if vector.shape == parent.shape]
    if len(donor_vectors) < 2:
        donor_vectors.extend([population[int(np.random.randint(0, population.shape[0]))].copy() for _ in range(2)])

    pbest_pool = sorted(candidates + strict_archive, key=lambda cand: _candidate_sort_key(cand, model))
    pbest_limit = max(1, int(np.ceil(0.25 * len(pbest_pool)))) if pbest_pool else 1
    pbest_source = pbest_pool[int(np.random.randint(0, pbest_limit))] if pbest_pool else candidates[int(parent_idx)]
    pbest = np.asarray(pbest_source.vector, dtype=float)

    donor_ids = np.random.choice(len(donor_vectors), size=2, replace=len(donor_vectors) < 2)
    d1 = donor_vectors[int(donor_ids[0])]
    d2 = donor_vectors[int(donor_ids[1])]
    mutant = parent + f_value * (pbest - parent) + f_value * (d1 - d2)
    mask = np.random.rand(parent.size) < cr_value
    mask[int(np.random.randint(0, parent.size))] = True
    child = np.where(mask, mutant, parent)
    if relaxed_archive and np.random.rand() < (0.12 + 0.25 * pressure):
        guide = relaxed_archive[int(np.random.randint(0, len(relaxed_archive)))].vector
        child = 0.75 * child + 0.25 * np.asarray(guide, dtype=float)
    return np.clip(child, lower, upper), f_value, cr_value


def _bridge_child(
    parent_idx: int,
    population: np.ndarray,
    candidates: list[Candidate],
    strict_archive: list[Candidate],
    relaxed_archive: list[Candidate],
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    lower: np.ndarray,
    upper: np.ndarray,
    pressure: float,
) -> np.ndarray:
    parent_vec = np.asarray(population[int(parent_idx)], dtype=float)
    partner_vec = _select_partner_vector(
        population, candidates, strict_archive, relaxed_archive, pressure, exclude_idx=parent_idx
    )

    parent_paths = decision_to_paths(parent_vec, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    partner_paths = decision_to_paths(partner_vec, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)

    clearance_target = float(model.get("droneSize", 1.0)) + 0.25 * float(model.get("safeDist", 10.0))
    turn_limit = float(model.get("maxTurnDeg", 75.0))
    child_paths: list[np.ndarray] = []

    for uav_idx in range(fleet_size):
        base = np.asarray(parent_paths[uav_idx], dtype=float)
        mate = np.asarray(partner_paths[uav_idx], dtype=float)
        internal_base = base[1:-1].copy()
        internal_mate = mate[1:-1].copy()
        if internal_base.shape[0] > 0:
            cut = int(np.random.randint(0, internal_base.shape[0]))
            internal_base[cut:] = internal_mate[cut:]
        full_path = np.vstack([base[:1], internal_base, base[-1:]]) if internal_base.size > 0 else base.copy()
        full_path = _smooth_path(full_path, passes=1)

        _, details = evaluate_path_details(full_path, model)
        min_clearance = float(details.get("minClearance", np.nan))
        if (
            float(details.get("collisionViolation", 0.0)) > 0.5
            or not np.isfinite(min_clearance)
            or min_clearance < clearance_target
        ):
            shortfall = clearance_target - min_clearance if np.isfinite(min_clearance) else clearance_target
            full_path = _lift_path(full_path, max(1.0, shortfall))
        if float(details.get("maxTurnDeg", 0.0)) > turn_limit:
            full_path = _smooth_path(full_path, passes=2)
        child_paths.append(full_path)

    vector = paths_to_decision(child_paths, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    return np.clip(np.asarray(vector, dtype=float).reshape(-1), lower, upper)


def _repair_child(
    parent_idx: int,
    population: np.ndarray,
    strict_archive: list[Candidate],
    model: dict[str, Any],
    fleet_size: int,
    n_waypoints: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    parent_vec = np.asarray(population[int(parent_idx)], dtype=float)
    paths = decision_to_paths(parent_vec, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    _, mission_details = evaluate_mission_details(paths, model)

    change_applied = False
    conflict_log = np.asarray(mission_details.get("conflictLog", np.zeros((0, 5), dtype=float)), dtype=float)
    if conflict_log.size > 0:
        worst_row = conflict_log[int(np.argmax(conflict_log[:, 4]))]
        left_idx = int(np.clip(round(worst_row[1]), 0, fleet_size - 1))
        right_idx = int(np.clip(round(worst_row[2]), 0, fleet_size - 1))
        left_key = _path_issue_key(paths[left_idx], model)
        right_key = _path_issue_key(paths[right_idx], model)
        target_idx = left_idx if left_key >= right_key else right_idx
        other_idx = right_idx if target_idx == left_idx else left_idx

        target_path = np.asarray(paths[target_idx], dtype=float).copy()
        other_path = np.asarray(paths[other_idx], dtype=float)
        if target_path.shape[0] > 2 and other_path.shape[0] > 2:
            delta = np.mean(target_path[1:-1, :2] - other_path[1:-1, :2], axis=0)
            norm = float(np.linalg.norm(delta))
            direction = _path_perpendicular(target_path) if norm <= 1e-9 else delta / norm
            shift = 0.45 * float(model.get("separationMin", 10.0))
            target_path[1:-1, 0] += direction[0] * shift
            target_path[1:-1, 1] += direction[1] * shift
            paths[target_idx] = target_path
            change_applied = True

    clearance_target = float(model.get("droneSize", 1.0)) + 0.25 * float(model.get("safeDist", 10.0))
    turn_limit = float(model.get("maxTurnDeg", 75.0))
    for uav_idx in range(fleet_size):
        path = np.asarray(paths[uav_idx], dtype=float).copy()
        _, details = evaluate_path_details(path, model)
        min_clearance = float(details.get("minClearance", np.nan))
        if (
            float(details.get("collisionViolation", 0.0)) > 0.5
            or not np.isfinite(min_clearance)
            or min_clearance < clearance_target
        ):
            shortfall = clearance_target - min_clearance if np.isfinite(min_clearance) else clearance_target
            path = _lift_path(path, max(1.0, shortfall))
            change_applied = True
        if float(details.get("maxTurnDeg", 0.0)) > turn_limit:
            path = _smooth_path(path, passes=2)
            change_applied = True
        paths[uav_idx] = path

    vector = paths_to_decision(paths, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
    vector = np.clip(np.asarray(vector, dtype=float).reshape(-1), lower, upper)
    if change_applied:
        return vector

    if strict_archive:
        elite = strict_archive[int(np.random.randint(0, min(5, len(strict_archive))))].vector
        vector = 0.70 * parent_vec + 0.30 * np.asarray(elite, dtype=float)
    else:
        noise = np.random.normal(0.0, 0.04, size=parent_vec.shape) * (upper - lower)
        vector = parent_vec + noise
    return np.clip(vector, lower, upper)


def _operator_bias(stage: str, progress: float, pressure: float, diversity: float) -> dict[str, float]:
    low_div = 1.0 - float(np.clip(diversity, 0.0, 1.0))
    progress = float(np.clip(progress, 0.0, 1.0))
    if stage == "repair":
        return {
            _OP_REPAIR: 0.30 + 0.35 * pressure,
            _OP_DE: 0.08 + 0.12 * low_div,
        }
    if stage == "exploit":
        return {
            _OP_BRIDGE: 0.16 + 0.24 * progress,
            _OP_DE: 0.08,
        }
    return {
        _OP_DE: 0.16 + 0.22 * (1.0 - progress) + 0.12 * low_div,
        _OP_BRIDGE: 0.10 + 0.10 * pressure,
    }


def _allowed_operators(stage: str) -> tuple[str, ...]:
    if stage == "repair":
        return (_OP_REPAIR, _OP_DE)
    if stage == "exploit":
        return (_OP_BRIDGE, _OP_DE)
    return (_OP_DE, _OP_BRIDGE)
