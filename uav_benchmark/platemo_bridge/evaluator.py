from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import _constraint_violation_vector, _evaluate_population
from uav_benchmark.core.mission_encoding import paths_to_decision
from uav_benchmark.io.matlab import load_mat, save_mat


def _finite_optimizer_objective(candidate_objective: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    objective = np.asarray(candidate_objective, dtype=float).reshape(-1)
    if objective.size == 4 and np.all(np.isfinite(objective)):
        return np.clip(objective, 0.0, 1.0)

    fallback = np.array(
        [
            float(details.get("makespan", 1.0)),
            float(details.get("energy", 1.0)),
            float(details.get("risk", 1.0)),
            float(details.get("turnPenalty", 1.0)),
        ],
        dtype=float,
    )
    fallback = np.nan_to_num(fallback, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(fallback, 0.0, 1.0)


def evaluate_batch(context_path: Path, request_path: Path, response_path: Path) -> None:
    with context_path.open("rb") as handle:
        context = pickle.load(handle)

    request = load_mat(request_path)
    if "PathStack" in request:
        stack = np.asarray(request["PathStack"], dtype=float)
        if stack.ndim == 2 and stack.shape[1] >= 3:
            stack = stack.reshape(1, stack.shape[0], stack.shape[1])
        decisions = []
        for idx in range(stack.shape[0]):
            path = stack[idx, :, :3]
            path = path[np.all(np.isfinite(path), axis=1)]
            if path.shape[0] < 2:
                continue
            decisions.append(
                paths_to_decision(
                    [path],
                    context["model"],
                    fleet_size=int(context["fleet_size"]),
                    n_waypoints=int(context["n_waypoints"]),
                )
            )
        decisions = np.asarray(decisions, dtype=float)
    elif "PopDec" in request:
        decisions = np.asarray(request["PopDec"], dtype=float)
    elif "Decs" in request:
        decisions = np.asarray(request["Decs"], dtype=float)
    else:
        raise KeyError(f"{request_path} does not contain PopDec or Decs")
    if decisions.ndim == 1:
        decisions = decisions.reshape(1, -1)
    if decisions.shape[0] == 0:
        save_mat(response_path, {"PopObj": np.zeros((0, 4), dtype=float), "PopCon": np.zeros((0, 1), dtype=float)})
        return

    candidates = _evaluate_population(
        decisions,
        context["model"],
        fleet_size=int(context["fleet_size"]),
        n_waypoints=int(context["n_waypoints"]),
    )
    pop_obj = np.vstack(
        [_finite_optimizer_objective(candidate.objective, candidate.details) for candidate in candidates]
    )
    pop_con = _constraint_violation_vector(candidates, context["model"]).reshape(-1, 1)
    save_mat(response_path, {"PopObj": pop_obj, "PopCon": pop_con})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UAV decisions for MATLAB PlatEMO bridge.")
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    evaluate_batch(args.context, args.request, args.response)


if __name__ == "__main__":
    main()
