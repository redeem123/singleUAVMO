from __future__ import annotations

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import _decision_to_direct_paths, _vectors_from_candidates
from uav_benchmark.algorithms.shared.nmopso_engine import NMOPSOEngine
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.core.mission_encoding import paths_to_decision


def test_vectors_from_candidates_uses_repaired_candidate_state() -> None:
    fallback = np.zeros((2, 3), dtype=float)
    candidates = [
        Candidate(vector=np.asarray([1.0, 2.0, 3.0]), objective=np.zeros(4), details={"feasible": 1.0}),
        Candidate(vector=np.asarray([4.0, 5.0, 6.0]), objective=np.ones(4), details={"feasible": 1.0}),
    ]

    vectors = _vectors_from_candidates(candidates, fallback)

    np.testing.assert_allclose(vectors, np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_nmopso_engine_syncs_population_after_repairing_evaluator() -> None:
    lower = np.zeros(3, dtype=float)
    upper = np.ones(3, dtype=float)

    def evaluate(vectors: np.ndarray) -> list[Candidate]:
        repaired = np.clip(np.asarray(vectors, dtype=float), 0.25, 0.75)
        return [
            Candidate(vector=row.copy(), objective=np.full(4, float(index + 1)), details={"feasible": 1.0})
            for index, row in enumerate(repaired)
        ]

    engine = NMOPSOEngine(
        model={},
        pop_size=2,
        lower=lower,
        upper=upper,
        fleet_size=1,
        n_waypoints=1,
        representation="cart",
        objective_count=4,
        evaluate_fn=evaluate,
        initial_population=np.asarray([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]], dtype=float),
    )
    engine.reset()

    np.testing.assert_allclose(engine.population, np.asarray([[0.25, 0.5, 0.75], [0.75, 0.25, 0.5]]))
    np.testing.assert_allclose(engine.pbest, engine.population)


def test_direct_matched_encoding_preserves_path_feasibility() -> None:
    model = {
        "H": np.zeros((20, 20), dtype=float),
        "xmin": 1.0,
        "xmax": 20.0,
        "ymin": 1.0,
        "ymax": 20.0,
        "zmin": 1.0,
        "zmax": 10.0,
        "safeDist": 2.0,
        "separationMin": 2.0,
        "droneSize": 1.0,
        "maxTurnDeg": 120.0,
        "hardCollisionConstraint": True,
        "nofly_c": np.zeros((0, 2), dtype=float),
        "nofly_r": np.zeros(0, dtype=float),
        "starts": np.asarray([[2.0, 2.0, 5.0]], dtype=float),
        "goals": np.asarray([[18.0, 18.0, 5.0]], dtype=float),
    }
    path = np.asarray(
        [[2.0, 2.0, 5.0], [6.0, 6.0, 5.0], [10.0, 10.0, 5.0], [14.0, 14.0, 5.0], [18.0, 18.0, 5.0]],
        dtype=float,
    )
    vector = paths_to_decision([path], model, fleet_size=1, n_waypoints=3)

    decoded = _decision_to_direct_paths(vector, model, fleet_size=1, n_waypoints=3)
    objective, details = evaluate_mission_details(decoded, model)

    assert np.all(np.isfinite(objective))
    assert float(details["feasible"]) > 0.5
    np.testing.assert_allclose(decoded[0][1:-1, :2], path[1:-1, :2])
