from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.algorithms.sac_smopso import _refresh_unconstrained_population, _reservoir_score
from uav_benchmark.algorithms.shared.pso_types import Candidate


def _candidate(
    vector: list[float],
    *,
    max_turn_deg: float,
    min_separation: float,
    min_clearance: float,
    objective_search: list[float],
) -> Candidate:
    return Candidate(
        vector=np.asarray(vector, dtype=float),
        objective=np.asarray(objective_search, dtype=float),
        details={
            "maxTurnDeg": float(max_turn_deg),
            "minSeparation": float(min_separation),
            "minClearance": float(min_clearance),
            "feasible": 1.0,
            "objective_search": np.asarray(objective_search, dtype=float),
        },
    )


class SACSMOPSOGeometryScoreTest(unittest.TestCase):
    def test_reservoir_score_respects_problem_specific_constraints(self) -> None:
        candidate = _candidate(
            [0.1, 0.2],
            max_turn_deg=90.0,
            min_separation=9.0,
            min_clearance=0.6,
            objective_search=[1.2, 1.1, 1.0, 1.0],
        )

        strict_score = _reservoir_score(
            candidate,
            separation_min=10.0,
            drone_size=1.0,
            max_turn_deg=75.0,
        )
        relaxed_score = _reservoir_score(
            candidate,
            separation_min=8.0,
            drone_size=0.5,
            max_turn_deg=100.0,
        )

        self.assertGreater(strict_score, relaxed_score)

    def test_unconstrained_population_order_tracks_live_geometry_limits(self) -> None:
        turn_limited = _candidate(
            [0.1, 0.2],
            max_turn_deg=85.0,
            min_separation=12.0,
            min_clearance=1.2,
            objective_search=[1.1, 1.0, 1.0, 1.0],
        )
        clearance_limited = _candidate(
            [0.3, 0.4],
            max_turn_deg=55.0,
            min_separation=7.0,
            min_clearance=0.45,
            objective_search=[1.0, 1.0, 1.0, 1.0],
        )
        fresh_vectors = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float)

        strict_state = {"vectors": np.zeros((0, 0), dtype=float), "candidates": []}
        _refresh_unconstrained_population(
            strict_state,
            fresh_vectors=fresh_vectors,
            fresh_candidates=[turn_limited, clearance_limited],
            capacity=2,
            model={"separationMin": 10.0, "droneSize": 1.0, "maxTurnDeg": 75.0},
        )
        self.assertTrue(np.allclose(strict_state["candidates"][0].vector, turn_limited.vector))

        relaxed_state = {"vectors": np.zeros((0, 0), dtype=float), "candidates": []}
        _refresh_unconstrained_population(
            relaxed_state,
            fresh_vectors=fresh_vectors,
            fresh_candidates=[turn_limited, clearance_limited],
            capacity=2,
            model={"separationMin": 7.0, "droneSize": 0.4, "maxTurnDeg": 90.0},
        )
        self.assertTrue(np.allclose(relaxed_state["candidates"][0].vector, clearance_limited.vector))


if __name__ == "__main__":
    unittest.main()
