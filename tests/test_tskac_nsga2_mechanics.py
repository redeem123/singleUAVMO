from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.algorithms.tskac_nsga2 import (
    _apply_eq20_top20,
    _assistant_model,
    _objective1_stats,
)


class TSKACNSGA2MechanicsTest(unittest.TestCase):
    def test_assistant_task_removes_threats_and_keeps_terrain(self) -> None:
        height_map = np.arange(9, dtype=float).reshape(3, 3)
        model = {
            "H": height_map,
            "threats": np.array([[1.0, 2.0, 0.0, 5.0]], dtype=float),
            "nofly_c": np.array([[2.0, 2.0]], dtype=float),
            "nofly_r": np.array([3.0], dtype=float),
            "nofly_h": np.array([10.0], dtype=float),
        }

        assistant = _assistant_model(model)

        np.testing.assert_array_equal(assistant["H"], height_map)
        self.assertEqual(assistant["threats"].shape, (0, 4))
        self.assertEqual(assistant["nofly_c"].shape, (0, 2))
        self.assertEqual(assistant["nofly_r"].shape, (0,))
        self.assertEqual(assistant["nofly_h"].shape, (0,))

    def test_objective_stats_use_real_constraint_feasibility(self) -> None:
        infeasible = Candidate(
            vector=np.zeros(1, dtype=float),
            objective=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
            details={"feasible": 0.0},
        )
        feasible = Candidate(
            vector=np.ones(1, dtype=float),
            objective=np.array([0.2, 0.2, 0.3, 0.4], dtype=float),
            details={"feasible": 1.0},
        )

        mean_obj1, feasible_ratio, mean_violation = _objective1_stats([infeasible, feasible], {})

        self.assertAlmostEqual(mean_obj1, 0.2)
        self.assertAlmostEqual(feasible_ratio, 0.5)
        self.assertGreater(mean_violation, 0.0)

    def test_top20_operator_preserves_elites_instead_of_pushing_them_to_bounds(self) -> None:
        vectors = np.arange(40, dtype=float).reshape(10, 4)
        lower = np.zeros(4, dtype=float)
        upper = np.full(4, 100.0, dtype=float)
        candidates = [
            Candidate(vector=vectors[idx], objective=np.array([float(idx), 0.0, 0.0, 0.0]), details={"feasible": 1.0})
            for idx in range(vectors.shape[0])
        ]

        moved = _apply_eq20_top20(vectors, candidates, lower, upper)

        np.testing.assert_array_equal(moved[:2], vectors[:2])
        self.assertFalse(np.array_equal(moved[-2:], vectors[-2:]))


if __name__ == "__main__":
    unittest.main()
