from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.algorithms.shared.mission_stats import build_mission_stats


def _base_model() -> dict[str, float | np.ndarray]:
    size = 20
    return {
        "xmin": 1.0,
        "xmax": float(size),
        "ymin": 1.0,
        "ymax": float(size),
        "zmin": 0.0,
        "zmax": 50.0,
        "safeDist": 10.0,
        "droneSize": 1.0,
        "maxTurnDeg": 75.0,
        "H": np.zeros((size, size), dtype=float),
    }


class SingleUavStatsTest(unittest.TestCase):
    def test_collision_makes_solution_infeasible(self) -> None:
        model = _base_model()
        feasible_path = np.array([[1.0, 1.0, 15.0], [10.0, 10.0, 15.0], [20.0, 20.0, 15.0]], dtype=float)
        colliding_path = np.array([[1.0, 1.0, 0.1], [10.0, 10.0, 0.1], [20.0, 20.0, 0.1]], dtype=float)

        mission_stats, feasible_mask = build_mission_stats([feasible_path, colliding_path], model)

        self.assertEqual(feasible_mask.tolist(), [True, False])
        self.assertEqual(np.asarray(mission_stats["collisionViolation"], dtype=float).tolist(), [0.0, 1.0])

    def test_turn_is_soft_penalty_and_not_reported_as_violation_flag(self) -> None:
        model = _base_model()
        model["maxTurnDeg"] = 45.0
        smooth_path = np.array([[1.0, 1.0, 15.0], [10.0, 10.0, 15.0], [20.0, 20.0, 15.0]], dtype=float)
        sharp_turn = np.array([[1.0, 1.0, 15.0], [10.0, 1.0, 15.0], [10.0, 10.0, 15.0]], dtype=float)

        mission_stats, feasible_mask = build_mission_stats([smooth_path, sharp_turn], model)

        self.assertEqual(feasible_mask.tolist(), [True, True])
        self.assertNotIn("turnViolation", mission_stats)
        max_turn = np.asarray(mission_stats["maxTurnDeg"], dtype=float).tolist()
        self.assertGreater(max_turn[1], max_turn[0])


if __name__ == "__main__":
    unittest.main()
