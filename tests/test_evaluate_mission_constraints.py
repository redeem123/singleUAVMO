from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.core.evaluate_mission import evaluate_mission_details


class EvaluateMissionConstraintsTest(unittest.TestCase):
    def test_separation_violation_reflected_in_conflict(self) -> None:
        model = {
            "xmin": 1.0,
            "xmax": 50.0,
            "ymin": 1.0,
            "ymax": 50.0,
            "zmin": 0.0,
            "zmax": 20.0,
            "safeDist": 5.0,
            "droneSize": 1.0,
            "separationMin": 5.0,
            "maxTurnDeg": 90.0,
            "H": np.zeros((50, 50), dtype=float),
        }
        # Two near-identical routes create separation violations.
        path_a = np.array([[2.0, 2.0, 10.0], [20.0, 20.0, 10.0], [40.0, 40.0, 10.0]], dtype=float)
        path_b = np.array([[2.3, 2.2, 10.0], [20.2, 20.0, 10.0], [40.2, 40.1, 10.0]], dtype=float)
        obj, details = evaluate_mission_details([path_a, path_b], model)
        self.assertTrue(np.isinf(obj).all())
        self.assertGreater(details["conflictRate"], 0.0)
        self.assertLess(details["minSeparation"], model["separationMin"])

    def test_turn_violation_is_recorded(self) -> None:
        model = {
            "xmin": 1.0,
            "xmax": 100.0,
            "ymin": 1.0,
            "ymax": 100.0,
            "zmin": 0.0,
            "zmax": 20.0,
            "safeDist": 5.0,
            "droneSize": 1.0,
            "separationMin": 3.0,
            "maxTurnDeg": 45.0,
            "H": np.zeros((100, 100), dtype=float),
        }
        # 90-degree bend exceeds maxTurnDeg and should be reflected in diagnostics.
        path = np.array([[10.0, 10.0, 10.0], [30.0, 10.0, 10.0], [30.0, 30.0, 10.0]], dtype=float)
        obj, details = evaluate_mission_details([path], model)
        self.assertTrue(np.all(np.isfinite(obj)))
        self.assertEqual(details["turnViolation"], 1.0)
        self.assertGreater(details["maxTurnDeg"], model["maxTurnDeg"])

    def test_collision_violation_is_hard_infeasible_when_enabled(self) -> None:
        model = {
            "xmin": 1.0,
            "xmax": 30.0,
            "ymin": 1.0,
            "ymax": 30.0,
            "zmin": 0.0,
            "zmax": 20.0,
            "safeDist": 5.0,
            "droneSize": 1.0,
            "separationMin": 3.0,
            "maxTurnDeg": 90.0,
            "hardCollisionConstraint": True,
            "H": np.zeros((30, 30), dtype=float),
        }
        # Entire path flies too close to terrain for drone body clearance.
        path = np.array([[5.0, 5.0, 0.2], [15.0, 15.0, 0.2], [25.0, 25.0, 0.2]], dtype=float)
        obj, details = evaluate_mission_details([path], model)
        self.assertTrue(np.isinf(obj).all())
        self.assertEqual(details["collisionViolation"], 1.0)
        self.assertLess(details["minClearance"], model["droneSize"])

    def test_collision_violation_can_be_soft_when_hard_constraint_disabled(self) -> None:
        model = {
            "xmin": 1.0,
            "xmax": 30.0,
            "ymin": 1.0,
            "ymax": 30.0,
            "zmin": 0.0,
            "zmax": 20.0,
            "safeDist": 5.0,
            "droneSize": 1.0,
            "separationMin": 3.0,
            "maxTurnDeg": 90.0,
            "hardCollisionConstraint": False,
            "H": np.zeros((30, 30), dtype=float),
        }
        path = np.array([[5.0, 5.0, 0.2], [15.0, 15.0, 0.2], [25.0, 25.0, 0.2]], dtype=float)
        obj, details = evaluate_mission_details([path], model)
        self.assertTrue(np.all(np.isfinite(obj)))
        self.assertEqual(details["collisionViolation"], 1.0)


if __name__ == "__main__":
    unittest.main()
