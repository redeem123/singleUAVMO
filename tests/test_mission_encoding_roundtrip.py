from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.core.mission_encoding import decision_size, decision_to_paths, paths_to_decision


class MissionEncodingRoundtripTest(unittest.TestCase):
    def test_roundtrip_shape_and_finiteness(self) -> None:
        model = {
            "xmin": 1.0,
            "xmax": 60.0,
            "ymin": 1.0,
            "ymax": 60.0,
            "zmin": 0.0,
            "zmax": 20.0,
            "safeH": 5.0,
            "H": np.zeros((60, 60), dtype=float),
            "starts": np.array([[2.0, 2.0, 5.0], [3.0, 6.0, 5.0], [4.0, 10.0, 5.0]], dtype=float),
            "goals": np.array([[58.0, 55.0, 5.0], [57.0, 50.0, 5.0], [56.0, 45.0, 5.0]], dtype=float),
        }
        fleet_size = 3
        n_waypoints = 5
        np.random.seed(7)
        d = decision_size(fleet_size, n_waypoints)
        lower = np.tile(np.array([model["xmin"], model["ymin"], model["zmin"]], dtype=float), fleet_size * n_waypoints)
        upper = np.tile(np.array([model["xmax"], model["ymax"], model["zmax"]], dtype=float), fleet_size * n_waypoints)
        decision = np.random.uniform(lower, upper)

        paths = decision_to_paths(decision, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        encoded = paths_to_decision(paths, model=model, fleet_size=fleet_size, n_waypoints=n_waypoints)
        self.assertEqual(encoded.shape, (d,))
        self.assertTrue(np.all(np.isfinite(encoded)))


if __name__ == "__main__":
    unittest.main()

