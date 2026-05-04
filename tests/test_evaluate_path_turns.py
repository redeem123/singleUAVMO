import unittest

import numpy as np

from uav_benchmark.core.evaluate_path import evaluate_path_details


class EvaluatePathTurnTests(unittest.TestCase):
    def test_duplicate_waypoint_does_not_hide_turn(self) -> None:
        model = {
            "H": np.zeros((220, 220), dtype=float),
            "xmin": 1.0,
            "xmax": 220.0,
            "ymin": 1.0,
            "ymax": 220.0,
            "zmin": 0.0,
            "zmax": 120.0,
            "safeDist": 20.0,
            "droneSize": 1.0,
            "maxTurnDeg": 75.0,
            "rmin": 0.0,
        }
        path = np.array(
            [
                [1.0, 1.0, 60.0],
                [200.0, 1.0, 60.0],
                [200.0, 1.0, 60.0],
                [200.0, 200.0, 60.0],
            ],
            dtype=float,
        )

        _objective, details = evaluate_path_details(path, model)

        self.assertGreater(float(details["maxTurnDeg"]), 89.0)


if __name__ == "__main__":
    unittest.main()
