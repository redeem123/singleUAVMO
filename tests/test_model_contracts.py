from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.exceptions import ModelValidationError
from uav_benchmark.model_contracts import validate_terrain_model


def _valid_model() -> dict[str, object]:
    return {
        "H": np.zeros((4, 4), dtype=float),
        "xmin": 0.0,
        "xmax": 4.0,
        "ymin": 0.0,
        "ymax": 4.0,
        "zmin": 0.0,
        "zmax": 10.0,
        "n": 3,
        "start": np.asarray([0.0, 0.0, 2.0]),
        "end": np.asarray([4.0, 4.0, 2.0]),
        "safeDist": 1.0,
        "droneSize": 1.0,
    }


class TerrainModelContractTest(unittest.TestCase):
    def test_valid_single_uav_model_passes(self) -> None:
        model = validate_terrain_model(_valid_model())
        self.assertEqual(model["xmax"], 4.0)

    def test_valid_fleet_model_passes(self) -> None:
        model = _valid_model()
        model.pop("start")
        model.pop("end")
        model["starts"] = np.asarray([[0.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
        model["goals"] = np.asarray([[4.0, 4.0, 2.0], [4.0, 3.0, 2.0]])
        model["fleetSize"] = 2.0
        validated = validate_terrain_model(model)
        self.assertEqual(validated["fleetSize"], 2.0)

    def test_missing_height_map_fails_fast(self) -> None:
        model = _valid_model()
        model.pop("H")
        with self.assertRaisesRegex(ModelValidationError, "H.*required"):
            validate_terrain_model(model)

    def test_invalid_bounds_fail_fast(self) -> None:
        model = _valid_model()
        model["xmin"] = 5.0
        with self.assertRaisesRegex(ModelValidationError, "xmin.*xmax"):
            validate_terrain_model(model)

    def test_fleet_size_must_match_endpoint_rows(self) -> None:
        model = _valid_model()
        model.pop("start")
        model.pop("end")
        model["starts"] = np.asarray([[0.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
        model["goals"] = np.asarray([[4.0, 4.0, 2.0], [4.0, 3.0, 2.0]])
        model["fleetSize"] = 3.0
        with self.assertRaisesRegex(ModelValidationError, "fleetSize.*match"):
            validate_terrain_model(model)


if __name__ == "__main__":
    unittest.main()
