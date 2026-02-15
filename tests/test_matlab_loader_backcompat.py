from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.io.matlab import load_terrain_struct, save_mat


class MatlabLoaderBackcompatTest(unittest.TestCase):
    def test_promotes_start_end_to_starts_goals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terrainStruct_tmp.mat"
            terrain_struct = {
                "start": np.array([1.0, 2.0, 3.0], dtype=float),
                "end": np.array([10.0, 12.0, 4.0], dtype=float),
                "xmin": 1.0,
                "xmax": 20.0,
                "ymin": 1.0,
                "ymax": 20.0,
                "zmin": 0.0,
                "zmax": 15.0,
                "X": np.arange(1, 21, dtype=float),
                "Y": np.arange(1, 21, dtype=float),
                "H": np.zeros((20, 20), dtype=float),
            }
            save_mat(path, {"terrainStruct": terrain_struct})
            loaded = load_terrain_struct(path)
            self.assertIn("starts", loaded)
            self.assertIn("goals", loaded)
            self.assertEqual(loaded["starts"].shape, (1, 3))
            self.assertEqual(loaded["goals"].shape, (1, 3))
            self.assertEqual(int(loaded["fleetSize"]), 1)


if __name__ == "__main__":
    unittest.main()

