from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from uav_benchmark.algorithms.gcnmoea import _cosine_similarity, run_gcnmoea
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


class GCNMOEAFleetSmokeTest(unittest.TestCase):
    def test_cosine_similarity_handles_nonfinite_and_large_values(self) -> None:
        left = np.array([[np.inf, 1.0, 0.0], [1e308, 1e308, 0.0]], dtype=float)
        right = np.array([[1.0, 0.0, 0.0], [1e308, -1e308, 0.0]], dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            similarity = _cosine_similarity(left, right)

        self.assertEqual(similarity.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(similarity)))
        self.assertTrue(np.all(similarity <= 1.0))
        self.assertTrue(np.all(similarity >= -1.0))

    def test_fleet_gcnmoea_writes_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=23, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_fleet_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
            )
            run_gcnmoea(multi, params)
            run_dir = Path(tmpdir) / "smoke_fleet_uav3" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "conflict_log.mat").exists())


if __name__ == "__main__":
    unittest.main()
