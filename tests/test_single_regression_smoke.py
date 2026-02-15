from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from uav_benchmark.algorithms.nmopso import run_nmopso
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct


class SingleRegressionSmokeTest(unittest.TestCase):
    def test_single_nmopso_still_runs(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=8,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_single",
                problem_index=1,
                mode="single",
            )
            run_nmopso(terrain, params)
            run_dir = Path(tmpdir) / "smoke_single" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "run_stats.mat").exists())


if __name__ == "__main__":
    unittest.main()

