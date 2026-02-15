from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from uav_benchmark.algorithms.multi_uav import run_multi_nmopso
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_multi_uav_terrain


class MultiSmokeTest(unittest.TestCase):
    def test_multi_nmopso_writes_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_multi_uav_terrain(terrain, fleet_size=3, seed=13, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_multi_uav3",
                problem_index=1,
                mode="multi",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
            )
            run_multi_nmopso(multi, params)
            run_dir = Path(tmpdir) / "smoke_multi_uav3" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "conflict_log.mat").exists())


if __name__ == "__main__":
    unittest.main()

