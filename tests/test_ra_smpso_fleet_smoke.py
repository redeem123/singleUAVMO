from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from uav_benchmark.algorithms.ra_smpso import run_ra_smpso
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


class RASMPSOFleetSmokeTest(unittest.TestCase):
    def test_fleet_ra_smpso_writes_artifacts_and_rl_trace(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=23, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=3,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_ra_smpso_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
                extra={"metricInterval": 1, "sacWarmupSteps": 2, "sacBatchSize": 4, "sacReplayCapacity": 64},
            )
            run_ra_smpso(multi, params)
            run_dir = Path(tmpdir) / "smoke_ra_smpso_uav3" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "rl_reward.mat").exists())
            self.assertTrue((run_dir / "rl_metadata.mat").exists())


if __name__ == "__main__":
    unittest.main()
