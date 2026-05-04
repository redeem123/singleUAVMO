from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from uav_benchmark.algorithms.apex_shade import run_fleet_lshade_cdp
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


class LSHADECDPFleetSmokeTest(unittest.TestCase):
    def test_fleet_lshade_cdp_writes_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=29, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_lshade_cdp_fleet_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                max_turn_deg=75.0,
                gpu_mode="off",
                extra={
                    "resumeExistingRuns": False,
                    "nRep": 6,
                },
            )
            run_fleet_lshade_cdp(multi, params)
            run_dir = Path(tmpdir) / "smoke_lshade_cdp_fleet_uav3" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "conflict_log.mat").exists())


if __name__ == "__main__":
    unittest.main()
