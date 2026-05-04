from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from uav_benchmark.algorithms.fastr_moea import _build_relaxed_model, run_fastr_moea
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


class FASTRMOEAFleetSmokeTest(unittest.TestCase):
    def test_relaxed_model_tightens_toward_strict_constraints(self) -> None:
        model = {
            "separationMin": 10.0,
            "safeDist": 20.0,
            "nofly_r": [12.0, 18.0],
            "turnSpikePenaltyWeight": 1.0,
        }
        params = BenchmarkParams(extra={})

        early = _build_relaxed_model(model, params, progress=0.0)
        late = _build_relaxed_model(model, params, progress=1.0)

        self.assertLess(float(early["separationMin"]), float(late["separationMin"]))
        self.assertLess(float(early["safeDist"]), float(late["safeDist"]))
        self.assertFalse(bool(early["hardCollisionConstraint"]))
        self.assertTrue(bool(late["hardCollisionConstraint"]))
        self.assertAlmostEqual(float(late["separationMin"]), float(model["separationMin"]))
        self.assertAlmostEqual(float(late["safeDist"]), float(model["safeDist"]))

    def test_fleet_fastr_moea_writes_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=31, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_fastr_fleet_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                max_turn_deg=75.0,
                gpu_mode="off",
                extra={
                    "resumeExistingRuns": False,
                    "nRep": 6,
                    "fastrLocalSearchShare": 0.0,
                },
            )
            run_fastr_moea(multi, params)
            run_dir = Path(tmpdir) / "smoke_fastr_fleet_uav3" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "conflict_log.mat").exists())


if __name__ == "__main__":
    unittest.main()
