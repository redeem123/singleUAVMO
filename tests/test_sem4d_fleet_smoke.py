from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.algorithms.sem4d import _candidate_from_paths, _shield_config, run_sem4d
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat, load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


class SEM4DFleetSmokeTest(unittest.TestCase):
    def test_sem4d_writes_shielded_fleet_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=17, separation_min=10.0)
        multi["dynamicObstacles"] = np.asarray(
            [[50.0, 50.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="sem4d_smoke_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                max_turn_deg=80.0,
                gpu_mode="off",
                extra={
                    "resumeExistingRuns": False,
                    "sem4dShieldIterations": 2,
                    "sem4dTimeSamples": 16,
                    "sem4dMaxShieldInsertions": 8,
                    "metricInterval": 1,
                },
            )
            scores = run_sem4d(multi, params)
            run_dir = Path(tmpdir) / "sem4d_smoke_uav3" / "Run_1"
            self.assertEqual(scores.shape, (1, 2))
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "conflict_log.mat").exists())
            self.assertTrue((run_dir / "sem4d_shield.mat").exists())
            shield = load_mat(run_dir / "sem4d_shield.mat")
            self.assertIn("correctionNorm", shield)
            self.assertIn("terrainCorrections", shield)
            self.assertIn("postShieldConflictRate", shield)

    def test_optional_dynamic_and_energy_extensions_are_hard_gates(self) -> None:
        model = {
            "H": np.zeros((20, 20), dtype=float),
            "xmin": 1.0,
            "xmax": 20.0,
            "ymin": 1.0,
            "ymax": 20.0,
            "zmin": 1.0,
            "zmax": 50.0,
            "safeH": 5.0,
            "safeDist": 2.0,
            "droneSize": 1.0,
            "separationMin": 2.0,
            "starts": np.asarray([[1.0, 10.0, 5.0]], dtype=float),
            "goals": np.asarray([[20.0, 10.0, 5.0]], dtype=float),
            "dynamicObstacles": np.asarray(
                [[10.0, 10.0, 5.0, 50.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=float,
            ),
            "hardCollisionConstraint": True,
            "maxTurnDeg": 120.0,
        }
        params = BenchmarkParams(
            fleet_size=1,
            safe_dist=2.0,
            drone_size=1.0,
            separation_min=2.0,
            max_turn_deg=120.0,
            extra={
                "sem4dEnergyMax": 1.0,
                "sem4dShieldIterations": 1,
                "sem4dTimeSamples": 8,
                "sem4dRepairGain": 0.05,
            },
        )
        path = np.asarray(
            [[1.0, 10.0, 5.0], [10.0, 10.0, 5.0], [20.0, 10.0, 5.0]],
            dtype=float,
        )
        candidate = _candidate_from_paths(
            vector=np.zeros(3, dtype=float),
            paths=[path],
            model=model,
            params=params,
            config=_shield_config(model, params),
            task_id=0,
        )
        self.assertFalse(np.all(np.isfinite(candidate.objective)))
        self.assertEqual(float(candidate.details["feasible"]), 0.0)
        self.assertEqual(float(candidate.details["postShieldDynamicObstacleViolation"]), 1.0)
        self.assertEqual(float(candidate.details["postShieldEnergyViolation"]), 1.0)


if __name__ == "__main__":
    unittest.main()
