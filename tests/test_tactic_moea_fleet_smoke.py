from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.algorithms.tactic_moea import _extract_topology, run_tactic_moea
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.core.evaluate_mission import evaluate_mission_details
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


class TACTICMOEAFleetSmokeTest(unittest.TestCase):
    def test_pair_conflict_topology_is_detected(self) -> None:
        model = {
            "xmin": 1.0,
            "xmax": 50.0,
            "ymin": 1.0,
            "ymax": 50.0,
            "zmin": 0.0,
            "zmax": 20.0,
            "safeDist": 5.0,
            "droneSize": 1.0,
            "separationMin": 5.0,
            "maxTurnDeg": 90.0,
            "H": np.zeros((50, 50), dtype=float),
            "_tacticObstacleMatrix": np.zeros((0, 4), dtype=float),
        }
        path_a = np.array([[2.0, 2.0, 10.0], [20.0, 20.0, 10.0], [40.0, 40.0, 10.0]], dtype=float)
        path_b = np.array([[2.3, 2.2, 10.0], [20.2, 20.0, 10.0], [40.2, 40.1, 10.0]], dtype=float)
        objective, details = evaluate_mission_details([path_a, path_b], model)
        candidate = Candidate(
            vector=np.zeros(6, dtype=float),
            objective=objective,
            details={**details, "paths": [path_a, path_b]},
        )

        topology = _extract_topology(candidate, model)

        self.assertEqual(topology.issue_code, 1)
        self.assertIn(topology.target_uav, {0, 1})
        self.assertGreater(topology.severity, 0.0)

    def test_fleet_tactic_moea_writes_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=37, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_tactic_fleet_uav3",
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
            run_tactic_moea(multi, params)
            run_dir = Path(tmpdir) / "smoke_tactic_fleet_uav3" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "conflict_log.mat").exists())


if __name__ == "__main__":
    unittest.main()
