from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from uav_benchmark.algorithms.sac_smopso import run_sac_smopso
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


class SACSMPSOFleetSmokeTest(unittest.TestCase):
    def test_fleet_sac_smopso_writes_artifacts_and_rl_trace(self) -> None:
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
                problem_name="smoke_sac_smopso_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
                extra={
                    "metricInterval": 1,
                    "sacWarmupSteps": 2,
                    "sacBatchSize": 4,
                    "sacReplayCapacity": 64,
                },
            )
            run_sac_smopso(multi, params)
            run_dir = Path(tmpdir) / "smoke_sac_smopso_uav3" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "conflict_log.mat").exists())
            self.assertTrue((run_dir / "rl_reward.mat").exists())
            self.assertTrue((run_dir / "rl_metadata.mat").exists())

    def test_single_uav_still_runs_sac_controller(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=5,
                runs=1,
                compute_metrics=False,
                results_dir=Path(tmpdir),
                problem_name="smoke_sac_smopso_uav1",
                problem_index=1,
                mode="fleet",
                fleet_size=1,
                separation_min=10.0,
                gpu_mode="off",
                extra={
                    "metricInterval": 1,
                    "sacWarmupSteps": 1,
                    "sacBatchSize": 4,
                    "sacReplayCapacity": 32,
                    "resumeExistingRuns": False,
                },
            )
            run_sac_smopso(terrain, params)
            run_dir = Path(tmpdir) / "smoke_sac_smopso_uav1" / "Run_1"
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "rl_reward.mat").exists())
            self.assertTrue((run_dir / "rl_metadata.mat").exists())

    def test_reward_uses_actual_sbx_eval_count(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=29, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=1,
                population=6,
                runs=1,
                compute_metrics=False,
                results_dir=Path(tmpdir),
                problem_name="reward_eval_count",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
                extra={"resumeExistingRuns": False},
            )
            captured: list[dict[str, float]] = []

            def _capture_reward(
                *, before: dict[str, float], after: dict[str, float], operator_stats: dict[str, float], population: int
            ) -> float:
                captured.append(dict(operator_stats))
                return 0.0

            with (
                mock.patch.dict("os.environ", {"SAC_SMOPSO_FORCE_SBX": "1.0"}, clear=False),
                mock.patch(
                    "uav_benchmark.algorithms.sac_smopso._reservoir_sbx_injection",
                    return_value={"effectCount": 2.0, "evalCount": 7.0},
                ),
                mock.patch(
                    "uav_benchmark.algorithms.sac_smopso._targeted_geometry_repair",
                    return_value={"effectCount": 0.0, "evalCount": 0.0},
                ),
                mock.patch("uav_benchmark.algorithms.sac_smopso._compute_reward", side_effect=_capture_reward),
            ):
                run_sac_smopso(multi, params)

            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["evalCount"], 13.0)

    def test_structured_initial_population_is_rebuilt_per_run(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=31, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=1,
                population=4,
                runs=2,
                compute_metrics=False,
                results_dir=Path(tmpdir),
                problem_name="seeded_init",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
                seed=101,
                extra={"resumeExistingRuns": False},
            )
            draws: list[float] = []

            def _fake_structured_initial_population(
                model: dict[str, object],
                fleet_size: int,
                n_waypoints: int,
                pop_size: int,
                lower: np.ndarray,
                upper: np.ndarray,
                *,
                separation_min: float,
                representation: str,
            ) -> np.ndarray:
                draws.append(float(np.random.rand()))
                return np.tile(np.asarray(lower, dtype=float), (int(pop_size), 1))

            with mock.patch(
                "uav_benchmark.algorithms.sac_smopso._structured_initial_population",
                side_effect=_fake_structured_initial_population,
            ) as mocked_init:
                run_sac_smopso(multi, params)

            self.assertEqual(mocked_init.call_count, 2)
            self.assertEqual(len(draws), 2)
            self.assertNotEqual(draws[0], draws[1])

    def test_force_repair_intensity_controls_repair_calls(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=33, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=1,
                population=4,
                runs=1,
                compute_metrics=False,
                results_dir=Path(tmpdir),
                problem_name="forced_repair",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
                extra={"resumeExistingRuns": False},
            )
            captured: list[float] = []

            def _capture_repair(*args: object, **kwargs: object) -> dict[str, float]:
                repair_intensity = kwargs["repair_intensity"]
                assert isinstance(repair_intensity, (float, int))
                captured.append(float(repair_intensity))
                return {"effectCount": 0.0, "evalCount": 0.0}

            with (
                mock.patch.dict("os.environ", {"SAC_SMOPSO_FORCE_REPAIR_INTENSITY": "0.0"}, clear=False),
                mock.patch(
                    "uav_benchmark.algorithms.sac_smopso._targeted_geometry_repair",
                    side_effect=_capture_repair,
                ),
            ):
                run_sac_smopso(multi, params)

            self.assertEqual(captured, [0.0])


if __name__ == "__main__":
    unittest.main()
