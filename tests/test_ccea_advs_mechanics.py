from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.algorithms.ccea_advs import (
    adapt_jade_means,
    advs_probabilities,
    jade_current_to_pbest_trials,
    run_ccea_advs,
    select_advs_variables,
    trim_archive,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat, load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


def _mat_string(value: object) -> str:
    arr = np.asarray(value).reshape(-1)
    return str(arr[0]).strip() if arr.size else ""


class CceaAdvsMechanicsTest(unittest.TestCase):
    def test_single_uav_infeasible_paths_are_prioritized(self) -> None:
        rng = np.random.default_rng(3)
        selection = select_advs_variables(
            np.asarray([False, True, False, True]),
            np.asarray([10.0, 0.0, 7.0, 0.0]),
            np.ones(3),
            nsel=2,
            rng=rng,
        )
        self.assertEqual(selection.reason, "single_constraints")
        self.assertEqual(selection.strategy, 0)
        self.assertEqual(selection.selected.tolist(), [1, 3])

    def test_conflict_greedy_selects_heaviest_uavs(self) -> None:
        rng = np.random.default_rng(4)
        selection = select_advs_variables(
            np.zeros(5, dtype=bool),
            np.asarray([0.0, 5.0, 2.0, 9.0, 1.0]),
            np.ones(3),
            nsel=3,
            rng=rng,
            forced_strategy=1,
        )
        self.assertEqual(selection.strategy, 1)
        self.assertEqual(selection.selected.tolist(), [3, 1, 2])

    def test_roulette_probability_increases_with_conflict_count(self) -> None:
        probs = advs_probabilities(np.asarray([0.0, 1.0, 5.0, 10.0]))
        self.assertGreater(probs[3], probs[2])
        self.assertGreater(probs[2], probs[1])
        self.assertEqual(probs[0], 0.0)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0)

    def test_random_strategy_selects_unique_ids_up_to_nsel(self) -> None:
        rng = np.random.default_rng(5)
        selection = select_advs_variables(
            np.zeros(4, dtype=bool),
            np.zeros(4, dtype=float),
            np.ones(3),
            nsel=8,
            rng=rng,
        )
        self.assertEqual(selection.strategy, 3)
        self.assertEqual(len(selection.selected), 4)
        self.assertEqual(len(set(selection.selected.tolist())), 4)

    def test_jade_trials_remain_within_bounds(self) -> None:
        rng = np.random.default_rng(6)
        population = rng.random((8, 6))
        fitness = np.linspace(0.0, 1.0, 8)
        lower = np.zeros(6)
        upper = np.ones(6)
        trials, sampled_f, sampled_cr = jade_current_to_pbest_trials(
            population,
            fitness,
            np.zeros((0, 6)),
            lower,
            upper,
            mu_f=0.5,
            mu_cr=0.5,
            p_best_rate=0.25,
            rng=rng,
        )
        self.assertEqual(trials.shape, population.shape)
        self.assertTrue(np.all(trials >= lower.reshape(1, -1)))
        self.assertTrue(np.all(trials <= upper.reshape(1, -1)))
        self.assertTrue(np.all(sampled_f > 0.0))
        self.assertTrue(np.all((sampled_cr >= 0.0) & (sampled_cr <= 1.0)))

    def test_archive_size_never_exceeds_population(self) -> None:
        rng = np.random.default_rng(7)
        archive = rng.random((12, 4))
        trimmed = trim_archive(archive, max_size=5, rng=rng)
        self.assertEqual(trimmed.shape, (5, 4))

    def test_successful_jade_parameters_update_adaptive_means(self) -> None:
        next_f, next_cr = adapt_jade_means(
            0.5,
            0.5,
            np.asarray([0.8, 0.9]),
            np.asarray([0.2, 0.4]),
            c=0.2,
        )
        self.assertGreater(next_f, 0.5)
        self.assertLess(next_cr, 0.5)


class CceaAdvsSmokeTest(unittest.TestCase):
    def test_smoke_run_writes_standard_artifacts_and_metadata(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 2
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=19, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=2,
                population=20,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="ccea_advs_smoke_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                max_turn_deg=80.0,
                gpu_mode="off",
                extra={
                    "resumeExistingRuns": False,
                    "metricInterval": 1,
                    "cceaAdvsNiter1": 1,
                    "cceaAdvsNiter2": 1,
                    "cceaAdvsJadeInnerIterations": 1,
                    "cceaAdvsNsel": 2,
                },
            )
            scores = run_ccea_advs(multi, params)
            run_dir = Path(tmpdir) / "ccea_advs_smoke_uav3" / "Run_1"
            self.assertEqual(scores.shape, (1, 2))
            self.assertTrue((run_dir / "final_popobj.mat").exists())
            self.assertTrue((run_dir / "fleet_paths.mat").exists())
            self.assertTrue((run_dir / "mission_stats.mat").exists())
            self.assertTrue((run_dir / "run_stats.mat").exists())
            self.assertTrue((run_dir / "gen_hv.mat").exists())
            self.assertTrue((run_dir / "ccea_advs_trace.mat").exists())
            stats = load_mat(run_dir / "run_stats.mat")
            self.assertEqual(_mat_string(stats["algorithmName"]), "CCEA-ADVS")
            self.assertEqual(_mat_string(stats["dubinsRefinement"]), "benchmark_approximation")
            self.assertEqual(int(np.asarray(stats["cceaAdvsNiter1"]).reshape(-1)[0]), 1)


if __name__ == "__main__":
    unittest.main()
