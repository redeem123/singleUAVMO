from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_support import install_numba_stub

install_numba_stub()

from uav_benchmark.analysis.metrics.compute import MetricConfig
from uav_benchmark.analysis.metrics.stats import statistical_analysis
from uav_benchmark.io.matlab import load_mat, save_mat, save_run_popobj


class StatisticalAnalysisDeterminismTest(unittest.TestCase):
    def test_sampling_respects_metric_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            run_dir = results_dir / "ALG" / "prob" / "Run_1"
            run_dir.mkdir(parents=True)
            pop_obj = np.array(
                [[0.01 * idx, 0.02 * idx, 0.03 * idx, 0.04 * idx] for idx in range(1, 21)],
                dtype=float,
            )
            save_run_popobj(run_dir / "final_popobj.mat", pop_obj, problem_index=1, objective_count=4)
            save_mat(
                run_dir / "mission_stats.mat",
                {
                    "feasible": np.ones(pop_obj.shape[0], dtype=float),
                    "collisionViolation": np.zeros(pop_obj.shape[0], dtype=float),
                    "separationViolation": np.zeros(pop_obj.shape[0], dtype=float),
                    "conflictRate": np.zeros(pop_obj.shape[0], dtype=float),
                },
            )

            cfg = MetricConfig(hv_samples=128, max_points=5, seed=17)

            np.random.seed(1)
            first = statistical_analysis(results_dir, cfg)
            first_hv = first["ALG"][0].mean_hv
            first_saved = load_mat(results_dir / "ALG" / "prob" / "final_hv.mat")["bestScores"]

            np.random.seed(999)
            second = statistical_analysis(results_dir, cfg)
            second_hv = second["ALG"][0].mean_hv
            second_saved = load_mat(results_dir / "ALG" / "prob" / "final_hv.mat")["bestScores"]

            self.assertAlmostEqual(first_hv, second_hv, places=12)
            np.testing.assert_allclose(first_saved, second_saved)
