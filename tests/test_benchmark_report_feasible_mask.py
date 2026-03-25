from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_support import install_numba_stub

install_numba_stub()

from uav_benchmark.analysis.metrics.report import (
    ReportConfig,
    _load_feasible_mask,
    _load_mission_metric,
    generate_benchmark_report,
)
from uav_benchmark.io.matlab import save_mat, save_run_popobj


class BenchmarkReportFeasibleMaskTest(unittest.TestCase):
    def test_prefers_mission_feasible_flag(self) -> None:
        pop_obj = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [np.inf, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_mat(
                run_dir / "mission_stats.mat",
                {
                    "feasible": np.array([1.0, 0.0, 1.0], dtype=float),
                    "turnViolation": np.array([0.0, 0.0, 0.0], dtype=float),
                    "separationViolation": np.array([0.0, 0.0, 0.0], dtype=float),
                },
            )
            mask = _load_feasible_mask(run_dir, pop_obj)
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_fallback_uses_violation_flags_when_feasible_missing(self) -> None:
        pop_obj = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_mat(
                run_dir / "mission_stats.mat",
                {
                    "collisionViolation": np.array([1.0, 0.0], dtype=float),
                    "separationViolation": np.array([0.0, 0.0], dtype=float),
                },
            )
            mask = _load_feasible_mask(run_dir, pop_obj)
        self.assertEqual(mask.tolist(), [False, True])

    def test_ignores_mismatched_feasible_vector_and_uses_full_length_violation_masks(self) -> None:
        pop_obj = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_mat(
                run_dir / "mission_stats.mat",
                {
                    # Legacy/stale scalar should be ignored when length != PopObj rows.
                    "feasible": np.array([0.0], dtype=float),
                    "turnViolation": np.array([0.0, 1.0, 0.0], dtype=float),
                    "separationViolation": np.array([0.0, 0.0, 0.0], dtype=float),
                    "collisionViolation": np.array([0.0, 1.0, 0.0], dtype=float),
                },
            )
            mask = _load_feasible_mask(run_dir, pop_obj)
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_mission_metric_uses_min_and_max_aggregations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_mat(
                run_dir / "mission_stats.mat",
                {
                    "minClearance": np.array([0.7, 0.25, 0.5], dtype=float),
                    "maxTurnDeg": np.array([15.0, 80.0, 45.0], dtype=float),
                },
            )
            self.assertEqual(_load_mission_metric(run_dir, "minClearance"), 0.25)
            self.assertEqual(_load_mission_metric(run_dir, "maxTurnDeg"), 80.0)

    def test_report_prefers_saved_fleet_metrics_over_path_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            run_dir = results_dir / "ALG" / "c_100_uav3" / "Run_1"
            run_dir.mkdir(parents=True)
            save_run_popobj(
                run_dir / "final_popobj.mat",
                np.array([[0.2, 0.3, 0.4, 0.5]], dtype=float),
                problem_index=1,
                objective_count=4,
            )
            save_mat(
                run_dir / "mission_stats.mat",
                {
                    "feasible": np.array([1.0], dtype=float),
                    "turnViolation": np.array([0.0], dtype=float),
                    "separationViolation": np.array([0.0], dtype=float),
                    "collisionViolation": np.array([0.0], dtype=float),
                    "minClearance": np.array([0.25], dtype=float),
                    "maxTurnDeg": np.array([91.0], dtype=float),
                    "conflictRate": np.array([0.0], dtype=float),
                    "minSeparation": np.array([10.0], dtype=float),
                    "makespan": np.array([0.2], dtype=float),
                    "energy": np.array([0.3], dtype=float),
                },
            )
            save_mat(run_dir / "run_stats.mat", {"runtimeSec": 1.0})
            report = generate_benchmark_report(
                ReportConfig(
                    project_root=root,
                    results_dir=results_dir,
                    baseline_algorithm="ALG",
                )
            )
            self.assertEqual(report["summary_rows"], 1)
            payload = (results_dir / "metrics" / "benchmark_metrics_summary.json").read_text(encoding="utf-8")
            self.assertIn('"min_clearance_min": 0.25', payload)
            self.assertIn('"max_turn_deg_max": 91.0', payload)


if __name__ == "__main__":
    unittest.main()
