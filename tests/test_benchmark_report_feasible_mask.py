from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.analysis.benchmark_report import _load_feasible_mask
from uav_benchmark.io.matlab import save_mat


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
                    "turnViolation": np.array([1.0, 0.0], dtype=float),
                    "separationViolation": np.array([0.0, 0.0], dtype=float),
                },
            )
            mask = _load_feasible_mask(run_dir, pop_obj)
        self.assertEqual(mask.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
