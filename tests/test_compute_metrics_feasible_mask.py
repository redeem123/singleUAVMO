from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.analysis.metrics.compute import _load_feasible_mask
from uav_benchmark.io.matlab import save_mat


class ComputeMetricsFeasibleMaskTest(unittest.TestCase):
    def test_uses_feasible_vector_when_shape_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_mat(run_dir / "mission_stats.mat", {"feasible": np.array([1.0, 0.0, 1.0], dtype=float)})
            mask = _load_feasible_mask(run_dir, 3)
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_ignores_mismatched_feasible_and_uses_violation_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_mat(
                run_dir / "mission_stats.mat",
                {
                    "feasible": np.array([0.0], dtype=float),
                    "turnViolation": np.array([0.0, 1.0, 0.0], dtype=float),  # ignored for feasibility
                    "separationViolation": np.array([0.0, 0.0, 0.0], dtype=float),
                    "collisionViolation": np.array([0.0, 0.0, 1.0], dtype=float),
                },
            )
            mask = _load_feasible_mask(run_dir, 3)
        self.assertEqual(mask.tolist(), [True, True, False])


if __name__ == "__main__":
    unittest.main()
