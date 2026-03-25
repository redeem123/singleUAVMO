from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.algorithms.shared.fleet_runner import _save_fleet_artifacts
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.io.matlab import load_mat


class ConflictLogArtifactTest(unittest.TestCase):
    def test_saves_detailed_conflicts_and_turn_violation_stats(self) -> None:
        path = np.array([[1.0, 1.0, 10.0], [5.0, 5.0, 10.0]], dtype=float)
        candidates = [
            Candidate(
                vector=np.zeros(3, dtype=float),
                objective=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
                details={
                    "paths": [path],
                    "feasible": 1.0,
                    "conflictRate": 0.25,
                    "minSeparation": 8.0,
                    "makespan": 0.1,
                    "energy": 0.2,
                    "risk": 0.3,
                    "maxTurnDeg": 20.0,
                    "turnViolation": 0.0,
                    "separationViolation": 0.0,
                    "collisionViolation": 0.0,
                    "minClearance": 2.0,
                    "conflictLog": np.array(
                        [
                            [0.0, 0.0, 1.0, 4.0, 1.0],
                            [1.0, 0.0, 1.0, 4.5, 0.5],
                        ],
                        dtype=float,
                    ),
                },
            ),
            Candidate(
                vector=np.ones(3, dtype=float),
                objective=np.array([0.4, 0.3, 0.2, 0.1], dtype=float),
                details={
                    "paths": [path],
                    "feasible": 0.0,
                    "conflictRate": 0.0,
                    "minSeparation": np.nan,
                    "makespan": 0.4,
                    "energy": 0.3,
                    "risk": 0.2,
                    "maxTurnDeg": 91.0,
                    "turnViolation": 1.0,
                    "separationViolation": 0.0,
                    "collisionViolation": 0.0,
                    "minClearance": 1.5,
                    "conflictLog": np.zeros((0, 5), dtype=float),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _save_fleet_artifacts(
                run_dir=run_dir,
                final_candidates=candidates,
                problem_index=1,
                objective_count=4,
                runtime_sec=0.01,
                gpu_backend="numpy:cpu",
                gpu_peak_bytes=0.0,
            )

            payload = load_mat(run_dir / "conflict_log.mat")
            np.testing.assert_allclose(
                np.asarray(payload["conflicts"], dtype=float),
                np.array(
                    [
                        [0.0, 0.0, 1.0, 4.0, 1.0],
                        [1.0, 0.0, 1.0, 4.5, 0.5],
                    ],
                    dtype=float,
                ),
            )
            np.testing.assert_allclose(np.asarray(payload["candidateIndex"], dtype=float).reshape(-1), [1.0, 1.0])
            np.testing.assert_allclose(np.asarray(payload["conflictRates"], dtype=float).reshape(-1), [0.25, 0.0])

            mission_stats = load_mat(run_dir / "mission_stats.mat")
            np.testing.assert_allclose(np.asarray(mission_stats["turnViolation"], dtype=float).reshape(-1), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
