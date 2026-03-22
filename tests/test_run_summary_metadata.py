from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_support import install_numba_stub

install_numba_stub()

from uav_benchmark.algorithms.shared.fleet_runner import _save_fleet_artifacts
from uav_benchmark.algorithms.shared.pso_types import Candidate


class RunSummaryMetadataTest(unittest.TestCase):
    def test_shared_fleet_writer_uses_run_path_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "ALG" / "prob" / "Run_1"
            candidate = Candidate(
                vector=np.array([0.1, 0.2], dtype=float),
                objective=np.array([0.2, 0.3, 0.4, 0.5], dtype=float),
                details={
                    "paths": [np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=float)],
                    "feasible": 1.0,
                    "conflictRate": 0.0,
                    "minSeparation": 10.0,
                    "makespan": 0.2,
                    "energy": 0.3,
                    "risk": 0.4,
                    "maxTurnDeg": 20.0,
                    "separationViolation": 0.0,
                    "collisionViolation": 0.0,
                    "minClearance": 0.9,
                },
            )

            _save_fleet_artifacts(
                run_dir=run_dir,
                final_candidates=[candidate],
                problem_index=1,
                objective_count=4,
                runtime_sec=1.0,
                gpu_backend="numpy:cpu",
                gpu_peak_bytes=0.0,
            )

            payload = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["algorithm"], "ALG")
            self.assertEqual(payload["metadata"]["problem"], "prob")
            self.assertEqual(payload["metadata"]["fleet_size"], 1)
