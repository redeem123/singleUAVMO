from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.algorithms.multi_uav import _resume_run_scores
from uav_benchmark.io.matlab import save_mat, save_run_popobj


class MultiResumeTest(unittest.TestCase):
    def test_resume_reads_popobj_written_by_save_run_popobj(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            save_run_popobj(
                run_dir / "final_popobj.mat",
                np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float),
                problem_index=1,
                objective_count=4,
            )
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=1,
                objective_count=4,
                compute_metrics=False,
            )
            self.assertIsNotNone(resume_scores)
            np.testing.assert_allclose(resume_scores, np.zeros(2, dtype=float))

    def test_resume_keeps_legacy_final_popobj_key_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            save_mat(run_dir / "final_popobj.mat", {"final_popobj": np.array([[1.0, 2.0, 3.0, 4.0]])})
            resume_scores = _resume_run_scores(
                run_dir=run_dir,
                problem_index=1,
                objective_count=4,
                compute_metrics=False,
            )
            self.assertIsNotNone(resume_scores)


if __name__ == "__main__":
    unittest.main()
