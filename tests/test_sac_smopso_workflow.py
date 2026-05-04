from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.algorithms.sac_smopso.workflow import (
    aggregate_records,
    first_positive_generation,
    load_rollout_summary,
    stage_preset,
    validate_multi_uav_fleet_sizes,
)
from uav_benchmark.io.matlab import save_mat


class SACSMOPSOWorkflowTest(unittest.TestCase):
    def test_stage_preset_stage2_matches_expected_budget(self) -> None:
        preset = stage_preset("stage2")
        self.assertEqual(preset["fleetSizes"], [3])
        self.assertEqual(preset["generations"], 16)
        self.assertEqual(preset["population"], 12)

    def test_paper_mixed_12_stays_multi_uav_only(self) -> None:
        preset = stage_preset("paper_mixed_12")
        self.assertEqual(preset["fleetSizes"], [2])
        self.assertNotIn(1, preset["fleetSizes"])

    def test_first_positive_generation_returns_first_hit(self) -> None:
        values = np.array([0.0, 0.2, 0.6, 0.9], dtype=float)
        self.assertEqual(first_positive_generation(values), 3)
        self.assertEqual(first_positive_generation(np.zeros(4, dtype=float)), 0)

    def test_validate_fleet_sizes_requires_multi_uav_only(self) -> None:
        self.assertEqual(validate_multi_uav_fleet_sizes([2, 3, 5]), [2, 3, 5])
        with self.assertRaises(ValueError):
            validate_multi_uav_fleet_sizes([1, 2])
        with self.assertRaises(ValueError):
            validate_multi_uav_fleet_sizes([0, 3])

    def test_load_rollout_summary_extracts_trace_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            save_mat(run_dir / "rl_feasible.mat", {"rl_feasible": np.array([0.0, 0.0, 1.0], dtype=float)})
            save_mat(run_dir / "rl_conflict.mat", {"rl_conflict": np.array([0.2, 0.1, 0.05], dtype=float)})
            save_mat(run_dir / "rl_reward.mat", {"rl_reward": np.array([1.0, 2.0, 3.0], dtype=float)})
            summary = load_rollout_summary(run_dir)
            self.assertEqual(summary["firstFeasibleGeneration"], 3.0)
            self.assertAlmostEqual(summary["finalTraceConflict"], 0.05)
            self.assertAlmostEqual(summary["meanReward"], 2.0)

    def test_aggregate_records_averages_requested_keys(self) -> None:
        records = [
            {"hypervolume": 1.0, "conflictMean": 0.4},
            {"hypervolume": 3.0, "conflictMean": 0.2},
        ]
        summary = aggregate_records(records, ("hypervolume", "conflictMean"))
        self.assertAlmostEqual(summary["hypervolumeMean"], 2.0)
        self.assertAlmostEqual(summary["conflictMeanMean"], 0.3)


if __name__ == "__main__":
    unittest.main()
