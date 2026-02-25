from __future__ import annotations

import unittest
from pathlib import Path

from uav_benchmark.algorithms.shared.fleet_runner import _resolve_run_indices, _should_write_final_hv
from uav_benchmark.benchmark import (
    _can_parallelize_runs,
    _max_parallel_worker_slots,
    _next_dispatchable_task,
)
from uav_benchmark.config import BenchmarkParams


class ParallelSchedulerTest(unittest.TestCase):
    def test_run_parallel_guard_allows_registered_algorithms(self) -> None:
        baseline = BenchmarkParams(mode="fleet", runs=4, extra={})
        self.assertTrue(_can_parallelize_runs("NMOPSO", baseline))
        self.assertTrue(_can_parallelize_runs("MOQGWO", baseline))

    def test_run_index_override_and_final_hv_skip_flag(self) -> None:
        params_default = BenchmarkParams(runs=4, extra={})
        self.assertEqual(_resolve_run_indices(params_default), (1, 2, 3, 4))
        self.assertTrue(_should_write_final_hv(params_default))

        params_override = BenchmarkParams(runs=4, run_indices=(2, 4, 2), write_final_hv=False)
        self.assertEqual(_resolve_run_indices(params_override), (2, 4))
        self.assertFalse(_should_write_final_hv(params_override))

    def test_dispatch_prefers_first_task_until_it_has_no_pending(self) -> None:
        pending = [[1, 2], [1, 2]]
        active = [0, 0]
        limit = [2, 2]

        first = _next_dispatchable_task(pending, active, limit)
        self.assertEqual(first, 0)

        # First task saturated, scheduler should spill to next task.
        active[0] = 2
        second = _next_dispatchable_task(pending, active, limit)
        self.assertEqual(second, 1)

        # First task has running jobs but no pending left; next task should run.
        pending[0] = []
        active[0] = 1
        third = _next_dispatchable_task(pending, active, limit)
        self.assertEqual(third, 1)

    def test_max_parallel_worker_slots_counts_run_level_parallelism(self) -> None:
        params_baseline = BenchmarkParams(mode="fleet", runs=4, extra={})
        tasks = [
            (Path("a"), 1, "NMOPSO", params_baseline),
            (Path("b"), 2, "NSGA-II", params_baseline),
            (Path("c"), 3, "MOQGWO", params_baseline),
        ]
        self.assertEqual(_max_parallel_worker_slots(tasks), 12)


if __name__ == "__main__":
    unittest.main()
