from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_support import install_numba_stub

install_numba_stub()

from uav_benchmark import benchmark


class BenchmarkWorkerPythonTest(unittest.TestCase):
    def test_configure_worker_python_executable_uses_current_interpreter(self) -> None:
        expected = str(Path(sys.executable).absolute())
        with mock.patch.object(benchmark.multiprocessing, "set_executable") as mocked_set_executable:
            actual = benchmark._configure_worker_python_executable()
        self.assertEqual(actual, expected)
        mocked_set_executable.assert_called_once_with(expected)

    def test_torch_accelerated_algorithms_do_not_parallelize_when_gpu_enabled(self) -> None:
        params = benchmark.BenchmarkParams(gpu_mode="force")
        self.assertFalse(benchmark._can_parallelize_runs("SAC-SMOPSO", params))
        self.assertFalse(benchmark._can_parallelize_runs("RA-SMPSO", params))
        self.assertFalse(benchmark._can_parallelize_runs("RA-NSGA-II", params))

    def test_torch_accelerated_algorithms_can_parallelize_when_gpu_off(self) -> None:
        params = benchmark.BenchmarkParams(gpu_mode="off")
        self.assertTrue(benchmark._can_parallelize_runs("SAC-SMOPSO", params))


if __name__ == "__main__":
    unittest.main()
