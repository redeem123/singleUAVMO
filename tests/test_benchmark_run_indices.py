from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_support import install_numba_stub

install_numba_stub()

import uav_benchmark.benchmark as benchmark_module
from uav_benchmark.config import BenchmarkParams


class _ImmediateResult:
    def __init__(self, func, args: tuple[object, ...]) -> None:
        self._error: Exception | None = None
        self._value = None
        try:
            self._value = func(*args)
        except Exception as exc:  # pragma: no cover - surfaced by get()
            self._error = exc

    def ready(self) -> bool:
        return True

    def get(self):
        if self._error is not None:
            raise self._error
        return self._value

    def wait(self, timeout: float | None = None) -> None:
        del timeout


class _ImmediatePool:
    def __init__(self, processes: int) -> None:
        self.processes = processes

    def __enter__(self) -> "_ImmediatePool":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    def apply_async(self, func, args: tuple[object, ...] = ()):
        return _ImmediateResult(func, args)


class BenchmarkRunIndicesTest(unittest.TestCase):
    def test_grouped_benchmark_honors_run_index_subset(self) -> None:
        executed: list[int] = []

        def fake_runner(model: dict, params: BenchmarkParams) -> None:
            del model
            assert params.run_indices is not None
            executed.append(int(params.run_indices[0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            problems_dir = project_root / "problems"
            problems_dir.mkdir()
            (problems_dir / "terrainStruct_demo.mat").touch()
            params = BenchmarkParams(
                runs=4,
                run_indices=(2, 4, 2),
                results_dir=project_root / "results",
                scenario_set="custom",
                extra={"algorithms": ["TEST-ALG"], "maxWorkers": 1},
            )

            with patch.dict(benchmark_module._ALGORITHM_REGISTRY, {"TEST-ALG": fake_runner}, clear=False):
                with patch.object(benchmark_module, "load_terrain_struct", return_value={}):
                    with patch.object(benchmark_module.multiprocessing, "Pool", _ImmediatePool):
                        benchmark_module.run_benchmark(project_root, params)

        self.assertEqual(executed, [2, 4])


if __name__ == "__main__":
    unittest.main()
