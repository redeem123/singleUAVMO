from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_support import install_numba_stub

install_numba_stub()

import uav_benchmark.benchmark as benchmark_module
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat, save_mat


def _valid_terrain() -> dict[str, object]:
    return {
        "H": np.zeros((4, 4), dtype=float),
        "xmin": 0.0,
        "xmax": 4.0,
        "ymin": 0.0,
        "ymax": 4.0,
        "zmin": 0.0,
        "zmax": 10.0,
        "n": 3,
        "start": np.asarray([0.0, 0.0, 2.0]),
        "end": np.asarray([4.0, 4.0, 2.0]),
    }


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

    def __enter__(self) -> _ImmediatePool:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    def apply_async(self, func, args: tuple[object, ...] = ()):
        return _ImmediateResult(func, args)


class BenchmarkRunIndicesTest(unittest.TestCase):
    def test_paper_medium_generated_scenarios_stay_under_results_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            params = BenchmarkParams(
                results_dir=project_root / "results",
                scenario_set="paper_medium",
                seed=3,
                separation_min=12.0,
            )

            with patch.object(benchmark_module, "save_fleet_scenarios", return_value=[]) as mocked_save:
                benchmark_module._maybe_generate_fleet_scenarios(project_root, params, (1, 3))

        self.assertEqual(mocked_save.call_args.kwargs["output_dir"], params.results_dir / "generated_problems")

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
                with patch.object(benchmark_module, "load_terrain_struct", return_value=_valid_terrain()):
                    with patch.object(benchmark_module.multiprocessing, "Pool", _ImmediatePool):
                        benchmark_module.run_benchmark(project_root, params)

        self.assertEqual(executed, [2, 4])

    def test_grouped_hv_summary_uses_only_requested_completed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            problem_dir = results_dir / "ALG" / "demo"
            for run_index in (2, 4):
                run_dir = problem_dir / f"Run_{run_index}"
                run_dir.mkdir(parents=True)
                save_mat(
                    run_dir / "final_popobj.mat",
                    {"PopObj": np.asarray([[float(run_index), 1.0, 2.0, 3.0]], dtype=float)},
                )
            params = BenchmarkParams(
                runs=4,
                run_indices=(2, 4),
                compute_metrics=True,
                results_dir=results_dir,
            )

            def fake_metric(metric_id: int, matrix: np.ndarray, problem_index: int, objective_count: int) -> float:
                del problem_index, objective_count
                return float(matrix[0, 0] * 10.0 + metric_id)

            with patch.object(benchmark_module, "cal_metric", side_effect=fake_metric):
                benchmark_module._write_grouped_run_hv_summary(
                    params=params,
                    algorithm_label="ALG",
                    problem_name="demo",
                    problem_index=3,
                )

            payload = load_mat(problem_dir / "final_hv.mat")
            self.assertEqual(np.asarray(payload["bestScores"], dtype=float).tolist(), [[21.0, 22.0], [41.0, 42.0]])
            self.assertEqual(np.asarray(payload["runIndices"], dtype=int).reshape(-1).tolist(), [2, 4])


if __name__ == "__main__":
    unittest.main()
