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
import uav_benchmark.cli as cli_module
from uav_benchmark.algorithms.nmopso import create_grid
from uav_benchmark.algorithms.shared.nmopso_engine import _sbx_mutation
from uav_benchmark.config import BenchmarkParams


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


class NMOPSOAblationFlowTest(unittest.TestCase):
    def test_create_grid_ignores_nonfinite_objective_bounds(self) -> None:
        costs = np.array(
            [
                [np.inf, 1.0, 2.0, 3.0],
                [5.0, np.nan, 4.0, np.inf],
                [6.0, 2.0, 5.0, 7.0],
            ],
            dtype=float,
        )

        lower, upper = create_grid(costs, n_grid=5, alpha=0.1)

        self.assertEqual(lower.shape, (4, 7))
        self.assertEqual(upper.shape, (4, 7))
        self.assertTrue(np.all(np.isfinite(lower[:, 1:])))
        self.assertTrue(np.all(np.isfinite(upper[:, :-1])))

    def test_sbx_mutation_uses_local_operator_defaults(self) -> None:
        parents = np.array([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]], dtype=float)
        lower = np.zeros(3, dtype=float)
        upper = np.ones(3, dtype=float)

        offspring = _sbx_mutation(parents, lower, upper)

        self.assertEqual(offspring.shape, parents.shape)
        self.assertTrue(np.all(offspring >= lower))
        self.assertTrue(np.all(offspring <= upper))

    def test_run_nmopso_ablation_uses_registry_and_legacy_runner(self) -> None:
        captured: list[tuple[dict, BenchmarkParams]] = []

        def fake_runner(model: dict, params: BenchmarkParams) -> None:
            captured.append((model, params))

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            problems_dir = project_root / "problems"
            problems_dir.mkdir()
            (problems_dir / "terrainStruct_demo.mat").touch()
            (problems_dir / "terrainStruct_demo_uav3.mat").touch()
            params = BenchmarkParams(results_dir=project_root / "results", seed=7)

            with patch.dict(benchmark_module._ALGORITHM_REGISTRY, {"NMOPSO": fake_runner}, clear=False):
                with patch.object(benchmark_module, "load_terrain_struct", return_value=_valid_terrain()):
                    benchmark_module.run_nmopso_ablation(project_root, params)

        self.assertEqual(len(captured), 1)
        model, run_params = captured[0]
        self.assertIsInstance(run_params, BenchmarkParams)
        assert isinstance(run_params, BenchmarkParams)
        self.assertEqual(run_params.problem_name, "demo")
        self.assertEqual(run_params.problem_index, 1)
        self.assertTrue(run_params.extra.get("ablationStudy"))
        self.assertTrue(run_params.extra.get("legacyPathRunner"))

        self.assertIsInstance(model, dict)
        assert isinstance(model, dict)
        self.assertEqual(model["safeDist"], 20.0)
        self.assertEqual(model["droneSize"], 1.0)

    def test_cli_ablation_enables_legacy_runner(self) -> None:
        captured: dict[str, object] = {}

        def fake_run_nmopso_ablation(project_root: Path, params: BenchmarkParams) -> None:
            captured["project_root"] = project_root
            captured["params"] = params

        with tempfile.TemporaryDirectory() as tmpdir:
            expected_project_root = Path(tmpdir).resolve()
            argv = [
                "uav-benchmark",
                "ablation",
                "--project-root",
                tmpdir,
                "--results-dir",
                str(Path(tmpdir) / "results"),
                "--generations",
                "2",
                "--population",
                "4",
                "--runs",
                "1",
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(cli_module, "run_nmopso_ablation", new=fake_run_nmopso_ablation):
                    cli_module.main()

        self.assertEqual(captured.get("project_root"), expected_project_root)
        run_params = captured.get("params")
        self.assertIsInstance(run_params, BenchmarkParams)
        assert isinstance(run_params, BenchmarkParams)
        self.assertTrue(run_params.extra.get("ablationStudy"))
        self.assertTrue(run_params.extra.get("legacyPathRunner"))


if __name__ == "__main__":
    unittest.main()
