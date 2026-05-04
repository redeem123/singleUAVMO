from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_support import install_numba_stub

install_numba_stub()

from uav_benchmark.benchmark import _requested_problem_names
from uav_benchmark.cli import _build_params
from uav_benchmark.config import BenchmarkParams


def _build_fleet_args(protocol_path: Path, fleet_size: int, fleet_sizes: str) -> argparse.Namespace:
    return argparse.Namespace(
        command="benchmark-fleet",
        project_root=".",
        results_dir="results",
        generations=300,
        population=80,
        runs=10,
        compute_metrics=False,
        safe_dist=20.0,
        drone_size=1.0,
        seed=11,
        extra_json="",
        mode="fleet",
        fleet_size=fleet_size,
        fleet_sizes=fleet_sizes,
        scenario_set="paper_medium",
        separation_min=10.0,
        max_turn_deg=75.0,
        evaluation_budget=0,
        gpu_mode="auto",
        protocol=str(protocol_path),
        plots_after=False,
    )


class ProtocolFleetOverrideTest(unittest.TestCase):
    def test_protocol_aliases_are_normalized(self) -> None:
        params = BenchmarkParams.from_mapping(
            {
                "problems": ["c_100"],
                "output_dir": "results/eval_mogwo_v10",
                "metrics": True,
                "extra": {
                    "mogwoUseDiversityFeedback": True,
                    "mogwoUseStepLimiter": True,
                },
            }
        )
        self.assertEqual(_requested_problem_names(params.extra), ("c_100",))
        self.assertEqual(params.results_dir, Path("results/eval_mogwo_v10"))
        self.assertTrue(params.compute_metrics)
        self.assertTrue(params.extra["mogwoUseDiversityFeedback"])
        self.assertTrue(params.extra["mogwoUseStepLimiter"])

    def test_protocol_string_booleans_are_parsed_by_value(self) -> None:
        params = BenchmarkParams.from_mapping(
            {
                "computeMetrics": "false",
                "useParallel": "off",
                "writeFinalHv": "0",
            }
        )
        self.assertFalse(params.compute_metrics)
        self.assertFalse(params.use_parallel)
        self.assertFalse(params.write_final_hv)

    def test_protocol_output_dir_is_preserved_when_cli_uses_default_results_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol.yaml"
            protocol_path.write_text("output_dir: results/from_protocol\n", encoding="utf-8")
            args = _build_fleet_args(protocol_path=protocol_path, fleet_size=1, fleet_sizes="1,3")
            args.results_dir = "results"
            params = _build_params(args)
            self.assertEqual(params.results_dir, Path("results/from_protocol"))

    def test_protocol_fleets_preserved_when_cli_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol.yaml"
            protocol_path.write_text(
                "mode: multi\nfleetSize: 2\nfleetSizes: [2, 4]\n",
                encoding="utf-8",
            )
            args = _build_fleet_args(protocol_path=protocol_path, fleet_size=1, fleet_sizes="1,3")
            params = _build_params(args)
            self.assertEqual(params.fleet_size, 2)
            self.assertEqual(params.fleet_sizes, (2, 4))

    def test_protocol_fleets_can_be_overridden_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol.yaml"
            protocol_path.write_text(
                "mode: multi\nfleetSize: 2\nfleetSizes: [2, 4]\n",
                encoding="utf-8",
            )
            args = _build_fleet_args(protocol_path=protocol_path, fleet_size=4, fleet_sizes="4,6")
            params = _build_params(args)
            self.assertEqual(params.fleet_size, 4)
            self.assertEqual(params.fleet_sizes, (4, 6))

    def test_explicit_fleet_size_default_value_suppresses_default_fleet_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol.yaml"
            protocol_path.write_text("mode: multi\nfleetSizes: [2, 4]\n", encoding="utf-8")
            args = _build_fleet_args(protocol_path=protocol_path, fleet_size=1, fleet_sizes="1,3")
            args._fleet_size_explicit = True
            args._fleet_sizes_explicit = False
            params = _build_params(args)
            self.assertEqual(params.fleet_size, 1)
            self.assertEqual(params.fleet_sizes, ())

    def test_protocol_non_fleet_defaults_are_preserved_when_cli_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol.yaml"
            protocol_path.write_text(
                (
                    "mode: multi\n"
                    "separationMin: 22.5\n"
                    "maxTurnDeg: 55.0\n"
                    "evaluationBudget: 99\n"
                    "scenarioSet: custom_suite\n"
                ),
                encoding="utf-8",
            )
            args = _build_fleet_args(protocol_path=protocol_path, fleet_size=1, fleet_sizes="1,3")
            params = _build_params(args)
            self.assertEqual(params.separation_min, 22.5)
            self.assertEqual(params.max_turn_deg, 55.0)
            self.assertEqual(params.evaluation_budget, 99)
            self.assertEqual(params.scenario_set, "custom_suite")

    def test_protocol_non_fleet_defaults_can_be_overridden_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol.yaml"
            protocol_path.write_text(
                (
                    "mode: multi\n"
                    "separationMin: 22.5\n"
                    "maxTurnDeg: 55.0\n"
                    "evaluationBudget: 99\n"
                    "scenarioSet: custom_suite\n"
                ),
                encoding="utf-8",
            )
            args = _build_fleet_args(protocol_path=protocol_path, fleet_size=1, fleet_sizes="1,3")
            args.separation_min = 14.0
            args.max_turn_deg = 80.0
            args.evaluation_budget = 123
            args.scenario_set = "paper_medium_alt"
            params = _build_params(args)
            self.assertEqual(params.separation_min, 14.0)
            self.assertEqual(params.max_turn_deg, 80.0)
            self.assertEqual(params.evaluation_budget, 123)
            self.assertEqual(params.scenario_set, "paper_medium_alt")


if __name__ == "__main__":
    unittest.main()
