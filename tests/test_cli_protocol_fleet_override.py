from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from uav_benchmark.cli import _build_params


def _build_multi_args(protocol_path: Path, fleet_size: int, fleet_sizes: str) -> argparse.Namespace:
    return argparse.Namespace(
        command="benchmark-multi",
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
        mode="multi",
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
    def test_protocol_fleets_preserved_when_cli_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol.yaml"
            protocol_path.write_text(
                "mode: multi\nfleetSize: 2\nfleetSizes: [2, 4]\n",
                encoding="utf-8",
            )
            args = _build_multi_args(protocol_path=protocol_path, fleet_size=3, fleet_sizes="3,5,8")
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
            args = _build_multi_args(protocol_path=protocol_path, fleet_size=4, fleet_sizes="4,6")
            params = _build_params(args)
            self.assertEqual(params.fleet_size, 4)
            self.assertEqual(params.fleet_sizes, (4, 6))


if __name__ == "__main__":
    unittest.main()
