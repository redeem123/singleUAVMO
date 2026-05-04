from __future__ import annotations

import unittest
from pathlib import Path

from uav_benchmark.benchmark import _build_benchmark_manifest, _plan_hash
from uav_benchmark.config import BenchmarkParams


class BenchmarkManifestTest(unittest.TestCase):
    def test_manifest_records_resolved_plan_and_stable_hash(self) -> None:
        params = BenchmarkParams(
            generations=2,
            population=6,
            runs=1,
            results_dir=Path("results/test"),
            seed=123,
            fleet_size=1,
            fleet_sizes=(1,),
            run_indices=(1,),
            extra={"algorithms": ["NMOPSO"], "problemNames": ["c_100"], "maxWorkers": 1},
        )
        tasks = [(Path("problems/terrainStruct_c_100.mat"), 1, "NMOPSO", "NMOPSO", params)]

        first = _build_benchmark_manifest(
            project_root=Path("."),
            params=params,
            fleet_sizes=(1,),
            tasks=tasks,
            n_workers=1,
            created_utc="2026-01-01T00:00:00+00:00",
        )
        second = _build_benchmark_manifest(
            project_root=Path("."),
            params=params,
            fleet_sizes=(1,),
            tasks=tasks,
            n_workers=1,
            created_utc="2026-01-02T00:00:00+00:00",
        )

        self.assertEqual(first["plan"]["algorithmsResolved"], ["NMOPSO"])
        self.assertEqual(first["plan"]["problemsResolved"], ["c_100"])
        self.assertEqual(first["plan"]["fleetSizesResolved"], [1])
        self.assertEqual(first["plan"]["parameters"]["seed"], 123)
        self.assertEqual(first["plan"]["parameters"]["runIndices"], [1])
        self.assertEqual(first["planHashSha256"], second["planHashSha256"])
        self.assertEqual(first["planHashSha256"], _plan_hash(first["plan"]))


if __name__ == "__main__":
    unittest.main()
