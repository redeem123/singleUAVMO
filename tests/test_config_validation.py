from __future__ import annotations

import unittest
from pathlib import Path
from typing import cast

from uav_benchmark.config import BenchmarkParams


class BenchmarkParamsValidationTest(unittest.TestCase):
    def test_mapping_values_are_normalized_to_runtime_types(self) -> None:
        params = BenchmarkParams.from_mapping(
            {
                "Generations": "8",
                "pop": "12",
                "Runs": "3",
                "output_dir": "results/protocol",
                "fleetSizes": "1, 3, 5",
                "runIndices": "0,2,4,2",
                "computeMetrics": "yes",
                "writeFinalHv": "false",
            }
        )

        self.assertEqual(params.generations, 8)
        self.assertEqual(params.population, 12)
        self.assertEqual(params.runs, 3)
        self.assertEqual(params.results_dir, Path("results/protocol"))
        self.assertEqual(params.fleet_sizes, (1, 3, 5))
        self.assertEqual(params.run_indices, (2, 4))
        self.assertTrue(params.compute_metrics)
        self.assertFalse(params.write_final_hv)

    def test_invalid_positive_integer_fields_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "generations must be >= 1"):
            BenchmarkParams(generations=0)

        with self.assertRaisesRegex(ValueError, "fleet_sizes must be >= 1"):
            BenchmarkParams.from_mapping({"fleetSizes": [1, 0, 3]})

    def test_invalid_extra_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "extra must be a mapping"):
            BenchmarkParams(extra=cast(dict, ["not", "a", "mapping"]))


if __name__ == "__main__":
    unittest.main()
