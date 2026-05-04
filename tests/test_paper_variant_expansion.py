from __future__ import annotations

import unittest

from uav_benchmark.benchmark import _variant_tasks_for_algorithm
from uav_benchmark.config import BenchmarkParams


class PaperVariantExpansionTest(unittest.TestCase):
    def test_state_representation_modes_expand_for_adaptive_algorithms(self) -> None:
        params = BenchmarkParams(
            extra={
                "stateRepresentationModes": ["flat", "TRFTS-HAND", "TRFTS"],
            }
        )
        variants = _variant_tasks_for_algorithm("SAC-SMOPSO", params)
        labels = [label for label, _runner, _params in variants]
        self.assertEqual(
            labels,
            [
                "SAC-SMOPSO__flat",
                "SAC-SMOPSO__TRFTS-HAND",
                "SAC-SMOPSO__TRFTS",
            ],
        )

    def test_policy_modes_are_normalized_to_sac_policy_mode(self) -> None:
        params = BenchmarkParams(
            extra={
                "policyModes": ["online", "frozen"],
                "stateRepresentation": "TRFTS",
            }
        )
        variants = _variant_tasks_for_algorithm("RA-NSGA-II", params)
        self.assertEqual(len(variants), 2)
        for label, _runner, variant_params in variants:
            self.assertIn(label, {"RA-NSGA-II__TRFTS__online", "RA-NSGA-II__TRFTS__frozen"})
            self.assertIn(variant_params.extra["sacPolicyMode"], {"online", "frozen"})

    def test_nonadaptive_algorithms_do_not_expand(self) -> None:
        params = BenchmarkParams(extra={"stateRepresentationModes": ["flat", "TRFTS"]})
        variants = _variant_tasks_for_algorithm("MOEAD", params)
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0][0], "MOEAD")


if __name__ == "__main__":
    unittest.main()
