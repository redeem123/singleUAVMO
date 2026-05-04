from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

from uav_benchmark.algorithms import resolve_algorithm_profile

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_config(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class CgpoPaperConfigConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.head2head = _load_config("paper_cgpo_swec_head2head.yaml")
        self.reference = _load_config("paper_cgpo_swec_reference_single_uav.yaml")
        self.ablation = _load_config("paper_cgpo_swec_ablation.yaml")
        self.smoke = _load_config("cgpo_ablation_smoke.yaml")

    def test_main_head2head_matches_cgpo_swec_profile(self) -> None:
        self.assertEqual(tuple(self.head2head["algorithms"]), resolve_algorithm_profile("cgpo-swec"))
        self.assertTrue(self.head2head["allowExperimentalAlgorithms"])

    def test_reference_config_is_separate_single_uav_context(self) -> None:
        expected = ("CGPO", *resolve_algorithm_profile("uav-reference-code"))
        self.assertEqual(tuple(self.reference["algorithms"]), expected)
        self.assertEqual(self.reference["fleetSizes"], [1])
        self.assertTrue(self.reference["allowExperimentalAlgorithms"])

        main_algorithms = set(self.head2head["algorithms"])
        reference_only = set(self.reference["algorithms"]) - {"CGPO"}
        self.assertFalse(main_algorithms.intersection(reference_only))

    def test_main_table_matches_curated_reviewer_baseline_roles(self) -> None:
        self.assertEqual(
            tuple(self.head2head["algorithms"]),
            (
                "CGPO",
                "NSGA-II",
                "MOEAD",
                "NMOPSO",
                "CCEA-ADVS",
                "TSKAC-NSGA-II",
                "DTAPP-IICR",
            ),
        )

    def test_main_table_uses_only_lean_cgpo(self) -> None:
        main_algorithms = set(self.head2head["algorithms"])
        self.assertIn("CGPO", main_algorithms)
        self.assertNotIn("CGPO-R", main_algorithms)
        self.assertNotIn("NSGA-II-CGPO-HYBRID", main_algorithms)
        self.assertNotIn("MOEAD-CGPO-HYBRID", main_algorithms)
        self.assertNotIn("NMOPSO-CGPO-HYBRID", main_algorithms)

    def test_head2head_ablation_and_reference_share_paper_protocol(self) -> None:
        shared_keys = (
            "runs",
            "generations",
            "population",
            "seed",
            "mode",
            "separationMin",
            "maxTurnDeg",
            "safeDist",
            "droneSize",
            "scenarioSet",
            "gpuMode",
            "problemNames",
            "computeMetrics",
            "hardCollisionConstraint",
            "nRep",
            "metricInterval",
            "resumeExistingRuns",
        )
        for key in shared_keys:
            with self.subTest(key=key):
                self.assertEqual(self.head2head[key], self.ablation[key])
                self.assertEqual(self.head2head[key], self.reference[key])

        self.assertEqual(self.head2head["fleetSizes"], [1, 3, 5])
        self.assertEqual(self.ablation["fleetSizes"], [1, 3, 5])

    def test_fairness_critical_flags_are_locked(self) -> None:
        for config_name, config in (
            ("head2head", self.head2head),
            ("reference", self.reference),
            ("ablation", self.ablation),
        ):
            with self.subTest(config=config_name):
                self.assertEqual(config["runs"], 30)
                self.assertEqual(config["generations"], 500)
                self.assertEqual(config["population"], 100)
                self.assertEqual(config["seed"], 11)
                self.assertEqual(config["problemNames"], ["c_100", "m_100", "s_120"])
                self.assertEqual(config["scenarioSet"], "paper_medium")
                self.assertEqual(config["gpuMode"], "off")
                self.assertTrue(config["computeMetrics"])
                self.assertTrue(config["hardCollisionConstraint"])
                self.assertEqual(config["separationMin"], 10.0)
                self.assertEqual(config["maxTurnDeg"], 75.0)
                self.assertEqual(config["safeDist"], 20.0)
                self.assertEqual(config["droneSize"], 1.0)
                self.assertEqual(config["nRep"], 100)
                self.assertEqual(config["metricInterval"], 10)
                self.assertTrue(config["resumeExistingRuns"])

        # The two flat configs (head-to-head and reference) must lock down the
        # lean three-mechanism CGPO protocol.
        for config_name, config in (("head2head", self.head2head), ("reference", self.reference)):
            with self.subTest(config=config_name):
                self.assertTrue(config["cgpoUseCigEdgeCoupling"])
                self.assertTrue(config["cgpoUsePpfPressure"])
                self.assertTrue(config["cgpoUseOvoVariation"])
                self.assertTrue(config["cgpoUseOvoCoordination"])

    def test_main_table_contains_neutral_baselines_not_legacy_research_families(self) -> None:
        main_algorithms = set(self.head2head["algorithms"])
        self.assertIn("CGPO", main_algorithms)
        self.assertNotIn("CGPO-R", main_algorithms)
        self.assertFalse(
            main_algorithms.intersection(
                {
                    "SAC-SMOPSO",
                    "RA-SMPSO",
                    "RA-NSGA-II",
                    "MOGWO",
                    "MOGWO-NO-ATTENTION",
                    "MOGWO-STANDARD-GWO",
                }
            )
        )

    def test_ablation_variants_are_paper_honest_three_mechanism_variants(self) -> None:
        expected_variants = [
            "full",
            "random_only",
            "no_cig_edge_coupling",
            "no_ppf_pressure",
            "no_ovo_variation",
            "no_ovo_fleet_coordination",
        ]
        self.assertEqual(self.ablation["algorithms"], ["CGPO"])
        self.assertEqual(self.ablation["cgpoAblationVariants"], expected_variants)
        self.assertTrue(self.ablation["cgpoTraceEnabled"])
        self.assertTrue(self.ablation["allowExperimentalAlgorithms"])

    def test_smoke_tracks_ablation_mechanisms_but_not_paper_budget(self) -> None:
        self.assertEqual(self.smoke["algorithms"], self.ablation["algorithms"])
        self.assertEqual(self.smoke["cgpoAblationVariants"], self.ablation["cgpoAblationVariants"])
        self.assertEqual(self.smoke["cgpoTraceEnabled"], self.ablation["cgpoTraceEnabled"])
        self.assertEqual(self.smoke["scenarioSet"], self.ablation["scenarioSet"])
        self.assertEqual(self.smoke["separationMin"], self.ablation["separationMin"])
        self.assertEqual(self.smoke["maxTurnDeg"], self.ablation["maxTurnDeg"])
        self.assertEqual(self.smoke["safeDist"], self.ablation["safeDist"])
        self.assertLess(self.smoke["generations"], self.ablation["generations"])
        self.assertLess(self.smoke["population"], self.ablation["population"])
        self.assertLess(self.smoke["runs"], self.ablation["runs"])


if __name__ == "__main__":
    unittest.main()
