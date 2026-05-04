from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from uav_benchmark.algorithms import resolve_algorithm_profile
from uav_benchmark.benchmark import (
    _algorithm_map,
    _normalize_algorithm_name,
    _requested_algorithms,
    _variant_tasks_for_algorithm,
)
from uav_benchmark.config import BenchmarkParams


class AlgorithmRegistrationTest(unittest.TestCase):
    def _assert_normalizes(self, expected: str, *names: str) -> None:
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(_normalize_algorithm_name(name), expected)

    def _assert_registered(self, *names: str) -> None:
        registered = [name for name, _runner in _algorithm_map()]
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, registered)
        only = [name for name, _runner in _algorithm_map(names)]
        self.assertEqual(only, list(names))

    def _assert_experimental(self, *names: str) -> None:
        registered = [name for name, _runner in _algorithm_map()]
        for name in names:
            with self.subTest(name=name):
                self.assertNotIn(name, registered)
        only = [name for name, _runner in _algorithm_map(names, allow_experimental=True)]
        self.assertEqual(only, list(names))

    def _assert_not_registered(self, *names: str) -> None:
        registered = [name for name, _runner in _algorithm_map()]
        for name in names:
            with self.subTest(name=name):
                self.assertNotIn(name, registered)
        try:
            only = [name for name, _runner in _algorithm_map(names)]
        except ValueError as exc:
            self.assertIn("Unknown algorithm", str(exc))
            only = []
        self.assertEqual(only, [])

    def test_benchmark_safe_algorithms_registered(self) -> None:
        self._assert_registered(
            "NMOPSO",
            "MOPSO",
            "SMPSO",
            "NSGA-II",
            "NSGA-III",
            "MOEAD",
            "SPEA2",
            "MFO-SPEA2",
            "GCNMOEA",
            "CMOSMA",
            "MO-MFEA",
            "MO-MFEA-II",
            "L-SHADE-CDP",
            "CMOEA-CD",
            "APSEA",
            "C-TSEA",
            "ToP",
            "CMOCSO",
            "C-TAEA",
            "Two_Arch2",
            "CMOEA-MS",
            "CMOEA-MSG",
            "CCMO",
            "URCMO",
        )

    def test_algorithm_package_import_does_not_load_experimental_backends(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys\nimport uav_benchmark.algorithms\nprint('uav_benchmark.algorithms.sac_smopso' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_benchmark_import_does_not_load_heavy_backends(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys\n"
            "import uav_benchmark.benchmark\n"
            "blocked = [name for name in ("
            "'torch', "
            "'uav_benchmark.algorithms.sac_smopso', "
            "'uav_benchmark.platemo_bridge'"
            ") if name in sys.modules]\n"
            "print(','.join(blocked))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "")

    def test_sac_family_name_normalization(self) -> None:
        self._assert_normalizes("SAC-SMOPSO", "sac-smopso", "sac_smopso", "sacsmopso")
        self._assert_normalizes("RA-SMPSO", "ra-smpso", "ra_smpso", "ra-sm-pso")
        self._assert_normalizes("RA-NSGA-II", "ra-nsga-ii", "ra_nsga2", "ra-nsga2")

    def test_sac_family_is_experimental(self) -> None:
        self._assert_experimental("SAC-SMOPSO", "RA-SMPSO", "RA-NSGA-II")

    def test_ccea_advs_name_normalization(self) -> None:
        self._assert_normalizes(
            "CCEA-ADVS",
            "ccea-advs",
            "ccea_advs",
            "cooperative-coevolution-advs",
            "adaptive-decision-variable-selection",
        )

    def test_ccea_advs_is_experimental(self) -> None:
        self._assert_experimental("CCEA-ADVS")

    def test_smpso_name_normalization(self) -> None:
        self._assert_normalizes("SMPSO", "smpso", "SM-PSO", "sm_pso")

    def test_smpso_registered_in_algorithm_map(self) -> None:
        self._assert_registered("SMPSO")

    def test_mfo_spea2_name_normalization(self) -> None:
        self._assert_normalizes("MFO-SPEA2", "mfo-spea2", "mfospea2", "mfo_spea2")

    def test_mfo_spea2_registered_in_algorithm_map(self) -> None:
        self._assert_registered("MFO-SPEA2")

    def test_gcnmoea_name_normalization(self) -> None:
        self._assert_normalizes("GCNMOEA", "gcnmoea", "GCN-MOEA", "gcn_moea")

    def test_gcnmoea_registered_in_algorithm_map(self) -> None:
        self._assert_registered("GCNMOEA")

    def test_lshade_cdp_name_normalization(self) -> None:
        self._assert_normalizes(
            "L-SHADE-CDP",
            "l-shade-cdp",
            "lshade-cdp",
            "lshade_cdp",
            "apex-shade",
            "apex_shade",
        )

    def test_lshade_cdp_registered_in_algorithm_map(self) -> None:
        self._assert_registered("L-SHADE-CDP")

    def test_platemo_cmoea_name_normalization(self) -> None:
        self._assert_normalizes("CMOEA-CD", "cmoea-cd", "cmoeacd", "cmoea_cd")
        self._assert_normalizes("APSEA", "apsea")
        self._assert_normalizes("C-TSEA", "c-tsea", "ctsea", "c_tsea")
        self._assert_normalizes("ToP", "top", "to-p")
        self._assert_normalizes("CMOCSO", "cmocso", "cmo-cso", "cmo_cso")
        self._assert_normalizes("C-TAEA", "c-taea", "ctaea", "c_taea")
        self._assert_normalizes("Two_Arch2", "two-arch2", "two_arch2", "twoarch2")
        self._assert_normalizes("CMOEA-MS", "cmoea-ms", "cmoeams", "cmoea_ms")
        self._assert_normalizes("CMOEA-MSG", "cmoea-msg", "cmoeamsg", "cmoea_msg")
        self._assert_normalizes("CCMO", "ccmo")
        self._assert_normalizes("URCMO", "urcmo", "ur-cmo", "ur_cmo")

    def test_platemo_cmoea_registered_in_algorithm_map(self) -> None:
        self._assert_registered(
            "CMOEA-CD",
            "APSEA",
            "C-TSEA",
            "ToP",
            "CMOCSO",
            "C-TAEA",
            "Two_Arch2",
            "CMOEA-MS",
            "CMOEA-MSG",
            "CCMO",
            "URCMO",
        )

    def test_mogwo_family_is_experimental(self) -> None:
        self._assert_normalizes("MOGWO", "mogwo", "a2mogwo")
        self._assert_normalizes("MOGWO-STANDARD-GWO", "mogwo-standard-gwo")
        self._assert_experimental("MOGWO", "MOGWO-NO-ATTENTION", "MOGWO-STANDARD-GWO")

    def test_tskac_nsga2_is_experimental(self) -> None:
        self._assert_normalizes("TSKAC-NSGA-II", "tskac-nsga-ii", "tskac_nsga_ii", "tskacnsga2")
        self._assert_experimental("TSKAC-NSGA-II")

    def test_moea_2de_is_experimental_reference_code(self) -> None:
        self._assert_normalizes(
            "MOEA-2DE",
            "moea-2de",
            "moea_2de",
            "moea2de",
            "dimension-exploration-discrepancy-evolution",
        )
        self._assert_experimental("MOEA-2DE")

    def test_moead_awa_astar_is_experimental_reference_code(self) -> None:
        self._assert_normalizes(
            "MOEAD-AWA-ASTAR",
            "moead-awa-astar",
            "moead_awa_astar",
            "moeadawa-astar",
            "moea/d-awa-a*",
            "heuristic-driven-moead-awa",
        )
        self._assert_experimental("MOEAD-AWA-ASTAR")

    def test_emmop_is_experimental_reference_code(self) -> None:
        self._assert_normalizes("EMMOP", "emmop", "e-mmop", "e_mmop")
        self._assert_experimental("EMMOP")

    def test_sem4d_is_experimental(self) -> None:
        self._assert_normalizes(
            "SEM-4D",
            "sem-4d",
            "sem4d",
            "sem_4d",
            "shielded-evolutionary-multitasking",
            "safety-shielded-evolutionary-multitasking",
        )
        self._assert_experimental("SEM-4D")
        self.assertEqual(resolve_algorithm_profile("sem4d")[0], "SEM-4D")

    def test_dtapp_iicr_is_experimental_reference_code(self) -> None:
        self._assert_normalizes(
            "DTAPP-IICR",
            "dtapp-iicr",
            "dtapp_iicr",
            "dtappiicr",
            "delivery-time-aware-prioritized-planning",
            "incremental-iterative-conflict-resolution",
        )
        self._assert_experimental("DTAPP-IICR")

    def test_cgpo_is_experimental(self) -> None:
        self._assert_normalizes(
            "CGPO",
            "cgpo",
            "constraint-graph-projection-optimizer",
            "constraint-graph-policy-optimizer",
        )
        self._assert_experimental("CGPO")
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            _algorithm_map(("CGPO-R",), allow_experimental=True)

    def test_paper_honest_cgpo_ablation_variants_expand_to_labels(self) -> None:
        """The paper-honest variants isolate the three published mechanisms.

        These variants stay on the lean CGPO surface so the ablation isolates
        the *named* mechanism cleanly.
        """
        params = BenchmarkParams(
            extra={
                "cgpoAblationVariants": [
                    "full",
                    "random_only",
                    "no_cig_edge_coupling",
                    "no_ppf_pressure",
                    "no_ovo_variation",
                    "no_ovo_fleet_coordination",
                ]
            }
        )
        variants = _variant_tasks_for_algorithm("CGPO", params)
        labels = [label for label, _runner, _params in variants]
        self.assertEqual(
            labels,
            [
                "CGPO_full",
                "CGPO_random_only",
                "CGPO_no_cig_edge_coupling",
                "CGPO_no_ppf_pressure",
                "CGPO_no_ovo_variation",
                "CGPO_no_ovo_fleet_coordination",
            ],
        )
        self.assertEqual([runner for _label, runner, _params in variants], ["CGPO"] * 6)
        # Lean CGPO_full: three mechanisms ON.
        self.assertTrue(variants[0][2].extra["cgpoUseCigEdgeCoupling"])
        self.assertTrue(variants[0][2].extra["cgpoUsePpfPressure"])
        self.assertTrue(variants[0][2].extra["cgpoUseOvoVariation"])
        self.assertNotIn("cgpoEnableLegacyEngineering", variants[0][2].extra)
        self.assertNotIn("cgpoRepairEnabled", variants[0][2].extra)
        self.assertNotIn("cgpoUseGfpProjection", variants[0][2].extra)
        # random_only: NSGA-II floor, all three mechanisms OFF.
        self.assertFalse(variants[1][2].extra["cgpoUseCigEdgeCoupling"])
        self.assertFalse(variants[1][2].extra["cgpoUsePpfPressure"])
        self.assertFalse(variants[1][2].extra["cgpoUseOvoVariation"])
        # Each isolation variant flips exactly the corresponding mechanism.
        self.assertFalse(variants[2][2].extra["cgpoUseCigEdgeCoupling"])
        self.assertFalse(variants[3][2].extra["cgpoUsePpfPressure"])
        self.assertFalse(variants[4][2].extra["cgpoUseOvoVariation"])
        self.assertFalse(variants[4][2].extra["cgpoUseOvoCoordination"])
        self.assertFalse(variants[5][2].extra["cgpoUseOvoCoordination"])
        # The fleet-coordination variant must keep the parent-blend OVO ON.
        self.assertNotIn("cgpoUseOvoVariation", variants[5][2].extra)

    def test_legacy_cgpo_variants_are_rejected(self) -> None:
        params = BenchmarkParams(
            extra={
                "cgpoAblationVariants": [
                    "with_repair",
                    "repair_only",
                    "no_gfp_projection",
                    "no_constructive_priors",
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "Unknown CGPO ablation variant"):
            _variant_tasks_for_algorithm("CGPO", params)

    def test_cgpo_rejects_removed_repair_gfp_constructive_variants(self) -> None:
        params = BenchmarkParams(
            extra={
                "cgpoAblationVariants": [
                    "constructive_only",
                    "gfp_only",
                    "constructive_gfp",
                    "lite_with_repair",
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "Unknown CGPO ablation variant"):
            _variant_tasks_for_algorithm("CGPO", params)

    def test_experimental_algorithms_not_registered(self) -> None:
        self._assert_not_registered("FASTR-MOEA", "TACTIC-MOEA")

    def test_experimental_algorithms_require_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "allowExperimentalAlgorithms=true"):
            _algorithm_map(("SAC-SMOPSO",))
        with self.assertRaisesRegex(ValueError, "allowExperimentalAlgorithms=true"):
            _algorithm_map(("MOGWO",))

    def test_unknown_algorithm_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            _algorithm_map(("NOT-A-REAL-ALGO",))

    def test_algorithm_profiles_expand_to_curated_sets(self) -> None:
        self.assertEqual(
            resolve_algorithm_profile("benchmark-core"),
            ("NSGA-II", "NSGA-III", "SMPSO", "SPEA2", "MOEAD", "NMOPSO", "L-SHADE-CDP"),
        )
        self.assertEqual(
            resolve_algorithm_profile("state-representation"),
            ("SAC-SMOPSO", "RA-SMPSO", "RA-NSGA-II"),
        )
        self.assertEqual(
            resolve_algorithm_profile("constraint-cmoea"),
            (
                "CMOEA-CD",
                "APSEA",
                "C-TSEA",
                "ToP",
                "CMOCSO",
                "C-TAEA",
                "Two_Arch2",
                "CMOEA-MS",
                "CMOEA-MSG",
                "CCMO",
                "URCMO",
            ),
        )
        self.assertEqual(
            resolve_algorithm_profile("cgpo-swec"),
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
        self.assertEqual(
            resolve_algorithm_profile("uav-reference-code"),
            ("MOEA-2DE", "EMMOP"),
        )

    def test_requested_algorithms_can_come_from_profile(self) -> None:
        requested = _requested_algorithms({"algorithmProfile": "benchmark-core"})
        self.assertEqual(
            requested,
            ("NSGA-II", "NSGA-III", "SMPSO", "SPEA2", "MOEAD", "NMOPSO", "L-SHADE-CDP"),
        )

    def test_requested_algorithms_merge_profile_and_explicit_names(self) -> None:
        requested = _requested_algorithms(
            {
                "algorithmProfile": "benchmark-core",
                "algorithms": ["CMOSMA", "NSGA-II"],
            }
        )
        self.assertEqual(
            requested,
            ("NSGA-II", "NSGA-III", "SMPSO", "SPEA2", "MOEAD", "NMOPSO", "L-SHADE-CDP", "CMOSMA"),
        )

    def test_unknown_algorithm_profile_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown algorithm profile"):
            _requested_algorithms({"algorithmProfile": "not-a-profile"})


if __name__ == "__main__":
    unittest.main()
