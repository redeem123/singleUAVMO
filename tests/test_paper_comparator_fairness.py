from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

from uav_benchmark.algorithms import resolve_algorithm_profile

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _load_config(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class PaperComparatorFairnessTest(unittest.TestCase):
    def test_selected_profile_is_exact_paper_comparator_set(self) -> None:
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

    def test_head2head_config_matches_profile_and_common_budget(self) -> None:
        cfg = _load_config("paper_cgpo_swec_head2head.yaml")
        self.assertEqual(tuple(cfg["algorithms"]), resolve_algorithm_profile("cgpo-swec"))
        self.assertTrue(cfg["allowExperimentalAlgorithms"])
        self.assertEqual(int(cfg["population"]), 100)
        self.assertEqual(int(cfg["generations"]), 500)
        self.assertEqual(int(cfg["seed"]), 11)
        self.assertEqual(cfg["fleetSizes"], [1, 3, 5])
        self.assertEqual(cfg["problemNames"], ["c_100", "m_100", "s_120"])

    def test_ea_and_moea_comparators_optimize_shared_benchmark_objective(self) -> None:
        sources = {
            "CGPO": "uav_benchmark/algorithms/cgpo/__init__.py",
            "NSGA-II/NMOPSO": "uav_benchmark/algorithms/shared/fleet_runner.py",
            "MOEAD": "uav_benchmark/algorithms/moead/__init__.py",
            "CCEA-ADVS": "uav_benchmark/algorithms/ccea_advs/__init__.py",
            "TSKAC-NSGA-II": "uav_benchmark/algorithms/tskac_nsga2/__init__.py",
            "MOEA-2DE": "uav_benchmark/algorithms/moea_2de/__init__.py",
            "EMMOP/reference legacy": "uav_benchmark/platemo_bridge/__init__.py",
        }
        for label, source in sources.items():
            with self.subTest(label=label):
                text = _read(source)
                self.assertIn('"benchmarkObjectiveDuringSearch": True', text)
                self.assertIn("_save_fleet_artifacts", text)

    def test_dtapp_is_explicitly_not_a_fair_ea_optimizer_comparator(self) -> None:
        text = _read("uav_benchmark/algorithms/dtapp_iicr/__init__.py")
        self.assertIn('"benchmarkObjectiveDuringSearch": False', text)
        self.assertIn('"finalPathReevaluatedByPython": True', text)
        self.assertIn('"nativePopulationLoop": False', text)
        self.assertIn('"nativeGenerationLoop": False', text)

    def test_scope_doc_labels_dtapp_separately_and_ccea_clean_room(self) -> None:
        text = _read("docs/algorithm_online_offline_scope.md")
        self.assertIn("Separate MAPF / preflight baseline", text)
        self.assertIn("`DTAPP-IICR`", text)
        self.assertIn("clean-room experimental", text)
        self.assertIn("benchmark approximation", text)


if __name__ == "__main__":
    unittest.main()
