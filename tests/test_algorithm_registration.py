from __future__ import annotations

import unittest

from uav_benchmark.benchmark import _algorithm_map, _normalize_algorithm_name


class AlgorithmRegistrationTest(unittest.TestCase):
    def test_smpso_name_normalization(self) -> None:
        self.assertEqual(_normalize_algorithm_name("smpso"), "SMPSO")
        self.assertEqual(_normalize_algorithm_name("SM-PSO"), "SMPSO")
        self.assertEqual(_normalize_algorithm_name("sm_pso"), "SMPSO")

    def test_smpso_registered_in_algorithm_map(self) -> None:
        names = [name for name, _runner in _algorithm_map()]
        self.assertIn("SMPSO", names)

        only_smpso = [name for name, _runner in _algorithm_map(("SMPSO",))]
        self.assertEqual(only_smpso, ["SMPSO"])

    def test_mfo_spea2_name_normalization(self) -> None:
        self.assertEqual(_normalize_algorithm_name("mfo-spea2"), "MFO-SPEA2")
        self.assertEqual(_normalize_algorithm_name("mfospea2"), "MFO-SPEA2")
        self.assertEqual(_normalize_algorithm_name("mfo_spea2"), "MFO-SPEA2")

    def test_mfo_spea2_registered_in_algorithm_map(self) -> None:
        names = [name for name, _runner in _algorithm_map()]
        self.assertIn("MFO-SPEA2", names)

        only = [name for name, _runner in _algorithm_map(("MFO-SPEA2",))]
        self.assertEqual(only, ["MFO-SPEA2"])

    def test_gcnmoea_name_normalization(self) -> None:
        self.assertEqual(_normalize_algorithm_name("gcnmoea"), "GCNMOEA")
        self.assertEqual(_normalize_algorithm_name("GCN-MOEA"), "GCNMOEA")
        self.assertEqual(_normalize_algorithm_name("gcn_moea"), "GCNMOEA")

    def test_gcnmoea_registered_in_algorithm_map(self) -> None:
        names = [name for name, _runner in _algorithm_map()]
        self.assertIn("GCNMOEA", names)

        only = [name for name, _runner in _algorithm_map(("GCNMOEA",))]
        self.assertEqual(only, ["GCNMOEA"])


if __name__ == "__main__":
    unittest.main()
