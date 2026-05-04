from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from uav_benchmark.algorithms.cgpo import run_cgpo
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat, load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain


class CGPOFleetSmokeTest(unittest.TestCase):
    def test_lean_cgpo_writes_three_mechanism_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_s_120.mat")
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=13, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=1,
                population=4,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_cgpo_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
            )
            run_cgpo(multi, params)
            run_dir = Path(tmpdir) / "smoke_cgpo_uav3" / "Run_1"
            for required in (
                "final_popobj.mat",
                "mission_stats.mat",
                "fleet_paths.mat",
                "conflict_log.mat",
                "rl_repair.mat",
                "rl_cig_mean_tension.mat",
                "rl_cig_pairwise_edges.mat",
                "rl_ppf_feasibility_pressure.mat",
                "rl_ovo_perturbation_scale.mat",
                "rl_offspring_feasible_ratio.mat",
                "rl_feasibility_pressure.mat",
                "cgpo_cig_mean_tension.mat",
                "cgpo_ppf_feasibility_pressure.mat",
                "cgpo_ovo_perturbation_scale.mat",
            ):
                self.assertTrue((run_dir / required).exists(), msg=f"missing artifact: {required}")

            tension_trace = load_mat(run_dir / "rl_cig_mean_tension.mat")
            self.assertEqual(tension_trace["rl_cig_mean_tension"].size, 1)

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            stats = summary["statistics"]
            self.assertEqual(summary["metadata"]["generations"], 1)
            self.assertEqual(summary["metadata"]["population"], 4)
            self.assertEqual(stats["cgpoMethodName"], "Constraint-Graph Policy Optimizer")
            controls = stats["cgpoControls"]
            # Lean CGPO defaults: three mechanisms ON.
            self.assertTrue(bool(controls["cigEdgeCouplingEnabled"]))
            self.assertTrue(bool(controls["cigPairwiseEdgesEnabled"]))
            self.assertTrue(bool(controls["ppfPressureEnabled"]))
            self.assertTrue(bool(controls["ovoVariationEnabled"]))
            self.assertTrue(bool(controls["ovoCoordinationEnabled"]))
            self.assertIn("cgpoInitialFeasibleRatio", stats)
            self.assertGreater(int(stats["cgpoCandidateEvaluations"]), 0)
            # Lean CGPO has no projection-proxy mission evaluations.
            self.assertEqual(int(stats["cgpoProjectionProxyEvaluations"]), 0)
            self.assertEqual(int(stats["cgpoTotalMissionEvaluations"]), int(stats["cgpoCandidateEvaluations"]))

    def test_random_only_variant_disables_three_mechanisms(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_s_120.mat")
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=17, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=1,
                population=4,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_cgpo_random_only_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
                extra={
                    "cgpoUseCigEdgeCoupling": False,
                    "cgpoUseCigTerrainEdges": False,
                    "cgpoUseCigObstacleEdges": False,
                    "cgpoUseCigTurnEdges": False,
                    "cgpoUseCigSmoothingEdges": False,
                    "cgpoUseCigPairwiseEdges": False,
                    "cgpoUsePpfPressure": False,
                    "cgpoUseOvoVariation": False,
                    "cgpoUseOvoCoordination": False,
                    "cgpoAblationVariant": "random_only",
                },
            )
            run_cgpo(multi, params)
            run_dir = Path(tmpdir) / "smoke_cgpo_random_only_uav3" / "Run_1"
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            controls = summary["statistics"]["cgpoControls"]
            self.assertFalse(bool(controls["cigEdgeCouplingEnabled"]))
            self.assertFalse(bool(controls["ppfPressureEnabled"]))
            self.assertFalse(bool(controls["ovoVariationEnabled"]))

    def test_no_ppf_pressure_variant_leaves_other_mechanisms_intact(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_s_120.mat")
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=19, separation_min=10.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            params = BenchmarkParams(
                generations=1,
                population=4,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir),
                problem_name="smoke_cgpo_no_ppf_uav3",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
                extra={"cgpoUsePpfPressure": False, "cgpoAblationVariant": "no_ppf_pressure"},
            )
            run_cgpo(multi, params)
            run_dir = Path(tmpdir) / "smoke_cgpo_no_ppf_uav3" / "Run_1"
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            controls = summary["statistics"]["cgpoControls"]
            self.assertFalse(bool(controls["ppfPressureEnabled"]))
            self.assertTrue(bool(controls["cigEdgeCouplingEnabled"]))
            self.assertTrue(bool(controls["ovoVariationEnabled"]))


if __name__ == "__main__":
    unittest.main()
