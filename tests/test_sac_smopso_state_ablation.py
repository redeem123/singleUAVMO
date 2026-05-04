from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import run_sac_smopso_state_ablation as state_ablation_script
from uav_benchmark.algorithms.sac_smopso import controller as sac_controller_module
from uav_benchmark.algorithms.sac_smopso import run_sac_smopso
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat, load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain
from uav_benchmark.utils.random import seed_everything

_TORCH_READY = bool(getattr(sac_controller_module, "_TORCH_AVAILABLE", False))


class SACSMOPSOStateAblationTest(unittest.TestCase):
    def _string_field(self, metadata: dict[str, object], key: str) -> str:
        value = np.asarray(metadata[key]).reshape(-1)
        return str(value[0]) if value.size > 0 else ""

    def _run_mode(self, mode: str) -> tuple[Path, dict[str, object]]:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=23, separation_min=10.0)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)

        seed_everything(17)
        params = BenchmarkParams(
            generations=3,
            population=6,
            runs=1,
            compute_metrics=True,
            results_dir=Path(tmpdir.name),
            problem_name=f"smoke_sac_smopso_{mode}",
            problem_index=1,
            mode="fleet",
            fleet_size=3,
            separation_min=10.0,
            gpu_mode="off",
            seed=17,
            extra={
                "metricInterval": 1,
                "stateRepresentation": mode,
                "sacWarmupSteps": 2,
                "sacBatchSize": 4,
                "sacReplayCapacity": 64,
            },
        )
        run_sac_smopso(multi, params)
        run_dir = Path(tmpdir.name) / f"smoke_sac_smopso_{mode}" / "Run_1"
        return run_dir, load_mat(run_dir / "rl_metadata.mat")

    def test_trfts_mode_emits_relational_trace_channels(self) -> None:
        flat_run_dir, flat_metadata = self._run_mode("flat")
        self.assertEqual(self._string_field(flat_metadata, "stateRepresentation"), "flat")
        self.assertEqual(self._string_field(flat_metadata, "stateEncoderMode"), "flat")
        self.assertEqual(float(np.asarray(flat_metadata["stateHasRelationalTokens"]).reshape(-1)[0]), 0.0)

        hand_run_dir, hand_metadata = self._run_mode("TRFTS-HAND")
        self.assertEqual(self._string_field(hand_metadata, "stateRepresentation"), "TRFTS-HAND")
        self.assertEqual(self._string_field(hand_metadata, "stateEncoderMode"), "handcrafted")
        self.assertEqual(float(np.asarray(hand_metadata["stateHasRelationalTokens"]).reshape(-1)[0]), 1.0)

        run_dir, metadata = self._run_mode("TRFTS")

        self.assertEqual(self._string_field(metadata, "stateRepresentation"), "TRFTS")
        self.assertEqual(self._string_field(metadata, "stateEncoderMode"), "learned")
        self.assertEqual(float(np.asarray(metadata["stateHasRelationalTokens"]).reshape(-1)[0]), 1.0)
        self.assertEqual(float(np.asarray(metadata["controllerOperatorHeadEnabled"]).reshape(-1)[0]), 0.0)

        topology = np.asarray(load_mat(run_dir / "rl_topology_summary.mat")["rl_topology_summary"], dtype=float)
        interaction = np.asarray(
            load_mat(run_dir / "rl_interaction_summary.mat")["rl_interaction_summary"], dtype=float
        )
        environment = np.asarray(
            load_mat(run_dir / "rl_environment_summary.mat")["rl_environment_summary"], dtype=float
        )
        state_global = np.asarray(load_mat(run_dir / "rl_state_global.mat")["rl_state_global"], dtype=float)

        flat_state_global = np.asarray(load_mat(flat_run_dir / "rl_state_global.mat")["rl_state_global"], dtype=float)
        hand_state_global = np.asarray(load_mat(hand_run_dir / "rl_state_global.mat")["rl_state_global"], dtype=float)
        self.assertTrue(np.all(np.isfinite(flat_state_global)))
        self.assertTrue(np.all(np.isfinite(hand_state_global)))

        self.assertEqual(topology.shape[1], 8)
        self.assertEqual(interaction.shape[1], 7)
        self.assertEqual(environment.shape[1], 8)
        self.assertTrue(np.any(np.abs(topology) > 1e-8))
        self.assertTrue(np.any(np.abs(interaction) > 1e-8))
        self.assertTrue(np.all(np.isfinite(environment)))
        self.assertTrue(np.any(np.abs(state_global[:, 19:]) > 1e-8))

    @unittest.skipUnless(_TORCH_READY, "Torch is required for frozen checkpoint ablation tests.")
    def test_frozen_ablation_case_reuses_checkpoint_hidden_dim(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=29, separation_min=10.0)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        checkpoint = Path(tmpdir.name) / "controller.pt"

        seed_everything(41)
        train_params = BenchmarkParams(
            generations=3,
            population=6,
            runs=1,
            compute_metrics=True,
            results_dir=Path(tmpdir.name) / "train_results",
            problem_name="pretrain_case",
            problem_index=1,
            mode="fleet",
            fleet_size=3,
            separation_min=10.0,
            gpu_mode="off",
            seed=41,
            extra={
                "resumeExistingRuns": False,
                "stateRepresentation": "TRFTS",
                "sacPolicyMode": "finetune",
                "sacCheckpointPath": str(checkpoint),
                "sacSaveCheckpoint": True,
                "sacHiddenDim": 128,
                "sacWarmupSteps": 2,
                "sacBatchSize": 4,
                "sacReplayCapacity": 64,
            },
        )
        run_sac_smopso(multi, train_params)
        self.assertTrue(checkpoint.exists())

        record = state_ablation_script._run_case(
            problem_file=project_root / "problems" / "terrainStruct_c_100.mat",
            problem_index=1,
            seed=101,
            state_representation="TRFTS",
            generations=3,
            population=6,
            fleet_size=3,
            separation_min=10.0,
            policy_mode="frozen",
            checkpoint=checkpoint,
            results_dir=Path(tmpdir.name) / "ablation_results",
            gpu_mode="off",
        )

        self.assertEqual(str(record["stateRepresentation"]), "TRFTS")
        self.assertEqual(str(record["stateEncoderMode"]), "learned")
        self.assertEqual(str(record["policyMode"]), "frozen")
        self.assertEqual(float(record["checkpointLoaded"]), 1.0)

    def test_frozen_ablation_rejects_missing_checkpoint(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        missing_checkpoint = Path(tmpdir.name) / "missing_controller.pt"

        with self.assertRaises(FileNotFoundError):
            state_ablation_script._run_case(
                problem_file=project_root / "problems" / "terrainStruct_c_100.mat",
                problem_index=1,
                seed=101,
                state_representation="TRFTS",
                generations=3,
                population=6,
                fleet_size=3,
                separation_min=10.0,
                policy_mode="frozen",
                checkpoint=missing_checkpoint,
                results_dir=Path(tmpdir.name) / "ablation_results",
                gpu_mode="off",
            )


if __name__ == "__main__":
    unittest.main()
