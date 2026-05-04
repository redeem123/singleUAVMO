from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import train_sac_smopso_controller as train_script
from uav_benchmark.algorithms.sac_smopso import controller as sac_controller_module
from uav_benchmark.algorithms.sac_smopso import run_sac_smopso
from uav_benchmark.algorithms.sac_smopso.controller import (
    ControllerConfig,
    HybridSACController,
    TemporalRelationalState,
    TemporalRelationalStateSpec,
)
from uav_benchmark.algorithms.sac_smopso.workflow import stage_preset
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_mat, load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain
from uav_benchmark.utils.random import seed_everything


def _make_state(scale: float = 1.0) -> TemporalRelationalState:
    return TemporalRelationalState(
        global_features=np.linspace(0.0, 1.0, 24, dtype=np.float32) * scale,
        population_tokens=np.full((12, 14), 0.1 * scale, dtype=np.float32),
        population_mask=np.concatenate([np.ones(6, dtype=np.float32), np.zeros(6, dtype=np.float32)]),
        archive_tokens=np.full((16, 14), 0.2 * scale, dtype=np.float32),
        archive_mask=np.concatenate([np.ones(8, dtype=np.float32), np.zeros(8, dtype=np.float32)]),
        topology_tokens=np.full((8, 8), 0.3 * scale, dtype=np.float32),
        topology_mask=np.concatenate([np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]),
        interaction_tokens=np.full((12, 7), 0.4 * scale, dtype=np.float32),
        interaction_mask=np.concatenate([np.ones(5, dtype=np.float32), np.zeros(7, dtype=np.float32)]),
        environment_tokens=np.full((16, 8), 0.25 * scale, dtype=np.float32),
        environment_mask=np.concatenate([np.ones(3, dtype=np.float32), np.zeros(13, dtype=np.float32)]),
        temporal_tokens=np.full((6, 24), 0.5 * scale, dtype=np.float32),
        temporal_mask=np.concatenate([np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]),
    )


_TORCH_READY = bool(getattr(sac_controller_module, "_TORCH_AVAILABLE", False))


class SACSMOPSOCheckpointSelectionTest(unittest.TestCase):
    def test_checkpoint_selection_prioritizes_constraints_before_hv(self) -> None:
        unsafe_high_hv = {
            "feasibleMeanMean": 0.7,
            "violationMeanMean": 0.0,
            "conflictMeanMean": 0.0,
            "hypervolumeMean": 0.9,
            "pureDiversityMean": 0.9,
            "firstFeasibleGenerationMean": 1.0,
        }
        feasible_low_hv = {
            "feasibleMeanMean": 0.8,
            "violationMeanMean": 1.0,
            "conflictMeanMean": 1.0,
            "hypervolumeMean": 0.1,
            "pureDiversityMean": 0.1,
            "firstFeasibleGenerationMean": 10.0,
        }
        equal_feasible_safe = {
            "feasibleMeanMean": 0.8,
            "violationMeanMean": 0.2,
            "conflictMeanMean": 0.1,
            "hypervolumeMean": 0.2,
            "pureDiversityMean": 0.1,
            "firstFeasibleGenerationMean": 10.0,
        }

        self.assertGreater(
            train_script._summary_selection_key(feasible_low_hv),
            train_script._summary_selection_key(unsafe_high_hv),
        )
        self.assertGreater(
            train_script._summary_selection_key(equal_feasible_safe),
            train_script._summary_selection_key(feasible_low_hv),
        )


@unittest.skipUnless(_TORCH_READY, "Torch is required for SAC-SMOPSO checkpoint tests.")
class SACSMOPSOPretrainingTest(unittest.TestCase):
    def _string_field(self, payload: dict[str, object], key: str) -> str:
        values = np.asarray(payload[key]).reshape(-1)
        return str(values[0]) if values.size > 0 else ""

    def test_controller_checkpoint_round_trip_enables_frozen_policy(self) -> None:
        spec = TemporalRelationalStateSpec(
            global_dim=24,
            population_dim=14,
            archive_dim=14,
            topology_dim=8,
            interaction_dim=7,
            environment_dim=8,
            temporal_dim=24,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "controller.pt"
            controller = HybridSACController(
                state_spec=spec,
                action_dim=12,
                operator_names=("base", "sbx", "de", "elite", "spread"),
                device_tag="cpu",
                config=ControllerConfig(warmup_steps=1, batch_size=2, replay_capacity=16),
                policy_mode="finetune",
            )
            state = _make_state(scale=1.0)
            next_state = _make_state(scale=0.9)
            action = controller.act(state)
            controller.observe(state, action, reward=0.4, next_state=next_state, done=False)
            controller.observe(next_state, action, reward=0.2, next_state=state, done=True)
            controller.save_checkpoint(checkpoint, extra_metadata={"tag": "unit"})

            frozen = HybridSACController(
                state_spec=spec,
                action_dim=12,
                operator_names=("base", "sbx", "de", "elite", "spread"),
                device_tag="cpu",
                config=ControllerConfig(warmup_steps=1, batch_size=2, replay_capacity=16),
                policy_mode="frozen",
            )
            metadata = frozen.load_checkpoint(checkpoint, load_optimizers=False)
            frozen_action = frozen.act(state, deterministic=True)

            self.assertEqual(metadata["tag"], "unit")
            self.assertTrue(frozen.can_use_policy())
            self.assertEqual(frozen.policy_mode, "frozen")
            self.assertEqual(frozen_action.source, "sac-mixed")
            self.assertEqual(frozen_action.continuous.shape, (12,))

            resumed = HybridSACController(
                state_spec=spec,
                action_dim=12,
                operator_names=("base", "sbx", "de", "elite", "spread"),
                device_tag="cpu",
                config=ControllerConfig(warmup_steps=1, batch_size=2, replay_capacity=16),
                policy_mode="finetune",
            )
            resumed.load_checkpoint(checkpoint, load_optimizers=True)
            resumed_metadata = resumed.metadata()
            self.assertEqual(len(resumed.replay), 2)
            self.assertEqual(resumed.training_steps, controller.training_steps)
            self.assertEqual(resumed.total_steps, controller.total_steps)
            self.assertEqual(resumed_metadata["controllerReplaySize"], 2.0)

    def test_checkpoint_load_ignores_legacy_operator_set_when_operator_head_disabled(self) -> None:
        spec = TemporalRelationalStateSpec(
            global_dim=24,
            population_dim=14,
            archive_dim=14,
            topology_dim=8,
            interaction_dim=7,
            environment_dim=8,
            temporal_dim=24,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "controller.pt"
            source = HybridSACController(
                state_spec=spec,
                action_dim=12,
                operator_names=("base", "sbx", "de", "elite", "spread"),
                device_tag="cpu",
                config=ControllerConfig(warmup_steps=1, batch_size=2, replay_capacity=16, use_operator_head=False),
                policy_mode="finetune",
            )
            state = _make_state(scale=1.0)
            next_state = _make_state(scale=0.8)
            action = source.act(state)
            source.observe(state, action, reward=0.3, next_state=next_state, done=False)
            source.observe(next_state, action, reward=0.1, next_state=state, done=True)
            source.save_checkpoint(checkpoint, extra_metadata={"tag": "legacy-ops"})

            resumed = HybridSACController(
                state_spec=spec,
                action_dim=12,
                operator_names=("base", "sbx"),
                device_tag="cpu",
                config=ControllerConfig(warmup_steps=1, batch_size=2, replay_capacity=16, use_operator_head=False),
                policy_mode="finetune",
            )
            metadata = resumed.load_checkpoint(checkpoint, load_optimizers=True)
            resumed_action = resumed.act(state, deterministic=True)

            self.assertEqual(metadata["tag"], "legacy-ops")
            self.assertEqual(len(resumed.replay), 2)
            self.assertTrue(resumed.can_use_policy())
            self.assertEqual(resumed_action.source, "sac-mixed")
            self.assertEqual(resumed_action.operator_probs.shape, (2,))

    def test_paper_stage_preset_covers_uav8_and_actor_only_anneal(self) -> None:
        preset = stage_preset("paper_stage4")
        self.assertEqual(preset["fleetSizes"], [8])
        self.assertEqual(preset["population"], 24)
        controller_config = preset["controllerConfig"]
        self.assertEqual(controller_config["sacLoadedPolicyMixEnd"], 1.0)
        self.assertGreaterEqual(controller_config["sacReplayCapacity"], 4096)

    def test_runner_can_finetune_then_reuse_frozen_checkpoint(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
        terrain["n"] = 3
        multi = make_fleet_terrain(terrain, fleet_size=3, seed=29, separation_min=10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "controller.pt"
            seed_everything(41)
            train_params = BenchmarkParams(
                generations=3,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir) / "train_results",
                problem_name="train_case",
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
                    "sacWarmupSteps": 2,
                    "sacBatchSize": 4,
                    "sacReplayCapacity": 64,
                },
            )
            run_sac_smopso(multi, train_params)
            self.assertTrue(checkpoint.exists())

            eval_params = BenchmarkParams(
                generations=3,
                population=6,
                runs=1,
                compute_metrics=True,
                results_dir=Path(tmpdir) / "eval_results",
                problem_name="eval_case",
                problem_index=1,
                mode="fleet",
                fleet_size=3,
                separation_min=10.0,
                gpu_mode="off",
                seed=43,
                extra={
                    "resumeExistingRuns": False,
                    "stateRepresentation": "TRFTS",
                    "sacPolicyMode": "frozen",
                    "sacCheckpointPath": str(checkpoint),
                    "sacDeterministicPolicy": True,
                    "sacWarmupSteps": 2,
                    "sacBatchSize": 4,
                    "sacReplayCapacity": 64,
                },
            )
            run_sac_smopso(multi, eval_params)
            metadata = load_mat(Path(tmpdir) / "eval_results" / "eval_case" / "Run_1" / "rl_metadata.mat")
            self.assertEqual(self._string_field(metadata, "policyMode"), "frozen")
            self.assertEqual(float(np.asarray(metadata["checkpointLoaded"], dtype=float).reshape(-1)[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
