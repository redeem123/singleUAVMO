from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import evaluate_sac_smopso_policy_modes as policy_eval_script
from uav_benchmark.algorithms.sac_smopso import controller as sac_controller_module
from uav_benchmark.algorithms.sac_smopso.controller import (
    ControllerConfig,
    HybridSACController,
    TemporalRelationalStateSpec,
)
from uav_benchmark.config import BenchmarkParams

_TORCH_READY = bool(getattr(sac_controller_module, "_TORCH_AVAILABLE", False))


@unittest.skipUnless(_TORCH_READY, "Torch is required for SAC policy-eval checkpoint tests.")
class SACSMOPSOPolicyEvalTest(unittest.TestCase):
    def test_run_mode_uses_hidden_dim_from_checkpoint(self) -> None:
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
                config=ControllerConfig(hidden_dim=96, warmup_steps=1, batch_size=2, replay_capacity=16),
                policy_mode="finetune",
            )
            controller.save_checkpoint(checkpoint, extra_metadata={"tag": "hidden96"})

            captured: dict[str, int] = {}

            def _fake_run_sac_smopso(_terrain: dict[str, object], params: BenchmarkParams) -> np.ndarray:
                captured["hidden_dim"] = int(params.extra["sacHiddenDim"])
                return np.zeros((1, 2), dtype=float)

            def _fake_load_mat(path: Path) -> dict[str, object]:
                if path.name == "rl_metadata.mat":
                    return {"policyMode": "frozen", "checkpointLoaded": 1.0}
                if path.name == "mission_stats.mat":
                    return {
                        "feasible": np.asarray([1.0], dtype=float),
                        "conflictRate": np.asarray([0.0], dtype=float),
                        "turnViolation": np.asarray([0.0], dtype=float),
                        "separationViolation": np.asarray([0.0], dtype=float),
                        "collisionViolation": np.asarray([0.0], dtype=float),
                    }
                raise AssertionError(f"Unexpected mat load: {path}")

            with (
                mock.patch.object(policy_eval_script, "run_sac_smopso", side_effect=_fake_run_sac_smopso),
                mock.patch.object(
                    policy_eval_script,
                    "load_mat",
                    side_effect=_fake_load_mat,
                ),
            ):
                record = policy_eval_script._run_mode(
                    terrain={"fleetSize": 3, "separationMin": 10.0, "safeDist": 10.0, "maxTurnDeg": 75.0},
                    problem_name="policy_eval_case",
                    problem_index=1,
                    seed=11,
                    mode="frozen",
                    source_checkpoint=checkpoint,
                    generations=3,
                    population=6,
                    results_dir=Path(tmpdir) / "results",
                    gpu_mode="off",
                    state_representation="TRFTS",
                )

            self.assertEqual(captured["hidden_dim"], 96)
            self.assertEqual(float(record["checkpointLoaded"]), 1.0)


if __name__ == "__main__":
    unittest.main()
