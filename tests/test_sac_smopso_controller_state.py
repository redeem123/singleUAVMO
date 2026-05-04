from __future__ import annotations

import unittest
from collections import deque

import numpy as np

from uav_benchmark.algorithms.sac_smopso.controller import (
    ControllerConfig,
    HybridSACController,
    TemporalRelationalState,
    TemporalRelationalStateSpec,
)
from uav_benchmark.algorithms.shared.adaptive_control import build_adaptive_state
from uav_benchmark.algorithms.shared.pso_types import Candidate


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


class SACSMOPSOControllerStateTest(unittest.TestCase):
    def test_controller_accepts_temporal_relational_state(self) -> None:
        spec = TemporalRelationalStateSpec(
            global_dim=24,
            population_dim=14,
            archive_dim=14,
            topology_dim=8,
            interaction_dim=7,
            environment_dim=8,
            temporal_dim=24,
        )
        controller = HybridSACController(
            state_spec=spec,
            action_dim=12,
            operator_names=("base", "sbx", "de", "elite", "spread"),
            device_tag="cpu",
            config=ControllerConfig(warmup_steps=1, batch_size=2, replay_capacity=16),
        )

        state = _make_state(scale=1.0)
        next_state = _make_state(scale=0.9)
        action = controller.act(state)

        self.assertEqual(action.continuous.shape, (12,))
        self.assertEqual(action.operator_probs.shape, (5,))
        self.assertEqual(state.summary_vector().shape[0], 99)

        controller.observe(state, action, reward=0.5, next_state=next_state, done=False)
        controller.observe(next_state, action, reward=0.1, next_state=state, done=True)
        metadata = controller.metadata()
        self.assertEqual(metadata["stateGlobalDim"], 24.0)
        self.assertEqual(metadata["stateTopologyDim"], 8.0)
        self.assertEqual(metadata["stateEnvironmentDim"], 8.0)
        self.assertEqual(metadata["encoderMode"], "learned")

    def test_constraint_pressure_state_encodes_closest_approach_timing(self) -> None:
        model = {
            "xmin": 0.0,
            "xmax": 100.0,
            "ymin": 0.0,
            "ymax": 100.0,
            "zmin": 0.0,
            "zmax": 30.0,
            "start": np.array([0.0, 0.0, 5.0]),
            "end": np.array([100.0, 100.0, 5.0]),
            "starts": np.array([[0.0, 0.0, 5.0], [0.0, 10.0, 5.0]]),
            "goals": np.array([[100.0, 100.0, 5.0], [100.0, 90.0, 5.0]]),
            "fleetSize": 2.0,
            "separationMin": 10.0,
            "maxTurnDeg": 75.0,
        }
        paths = [
            np.array([[0.0, 0.0, 5.0], [50.0, 50.0, 5.0], [100.0, 100.0, 5.0]], dtype=float),
            np.array([[0.0, 10.0, 5.0], [50.0, 52.0, 5.0], [100.0, 90.0, 5.0]], dtype=float),
        ]
        candidate = Candidate(
            vector=np.zeros(18, dtype=float),
            objective=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            details={
                "paths": paths,
                "feasible": 0.0,
                "conflictRate": 0.25,
                "minClearance": 20.0,
                "minSeparation": 2.0,
            },
        )
        state = build_adaptive_state(
            candidates=[candidate],
            archive_candidates=[],
            model=model,
            generation=1,
            total_generations=4,
            last_metrics={},
            algorithm_features=np.zeros(6, dtype=float),
            history=deque(),
            state_representation="TRFTS-CP",
        )

        token = state.interaction_tokens[0]
        self.assertEqual(float(state.interaction_mask[0]), 1.0)
        self.assertGreater(float(token[0]), 0.9)
        self.assertAlmostEqual(float(token[1]), 0.5, places=6)
        self.assertGreater(float(token[2]), 0.0)
        self.assertTrue(np.allclose(token[4:7], 0.0))
        self.assertAlmostEqual(float(state.global_features[22]), float(token[0]), places=6)


if __name__ == "__main__":
    unittest.main()
