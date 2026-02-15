from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_benchmark.algorithms.rl_controller import LinUCBController


class RLControllerDeterminismTest(unittest.TestCase):
    def test_action_sequence_reproducible(self) -> None:
        features = np.array([0.2, 0.7, 0.1, 0.02, 0.3, 0.0], dtype=float)

        np.random.seed(1234)
        controller_a = LinUCBController(n_features=6, alpha=1.0, warmup_steps=3)
        actions_a = []
        for _ in range(8):
            idx, _action, _theta = controller_a.select_action(features)
            controller_a.update(idx, features, reward=0.1)
            actions_a.append(idx)

        np.random.seed(1234)
        controller_b = LinUCBController(n_features=6, alpha=1.0, warmup_steps=3)
        actions_b = []
        for _ in range(8):
            idx, _action, _theta = controller_b.select_action(features)
            controller_b.update(idx, features, reward=0.1)
            actions_b.append(idx)

        self.assertEqual(actions_a, actions_b)

    def test_allowed_action_mask_respected(self) -> None:
        features = np.array([0.4, 0.5, 0.2, 0.2, 0.2, 0.1], dtype=float)
        controller = LinUCBController(n_features=6, alpha=0.5, warmup_steps=5)
        for _ in range(10):
            idx, _action, _theta = controller.select_action(features, allowed_actions=[2, 5])
            controller.update(idx, features, reward=0.1)
            self.assertIn(idx, {2, 5})

    def test_checkpoint_load_freeze_blocks_updates(self) -> None:
        features = np.array([0.1, 0.6, 0.0, 0.3, 0.2, 0.0], dtype=float)
        controller = LinUCBController(n_features=6, alpha=1.0, warmup_steps=0)
        idx, _action, _theta = controller.select_action(features)
        controller.update(idx, features, reward=0.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "linucb_policy.npz"
            controller.save(checkpoint)
            restored = LinUCBController(n_features=6, alpha=1.0, warmup_steps=0)
            loaded = restored.load(checkpoint, freeze=True)
            self.assertTrue(loaded)
            before = np.asarray(restored._a[0], dtype=float).copy()  # noqa: SLF001
            restored.update(0, features, reward=1.0)
            after = np.asarray(restored._a[0], dtype=float)  # noqa: SLF001
            self.assertTrue(np.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
