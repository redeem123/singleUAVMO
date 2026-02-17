from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.core.rng_utils import ensure_rng


class EnsureRngTest(unittest.TestCase):
    def test_none_rng_respects_global_seed(self) -> None:
        state = np.random.get_state()
        try:
            np.random.seed(12345)
            first = ensure_rng().integers(0, 1_000_000, size=12)
            np.random.seed(12345)
            second = ensure_rng().integers(0, 1_000_000, size=12)
            np.testing.assert_array_equal(first, second)
        finally:
            np.random.set_state(state)

    def test_passthrough_generator_identity(self) -> None:
        rng = np.random.default_rng(7)
        self.assertIs(ensure_rng(rng), rng)


if __name__ == "__main__":
    unittest.main()
