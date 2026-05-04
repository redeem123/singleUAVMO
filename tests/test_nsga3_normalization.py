from __future__ import annotations

import unittest
import warnings

import numpy as np

from uav_benchmark.core.nsga3_ops import _last_selection, _pairwise_cosine_distance, uniform_point


class NSGA3NormalizationTest(unittest.TestCase):
    def test_pairwise_cosine_distance_handles_nonfinite_and_large_values(self) -> None:
        lhs = np.array([[np.inf, 1.0, 0.0], [1e308, 1e308, 0.0]], dtype=float)
        rhs = np.array([[1.0, 0.0, 0.0], [1e308, -1e308, 0.0]], dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            distance = _pairwise_cosine_distance(lhs, rhs)

        self.assertEqual(distance.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(distance)))
        self.assertTrue(np.all(distance >= 0.0))
        self.assertTrue(np.all(distance <= 2.0))

    def test_last_selection_handles_nonfinite_objectives_without_warning(self) -> None:
        reference_points, _ = uniform_point(6, 4)
        pop_obj_first = np.array([[0.2, 0.3, 0.4, 0.5]], dtype=float)
        pop_obj_last = np.array(
            [
                [np.inf, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3, np.inf],
            ],
            dtype=float,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            chosen = _last_selection(
                pop_obj_first,
                pop_obj_last,
                2,
                reference_points,
                np.zeros(4, dtype=float),
            )

        self.assertEqual(chosen.shape, (3,))
        self.assertEqual(int(np.sum(chosen)), 2)


if __name__ == "__main__":
    unittest.main()
