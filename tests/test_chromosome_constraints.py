from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.core.chromosome import Chromosome


class ChromosomeConstraintTest(unittest.TestCase):
    def test_constraint_violation_uses_configured_turn_limit(self) -> None:
        path = np.array(
            [
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 10.0],
                [20.0, 0.0, 10.0],
                [20.0, 10.0, 10.0],
                [20.0, 20.0, 10.0],
            ],
            dtype=float,
        )
        chromosome = Chromosome(rnvec=path.copy(), path=path.copy())

        chromosome.compute_constraint_violation({"maxTurnDeg": 120.0})
        self.assertEqual(chromosome.cons, 0.0)

        chromosome.compute_constraint_violation({"maxTurnDeg": 45.0})
        self.assertGreater(chromosome.cons, 0.0)


if __name__ == "__main__":
    unittest.main()
