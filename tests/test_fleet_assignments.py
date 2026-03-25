from __future__ import annotations

import unittest

import numpy as np

from uav_benchmark.problem_generation.fleet_assignments import sample_homogeneous_assignments


def _terrain(size: float) -> dict[str, float | np.ndarray]:
    return {
        "start": np.array([1.0, 1.0, 5.0], dtype=float),
        "end": np.array([size, size, 5.0], dtype=float),
        "xmin": 1.0,
        "xmax": float(size),
        "ymin": 1.0,
        "ymax": float(size),
        "zmin": 0.0,
        "zmax": 20.0,
    }


class FleetAssignmentsTest(unittest.TestCase):
    def test_generated_assignments_respect_requested_separation(self) -> None:
        assignment = sample_homogeneous_assignments(
            terrain=_terrain(200.0),
            fleet_size=5,
            seed=7,
            separation_min=30.0,
        )
        for points in (assignment.starts, assignment.goals):
            for index in range(points.shape[0]):
                for other_index in range(index + 1, points.shape[0]):
                    distance = float(np.linalg.norm(points[index, :2] - points[other_index, :2]))
                    self.assertGreaterEqual(distance, 30.0 - 1e-6)

    def test_raises_when_requested_separation_cannot_fit(self) -> None:
        with self.assertRaises(ValueError):
            sample_homogeneous_assignments(
                terrain=_terrain(10.0),
                fleet_size=5,
                seed=3,
                separation_min=20.0,
            )


if __name__ == "__main__":
    unittest.main()
