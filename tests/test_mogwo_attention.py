from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import uav_benchmark.algorithms.mogwo as mogwo_module
from uav_benchmark.algorithms.mogwo import (
    QGWO_Engine,
    run_fleet_mogwo,
    run_fleet_mogwo_standard_gwo,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain

np = mogwo_module.np


def _tiny_fleet_terrain() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
    terrain["n"] = 2
    return make_fleet_terrain(terrain, fleet_size=3, seed=17, separation_min=10.0)


def _tiny_params(results_dir: Path, problem_name: str) -> BenchmarkParams:
    return BenchmarkParams(
        generations=1,
        population=4,
        runs=1,
        compute_metrics=False,
        results_dir=results_dir,
        problem_name=problem_name,
        problem_index=1,
        mode="fleet",
        fleet_size=3,
        separation_min=10.0,
        max_turn_deg=75.0,
        gpu_mode="off",
        extra={
            "resumeExistingRuns": False,
            "nRep": 4,
            "nGrid": 5,
        },
    )


class MOGWOAttentionTest(unittest.TestCase):
    def test_attention_weights_are_finite_and_row_normalized(self) -> None:
        engine = QGWO_Engine(
            lower=np.zeros(6, dtype=float),
            upper=np.ones(6, dtype=float),
            pop_size=4,
            use_attention=True,
        )
        engine.leaders = np.asarray(
            [
                [0.1, 0.3, 0.5, 0.2, 0.8, 0.4],
                [0.4, 0.8, 0.2, 0.6, 0.1, 0.7],
                [0.7, 0.2, 0.9, 0.1, 0.3, 0.5],
            ],
            dtype=float,
        )
        engine.set_attention_context(
            wolf_objectives=np.asarray(
                [
                    [0.2, 0.4, 0.3, 0.5],
                    [0.9, 0.1, 0.2, 0.7],
                    [0.5, 0.6, 0.5, 0.4],
                    [0.3, 0.2, 0.8, 0.1],
                ],
                dtype=float,
            ),
            feasibility_pressure=0.35,
            leader_objectives=np.asarray(
                [
                    [0.3, 0.5, 0.4, 0.6],
                    [0.8, 0.2, 0.2, 0.6],
                    [0.4, 0.4, 0.7, 0.2],
                ],
                dtype=float,
            ),
        )
        weights = engine._attention_weights()
        self.assertEqual(weights.shape, (4, 3))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.allclose(np.sum(weights, axis=1), np.ones(4, dtype=float), atol=1e-8))

    def test_higher_feasibility_pressure_increases_best_ranked_leader_weight(self) -> None:
        engine = QGWO_Engine(
            lower=np.zeros(3, dtype=float),
            upper=np.ones(3, dtype=float),
            pop_size=1,
            use_attention=True,
        )
        engine.leaders = np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
            ],
            dtype=float,
        )
        wolf_obj = np.asarray([[0.9, 0.9, 0.9, 0.9]], dtype=float)
        leader_obj = np.asarray(
            [
                [0.1, 0.1, 0.1, 0.1],  # globally best rank
                [0.5, 0.5, 0.5, 0.5],
                [0.9, 0.9, 0.9, 0.9],  # locally closest to wolf
            ],
            dtype=float,
        )

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=0.0,
            leader_objectives=leader_obj,
        )
        low_pressure_weights = engine._attention_weights()[0]

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=1.0,
            leader_objectives=leader_obj,
        )
        high_pressure_weights = engine._attention_weights()[0]
        self.assertGreater(float(high_pressure_weights[0]), float(low_pressure_weights[0]) + 1e-6)
        self.assertLess(float(high_pressure_weights[2]), float(low_pressure_weights[2]) - 1e-6)

    def test_low_diversity_biases_attention_towards_sparse_leaders(self) -> None:
        engine = QGWO_Engine(
            lower=np.zeros(3, dtype=float),
            upper=np.ones(3, dtype=float),
            pop_size=1,
            use_attention=True,
            use_diversity_feedback=True,
        )
        engine.leaders = np.asarray(
            [
                [0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4],
            ],
            dtype=float,
        )
        wolf_obj = np.asarray([[0.4, 0.4, 0.4, 0.4]], dtype=float)
        leader_obj = np.asarray(
            [
                [0.4, 0.4, 0.4, 0.4],
                [0.4, 0.4, 0.4, 0.4],
                [0.4, 0.4, 0.4, 0.4],
            ],
            dtype=float,
        )
        occupancy = np.asarray([6.0, 1.0, 1.0], dtype=float)

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=0.0,
            leader_objectives=leader_obj,
            diversity_level=0.95,
            leader_occupancy=occupancy,
        )
        high_div_weights = engine._attention_weights()[0]

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=0.0,
            leader_objectives=leader_obj,
            diversity_level=0.05,
            leader_occupancy=occupancy,
        )
        low_div_weights = engine._attention_weights()[0]
        sparse_weight_high_div = float(high_div_weights[1] + high_div_weights[2])
        sparse_weight_low_div = float(low_div_weights[1] + low_div_weights[2])
        self.assertGreater(sparse_weight_low_div, sparse_weight_high_div + 1e-6)

    def test_step_limiter_reduces_update_when_feasibility_pressure_is_high(self) -> None:
        leaders = np.asarray(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                [0.3, 0.7, 0.3, 0.7, 0.3, 0.7],
            ],
            dtype=float,
        )
        positions = np.asarray(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.4, 0.6, 0.4, 0.6, 0.4, 0.6],
            ],
            dtype=float,
        )
        wolf_obj = np.asarray(
            [
                [0.8, 0.8, 0.8, 0.8],
                [0.2, 0.2, 0.2, 0.2],
            ],
            dtype=float,
        )
        leader_obj = np.asarray(
            [
                [0.2, 0.2, 0.2, 0.2],
                [0.5, 0.5, 0.5, 0.5],
                [0.8, 0.8, 0.8, 0.8],
            ],
            dtype=float,
        )

        engine_low_p = QGWO_Engine(
            lower=np.zeros(6, dtype=float),
            upper=np.ones(6, dtype=float),
            pop_size=2,
            use_attention=True,
            use_step_limiter=True,
            use_attention_guard=False,
        )
        engine_low_p.positions = positions.copy()
        engine_low_p.leaders = leaders.copy()
        engine_low_p.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=0.0,
            leader_objectives=leader_obj,
            diversity_level=0.8,
            leader_occupancy=np.asarray([1.0, 2.0, 3.0], dtype=float),
        )

        engine_high_p = QGWO_Engine(
            lower=np.zeros(6, dtype=float),
            upper=np.ones(6, dtype=float),
            pop_size=2,
            use_attention=True,
            use_step_limiter=True,
            use_attention_guard=False,
        )
        engine_high_p.positions = positions.copy()
        engine_high_p.leaders = leaders.copy()
        engine_high_p.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=1.0,
            leader_objectives=leader_obj,
            diversity_level=0.2,
            leader_occupancy=np.asarray([1.0, 2.0, 3.0], dtype=float),
        )

        np.random.seed(19)
        low_next = engine_low_p.step(1, 10)
        np.random.seed(19)
        high_next = engine_high_p.step(1, 10)

        low_delta = float(np.mean(np.linalg.norm(low_next - positions, axis=1)))
        high_delta = float(np.mean(np.linalg.norm(high_next - positions, axis=1)))
        self.assertLess(high_delta, low_delta - 1e-6)

    def test_default_variant_calls_attention_context(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(Path(tmpdir), problem_name="default_attention_context")
            call_count = 0
            original = mogwo_module.QGWO_Engine.set_attention_context

            def wrapped(self, **kwargs):  # type: ignore[no-untyped-def]
                nonlocal call_count
                call_count += 1
                return original(self, **kwargs)

            with patch.object(mogwo_module.QGWO_Engine, "set_attention_context", new=wrapped):
                run_fleet_mogwo(terrain, params)
            self.assertGreater(call_count, 0)

    def test_standard_gwo_does_not_call_attention_context(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(Path(tmpdir), problem_name="standard_gwo_no_attention_context")

            def fail_if_called(self, **kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("set_attention_context must not be called for MOGWO-STANDARD-GWO.")

            with patch.object(mogwo_module.QGWO_Engine, "set_attention_context", new=fail_if_called):
                run_fleet_mogwo_standard_gwo(terrain, params)

if __name__ == "__main__":
    unittest.main()
