from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import uav_benchmark.algorithms.moqgwo as moqgwo_module
from uav_benchmark.algorithms.moqgwo import (
    QGWO_Engine,
    run_fleet_moqgwo_no_atlas,
    run_fleet_moqgwo_standard_gwo,
)
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain

np = moqgwo_module.np


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


class MOQGWOAttentionTest(unittest.TestCase):
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
            wolf_risk=np.asarray([0.1, 0.8, 0.3, 0.5], dtype=float),
            feasibility_pressure=0.35,
            leader_objectives=np.asarray(
                [
                    [0.3, 0.5, 0.4, 0.6],
                    [0.8, 0.2, 0.2, 0.6],
                    [0.4, 0.4, 0.7, 0.2],
                ],
                dtype=float,
            ),
            leader_risk=np.asarray([0.2, 0.7, 0.4], dtype=float),
            wolf_topology=np.asarray([5, 2, 7, 3], dtype=int),
            wolf_robust=np.asarray([2, 1, 4, 3], dtype=int),
            leader_topology=np.asarray([5, 7, 3], dtype=int),
            leader_robust=np.asarray([2, 3, 1], dtype=int),
            atlas_enabled=True,
            atlas_robust_bins=4,
        )
        weights = engine._attention_weights()
        self.assertEqual(weights.shape, (4, 3))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.allclose(np.sum(weights, axis=1), np.ones(4, dtype=float), atol=1e-8))

    def test_higher_feasibility_pressure_increases_safe_leader_weight(self) -> None:
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
        wolf_obj = np.asarray([[0.4, 0.5, 0.6, 0.7]], dtype=float)
        leader_obj = np.repeat(wolf_obj, 3, axis=0)
        leader_risk = np.asarray([0.0, 1.0, 1.0], dtype=float)

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            wolf_risk=np.asarray([0.2], dtype=float),
            feasibility_pressure=0.0,
            leader_objectives=leader_obj,
            leader_risk=leader_risk,
            atlas_enabled=False,
        )
        low_pressure_weight = float(engine._attention_weights()[0, 0])

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            wolf_risk=np.asarray([0.2], dtype=float),
            feasibility_pressure=1.0,
            leader_objectives=leader_obj,
            leader_risk=leader_risk,
            atlas_enabled=False,
        )
        high_pressure_weight = float(engine._attention_weights()[0, 0])
        self.assertGreater(high_pressure_weight, low_pressure_weight + 1e-6)

    def test_atlas_channel_rewards_topology_match(self) -> None:
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
        wolf_obj = np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=float)
        leader_obj = np.repeat(wolf_obj, 3, axis=0)
        leader_risk = np.asarray([0.0, 0.0, 0.0], dtype=float)

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            wolf_risk=np.asarray([0.1], dtype=float),
            feasibility_pressure=0.0,
            leader_objectives=leader_obj,
            leader_risk=leader_risk,
            wolf_topology=np.asarray([5], dtype=int),
            wolf_robust=np.asarray([2], dtype=int),
            leader_topology=np.asarray([5, 8, 8], dtype=int),
            leader_robust=np.asarray([2, 2, 3], dtype=int),
            atlas_enabled=False,
            atlas_robust_bins=4,
        )
        weight_no_atlas = float(engine._attention_weights()[0, 0])

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            wolf_risk=np.asarray([0.1], dtype=float),
            feasibility_pressure=0.0,
            leader_objectives=leader_obj,
            leader_risk=leader_risk,
            wolf_topology=np.asarray([5], dtype=int),
            wolf_robust=np.asarray([2], dtype=int),
            leader_topology=np.asarray([5, 8, 8], dtype=int),
            leader_robust=np.asarray([2, 2, 3], dtype=int),
            atlas_enabled=True,
            atlas_robust_bins=4,
        )
        weights_atlas = engine._attention_weights()[0]
        self.assertGreater(float(weights_atlas[0]), weight_no_atlas + 1e-6)
        self.assertGreater(float(weights_atlas[0]), float(weights_atlas[1]))

    def test_no_atlas_variant_disables_atlas_channel_in_context(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(Path(tmpdir), problem_name="no_atlas_attention")
            seen_flags: list[bool] = []
            original = moqgwo_module.QGWO_Engine.set_attention_context

            def wrapped(self, **kwargs):  # type: ignore[no-untyped-def]
                seen_flags.append(bool(kwargs.get("atlas_enabled", False)))
                return original(self, **kwargs)

            with patch.object(moqgwo_module.QGWO_Engine, "set_attention_context", new=wrapped):
                run_fleet_moqgwo_no_atlas(terrain, params)
            self.assertTrue(seen_flags)
            self.assertTrue(all(flag is False for flag in seen_flags))

    def test_no_atlas_variant_disables_topology_in_rtcs(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(Path(tmpdir), problem_name="no_atlas_rtcs")
            params.extra["moqgwoInitBias"] = True
            seen_flags: list[bool] = []
            original = moqgwo_module._rtcs_initialize_population

            def wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
                seen_flags.append(bool(kwargs.get("use_topology", True)))
                return original(*args, **kwargs)

            with patch.object(moqgwo_module, "_rtcs_initialize_population", new=wrapped):
                run_fleet_moqgwo_no_atlas(terrain, params)
            self.assertTrue(seen_flags)
            self.assertTrue(all(flag is False for flag in seen_flags))

    def test_standard_gwo_does_not_call_attention_context(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(Path(tmpdir), problem_name="standard_gwo_no_attention_context")

            def fail_if_called(self, **kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("set_attention_context must not be called for MOQGWO-STANDARD-GWO.")

            with patch.object(moqgwo_module.QGWO_Engine, "set_attention_context", new=fail_if_called):
                run_fleet_moqgwo_standard_gwo(terrain, params)

    def test_standard_gwo_does_not_call_rtcs_initializer(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(Path(tmpdir), problem_name="standard_gwo_no_rtcs")

            def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("_rtcs_initialize_population must not be called for MOQGWO-STANDARD-GWO.")

            with patch.object(moqgwo_module, "_rtcs_initialize_population", new=fail_if_called):
                run_fleet_moqgwo_standard_gwo(terrain, params)


if __name__ == "__main__":
    unittest.main()
