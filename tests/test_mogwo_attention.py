from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import uav_benchmark.algorithms.mogwo as mogwo_module
from uav_benchmark.algorithms.mogwo import (
    QGWO_Engine,
    run_fleet_mogwo,
    run_fleet_mogwo_no_attention,
    run_fleet_mogwo_standard_gwo,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.config import BenchmarkParams
from uav_benchmark.io.matlab import load_terrain_struct
from uav_benchmark.problem_generation.generate import make_fleet_terrain

np = mogwo_module.np


def _tiny_fleet_terrain() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    terrain = load_terrain_struct(project_root / "problems" / "terrainStruct_c_100.mat")
    terrain["n"] = 2
    return make_fleet_terrain(terrain, fleet_size=3, seed=17, separation_min=10.0)


def _tiny_params(
    results_dir: Path,
    problem_name: str,
    *,
    generations: int = 1,
    extra: dict[str, object] | None = None,
) -> BenchmarkParams:
    merged_extra: dict[str, object] = {
        "resumeExistingRuns": False,
        "nRep": 4,
        "nGrid": 5,
    }
    if extra:
        merged_extra.update(extra)
    return BenchmarkParams(
        generations=generations,
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
        extra=merged_extra,
    )


class MOGWOAttentionTest(unittest.TestCase):
    def test_constrained_archive_prefers_feasible_candidate(self) -> None:
        model = {"separationMin": 10.0, "droneSize": 1.0}
        feasible = Candidate(
            vector=np.zeros(2, dtype=float),
            objective=np.full(4, 0.8, dtype=float),
            details={
                "feasible": 1.0,
                "separationViolation": 0.0,
                "collisionViolation": 0.0,
            },
        )
        mildly_infeasible = Candidate(
            vector=np.ones(2, dtype=float),
            objective=np.zeros(4, dtype=float),
            details={
                "feasible": 0.0,
                "separationViolation": 1.0,
                "minSeparation": 9.5,
                "collisionViolation": 0.0,
            },
        )

        archive = mogwo_module._update_archive(
            [],
            [feasible, mildly_infeasible],
            model,
            max_size=1,
            divisions=5,
            use_constraints=True,
        )

        self.assertEqual(len(archive), 1)
        self.assertGreater(float(archive[0].details.get("feasible", 0.0)), 0.5)

    def test_terrain_seed_population_respects_bounds(self) -> None:
        terrain = _tiny_fleet_terrain()
        lower, upper = mogwo_module._build_bounds(
            terrain,
            fleet_size=3,
            n_waypoints=int(terrain["n"]),
        )
        seeds, fraction = mogwo_module._terrain_seed_population(
            terrain,
            lower=lower,
            upper=upper,
            pop_size=6,
            fleet_size=3,
            n_waypoints=int(terrain["n"]),
        )
        self.assertEqual(seeds.shape[1], lower.size)
        self.assertGreater(seeds.shape[0], 0)
        self.assertGreater(fraction, 0.0)
        self.assertTrue(np.all(seeds >= (lower - 1e-9)))
        self.assertTrue(np.all(seeds <= (upper + 1e-9)))

    def test_relaxed_constraint_archive_prefers_near_feasible_guides(self) -> None:
        model = {"separationMin": 10.0, "droneSize": 1.0}
        feasible = Candidate(
            vector=np.asarray([0.2, 0.2, 0.2], dtype=float),
            objective=np.full(4, 0.2, dtype=float),
            details={
                "feasible": 1.0,
                "separationViolation": 0.0,
                "collisionViolation": 0.0,
            },
        )
        near_feasible = Candidate(
            vector=np.asarray([0.4, 0.4, 0.4], dtype=float),
            objective=np.full(4, 0.35, dtype=float),
            details={
                "feasible": 0.0,
                "separationViolation": 1.0,
                "minSeparation": 9.7,
                "collisionViolation": 0.0,
            },
        )
        far_infeasible = Candidate(
            vector=np.asarray([0.9, 0.9, 0.9], dtype=float),
            objective=np.full(4, 0.1, dtype=float),
            details={
                "feasible": 0.0,
                "separationViolation": 1.0,
                "minSeparation": 6.0,
                "collisionViolation": 0.0,
            },
        )

        relaxed = mogwo_module._update_relaxed_constraint_archive(
            [],
            [near_feasible, far_infeasible],
            [feasible],
            [near_feasible, far_infeasible],
            model,
            max_size=3,
            divisions=4,
            relaxation_eps=0.2,
        )

        self.assertTrue(relaxed)
        relaxed_vectors = {tuple(np.asarray(candidate.vector, dtype=float)) for candidate in relaxed}
        self.assertIn(tuple(feasible.vector), relaxed_vectors)
        self.assertIn(tuple(near_feasible.vector), relaxed_vectors)
        self.assertNotIn(tuple(far_infeasible.vector), relaxed_vectors)

    def test_feedback_relaxation_threshold_shrinks_as_feasibility_recovers(self) -> None:
        model = {"separationMin": 10.0, "droneSize": 1.0}
        archive_unconstrained = [
            Candidate(
                vector=np.asarray([0.4, 0.4, 0.4], dtype=float),
                objective=np.full(4, 0.4, dtype=float),
                details={
                    "feasible": 0.0,
                    "separationViolation": 1.0,
                    "minSeparation": 9.6,
                    "collisionViolation": 0.0,
                },
            ),
            Candidate(
                vector=np.asarray([0.8, 0.8, 0.8], dtype=float),
                objective=np.full(4, 0.6, dtype=float),
                details={
                    "feasible": 0.0,
                    "separationViolation": 1.0,
                    "minSeparation": 8.5,
                    "collisionViolation": 0.0,
                },
            ),
            Candidate(
                vector=np.asarray([0.6, 0.6, 0.6], dtype=float),
                objective=np.full(4, 0.5, dtype=float),
                details={
                    "feasible": 0.0,
                    "separationViolation": 1.0,
                    "minSeparation": 6.0,
                    "collisionViolation": 0.0,
                },
            ),
            Candidate(
                vector=np.asarray([0.9, 0.9, 0.9], dtype=float),
                objective=np.full(4, 0.7, dtype=float),
                details={
                    "feasible": 0.0,
                    "separationViolation": 1.0,
                    "minSeparation": 3.0,
                    "collisionViolation": 0.0,
                },
            ),
        ]

        low_feas_eps = mogwo_module._feedback_relaxation_threshold(
            candidates=list(archive_unconstrained),
            archive_unconstrained=archive_unconstrained,
            model=model,
            feasible_ratio=0.20,
            previous_feasible_ratio=0.15,
            generation=20,
            max_generations=100,
        )
        high_feas_eps = mogwo_module._feedback_relaxation_threshold(
            candidates=list(archive_unconstrained),
            archive_unconstrained=archive_unconstrained,
            model=model,
            feasible_ratio=0.85,
            previous_feasible_ratio=0.80,
            generation=20,
            max_generations=100,
        )

        self.assertGreater(low_feas_eps, high_feas_eps)
        self.assertGreater(low_feas_eps, 0.0)

    def test_attention_weights_are_finite_and_row_normalized(self) -> None:
        engine = QGWO_Engine(
            lower=np.zeros(6, dtype=float),
            upper=np.ones(6, dtype=float),
            pop_size=4,
            use_attention=True,
        )
        engine.leaders = np.asarray(  # type: ignore[assignment]
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

    def test_relay_activation_is_reported_in_stats(self) -> None:
        engine = QGWO_Engine(
            lower=np.zeros(3, dtype=float),
            upper=np.ones(3, dtype=float),
            pop_size=1,
            use_attention=True,
        )
        engine.leaders = np.asarray(  # type: ignore[assignment]
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
                [0.1, 0.1, 0.1, 0.1],
                [0.5, 0.5, 0.5, 0.5],
                [0.9, 0.9, 0.9, 0.9],
            ],
            dtype=float,
        )
        relay_guides = np.asarray([[0.95, 0.95, 0.95]], dtype=float)

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=0.20,
            leader_objectives=leader_obj,
            diversity_level=0.2,
            relay_guides=relay_guides,
            relay_activation=0.25,
        )
        weights_low = engine._attention_weights()[0]
        self.assertTrue(np.allclose(weights_low, np.full(3, 1.0 / 3.0), atol=1e-8))
        self.assertAlmostEqual(float(engine.last_attention_stats.get("stage_activation", 0.0)), 0.25, places=6)

        engine.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=0.20,
            leader_objectives=leader_obj,
            diversity_level=0.2,
            relay_guides=relay_guides,
            relay_activation=0.80,
        )
        _ = engine._attention_weights()
        self.assertAlmostEqual(float(engine.last_attention_stats.get("stage_activation", 0.0)), 0.80, places=6)

    def test_zero_relay_activation_falls_back_to_uniform_weights(self) -> None:
        engine = QGWO_Engine(
            lower=np.zeros(3, dtype=float),
            upper=np.ones(3, dtype=float),
            pop_size=2,
            use_attention=True,
        )
        engine.set_attention_context(
            wolf_objectives=np.asarray(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.6, 0.7, 0.8, 0.9],
                ],
                dtype=float,
            ),
            feasibility_pressure=0.95,
            leader_objectives=np.asarray(
                [
                    [0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2],
                    [0.7, 0.6, 0.5, 0.4],
                ],
                dtype=float,
            ),
            diversity_level=0.2,
            leader_occupancy=np.asarray([1.0, 3.0, 5.0], dtype=float),
        )
        weights = engine._attention_weights()
        self.assertTrue(np.allclose(weights, np.full((2, 3), 1.0 / 3.0), atol=1e-8))
        self.assertEqual(engine.last_attention_stats.get("attention_guard_active"), 0.0)
        self.assertLess(float(engine.last_attention_stats.get("stage_activation", 1.0)), 1e-9)

    def test_topology_relay_guides_prefer_low_violation_pool_under_low_feasibility(self) -> None:
        model = {"separationMin": 10.0, "droneSize": 1.0}
        lower = np.zeros(3, dtype=float)
        upper = np.ones(3, dtype=float)
        low_violation = Candidate(
            vector=np.asarray([0.2, 0.2, 0.2], dtype=float),
            objective=np.full(4, 0.4, dtype=float),
            details={
                "feasible": 0.0,
                "separationViolation": 1.0,
                "minSeparation": 9.5,
                "collisionViolation": 0.0,
            },
        )
        high_violation = Candidate(
            vector=np.asarray([0.85, 0.85, 0.85], dtype=float),
            objective=np.full(4, 0.6, dtype=float),
            details={
                "feasible": 0.0,
                "separationViolation": 1.0,
                "minSeparation": 3.0,
                "collisionViolation": 0.0,
            },
        )
        candidates = [
            Candidate(
                vector=np.asarray([0.18, 0.18, 0.18], dtype=float),
                objective=np.full(4, 0.5, dtype=float),
                details={"feasible": 0.0, "separationViolation": 1.0, "minSeparation": 8.0},
            ),
            Candidate(
                vector=np.asarray([0.22, 0.22, 0.22], dtype=float),
                objective=np.full(4, 0.5, dtype=float),
                details={"feasible": 0.0, "separationViolation": 1.0, "minSeparation": 8.2},
            ),
        ]

        guides, activation, pool_share = mogwo_module._topology_relay_guides(
            pack_positions=np.stack([candidate.vector for candidate in candidates], axis=0),
            candidates=candidates,
            archive=[],
            archive_unconstrained=[high_violation, low_violation],
            relaxation_archive=[],
            model=model,
            lower=lower,
            upper=upper,
            feasible_ratio=0.20,
            diversity_level=0.30,
            relaxation_eps=0.0,
        )

        self.assertTrue(np.allclose(guides, np.broadcast_to(low_violation.vector, guides.shape), atol=1e-8))
        self.assertGreater(activation, 0.35)
        self.assertEqual(pool_share, 0.0)

    def test_relay_guides_shift_step_towards_assisting_pool(self) -> None:
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
        relay_guides = np.asarray(
            [
                [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
                [0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
            ],
            dtype=float,
        )

        engine_with_relay = QGWO_Engine(
            lower=np.zeros(6, dtype=float),
            upper=np.ones(6, dtype=float),
            pop_size=2,
            use_attention=True,
        )
        engine_with_relay.positions = positions.copy()
        engine_with_relay.leaders = leaders.copy()  # type: ignore[assignment]
        engine_with_relay.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=0.38,
            leader_objectives=leader_obj,
            diversity_level=0.2,
            leader_occupancy=np.asarray([1.0, 2.0, 3.0], dtype=float),
            relay_guides=relay_guides,
            relay_activation=1.0,
        )

        engine_without_relay = QGWO_Engine(
            lower=np.zeros(6, dtype=float),
            upper=np.ones(6, dtype=float),
            pop_size=2,
            use_attention=True,
        )
        engine_without_relay.positions = positions.copy()
        engine_without_relay.leaders = leaders.copy()  # type: ignore[assignment]
        engine_without_relay.set_attention_context(
            wolf_objectives=wolf_obj,
            feasibility_pressure=0.38,
            leader_objectives=leader_obj,
            diversity_level=0.2,
            leader_occupancy=np.asarray([1.0, 2.0, 3.0], dtype=float),
            relay_guides=relay_guides,
            relay_activation=0.0,
        )

        np.random.seed(19)
        guided_next = engine_with_relay.step(1, 10)
        np.random.seed(19)
        base_next = engine_without_relay.step(1, 10)

        guided_distance = float(np.mean(np.linalg.norm(guided_next - relay_guides, axis=1)))
        base_distance = float(np.mean(np.linalg.norm(base_next - relay_guides, axis=1)))
        self.assertLess(guided_distance, base_distance - 1e-6)

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

    def test_standard_gwo_uses_standard_variant_controls(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(Path(tmpdir), problem_name="standard_gwo_controls")
            select_calls: list[tuple[bool, bool]] = []
            saved_metadata: dict[str, object] = {}
            original = mogwo_module._select_leaders

            def wrapped_select(  # type: ignore[no-untyped-def]
                archive,
                divisions,
                use_advanced_archive=True,
                use_mean_selection=False,
                model=None,
                relaxation_eps=0.0,
            ):
                select_calls.append((bool(use_advanced_archive), bool(use_mean_selection)))
                return original(
                    archive,
                    divisions,
                    use_advanced_archive=use_advanced_archive,
                    use_mean_selection=use_mean_selection,
                    model=model,
                    relaxation_eps=relaxation_eps,
                )

            def capture_save(**kwargs):  # type: ignore[no-untyped-def]
                saved_metadata.update(kwargs.get("run_metadata", {}))

            with patch.object(mogwo_module, "_select_leaders", new=wrapped_select):
                with patch.object(mogwo_module, "_save_fleet_artifacts", new=capture_save):
                    run_fleet_mogwo_standard_gwo(terrain, params)

            self.assertTrue(select_calls)
            self.assertTrue(any((not advanced) and mean for advanced, mean in select_calls))
            self.assertEqual(saved_metadata.get("algorithmName"), "MOGWO-STANDARD-GWO")
            self.assertEqual(saved_metadata.get("mogwoVariant"), "standard_gwo")
            self.assertEqual(saved_metadata.get("mogwoUseAdvancedArchive"), 0.0)
            self.assertEqual(saved_metadata.get("mogwoUseMeanSelection"), 1.0)

    def test_no_attention_metadata_uses_variant_algorithm_name(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(Path(tmpdir), problem_name="no_attention_metadata")
            saved_metadata: dict[str, object] = {}

            def capture_save(**kwargs):  # type: ignore[no-untyped-def]
                saved_metadata.update(kwargs.get("run_metadata", {}))

            with patch.object(mogwo_module, "_save_fleet_artifacts", new=capture_save):
                run_fleet_mogwo_no_attention(terrain, params)

            self.assertEqual(saved_metadata.get("algorithmName"), "MOGWO-NO-ATTENTION")
            self.assertEqual(saved_metadata.get("mogwoVariant"), "no_attention")
            self.assertEqual(saved_metadata.get("mogwoComponentRepairRestart"), 1.0)
            self.assertEqual(saved_metadata.get("mogwoComponentFeedbackRelaxation"), 1.0)
            self.assertEqual(saved_metadata.get("mogwoComponentAdaptiveAttention"), 0.0)
            self.assertEqual(saved_metadata.get("mogwoComponentTerrainSeeding"), 0.0)
            self.assertEqual(saved_metadata.get("mogwoComponentDualArchiveExplorer"), 1.0)

    def test_blended_population_becomes_next_pack_state(self) -> None:
        terrain = _tiny_fleet_terrain()
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _tiny_params(
                Path(tmpdir),
                problem_name="blended_pack_state",
                generations=2,
            )
            step_inputs: list[np.ndarray] = []

            def fake_evaluate(population, model, fleet_size, n_waypoints, representation="cart"):  # type: ignore[no-untyped-def]
                del model, fleet_size, n_waypoints, representation
                candidates: list[Candidate] = []
                for index, vector in enumerate(np.asarray(population, dtype=float)):
                    objective = np.full(4, float(index) / max(1, population.shape[0]), dtype=float)
                    candidates.append(
                        Candidate(
                            vector=vector.copy(),
                            objective=objective,
                            details={
                                "feasible": 1.0,
                                "separationViolation": 0.0,
                                "collisionViolation": 0.0,
                                "makespan": float(objective[0]),
                                "energy": float(objective[1]),
                                "risk": float(objective[2]),
                                "turnPenalty": float(objective[3]),
                                "paths": [],
                            },
                        )
                    )
                return candidates

            def fake_update_archive(archive, new_cands, model, max_size, divisions, use_constraints=True):  # type: ignore[no-untyped-def]
                del archive, model, divisions, use_constraints
                return list(new_cands[:max_size])

            def fake_select_leaders(
                archive, divisions, use_advanced_archive=True, use_mean_selection=False, model=None, relaxation_eps=0.0
            ):  # type: ignore[no-untyped-def]
                del divisions, use_advanced_archive, use_mean_selection, model, relaxation_eps
                leader_count = min(3, len(archive))
                leaders = [archive[idx].vector.copy() for idx in range(leader_count)]
                indices = [idx for idx in range(leader_count)]
                while len(leaders) < 3:
                    leaders.append(leaders[0].copy())
                    indices.append(indices[0])
                return np.stack(leaders), np.asarray(indices, dtype=int), np.ones(3, dtype=float)

            def fake_explorer(**kwargs):  # type: ignore[no-untyped-def]
                lower = np.asarray(kwargs["lower"], dtype=float)
                offspring_count = int(kwargs["offspring_count"])
                return np.full((offspring_count, lower.size), 9.0, dtype=float)

            def fake_step(self, generation, max_generations):  # type: ignore[no-untyped-def]
                del max_generations
                step_inputs.append(self.positions.copy())
                self.positions = np.full_like(self.positions, float(generation))
                return self.positions

            def capture_save(**kwargs):  # type: ignore[no-untyped-def]
                del kwargs

            with patch.object(mogwo_module, "_evaluate_population", new=fake_evaluate):
                with patch.object(mogwo_module, "_update_archive", new=fake_update_archive):
                    with patch.object(mogwo_module, "_select_leaders", new=fake_select_leaders):
                        with patch.object(mogwo_module, "_adaptive_explorer_ratio", new=lambda *_args, **_kwargs: 0.5):
                            with patch.object(mogwo_module, "_adaptive_archive_explorer", new=fake_explorer):
                                with patch.object(mogwo_module.QGWO_Engine, "step", new=fake_step):
                                    with patch.object(mogwo_module, "_save_fleet_artifacts", new=capture_save):
                                        run_fleet_mogwo_no_attention(terrain, params)

            self.assertGreaterEqual(len(step_inputs), 2)
            expected = np.full_like(step_inputs[0], 1.0)
            expected[params.population // 2 :] = 9.0
            self.assertTrue(np.allclose(step_inputs[1], expected))


if __name__ == "__main__":
    unittest.main()
