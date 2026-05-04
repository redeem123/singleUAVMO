from __future__ import annotations

import numpy as np

from uav_benchmark.algorithms.cgpo import _AblationControls, _controls_from_extra, _project_fleet
from uav_benchmark.algorithms.cgpo.cig import build_constraint_interaction_graph
from uav_benchmark.algorithms.cgpo.ppf import compute_pareto_pressure_field


def _flat_model() -> dict[str, object]:
    return {
        "H": np.zeros((10, 10), dtype=float),
        "xmin": 1.0,
        "xmax": 10.0,
        "ymin": 1.0,
        "ymax": 10.0,
        "zmin": 1.0,
        "zmax": 10.0,
        "safeDist": 2.0,
        "separationMin": 2.0,
        "droneSize": 1.0,
        "maxTurnDeg": 120.0,
        "hardCollisionConstraint": True,
        "nofly_c": np.asarray([[5.0, 5.0]], dtype=float),
        "nofly_r": np.asarray([1.0], dtype=float),
    }


# ---------------------------------------------------------------------------
# CIG: typed edge wiring is independent of PPF after the redesign.
# ---------------------------------------------------------------------------


def test_cig_marks_no_fly_obstacle_edges() -> None:
    model = _flat_model()
    path = np.asarray([[1.0, 1.0, 5.0], [5.0, 5.0, 5.0], [9.0, 9.0, 5.0]], dtype=float)

    graph = build_constraint_interaction_graph([path], model)

    assert graph.obstacle_edges > 0
    assert "obstacle" in graph.active_constraint_types
    assert np.linalg.norm(graph.tension[0, 1, :2]) > 0.0
    assert graph.node_features.shape[1] == 8
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_weight.size == graph.edge_type.size == graph.edge_index.shape[1]
    assert graph.edge_index.shape[1] > 0


def test_cig_edge_family_switches_are_independent() -> None:
    model = _flat_model()
    path_a = np.asarray([[1.0, 1.0, 5.0], [5.0, 5.0, 5.0], [9.0, 9.0, 5.0]], dtype=float)
    path_b = np.asarray([[1.0, 2.0, 5.0], [5.0, 5.5, 5.0], [9.0, 8.0, 5.0]], dtype=float)

    pairwise_only = build_constraint_interaction_graph(
        [path_a, path_b],
        model,
        use_terrain_edges=False,
        use_obstacle_edges=False,
        use_turn_edges=False,
        use_smoothing_edges=False,
        use_pairwise_edges=True,
    )
    smoothing_only = build_constraint_interaction_graph(
        [path_a, path_b],
        model,
        use_terrain_edges=False,
        use_obstacle_edges=False,
        use_turn_edges=False,
        use_smoothing_edges=True,
        use_pairwise_edges=False,
    )

    assert pairwise_only.pairwise_edges > 0
    assert pairwise_only.objective_edges == 0
    assert pairwise_only.obstacle_edges == 0
    assert smoothing_only.objective_edges > 0
    assert smoothing_only.pairwise_edges == 0
    assert smoothing_only.obstacle_edges == 0


def test_cig_smoothing_weight_no_longer_depends_on_objective_weights() -> None:
    """After the redesign CIG must be independent of PPF's objective spread."""
    model = _flat_model()
    path = np.asarray([[1.0, 1.0, 5.0], [3.0, 4.0, 5.0], [6.0, 6.0, 5.0], [9.0, 9.0, 5.0]], dtype=float)

    g_zero = build_constraint_interaction_graph([path], model, objective_weights=np.zeros(4))
    g_uniform = build_constraint_interaction_graph([path], model, objective_weights=np.ones(4))
    g_skewed = build_constraint_interaction_graph([path], model, objective_weights=np.array([5.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(g_zero.scalar_tension, g_uniform.scalar_tension)
    np.testing.assert_allclose(g_zero.scalar_tension, g_skewed.scalar_tension)
    assert g_zero.objective_edges == g_uniform.objective_edges == g_skewed.objective_edges


# ---------------------------------------------------------------------------
# PPF: validation contract.
# ---------------------------------------------------------------------------


def test_ppf_rejects_mismatched_candidate_shapes() -> None:
    model = _flat_model()
    path = np.asarray([[1.0, 1.0, 5.0], [4.0, 1.0, 5.0], [9.0, 1.0, 5.0]], dtype=float)
    graph = build_constraint_interaction_graph([path], model)

    with np.testing.assert_raises(ValueError):
        compute_pareto_pressure_field(
            objective=np.zeros((2, 4), dtype=float),
            violations=np.zeros(1, dtype=float),
            feasible=np.ones(2, dtype=bool),
            graphs=[graph, graph],
        )

    with np.testing.assert_raises(ValueError):
        compute_pareto_pressure_field(
            objective=np.zeros((2, 4), dtype=float),
            violations=np.zeros(2, dtype=float),
            feasible=np.ones(2, dtype=bool),
            graphs=[graph],
        )


# ---------------------------------------------------------------------------
# Ablation control plumbing.
# ---------------------------------------------------------------------------


def test_default_controls_are_lean_three_mechanism_cgpo() -> None:
    """Defaults enable CIG/PPF/OVO only."""
    controls = _controls_from_extra({})

    assert isinstance(controls, _AblationControls)
    # Published mechanisms are ON by default.
    assert controls.cig_edge_coupling_enabled
    assert controls.ppf_pressure_enabled
    assert controls.ovo_variation_enabled
    assert controls.ovo_coordination_enabled


def test_lean_cgpo_uses_pure_domain_projection_for_offspring() -> None:
    """The lean runner only clips to bounds; no implicit repair on offspring."""
    model = _flat_model()
    out_of_bounds = [
        np.asarray(
            [
                [-10.0, -10.0, -5.0],
                [5.0, 5.0, 200.0],
                [12.0, 12.0, -5.0],
            ],
            dtype=float,
        )
    ]

    projected = _project_fleet(out_of_bounds, model)

    assert len(projected) == 1
    arr = projected[0]
    assert arr.shape == out_of_bounds[0].shape
    assert np.all(arr[:, 0] >= model["xmin"])
    assert np.all(arr[:, 0] <= model["xmax"])
    assert np.all(arr[:, 1] >= model["ymin"])
    assert np.all(arr[:, 1] <= model["ymax"])
    ground = np.zeros(arr.shape[0], dtype=float)
    assert np.all(arr[:, 2] >= ground + model["zmin"])
    assert np.all(arr[:, 2] <= ground + model["zmax"])
