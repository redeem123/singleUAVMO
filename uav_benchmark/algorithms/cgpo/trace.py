from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class CGPOTrace:
    """Per-generation diagnostics for lean CGPO mechanisms.

    Only the three retained mechanisms (CIG / PPF / OVO) are active.  The
    projection fields remain zero-valued output columns for artifact schema
    compatibility.
    """

    # CIG diagnostics
    cig_mean_tension: list[float] = field(default_factory=list)
    cig_max_tension: list[float] = field(default_factory=list)
    cig_terrain_edges: list[float] = field(default_factory=list)
    cig_obstacle_edges: list[float] = field(default_factory=list)
    cig_turn_edges: list[float] = field(default_factory=list)
    cig_smoothing_edges: list[float] = field(default_factory=list)
    cig_pairwise_edges: list[float] = field(default_factory=list)
    # PPF diagnostics
    ppf_feasibility_pressure: list[float] = field(default_factory=list)
    ppf_boundary_mass: list[float] = field(default_factory=list)
    ppf_pressure_entropy: list[float] = field(default_factory=list)
    # OVO diagnostics
    ovo_perturbation_scale: list[float] = field(default_factory=list)
    ovo_coordinated_clusters: list[float] = field(default_factory=list)
    # Population diagnostics
    offspring_feasible_ratio: list[float] = field(default_factory=list)
    candidate_evaluations: list[float] = field(default_factory=list)
    # Zero-valued compatibility diagnostics.
    gfp_projection_norm: list[float] = field(default_factory=list)
    gfp_violation_delta: list[float] = field(default_factory=list)
    gfp_acceptance_rate: list[float] = field(default_factory=list)
    projection_proxy_evaluations: list[float] = field(default_factory=list)

    def as_trace(self) -> dict[str, np.ndarray]:
        return {
            "cig_mean_tension": np.asarray(self.cig_mean_tension, dtype=float),
            "cig_max_tension": np.asarray(self.cig_max_tension, dtype=float),
            "cig_terrain_edges": np.asarray(self.cig_terrain_edges, dtype=float),
            "cig_obstacle_edges": np.asarray(self.cig_obstacle_edges, dtype=float),
            "cig_turn_edges": np.asarray(self.cig_turn_edges, dtype=float),
            "cig_smoothing_edges": np.asarray(self.cig_smoothing_edges, dtype=float),
            "cig_pairwise_edges": np.asarray(self.cig_pairwise_edges, dtype=float),
            "ppf_feasibility_pressure": np.asarray(self.ppf_feasibility_pressure, dtype=float),
            "ppf_boundary_mass": np.asarray(self.ppf_boundary_mass, dtype=float),
            "ppf_pressure_entropy": np.asarray(self.ppf_pressure_entropy, dtype=float),
            "ovo_perturbation_scale": np.asarray(self.ovo_perturbation_scale, dtype=float),
            "ovo_coordinated_clusters": np.asarray(self.ovo_coordinated_clusters, dtype=float),
            "offspring_feasible_ratio": np.asarray(self.offspring_feasible_ratio, dtype=float),
            "candidate_evaluations": np.asarray(self.candidate_evaluations, dtype=float),
            "gfp_projection_norm": np.asarray(self.gfp_projection_norm, dtype=float),
            "gfp_violation_delta": np.asarray(self.gfp_violation_delta, dtype=float),
            "gfp_acceptance_rate": np.asarray(self.gfp_acceptance_rate, dtype=float),
            "projection_proxy_evaluations": np.asarray(self.projection_proxy_evaluations, dtype=float),
        }

    def as_rl_trace(self) -> dict[str, np.ndarray]:
        """Legacy artifact compatibility for the shared result writer."""
        return self.as_trace()
