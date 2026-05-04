from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.shared.nmopso_helpers import (
    _candidate_feasible_flags,
    _candidate_matrix,
    _finite_mean,
    _fixed_hv_reference,
)
from uav_benchmark.algorithms.shared.pso_types import Candidate
from uav_benchmark.core.metrics import cal_metric
from uav_benchmark.core.r2_archive import r2_indicator


class NMOPSOFeatureMixin:
    """Feature, attention, and state helpers for NMOPSOEngine."""

    model: dict[str, Any]
    feature_mode: str
    objective_count: int
    pop_size: int
    dimensions: int
    lower: np.ndarray
    upper: np.ndarray
    population: np.ndarray
    velocity: np.ndarray
    velocity_limit_base: np.ndarray
    pbest: np.ndarray
    candidates: list[Candidate]
    archive: list[Candidate]
    current_obj: np.ndarray
    use_r2_archive: bool
    r2_weights: np.ndarray
    r2_z_ideal: np.ndarray
    hv_ref_point: np.ndarray | None
    metric_rng: np.random.Generator

    def _candidate_centroid(self, candidate: Candidate) -> np.ndarray:
        """Estimate a 3D spatial centroid from candidate telemetry."""
        if self.feature_mode != "path":
            vec = np.asarray(candidate.vector, dtype=float).reshape(-1)
            if vec.size >= 3:
                usable = vec[: (vec.size // 3) * 3]
                if usable.size >= 3:
                    reshaped = usable.reshape(-1, 3)
                    centroid = np.mean(reshaped, axis=0)
                    if np.all(np.isfinite(centroid)):
                        return centroid
            return np.zeros(3, dtype=float)

        details = candidate.details if isinstance(candidate.details, dict) else {}
        paths = details.get("paths", [])
        points: list[np.ndarray] = []
        for path in paths:
            arr = np.asarray(path, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            xyz = arr[:, :3]
            finite_mask = np.all(np.isfinite(xyz), axis=1)
            if np.any(finite_mask):
                points.append(xyz[finite_mask])
        if points:
            merged = np.vstack(points)
            return np.mean(merged, axis=0)

        vec = np.asarray(candidate.vector, dtype=float).reshape(-1)
        if vec.size >= 3:
            usable = vec[: (vec.size // 3) * 3]
            if usable.size >= 3:
                reshaped = usable.reshape(-1, 3)
                centroid = np.mean(reshaped, axis=0)
                if np.all(np.isfinite(centroid)):
                    return centroid
        return np.zeros(3, dtype=float)

    def _centroids_from_vectors(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        n_rows = matrix.shape[0]
        c0 = np.mean(matrix[:, 0::3], axis=1) if matrix.shape[1] > 0 else np.zeros(n_rows, dtype=float)
        c1 = np.mean(matrix[:, 1::3], axis=1) if matrix.shape[1] > 1 else np.zeros(n_rows, dtype=float)
        c2 = np.mean(matrix[:, 2::3], axis=1) if matrix.shape[1] > 2 else np.zeros(n_rows, dtype=float)
        out = np.stack([c0, c1, c2], axis=1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_objectives(self, obj: np.ndarray) -> np.ndarray:
        if obj.size == 0:
            return np.zeros((0, self.objective_count), dtype=float)
        finite = np.where(np.isfinite(obj), obj, np.nan)
        col_min = np.nanmin(finite, axis=0)
        col_max = np.nanmax(finite, axis=0)
        col_min = np.where(np.isfinite(col_min), col_min, 0.0)
        col_max = np.where(np.isfinite(col_max), col_max, col_min + 1.0)
        span = np.maximum(col_max - col_min, 1e-9)
        safe_obj = np.where(np.isfinite(obj), obj, col_max + span)
        norm = (safe_obj - col_min) / span
        return np.clip(norm, 0.0, 1.0)

    def _normalize_centroids(self, centroids: np.ndarray) -> np.ndarray:
        if centroids.size == 0:
            return np.zeros((0, 3), dtype=float)
        lower = np.array(
            [
                float(self.model.get("xmin", np.nan)),
                float(self.model.get("ymin", np.nan)),
                float(self.model.get("zmin", np.nan)),
            ],
            dtype=float,
        )
        upper = np.array(
            [
                float(self.model.get("xmax", np.nan)),
                float(self.model.get("ymax", np.nan)),
                float(self.model.get("zmax", np.nan)),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            finite = np.where(np.isfinite(centroids), centroids, np.nan)
            lower = np.nanmin(finite, axis=0)
            upper = np.nanmax(finite, axis=0)
            lower = np.where(np.isfinite(lower), lower, 0.0)
            upper = np.where(np.isfinite(upper), upper, lower + 1.0)
        span = np.maximum(upper - lower, 1e-9)
        safe = np.where(np.isfinite(centroids), centroids, lower)
        norm = (safe - lower) / span
        return np.clip(norm, 0.0, 1.0)

    def get_particle_features(self) -> np.ndarray:
        """Return per-particle features with shape (N_pop, M+4)."""
        if self.pop_size <= 0:
            return np.zeros((0, self.objective_count + 4), dtype=float)

        obj = np.asarray(self.current_obj, dtype=float)
        if obj.shape != (self.pop_size, self.objective_count):
            obj = _candidate_matrix(self.candidates)
            if obj.shape != (self.pop_size, self.objective_count):
                pad = np.zeros((self.pop_size, self.objective_count), dtype=float)
                rows = min(self.pop_size, obj.shape[0])
                cols = min(self.objective_count, obj.shape[1]) if obj.ndim == 2 else 0
                if rows > 0 and cols > 0:
                    pad[:rows, :cols] = obj[:rows, :cols]
                obj = pad

        norm_obj = self._normalize_objectives(obj)
        if self.feature_mode == "path":
            centroids = np.zeros((self.pop_size, 3), dtype=float)
            for idx in range(min(self.pop_size, len(self.candidates))):
                centroids[idx] = self._candidate_centroid(self.candidates[idx])
        else:
            if self.population.size and self.population.shape[0] == self.pop_size:
                centroids = self._centroids_from_vectors(self.population)
            else:
                candidate_vectors = np.zeros((self.pop_size, self.dimensions), dtype=float)
                for idx in range(min(self.pop_size, len(self.candidates))):
                    candidate_vectors[idx] = np.asarray(self.candidates[idx].vector, dtype=float)
                centroids = self._centroids_from_vectors(candidate_vectors)
        norm_centroids = self._normalize_centroids(centroids)

        vel_mag = (
            np.linalg.norm(np.asarray(self.velocity, dtype=float), axis=1)
            if self.velocity.size
            else np.zeros(self.pop_size, dtype=float)
        )
        denom = float(np.linalg.norm(self.velocity_limit_base))
        if not np.isfinite(denom) or denom <= 1e-9:
            denom = max(float(np.nanmax(vel_mag)) if vel_mag.size else 1.0, 1e-9)
        vel_norm = np.clip(vel_mag / denom, 0.0, 1.0).reshape(-1, 1)

        feat = np.concatenate([norm_obj, norm_centroids, vel_norm], axis=1)
        return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)

    def get_archive_features(self) -> np.ndarray:
        """Return per-archive features with shape (N_arch, M+4)."""
        n_arch = len(self.archive)
        if n_arch == 0:
            return np.zeros((0, self.objective_count + 4), dtype=float)

        obj = _candidate_matrix(self.archive)
        if obj.shape[1] != self.objective_count:
            pad = np.zeros((n_arch, self.objective_count), dtype=float)
            cols = min(self.objective_count, obj.shape[1]) if obj.ndim == 2 else 0
            if cols > 0:
                pad[:, :cols] = obj[:, :cols]
            obj = pad
        norm_obj = self._normalize_objectives(obj)

        if self.feature_mode == "path":
            centroids = np.zeros((n_arch, 3), dtype=float)
            for idx, candidate in enumerate(self.archive):
                centroids[idx] = self._candidate_centroid(candidate)
        else:
            archive_vectors = np.stack([np.asarray(c.vector, dtype=float) for c in self.archive], axis=0)
            centroids = self._centroids_from_vectors(archive_vectors)
        norm_centroids = self._normalize_centroids(centroids)
        surrogate_speed = np.zeros((n_arch, 1), dtype=float)
        feat = np.concatenate([norm_obj, norm_centroids, surrogate_speed], axis=1)
        return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)

    def attention_leader_select(self, attention_weights: np.ndarray | None) -> np.ndarray | None:
        """Compute leader vectors from attention weights.

        Returns ``None`` when weights are invalid so caller can fall back to
        the original sampling path.
        """
        if attention_weights is None:
            return None
        if not self.archive:
            return self.pbest.copy()

        weights = np.asarray(attention_weights, dtype=float)
        n_arch = len(self.archive)
        if weights.shape != (self.pop_size, n_arch):
            return None

        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = np.clip(weights, 0.0, None)
        row_sum = np.sum(weights, axis=1, keepdims=True)
        invalid = row_sum[:, 0] <= 1e-12
        if np.any(invalid):
            weights[invalid] = 1.0 / max(1, n_arch)
            row_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / np.maximum(row_sum, 1e-12)

        archive_vectors = np.stack([c.vector for c in self.archive], axis=0)
        leaders = weights @ archive_vectors
        if leaders.shape != (self.pop_size, self.dimensions):
            return None
        return np.clip(leaders, self.lower, self.upper)

    def r2_before(self) -> float:
        """Compute current R2 indicator for FRRMAB credit."""
        if not self.use_r2_archive or not self.archive:
            return 0.0
        arch_obj = _candidate_matrix(self.archive)
        feas = arch_obj[np.all(np.isfinite(arch_obj), axis=1)]
        if feas.size == 0:
            return 0.0
        return r2_indicator(feas, self.r2_weights, self.r2_z_ideal)

    def state_features(
        self,
        generation: int,
        total_generations: int,
        last_hv: float,
        stagnation: int,
        diversity_ref: float,
    ) -> np.ndarray:
        """Build the 6-dim optimizer state vector."""
        finite_archive = self._finite_archive_matrix()
        if self.hv_ref_point is None and finite_archive.size > 0:
            self.hv_ref_point = _fixed_hv_reference(finite_archive)
        hv_now = (
            cal_metric(1, finite_archive, 0, self.objective_count, ref_point=self.hv_ref_point, rng=self.metric_rng)
            if finite_archive.size > 0
            else 0.0
        )
        diversity = float(np.mean(np.std(finite_archive, axis=0))) if finite_archive.size > 0 else 0.0
        feasible_ratio = float(np.mean(_candidate_feasible_flags(self.candidates, self.current_obj)))

        # 'conflictRate' tracks UAV-to-UAV collisions
        conflict_rate = _finite_mean(
            [float(getattr(c, "details", {}).get("conflictRate", 0.0)) for c in self.candidates],
            default=0.0,
        )
        hv_slope = hv_now - last_hv
        return np.array(
            [
                generation / max(1, total_generations),
                np.clip(feasible_ratio, 0.0, 1.0),
                np.clip(max(0.0, conflict_rate) / 0.02, 0.0, 1.0),
                0.5 * (np.tanh(hv_slope / 0.01) + 1.0),
                np.clip(np.log1p(max(0.0, diversity)) / np.log1p(3.0 * diversity_ref), 0.0, 1.0),
                min(1.0, stagnation / max(1, total_generations)),
            ],
            dtype=float,
        )

    def _finite_archive_matrix(self) -> np.ndarray:
        matrix = _candidate_matrix(self.archive)
        if matrix.size == 0:
            return matrix
        return matrix[np.all(np.isfinite(matrix), axis=1)]

    @property
    def archive_candidates(self) -> list[Candidate]:
        return self.archive
