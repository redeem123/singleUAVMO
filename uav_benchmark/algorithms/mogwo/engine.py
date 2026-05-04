from __future__ import annotations

import numpy as np

from uav_benchmark.algorithms.mogwo.constants import (
    _ATTN_EPS,
    _ATTN_STEP_MAX,
    _ATTN_STEP_MIN,
    _ATTN_TAU_OBJ,
    _fit_matrix,
    _nonlinear_convergence_factor,
)


class QGWO_Engine:
    """Grey Wolf Optimizer core with topology-assisted relay guidance."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        pop_size: int,
        use_attention: bool = True,
        use_feasibility_pressure: bool = True,
        use_diversity_feedback: bool = True,
        use_step_limiter: bool = True,
        use_attention_guard: bool = True,
        # Sensitivity Parameters
        attn_tau_obj: float = _ATTN_TAU_OBJ,
        attn_step_min: float = _ATTN_STEP_MIN,
        attn_step_max: float = _ATTN_STEP_MAX,
        # Surgical Flags
        use_attn_feas_boost: bool = True,
        use_attn_div_boost: bool = True,
        use_step_feas_driver: bool = True,
        use_step_div_driver: bool = True,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.dim = lower.size
        self.pop_size = pop_size
        self.use_attention = bool(use_attention)
        self.use_feasibility_pressure = bool(use_feasibility_pressure)
        self.use_diversity_feedback = bool(use_diversity_feedback)
        self.use_step_limiter = bool(use_step_limiter)
        self.use_attention_guard = bool(use_attention_guard)

        self.attn_tau_obj = float(attn_tau_obj)
        self.attn_step_min = float(attn_step_min)
        self.attn_step_max = float(attn_step_max)

        self.use_attn_feas_boost = bool(use_attn_feas_boost)
        self.use_attn_div_boost = bool(use_attn_div_boost)
        self.use_step_feas_driver = bool(use_step_feas_driver)
        self.use_step_div_driver = bool(use_step_div_driver)

        self.positions = np.random.uniform(lower, upper, size=(pop_size, self.dim))
        self.leaders = np.zeros((3, self.dim))
        self._wolf_objectives = np.zeros((self.pop_size, 4), dtype=float)
        self._leader_objectives = np.zeros((3, 4), dtype=float)
        self._feasibility_pressure = 0.0
        self._diversity_level = 0.5
        self._leader_occupancy = np.ones(3, dtype=float)
        self._relay_guides = np.zeros((self.pop_size, self.dim), dtype=float)
        self._relay_activation = 0.0
        self._relay_pool_feasible_share = 0.0
        self.last_attention_stats: dict[str, float] = {
            "entropy_mean": float(np.log(3.0)),
            "lambda_feasibility": 0.0,
            "lambda_diversity": 0.0,
            "diversity_level": 0.5,
            "tau_effective": 0.0,
            "step_scale": 1.0,
            "attention_guard_active": 0.0,
            "stage_activation": 0.0,
            "relay_pool_feasible_share": 0.0,
        }

    def set_attention_context(
        self,
        *,
        wolf_objectives: np.ndarray,
        feasibility_pressure: float,
        leader_objectives: np.ndarray,
        diversity_level: float | None = None,
        leader_occupancy: np.ndarray | None = None,
        relay_guides: np.ndarray | None = None,
        relay_activation: float | None = None,
        relay_pool_feasible_share: float | None = None,
        wolf_risk: np.ndarray | None = None,
        leader_risk: np.ndarray | None = None,
    ) -> None:
        if wolf_risk is not None or leader_risk is not None:
            import warnings

            warnings.warn(
                "set_attention_context: wolf_risk and leader_risk are not used by the "
                "attention mechanism and are ignored. Pass None or omit them.",
                stacklevel=2,
            )
        self._wolf_objectives = np.clip(_fit_matrix(wolf_objectives, self.pop_size, 4, fill=1.0), 0.0, 1.0)
        self._leader_objectives = np.clip(_fit_matrix(leader_objectives, 3, 4, fill=1.0), 0.0, 1.0)
        self._feasibility_pressure = (
            float(np.clip(feasibility_pressure, 0.0, 1.0)) if self.use_feasibility_pressure else 0.0
        )
        if diversity_level is None:
            self._diversity_level = 0.5
        else:
            self._diversity_level = float(np.clip(diversity_level, 0.0, 1.0))
        if leader_occupancy is None:
            self._leader_occupancy = np.ones(3, dtype=float)
        else:
            occ_raw = np.asarray(leader_occupancy, dtype=float).reshape(-1)
            occ = np.ones(3, dtype=float)
            use = min(3, occ_raw.size)
            if use > 0:
                occ[:use] = occ_raw[:use]
            occ[~np.isfinite(occ)] = 1.0
            self._leader_occupancy = np.clip(occ, 1.0, np.inf)
        if relay_guides is None:
            self._relay_guides = np.broadcast_to(self.leaders[2].reshape(1, -1), (self.pop_size, self.dim)).copy()
        else:
            self._relay_guides = np.clip(
                _fit_matrix(relay_guides, self.pop_size, self.dim, fill=0.0),
                self.lower[None, :],
                self.upper[None, :],
            )
        self._relay_activation = float(np.clip(relay_activation if relay_activation is not None else 0.0, 0.0, 1.0))
        self._relay_pool_feasible_share = float(
            np.clip(relay_pool_feasible_share if relay_pool_feasible_share is not None else 0.0, 0.0, 1.0)
        )

    @staticmethod
    def _normalize_channel_rows(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            return np.zeros((0, 0), dtype=float)
        mean = np.mean(scores, axis=1, keepdims=True)
        std = np.std(scores, axis=1, keepdims=True)
        centered = scores - mean
        good = std > _ATTN_EPS
        out = np.zeros_like(scores)
        if np.any(good):
            out = np.divide(centered, np.where(good, std, 1.0))
        return out

    @staticmethod
    def _stage_activation(p: float, d: float) -> float:
        low_div = 1.0 - float(np.clip(d, 0.0, 1.0))
        if p < 0.45:
            return float(np.clip(0.35 + 0.65 * (0.45 - p) / 0.45, 0.35, 1.0))
        return float(np.clip(0.20 + 0.70 * low_div, 0.20, 0.85))

    def _attention_weights(self) -> np.ndarray:
        d = float(np.clip(self._diversity_level, 0.0, 1.0))
        stage_activation = self._relay_activation if self.use_attention else 0.0
        weights = np.full((self.pop_size, 3), 1.0 / 3.0, dtype=float)
        self.last_attention_stats = {
            "entropy_mean": float(np.log(3.0)),
            "lambda_feasibility": 0.0,
            "lambda_diversity": 0.0,
            "diversity_level": float(d),
            "tau_effective": 0.0,
            "step_scale": 1.0,
            "attention_guard_active": 0.0,
            "lambda_safe": 0.0,
            "stage_activation": float(stage_activation),
            "relay_pool_feasible_share": float(self._relay_pool_feasible_share),
        }
        return weights

    def _step_scale(self) -> float:
        return 1.0

    # -- One generation step --------------------------------------------
    def step(self, generation: int, max_generations: int) -> np.ndarray:
        """Vectorised MOGWO update with optional topology-assisted relay guidance."""
        if self.use_attention:
            a = _nonlinear_convergence_factor(generation, max_generations)
        else:
            a = 2.0 - generation * (2.0 / max_generations)

        # Standard GWO estimate from 3 leaders
        X_terms = np.zeros((3, self.pop_size, self.dim), dtype=float)
        relay_target = np.broadcast_to(self.leaders[2].reshape(1, -1), (self.pop_size, self.dim)).copy()
        if self.use_attention and self._relay_guides.shape == self.positions.shape:
            relay_target = (1.0 - self._relay_activation) * relay_target + self._relay_activation * self._relay_guides

        for j in range(3):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            target = (
                relay_target if j == 2 else np.broadcast_to(self.leaders[j].reshape(1, -1), (self.pop_size, self.dim))
            )
            D = np.abs(C * target - self.positions)
            X_terms[j] = target - A * D
        X_GWO = np.mean(X_terms, axis=0)

        if self.use_attention:
            _ = self._attention_weights()
            new_positions = X_GWO
        else:
            self.last_attention_stats = {
                "entropy_mean": float(np.log(3.0)),
                "lambda_feasibility": 0.0,
                "lambda_diversity": 0.0,
                "diversity_level": float(np.clip(self._diversity_level, 0.0, 1.0)),
                "tau_effective": 0.0,
                "lambda_safe": 0.0,
                "step_scale": 1.0,
                "attention_guard_active": 0.0,
                "stage_activation": 0.0,
                "relay_pool_feasible_share": 0.0,
            }
            new_positions = X_GWO

        step_scale = self._step_scale()
        self.last_attention_stats["step_scale"] = float(step_scale)

        # Sanitize and clip
        finite_mask = np.isfinite(new_positions)
        if not np.all(finite_mask):
            center = 0.5 * (self.lower + self.upper)
            new_positions = np.where(finite_mask, new_positions, center)

        self.positions = np.clip(new_positions, self.lower, self.upper)
        return self.positions
