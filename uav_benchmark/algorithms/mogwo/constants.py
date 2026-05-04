from __future__ import annotations

import numpy as np

_ATTN_TAU_OBJ = 0.20
_ATTN_BLEND_EPS = 0.03
_ATTN_ROW_DEGENERATE_EPS = 1e-6
_ATTN_EPS = 1e-12
_ATTN_FEAS_LAMBDA_MAX = 0.82
_ATTN_DIVERSITY_LAMBDA_MAX = 0.55
_ATTN_STEP_MIN = 0.18
_ATTN_STEP_MAX = 0.92
_ATTN_GUARD_PRESSURE = 0.85
_ATTN_STAGE_PRESSURE_CENTER = 0.38
_ATTN_STAGE_PRESSURE_HALF_WIDTH = 0.26
_ATTN_STAGE_DIVERSITY_MAX = 0.48
_ATTN_STAGE_MIN_ACTIVATION = 0.15
_DIVERSITY_EPS = 1e-9
_COMPONENT_SEED_FRACTION = 0.06
_RESEED_INTERVAL = 4
_RESEED_TRIGGER = 0.95
_REPAIR_RATE = 0.25
_EXPLORER_RATIO_MIN = 0.12
_EXPLORER_RATIO_MAX = 0.48
_CAUCHY_SCALE_BASE = 0.012
_CONVERGENCE_POWER = 1.65
_RELAX_SHARE_MIN = 0.06
_RELAX_SHARE_MAX = 0.48
_RELAX_PROGRESS_POWER = 1.55
_RELAX_INFUSION_MIN = 0.18
_RELAX_INFUSION_MAX = 0.58


def _fit_matrix(values: np.ndarray, rows: int, cols: int, fill: float) -> np.ndarray:
    out = np.full((rows, cols), float(fill), dtype=float)
    raw = np.asarray(values, dtype=float)
    if raw.size == 0:
        return out
    raw = raw.reshape(-1, cols) if raw.ndim != 2 else raw
    if raw.shape[1] != cols:
        return out
    use = min(rows, raw.shape[0])
    out[:use] = raw[:use]
    return out


def _nonlinear_convergence_factor(generation: int, max_generations: int) -> float:
    if max_generations <= 0:
        return 0.0
    progress = float(np.clip(float(generation) / float(max_generations), 0.0, 1.0))
    return float(max(0.0, 2.0 * (1.0 - progress**_CONVERGENCE_POWER)))
