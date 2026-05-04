from __future__ import annotations

import numpy as np

from uav_benchmark.algorithms.sac_smopso.constants import _ACTION_KEYS, _ACTION_LOWER, _ACTION_UPPER


def _map_continuous_action(raw: np.ndarray) -> dict[str, float]:
    """Map the controller's continuous action to named PSO hyper-parameters."""
    scaled = 0.5 * (np.clip(np.asarray(raw, dtype=float).reshape(-1), -1.0, 1.0) + 1.0)
    values = _ACTION_LOWER + scaled * (_ACTION_UPPER - _ACTION_LOWER)
    return {key: float(values[index]) for index, key in enumerate(_ACTION_KEYS)}
