from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Candidate:
    vector: np.ndarray
    objective: np.ndarray
    details: dict[str, Any]


__all__ = ["Candidate"]
