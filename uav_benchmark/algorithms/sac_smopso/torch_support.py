from __future__ import annotations

from typing import Any

torch: Any
F: Any
nn: Any
Normal: Any
_TORCH_AVAILABLE: bool

try:
    import torch as _torch  # type: ignore[import-not-found]
    import torch.nn.functional as _F  # type: ignore[import-not-found]
    from torch import nn as _nn  # type: ignore[import-not-found]
    from torch.distributions import Normal as _Normal  # type: ignore[import-not-found]

    torch = _torch
    F = _F
    nn = _nn
    Normal = _Normal
    _TORCH_AVAILABLE = True
except (ImportError, OSError, RuntimeError):  # pragma: no cover - torch is optional at import time
    torch = None
    F = None
    nn = None
    Normal = None
    _TORCH_AVAILABLE = False

__all__ = ["F", "Normal", "_TORCH_AVAILABLE", "nn", "torch"]
