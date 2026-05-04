from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any


def install_numba_stub() -> None:
    if "numba" in sys.modules:
        return

    module = ModuleType("numba")

    def njit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        del kwargs
        if args and callable(args[0]) and len(args) == 1:
            return args[0]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    module.njit = njit  # type: ignore[attr-defined]
    sys.modules["numba"] = module
