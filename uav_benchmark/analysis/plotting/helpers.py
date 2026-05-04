from __future__ import annotations


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting") from exc
    return plt
