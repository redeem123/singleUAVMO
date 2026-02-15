from __future__ import annotations

import sys
from pathlib import Path


def _numpy_is_broken() -> bool:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return True
    if hasattr(np, "__version__"):
        return False
    spec = getattr(np, "__spec__", None)
    origin = getattr(spec, "origin", None) if spec is not None else None
    return origin == "namespace" or getattr(np, "__file__", None) is None


def _latest_site_packages(formula: str, relative_path: str) -> str | None:
    cellar = Path("/opt/homebrew/Cellar")
    formula_dir = cellar / formula
    if not formula_dir.exists():
        return None
    versions = sorted((entry for entry in formula_dir.iterdir() if entry.is_dir()), reverse=True)
    for version_dir in versions:
        candidate = version_dir / relative_path
        if candidate.exists():
            return str(candidate)
    return None


def bootstrap_homebrew_science_stack() -> None:
    if not _numpy_is_broken():
        return
    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        _latest_site_packages("numpy", f"lib/{python_tag}/site-packages"),
        _latest_site_packages("scipy", f"lib/{python_tag}/site-packages"),
        _latest_site_packages("python-matplotlib", f"libexec/lib/{python_tag}/site-packages"),
    ]
    for candidate in reversed([item for item in candidates if item]):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    if "numpy" in sys.modules:
        del sys.modules["numpy"]
    if "scipy" in sys.modules:
        del sys.modules["scipy"]

