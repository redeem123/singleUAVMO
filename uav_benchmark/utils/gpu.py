from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GPUInfo:
    enabled: bool
    backend: str
    device: str
    reason: str


def resolve_gpu(mode: str) -> GPUInfo:
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "off", "force"}:
        normalized = "auto"
    if normalized == "off":
        return GPUInfo(enabled=False, backend="numpy", device="cpu", reason="disabled by user")

    # Prefer CuPy for CUDA, then PyTorch CUDA/MPS.
    try:
        import cupy  # type: ignore

        device_count = int(cupy.cuda.runtime.getDeviceCount())
        if device_count > 0:
            return GPUInfo(enabled=True, backend="cupy", device="cuda:0", reason="cupy detected")
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return GPUInfo(enabled=True, backend="torch", device="cuda:0", reason="torch cuda available")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return GPUInfo(enabled=True, backend="torch", device="mps", reason="torch mps available")
    except Exception:
        pass

    if normalized == "force":
        return GPUInfo(enabled=False, backend="numpy", device="cpu", reason="force requested but no GPU backend found")
    return GPUInfo(enabled=False, backend="numpy", device="cpu", reason="auto fallback to CPU")

