"""Device management utilities for CPU / CUDA / MPS execution."""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(*, prefer: str | None = None) -> Any:
    """Auto-detect the best available device and return a :class:`torch.device`.

    Detection order (highest to lowest priority):
    1. CUDA (if available)
    2. MPS  (Apple Silicon, if available)
    3. CPU

    Parameters
    ----------
    prefer:
        Optional override.  When set to ``"cpu"``, ``"cuda"``, or
        ``"mps"`` the function returns that device directly (raising
        :class:`RuntimeError` if the requested backend is unavailable).

    Returns
    -------
    torch.device
        The selected device object.

    Raises
    ------
    RuntimeError
        If *prefer* names a backend that is not available.
    """
    import torch

    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if prefer == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but not available")
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data transfer
# ---------------------------------------------------------------------------

def to_device(
    data: Any,
    device: Any | None = None,
) -> Any:
    """Move a tensor (or NumPy array) to the target device.

    * If *data* is a :class:`torch.Tensor` it is moved via ``.to(device)``.
    * If *data* is a :class:`numpy.ndarray` it is first converted to a
      tensor and then moved.
    * Nested :class:`dict` and :class:`list`/:class:`tuple` structures are
      handled recursively.

    Parameters
    ----------
    data:
        Tensor, array, or nested structure to move.
    device:
        Target device.  When *None*, :func:`get_device` is called to
        auto-detect the best device.

    Returns
    -------
    Any
        The data residing on *device*.
    """
    import torch

    if device is None:
        device = get_device()

    if isinstance(data, torch.Tensor):
        # MPS does not support float64 -- down-cast when necessary.
        if device.type == "mps" and data.dtype == torch.float64:
            data = data.float()
        return data.to(device)
    if isinstance(data, np.ndarray):
        t = torch.from_numpy(data)
        if device.type == "mps" and t.dtype == torch.float64:
            t = t.float()
        return t.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        moved = [to_device(v, device) for v in data]
        return type(data)(moved)
    return data


# ---------------------------------------------------------------------------
# Device information
# ---------------------------------------------------------------------------

def device_info(device: Any | None = None) -> dict[str, Any]:
    """Return a summary dict describing the target device.

    The returned dictionary always contains:

    * ``"device_type"`` -- one of ``"cuda"``, ``"mps"``, ``"cpu"``.
    * ``"device_name"`` -- human-readable device name.

    For CUDA devices additional keys are present:

    * ``"cuda_device_name"`` -- GPU model string.
    * ``"memory_total_mb"`` -- total GPU memory in MiB.
    * ``"memory_allocated_mb"`` -- currently allocated memory.
    * ``"memory_reserved_mb"`` -- memory held by the caching allocator.

    Parameters
    ----------
    device:
        Device to query.  *None* auto-detects via :func:`get_device`.

    Returns
    -------
    dict[str, Any]
        A flat dictionary with device metadata.
    """
    import torch

    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    info: dict[str, Any] = {
        "device_type": device.type,
        "device_name": str(device),
    }

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        info["cuda_device_name"] = torch.cuda.get_device_name(idx)
        info["memory_total_mb"] = round(
            torch.cuda.get_device_properties(idx).total_mem / (1024**2), 1
        )
        info["memory_allocated_mb"] = round(
            torch.cuda.memory_allocated(idx) / (1024**2), 1
        )
        info["memory_reserved_mb"] = round(
            torch.cuda.memory_reserved(idx) / (1024**2), 1
        )
    elif device.type == "mps":
        # MPS exposes limited introspection; include what is available.
        if hasattr(torch.mps, "current_allocated_memory"):
            info["memory_allocated_bytes"] = torch.mps.current_allocated_memory()

    return info
