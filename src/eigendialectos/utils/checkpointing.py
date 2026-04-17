"""Model and result checkpointing with metadata tracking."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_checkpoint(
    state_dict: dict[str, Any],
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a checkpoint (state dict + optional metadata) to disk.

    The checkpoint is saved as a PyTorch file via :func:`torch.save`.
    If *metadata* is provided it is embedded inside the saved object
    under the ``"_metadata"`` key **and** written as a companion
    ``.meta.json`` sidecar so it can be inspected without loading the
    full checkpoint.

    Parameters
    ----------
    state_dict:
        Typically ``model.state_dict()`` or any picklable mapping.
    path:
        Destination file path (``.pt`` extension recommended).
    metadata:
        Optional metadata dict (e.g. epoch, loss, config hash).

    Returns
    -------
    Path
        The resolved path that was written.
    """
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = dict(state_dict)

    if metadata is not None:
        stamped_meta = {
            **metadata,
            "saved_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
        payload["_metadata"] = stamped_meta

        # Write a human-readable sidecar.
        meta_path = p.with_suffix(p.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(stamped_meta, f, indent=2, ensure_ascii=False, default=str)
    else:
        payload["_metadata"] = {
            "saved_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }

    torch.save(payload, p)
    return p


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint previously saved with :func:`save_checkpoint`.

    Parameters
    ----------
    path:
        Path to the ``.pt`` checkpoint file.

    Returns
    -------
    dict[str, Any]
        The full payload dict, including the ``"_metadata"`` key if
        present.
    """
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def list_checkpoints(
    directory: str | Path,
    *,
    pattern: str = "*.pt",
) -> list[Path]:
    """List checkpoint files in *directory*, sorted newest-first.

    Parameters
    ----------
    directory:
        Folder to scan.
    pattern:
        Glob pattern used to match checkpoint files (default ``*.pt``).

    Returns
    -------
    list[Path]
        Checkpoint paths sorted by modification time (most recent first).
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    files = list(d.glob(pattern))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def cleanup_old_checkpoints(
    directory: str | Path,
    *,
    keep: int = 5,
    pattern: str = "*.pt",
) -> list[Path]:
    """Remove old checkpoints, keeping the *keep* most recent.

    Companion ``.meta.json`` sidecar files are also removed when their
    parent checkpoint is deleted.

    Parameters
    ----------
    directory:
        Folder containing checkpoints.
    keep:
        Number of newest checkpoints to retain (default ``5``).
    pattern:
        Glob pattern for checkpoint files.

    Returns
    -------
    list[Path]
        Paths of files that were deleted.
    """
    all_ckpts = list_checkpoints(directory, pattern=pattern)
    to_remove = all_ckpts[keep:]
    removed: list[Path] = []

    for ckpt in to_remove:
        # Remove sidecar first (if it exists).
        sidecar = ckpt.with_suffix(ckpt.suffix + ".meta.json")
        if sidecar.exists():
            sidecar.unlink()
            removed.append(sidecar)
        ckpt.unlink()
        removed.append(ckpt)

    return removed
