"""Utility helpers for EigenDialectos: I/O, logging, reproducibility, GPU, checkpointing."""

from __future__ import annotations

from eigendialectos.utils.checkpointing import (
    cleanup_old_checkpoints,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from eigendialectos.utils.gpu import device_info, get_device, to_device
from eigendialectos.utils.io import (
    cache_result,
    ensure_dir,
    load_json,
    load_numpy,
    load_pickle,
    save_json,
    save_numpy,
    save_pickle,
)
from eigendialectos.utils.logging import get_logger, log_metric, log_phase
from eigendialectos.utils.reproducibility import (
    get_deterministic_config,
    reproducible_context,
    set_global_seed,
)

__all__ = [
    # io
    "save_pickle",
    "load_pickle",
    "save_numpy",
    "load_numpy",
    "save_json",
    "load_json",
    "ensure_dir",
    "cache_result",
    # logging
    "get_logger",
    "log_phase",
    "log_metric",
    # reproducibility
    "set_global_seed",
    "get_deterministic_config",
    "reproducible_context",
    # gpu
    "get_device",
    "to_device",
    "device_info",
    # checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
    "cleanup_old_checkpoints",
]
