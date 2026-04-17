"""Structured logging utilities for EigenDialectos."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(name)-28s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Keep a registry so each name is configured only once.
_configured_loggers: set[str] = set()


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted :class:`~logging.Logger`.

    All loggers produced by this function share the same format string
    and write to *stderr* so that regular program output on *stdout*
    stays clean.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__`` of the calling module).
    level:
        Logging level (default ``INFO``).

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    if name not in _configured_loggers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent messages from propagating to the root logger and being
        # duplicated when a parent logger also has handlers.
        logger.propagate = False
        _configured_loggers.add(name)

    return logger


# ---------------------------------------------------------------------------
# Phase context manager
# ---------------------------------------------------------------------------

@contextmanager
def log_phase(
    phase_name: str,
    *,
    logger: logging.Logger | None = None,
) -> Generator[None, None, None]:
    """Context manager that logs the start, end, and duration of a phase.

    Parameters
    ----------
    phase_name:
        Human-readable label for the phase (e.g. ``"embedding extraction"``).
    logger:
        Logger to use.  When *None* a module-level logger is used.

    Examples
    --------
    >>> with log_phase("PCA reduction"):
    ...     result = run_pca(data)
    """
    _logger = logger or get_logger("eigendialectos.phase")
    _logger.info("Phase [%s] started", phase_name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _logger.info(
            "Phase [%s] finished in %.3f s",
            phase_name,
            elapsed,
        )


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------

_metric_logger: logging.Logger | None = None


def log_metric(
    name: str,
    value: Any,
    step: int | None = None,
) -> None:
    """Log a single metric value in a structured, grep-friendly format.

    Output lines look like::

        METRIC | cosine_sim=0.9321 | step=12

    Parameters
    ----------
    name:
        Metric name (e.g. ``"loss"``, ``"cosine_sim"``).
    value:
        Metric value (usually numeric).
    step:
        Optional training / evaluation step counter.
    """
    global _metric_logger  # noqa: PLW0603
    if _metric_logger is None:
        _metric_logger = get_logger("eigendialectos.metrics")

    parts = [f"METRIC | {name}={value}"]
    if step is not None:
        parts.append(f"step={step}")
    _metric_logger.info(" | ".join(parts))
