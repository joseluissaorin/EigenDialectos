"""Reproducibility utilities: global seeding and deterministic execution."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import numpy as np


# ---------------------------------------------------------------------------
# Global seed
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Set seeds for all relevant random number generators.

    This sets seeds for:
    * :mod:`random` (Python stdlib)
    * :mod:`numpy.random`
    * :mod:`torch` (manual seed + CUDA seed), **if** PyTorch is installed

    It also configures common environment variables that libraries such as
    PyTorch and various CUDA tools honour for reproducibility.

    Parameters
    ----------
    seed:
        The seed value (default ``42``).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Deterministic config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeterministicConfig:
    """Settings required for fully deterministic PyTorch execution."""

    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    use_deterministic_algorithms: bool = True
    warn_only: bool = True
    cublas_workspace_config: str = ":4096:8"
    extra_env: dict[str, str] = field(default_factory=dict)

    def apply(self) -> None:
        """Apply deterministic settings to the current process.

        Safe to call even when PyTorch is not installed -- the function
        will simply set the environment variables and return.
        """
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.cublas_workspace_config
        for k, v in self.extra_env.items():
            os.environ[k] = v

        try:
            import torch

            torch.backends.cudnn.deterministic = self.cudnn_deterministic
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
            torch.use_deterministic_algorithms(
                self.use_deterministic_algorithms,
                warn_only=self.warn_only,
            )
        except ImportError:
            pass


def get_deterministic_config() -> DeterministicConfig:
    """Return the recommended deterministic configuration.

    Call :meth:`DeterministicConfig.apply` on the returned object to
    activate the settings in the current process.

    Returns
    -------
    DeterministicConfig
        A frozen dataclass with the recommended values.
    """
    return DeterministicConfig()


# ---------------------------------------------------------------------------
# Reproducible context
# ---------------------------------------------------------------------------

@contextmanager
def reproducible_context(seed: int = 42) -> Generator[None, None, None]:
    """Context manager that seeds RNGs, applies deterministic config, and restores state on exit.

    Parameters
    ----------
    seed:
        Seed value used for the duration of the block.

    Examples
    --------
    >>> with reproducible_context(seed=123):
    ...     samples = np.random.randn(10)
    """
    # Capture current states.
    py_state = random.getstate()
    np_state = np.random.get_state()

    torch_state = None
    torch_cuda_state = None
    old_cudnn_deterministic = None
    old_cudnn_benchmark = None
    try:
        import torch

        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_state = torch.cuda.get_rng_state_all()
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        old_cudnn_benchmark = torch.backends.cudnn.benchmark
    except ImportError:
        torch = None  # type: ignore[assignment]

    # Apply seeds + deterministic config.
    set_global_seed(seed)
    get_deterministic_config().apply()

    try:
        yield
    finally:
        # Restore prior states.
        random.setstate(py_state)
        np.random.set_state(np_state)

        if torch is not None:
            torch.random.set_rng_state(torch_state)
            if torch_cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(torch_cuda_state)
            if old_cudnn_deterministic is not None:
                torch.backends.cudnn.deterministic = old_cudnn_deterministic
            if old_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = old_cudnn_benchmark
