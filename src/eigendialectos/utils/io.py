"""I/O helpers: serialisation, directory management, and disk-based caching."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create parent directories for *path* if they do not exist.

    Parameters
    ----------
    path:
        File or directory path whose parent tree should exist.

    Returns
    -------
    Path
        The resolved :class:`~pathlib.Path` object.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------------

def save_pickle(obj: Any, path: str | Path) -> Path:
    """Serialise *obj* to a pickle file.

    Parameters
    ----------
    obj:
        Any picklable Python object.
    path:
        Destination file path (parent dirs created automatically).

    Returns
    -------
    Path
        The resolved path that was written.
    """
    p = ensure_dir(path)
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_pickle(path: str | Path) -> Any:
    """Deserialise a pickle file.

    Parameters
    ----------
    path:
        Path to the ``.pkl`` / ``.pickle`` file.

    Returns
    -------
    Any
        The deserialised object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


# ---------------------------------------------------------------------------
# Numpy
# ---------------------------------------------------------------------------

def save_numpy(arr: np.ndarray, path: str | Path) -> Path:
    """Save a NumPy array to disk in ``.npy`` format.

    Parameters
    ----------
    arr:
        The array to persist.
    path:
        Destination file path.

    Returns
    -------
    Path
        The resolved path that was written.
    """
    p = ensure_dir(path)
    np.save(p, arr)
    return p


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from a ``.npy`` file.

    Parameters
    ----------
    path:
        Path to the ``.npy`` file.

    Returns
    -------
    np.ndarray
        The loaded array.
    """
    return np.load(path)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write *data* as a JSON file (UTF-8, human-readable).

    Parameters
    ----------
    data:
        JSON-serialisable object.
    path:
        Destination file path.
    indent:
        Number of spaces for pretty-printing (default ``2``).

    Returns
    -------
    Path
        The resolved path that was written.
    """
    p = ensure_dir(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return p


def load_json(path: str | Path) -> Any:
    """Load a JSON file.

    Parameters
    ----------
    path:
        Path to the ``.json`` file.

    Returns
    -------
    Any
        The parsed JSON structure.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Disk-based caching
# ---------------------------------------------------------------------------

def cache_result(
    key: str,
    compute_fn: Callable[[], T],
    cache_dir: str | Path = ".cache",
) -> T:
    """Return a cached result or compute, cache, and return it.

    The cache file is derived from *key* via a SHA-256 hash so that
    arbitrary strings (including file paths) can be used safely.

    Parameters
    ----------
    key:
        A unique identifier for the cached computation.
    compute_fn:
        A zero-argument callable that produces the value when the cache
        misses.
    cache_dir:
        Directory where cache files are stored.

    Returns
    -------
    T
        The (possibly cached) result of *compute_fn*.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    hashed = hashlib.sha256(key.encode("utf-8")).hexdigest()
    file_path = cache_path / f"{hashed}.pkl"

    if file_path.exists():
        return load_pickle(file_path)

    result = compute_fn()
    save_pickle(result, file_path)
    return result
