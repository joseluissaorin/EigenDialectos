"""Global model registry for embedding backends.

Usage::

    from eigendialectos.embeddings.registry import register_model, get_model

    register_model("fasttext", FastTextModel)
    model = get_model("fasttext")
"""

from __future__ import annotations

from typing import Type

from eigendialectos.embeddings.base import EmbeddingModel

_REGISTRY: dict[str, Type[EmbeddingModel]] = {}


def register_model(name: str, cls: Type[EmbeddingModel]) -> None:
    """Register an embedding model class under *name*.

    Parameters
    ----------
    name:
        Case-insensitive lookup key (stored lower-cased).
    cls:
        A concrete subclass of :class:`EmbeddingModel`.

    Raises
    ------
    TypeError
        If *cls* is not a subclass of ``EmbeddingModel``.
    ValueError
        If *name* is already registered.
    """
    if not (isinstance(cls, type) and issubclass(cls, EmbeddingModel)):
        raise TypeError(
            f"Expected a subclass of EmbeddingModel, got {cls!r}"
        )
    key = name.lower().strip()
    if key in _REGISTRY:
        raise ValueError(
            f"Model '{key}' is already registered "
            f"(current: {_REGISTRY[key].__name__})"
        )
    _REGISTRY[key] = cls


def get_model(name: str, **kwargs) -> EmbeddingModel:
    """Instantiate a registered model by *name*.

    Parameters
    ----------
    name:
        Case-insensitive name previously passed to :func:`register_model`.
    **kwargs:
        Forwarded to the model constructor.

    Returns
    -------
    EmbeddingModel
        A freshly constructed model instance.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown model '{key}'. Available: {available}"
        )
    return _REGISTRY[key](**kwargs)


def list_available() -> list[str]:
    """Return a sorted list of registered model names."""
    return sorted(_REGISTRY)


def clear_registry() -> None:
    """Remove all entries.  Mainly useful in tests."""
    _REGISTRY.clear()
