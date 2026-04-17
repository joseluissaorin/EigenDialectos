"""Global registry of corpus data sources."""

from __future__ import annotations

from typing import Type

from eigendialectos.corpus.base import CorpusSource

_REGISTRY: dict[str, Type[CorpusSource]] = {}


def register_source(name: str, cls: Type[CorpusSource]) -> None:
    """Register a :class:`CorpusSource` subclass under *name*.

    Raises
    ------
    ValueError
        If *name* is already registered.
    TypeError
        If *cls* is not a subclass of :class:`CorpusSource`.
    """
    if not (isinstance(cls, type) and issubclass(cls, CorpusSource)):
        raise TypeError(
            f"Expected a CorpusSource subclass, got {cls!r}"
        )
    if name in _REGISTRY:
        raise ValueError(
            f"Source '{name}' is already registered ({_REGISTRY[name]!r})"
        )
    _REGISTRY[name] = cls


def get_source(name: str) -> CorpusSource:
    """Instantiate and return the :class:`CorpusSource` registered as *name*.

    Raises
    ------
    KeyError
        If *name* has not been registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown source '{name}'. Available: {available}"
        )
    return _REGISTRY[name]()


def list_available() -> list[str]:
    """Return a sorted list of all registered source names."""
    return sorted(_REGISTRY)


def clear_registry() -> None:
    """Remove all registered sources (useful in tests)."""
    _REGISTRY.clear()
