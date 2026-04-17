"""Abstract base class for corpus data sources."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterator

from eigendialectos.constants import DialectCode
from eigendialectos.types import DialectSample


class CorpusSource(abc.ABC):
    """Abstract interface for a downloadable corpus data source.

    Subclasses must implement methods to download raw data, load it as
    :class:`DialectSample` instances, declare which dialect codes are
    covered, and provide a citation string.
    """

    @abc.abstractmethod
    def download(self, output_dir: Path) -> Path:
        """Download raw data to *output_dir* and return the artefact path.

        Parameters
        ----------
        output_dir:
            Local directory where data should be written.

        Returns
        -------
        Path
            Path to the downloaded artefact (file or directory).
        """

    @abc.abstractmethod
    def load(self, path: Path) -> Iterator[DialectSample]:
        """Yield :class:`DialectSample` instances from a previously
        downloaded artefact.

        Parameters
        ----------
        path:
            Local path returned by a prior call to :meth:`download`.
        """

    @abc.abstractmethod
    def dialect_codes(self) -> list[DialectCode]:
        """Return the dialect codes this source covers."""

    @abc.abstractmethod
    def citation(self) -> str:
        """Return a human-readable citation / attribution string."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def name(self) -> str:
        """Return the source's class name as a readable identifier."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        codes = ", ".join(c.value for c in self.dialect_codes())
        return f"<{self.name()} dialects=[{codes}]>"
