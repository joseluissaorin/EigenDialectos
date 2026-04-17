"""Front-end of the SDC compiler: text → multi-level representation."""

from __future__ import annotations

from eigendialectos.corpus.parsing.multi_level import MultiLevelParser
from eigendialectos.types import ParsedText


class CompilerParser:
    """Front-end parser for the Spectral Dialectal Compiler.

    Thin wrapper around MultiLevelParser that provides the compiler
    interface expected by SpectralTransformer.
    """

    def __init__(self) -> None:
        self._parser = MultiLevelParser()

    def parse(self, text: str) -> ParsedText:
        """Parse text into all 5 linguistic levels."""
        return self._parser.parse(text)
