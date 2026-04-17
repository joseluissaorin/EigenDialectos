"""EigenDialectos facade -- thin entry point delegating to specialised modules."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from eigen3.decomposition import eigendecompose, eigenspectrum
from eigen3.scorer import DialectScorer
from eigen3.analyzer import analyze_text
from eigen3.algebra import compose_dialects, analogy_dialects, interpolate_spectrum
from eigen3.compiler import compile as sdc_compile
from eigen3.eigenfield import EigenvalueField
from eigen3.types import (
    AlphaVector, AnalysisResult, ComposeResult,
    EigenDecomp, EigenSpectrum, ScoreResult, TransformResult,
)

logger = logging.getLogger(__name__)


class EigenDialectos:
    """Main facade for the eigen3 system.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray]
        Per-variety embedding matrices {variety: (vocab_size, dim)}.
    vocab : list[str]
        Shared vocabulary aligned with embedding rows.
    W_dict : dict[str, np.ndarray]
        Per-variety transformation matrices {variety: (dim, dim)}.
    reference : str
        Reference variety (default "ES_PEN").
    """

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        vocab: list[str],
        W_dict: dict[str, np.ndarray],
        reference: str = "ES_PEN",
    ) -> None:
        self._embeddings = embeddings
        self._vocab = vocab
        self._W_dict = W_dict
        self._reference = reference

        self._decomps: dict[str, EigenDecomp] = {
            v: eigendecompose(W, variety=v) for v, W in W_dict.items()
        }
        self._spectra: dict[str, EigenSpectrum] = {
            v: eigenspectrum(d.eigenvalues) for v, d in self._decomps.items()
        }
        self._scorer = DialectScorer(
            embeddings=embeddings, vocab=vocab,
            decomps=self._decomps, reference=reference,
        )
        logger.info(
            "EigenDialectos ready: %d varieties, vocab=%d",
            len(self._decomps), len(vocab),
        )

    @property
    def varieties(self) -> list[str]:
        """Sorted list of available dialect variety codes."""
        return sorted(self._decomps.keys())

    @property
    def n_modes(self) -> int:
        """Number of eigenmodes (from the first decomposition)."""
        return next(iter(self._decomps.values())).n_modes

    def score(self, text: str, temperature: float = 1.0) -> ScoreResult:
        """Score *text* against all dialects."""
        return self._scorer.score(text, temperature=temperature)

    def classify(self, text: str) -> str:
        """Return the most-probable dialect code for *text*."""
        return self._scorer.classify(text)

    def transform(
        self, text: str, source: str, target: str,
        alpha: Optional[AlphaVector] = None,
    ) -> TransformResult:
        """Transform *text* from *source* dialect to *target*."""
        decomp = self._decomps[target]
        return sdc_compile(
            text=text, source=source, target=target,
            embeddings=self._embeddings, vocab=self._vocab,
            decomp=decomp, alpha=alpha,
        )

    def analyze(self, text: str) -> AnalysisResult:
        """Per-mode linguistic analysis of *text*."""
        return analyze_text(
            text=text, embeddings=self._embeddings,
            vocab=self._vocab, decomps=self._decomps,
        )

    def compose(self, weights: dict[str, float]) -> ComposeResult:
        """Compose a synthetic dialect from weighted spectra."""
        ref = self._decomps[self._reference]
        return compose_dialects(self._decomps, weights, ref)

    def analogy(self, a: str, b: str, c: str) -> ComposeResult:
        """Dialect analogy: *a* is to *b* as *c* is to ?"""
        ref = self._decomps[self._reference]
        return analogy_dialects(self._decomps, a, b, c, ref)

    def interpolate(self, variety_a: str, variety_b: str, t: float) -> np.ndarray:
        """Interpolate eigenvalue magnitudes between two varieties."""
        mag_a = self._spectra[variety_a].magnitudes
        mag_b = self._spectra[variety_b].magnitudes
        return interpolate_spectrum(mag_a, mag_b, t)

    def eigenfield(self, lat: float, lon: float) -> np.ndarray:
        """Query the continuous eigenvalue field at *(lat, lon)*."""
        spectra = {v: s.magnitudes for v, s in self._spectra.items()}
        return EigenvalueField(spectra).field_at(lat, lon)
