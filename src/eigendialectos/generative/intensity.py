"""Dialectal intensity control and threshold discovery.

Provides tools to sweep the alpha parameter, visualise its effect on
embeddings, and perform binary-search discovery of perceptual thresholds.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import ALPHA_RANGE
from eigendialectos.generative.dial import dial_transform_embedding
from eigendialectos.types import EigenDecomposition


class DialectClassifier(Protocol):
    """Protocol for a dialect classifier used in threshold discovery."""

    def predict(self, embedding: npt.NDArray[np.float64]) -> bool:
        """Return True if the embedding is recognised as the target dialect."""
        ...  # pragma: no cover


class IntensityController:
    """Controls dialectal intensity and discovers perceptual thresholds.

    Parameters
    ----------
    tolerance : float
        Convergence tolerance for binary-search threshold discovery.
    max_iterations : int
        Maximum number of binary-search iterations.
    """

    def __init__(
        self,
        tolerance: float = 0.01,
        max_iterations: int = 50,
    ) -> None:
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def generate_at_intensity(
        self,
        embedding: npt.NDArray[np.floating],
        eigen: EigenDecomposition,
        alpha: float,
    ) -> npt.NDArray[np.float64]:
        """Apply the DIAL transform at a specific intensity.

        Parameters
        ----------
        embedding : ndarray
            Input embedding vector or batch.
        eigen : EigenDecomposition
            Eigendecomposition of the target dialect transform.
        alpha : float
            Dialectal intensity.

        Returns
        -------
        ndarray of float64
            Transformed embedding.
        """
        return dial_transform_embedding(embedding, eigen, alpha)

    def sweep_intensities(
        self,
        embedding: npt.NDArray[np.floating],
        eigen: EigenDecomposition,
        start: float = 0.0,
        stop: float = 1.5,
        step: float = 0.1,
    ) -> list[tuple[float, npt.NDArray[np.float64]]]:
        """Generate embeddings across a range of alpha values.

        Parameters
        ----------
        embedding : ndarray
            Input embedding vector or batch.
        eigen : EigenDecomposition
            Eigendecomposition of the target dialect transform.
        start, stop, step : float
            Alpha range specification.

        Returns
        -------
        list of (alpha, transformed_embedding)
            Pairs of intensity value and resulting embedding.
        """
        alphas = np.arange(start, stop, step)
        results: list[tuple[float, npt.NDArray[np.float64]]] = []
        for a in alphas:
            transformed = self.generate_at_intensity(embedding, eigen, float(a))
            results.append((float(a), transformed))
        return results

    def find_recognition_threshold(
        self,
        embedding: npt.NDArray[np.floating],
        eigen: EigenDecomposition,
        classifier: Any,
        low: float = 0.0,
        high: float = 2.0,
    ) -> float:
        """Find the alpha where *classifier* first recognises the dialect.

        Uses binary search over [low, high].  The classifier must expose a
        ``predict(embedding) -> bool`` interface that returns ``True`` when
        the dialect is recognised.

        Parameters
        ----------
        embedding : ndarray
            Input embedding.
        eigen : EigenDecomposition
            Eigendecomposition of the target dialect transform.
        classifier : object
            Dialect classifier with a ``predict`` method.
        low, high : float
            Search bounds for alpha.

        Returns
        -------
        float
            The smallest alpha (within *self.tolerance*) at which the
            classifier recognises the dialect.  Returns *high* if the
            dialect is never recognised.
        """
        for _ in range(self.max_iterations):
            if high - low < self.tolerance:
                break

            mid = (low + high) / 2.0
            transformed = self.generate_at_intensity(embedding, eigen, mid)
            if classifier.predict(transformed):
                high = mid
            else:
                low = mid

        return high

    def find_naturalness_threshold(
        self,
        embedding: npt.NDArray[np.floating],
        eigen: EigenDecomposition,
        quality_fn: Callable[[npt.NDArray[np.float64]], float],
        quality_floor: float = 0.5,
        low: float = 0.0,
        high: float = 2.0,
    ) -> float:
        """Find the alpha where quality drops below *quality_floor*.

        Uses binary search to locate the intensity at which *quality_fn*
        first returns a value below *quality_floor*.

        Parameters
        ----------
        embedding : ndarray
            Input embedding.
        eigen : EigenDecomposition
            Eigendecomposition of the target dialect transform.
        quality_fn : callable
            A function that takes a transformed embedding and returns a
            quality score in [0, 1], where higher is better.
        quality_floor : float
            Minimum acceptable quality score.
        low, high : float
            Search bounds for alpha.

        Returns
        -------
        float
            The largest alpha (within *self.tolerance*) for which quality
            remains above *quality_floor*.  Returns *high* if quality never
            drops below the floor.
        """
        # Verify that quality at low is acceptable
        transformed_low = self.generate_at_intensity(embedding, eigen, low)
        if quality_fn(transformed_low) < quality_floor:
            return low

        for _ in range(self.max_iterations):
            if high - low < self.tolerance:
                break

            mid = (low + high) / 2.0
            transformed = self.generate_at_intensity(embedding, eigen, mid)
            quality = quality_fn(transformed)

            if quality >= quality_floor:
                low = mid
            else:
                high = mid

        return low
