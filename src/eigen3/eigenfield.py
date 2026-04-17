"""Spatial eigenvalue interpolation using inverse distance weighting (IDW).

Provides the EigenvalueField class for interpolating eigenvalue spectra
at arbitrary geographic coordinates, computing field gradients (dialectal
change rates), and estimating uncertainty.
"""

from __future__ import annotations

import logging

import numpy as np

from eigen3.constants import DIALECT_COORDINATES

logger = logging.getLogger(__name__)


class EigenvalueField:
    """Continuous eigenvalue field over geographic space via IDW interpolation.

    Given discrete eigenvalue spectra at known dialect coordinates,
    produces a continuous field that can be queried at any (lat, lon).

    Parameters
    ----------
    spectra : dict[str, np.ndarray]
        Variety name -> 1-D array of eigenvalue magnitudes.
    coordinates : dict[str, tuple[float, float]]
        Variety name -> (latitude, longitude). Defaults to
        ``DIALECT_COORDINATES`` from ``eigen3.constants``.
    power : float
        IDW power parameter. Higher values give more local interpolation.
    """

    def __init__(
        self,
        spectra: dict[str, np.ndarray],
        coordinates: dict[str, tuple[float, float]] | None = None,
        power: float = 2.0,
    ) -> None:
        if coordinates is None:
            coordinates = DIALECT_COORDINATES

        # Only keep varieties present in both spectra and coordinates
        common = sorted(set(spectra.keys()) & set(coordinates.keys()))
        if not common:
            raise ValueError("No overlap between spectra keys and coordinate keys.")

        self._labels = common
        self._spectra = np.stack([spectra[v] for v in common], axis=0)  # (n, k)
        self._coords = np.array(
            [coordinates[v] for v in common], dtype=np.float64
        )  # (n, 2)
        self._power = power
        self._n = len(common)
        self._k = self._spectra.shape[1]

    @property
    def labels(self) -> list[str]:
        """Variety labels in internal order."""
        return list(self._labels)

    @property
    def spectrum_dim(self) -> int:
        """Dimensionality of the eigenvalue spectra."""
        return self._k

    # ------------------------------------------------------------------
    # IDW weights
    # ------------------------------------------------------------------

    def _idw_weights(self, lat: float, lon: float) -> np.ndarray:
        """Compute IDW weights for a query point.

        If the query point coincides with a known point (distance < 1e-12),
        all weight is assigned to that point.

        Returns
        -------
        np.ndarray
            (n,) weight vector summing to 1.
        """
        query = np.array([lat, lon], dtype=np.float64)
        dists = np.linalg.norm(self._coords - query, axis=1)  # (n,)

        # Check for exact (or near-exact) coincidence
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < 1e-12:
            w = np.zeros(self._n, dtype=np.float64)
            w[min_idx] = 1.0
            return w

        inv_d = 1.0 / np.power(dists, self._power)
        return inv_d / inv_d.sum()

    # ------------------------------------------------------------------
    # Field evaluation
    # ------------------------------------------------------------------

    def field_at(self, lat: float, lon: float) -> np.ndarray:
        """IDW-interpolated eigenvalue spectrum at any coordinate.

        Parameters
        ----------
        lat, lon : float
            Query latitude and longitude.

        Returns
        -------
        np.ndarray
            (k,) interpolated eigenvalue magnitude array.
        """
        w = self._idw_weights(lat, lon)
        return w @ self._spectra  # (k,)

    # ------------------------------------------------------------------
    # Field gradient (dialectal change rate)
    # ------------------------------------------------------------------

    def field_gradient(
        self,
        lat: float,
        lon: float,
        delta: float = 0.1,
    ) -> np.ndarray:
        """Numerical gradient of the eigenvalue field.

        Uses central differences in latitude and longitude.

        Parameters
        ----------
        lat, lon : float
            Query point.
        delta : float
            Step size in degrees for finite differences.

        Returns
        -------
        np.ndarray
            (2, k) array where row 0 is d(spectrum)/d(lat)
            and row 1 is d(spectrum)/d(lon).
        """
        f_lat_plus = self.field_at(lat + delta, lon)
        f_lat_minus = self.field_at(lat - delta, lon)
        f_lon_plus = self.field_at(lat, lon + delta)
        f_lon_minus = self.field_at(lat, lon - delta)

        grad_lat = (f_lat_plus - f_lat_minus) / (2.0 * delta)
        grad_lon = (f_lon_plus - f_lon_minus) / (2.0 * delta)

        return np.stack([grad_lat, grad_lon], axis=0)

    # ------------------------------------------------------------------
    # Uncertainty estimate
    # ------------------------------------------------------------------

    def field_uncertainty(self, lat: float, lon: float) -> float:
        """Distance-based uncertainty at a query point.

        High values indicate that the query point is far from all known
        dialect locations, so the interpolation is unreliable.

        Computed as the harmonic mean of distances to all known points,
        normalised by the median inter-point distance.

        Parameters
        ----------
        lat, lon : float
            Query point.

        Returns
        -------
        float
            Uncertainty score (dimensionless). 0 at known points;
            increases with distance from known locations.
        """
        query = np.array([lat, lon], dtype=np.float64)
        dists = np.linalg.norm(self._coords - query, axis=1)

        # Minimum distance to any known point
        d_min = float(dists.min())

        # Normalisation: median pairwise distance among known points
        if self._n < 2:
            return d_min

        pairwise: list[float] = []
        for i in range(self._n):
            for j in range(i + 1, self._n):
                pairwise.append(float(
                    np.linalg.norm(self._coords[i] - self._coords[j])
                ))
        d_median = float(np.median(pairwise))

        if d_median < 1e-12:
            return d_min

        return d_min / d_median

    # ------------------------------------------------------------------
    # Grid evaluation (convenience)
    # ------------------------------------------------------------------

    def evaluate_grid(
        self,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        resolution: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the eigenvalue field on a regular lat/lon grid.

        Parameters
        ----------
        lat_range : tuple[float, float]
            (min_lat, max_lat).
        lon_range : tuple[float, float]
            (min_lon, max_lon).
        resolution : int
            Number of grid points per axis.

        Returns
        -------
        lats : np.ndarray
            (resolution,) latitude values.
        lons : np.ndarray
            (resolution,) longitude values.
        field : np.ndarray
            (resolution, resolution, k) interpolated spectra.
        """
        lats = np.linspace(lat_range[0], lat_range[1], resolution)
        lons = np.linspace(lon_range[0], lon_range[1], resolution)
        field = np.zeros((resolution, resolution, self._k), dtype=np.float64)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                field[i, j] = self.field_at(lat, lon)

        return lats, lons, field
