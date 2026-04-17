"""Continuous eigenvalue field λ_k(x,y) over geographic space.

Uses Gaussian Process regression to interpolate eigenvalues from
discrete dialect points to continuous geography, enabling:
- Generation for any point on the map
- Isogloss detection as contour lines
- Dialect continua modeling
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from eigendialectos.types import EigenFieldResult

logger = logging.getLogger(__name__)


class EigenvalueField:
    """Continuous eigenvalue field over geographic coordinates.

    Fits one Gaussian Process per eigenvalue dimension, interpolating
    from discrete dialect measurement points to a continuous surface.
    """

    def __init__(
        self,
        kernel_lengthscale: float = 10.0,
        kernel_variance: float = 1.0,
        noise_variance: float = 0.01,
    ) -> None:
        """
        Parameters
        ----------
        kernel_lengthscale : float
            RBF kernel lengthscale (degrees). Controls smoothness.
        kernel_variance : float
            RBF kernel signal variance.
        noise_variance : float
            Observation noise variance.
        """
        self.kernel_lengthscale = kernel_lengthscale
        self.kernel_variance = kernel_variance
        self.noise_variance = noise_variance

        self._coordinates: npt.NDArray[np.float64] | None = None
        self._eigenvalues: npt.NDArray[np.float64] | None = None
        self._K_inv: list[npt.NDArray[np.float64]] = []
        self._alpha: list[npt.NDArray[np.float64]] = []
        self._n_eigenvalues: int = 0
        self._fitted_lengthscales: npt.NDArray[np.float64] = np.array([])

    def _rbf_kernel(
        self,
        X1: npt.NDArray[np.float64],
        X2: npt.NDArray[np.float64],
        lengthscale: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute RBF (squared exponential) kernel matrix.

        k(x, x') = σ² · exp(-||x - x'||² / (2·ℓ²))
        """
        ls = lengthscale or self.kernel_lengthscale
        # Pairwise squared distances
        sq_dists = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1, keepdims=True).T
            - 2 * X1 @ X2.T
        )
        return self.kernel_variance * np.exp(-sq_dists / (2 * ls ** 2))

    def fit(
        self,
        coordinates: npt.NDArray[np.float64],
        eigenvalues: npt.NDArray[np.float64],
    ) -> EigenvalueField:
        """Fit Gaussian Process models for eigenvalue interpolation.

        Parameters
        ----------
        coordinates : ndarray, shape (n_dialects, 2)
            Geographic coordinates (lat, lon) for each dialect.
        eigenvalues : ndarray, shape (n_dialects, n_eigenvalues)
            Eigenvalue arrays. Uses absolute values of eigenvalues.

        Returns
        -------
        self
        """
        self._coordinates = coordinates.astype(np.float64)
        self._eigenvalues = np.abs(eigenvalues.astype(np.float64))
        n_dialects, self._n_eigenvalues = self._eigenvalues.shape

        self._K_inv = []
        self._alpha = []
        self._fitted_lengthscales = np.full(self._n_eigenvalues, self.kernel_lengthscale)

        for k in range(self._n_eigenvalues):
            y_k = self._eigenvalues[:, k]

            # Compute kernel matrix
            K = self._rbf_kernel(self._coordinates, self._coordinates)
            K_noise = K + self.noise_variance * np.eye(n_dialects)

            # Cholesky solve for efficiency
            try:
                L = np.linalg.cholesky(K_noise)
                alpha_k = np.linalg.solve(L.T, np.linalg.solve(L, y_k))
                K_inv_k = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_dialects)))
            except np.linalg.LinAlgError:
                # Fall back to direct inverse
                K_inv_k = np.linalg.pinv(K_noise)
                alpha_k = K_inv_k @ y_k

            self._K_inv.append(K_inv_k)
            self._alpha.append(alpha_k)

        logger.info(
            "Fitted eigenvalue field: %d dialects, %d eigenvalue dimensions",
            n_dialects,
            self._n_eigenvalues,
        )

        return self

    def predict(
        self,
        grid_coords: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Predict eigenvalues at new geographic points.

        Parameters
        ----------
        grid_coords : ndarray, shape (n_points, 2)
            Geographic coordinates to predict at.

        Returns
        -------
        predictions : ndarray, shape (n_points, n_eigenvalues)
        uncertainties : ndarray, shape (n_points, n_eigenvalues)
        """
        if self._coordinates is None:
            raise RuntimeError("Call fit() first")

        n_points = grid_coords.shape[0]
        predictions = np.zeros((n_points, self._n_eigenvalues))
        uncertainties = np.zeros((n_points, self._n_eigenvalues))

        for k in range(self._n_eigenvalues):
            # Cross-kernel: k(x_new, X_train)
            K_star = self._rbf_kernel(grid_coords, self._coordinates)

            # Predictive mean: K_star @ α
            predictions[:, k] = K_star @ self._alpha[k]

            # Predictive variance: k(x_new, x_new) - K_star @ K_inv @ K_star^T
            K_ss = self.kernel_variance * np.ones(n_points)  # diagonal of self-kernel
            v = K_star @ self._K_inv[k]
            var = K_ss - np.sum(v * K_star, axis=1)
            uncertainties[:, k] = np.maximum(var, 0.0)

        return predictions, uncertainties

    def compute_field(
        self,
        resolution: int = 50,
        padding: float = 5.0,
    ) -> EigenFieldResult:
        """Compute eigenvalue field over a geographic grid.

        Parameters
        ----------
        resolution : int
            Grid resolution in each dimension.
        padding : float
            Degrees of padding around the data extent.

        Returns
        -------
        EigenFieldResult
        """
        if self._coordinates is None:
            raise RuntimeError("Call fit() first")

        lat_min, lon_min = self._coordinates.min(axis=0) - padding
        lat_max, lon_max = self._coordinates.max(axis=0) + padding

        grid_lat = np.linspace(lat_min, lat_max, resolution)
        grid_lon = np.linspace(lon_min, lon_max, resolution)
        lat_mesh, lon_mesh = np.meshgrid(grid_lat, grid_lon, indexing="ij")
        grid_coords = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])

        predictions, uncerts = self.predict(grid_coords)

        # Reshape to (n_eigenvalues, resolution, resolution)
        eigenvalue_surfaces = np.zeros((self._n_eigenvalues, resolution, resolution))
        uncertainty_surfaces = np.zeros((self._n_eigenvalues, resolution, resolution))

        for k in range(self._n_eigenvalues):
            eigenvalue_surfaces[k] = predictions[:, k].reshape(resolution, resolution)
            uncertainty_surfaces[k] = uncerts[:, k].reshape(resolution, resolution)

        return EigenFieldResult(
            coordinates=self._coordinates,
            eigenvalue_surfaces=eigenvalue_surfaces,
            gp_lengthscales=self._fitted_lengthscales,
            uncertainties=uncertainty_surfaces,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
        )

    def find_isoglosses(
        self,
        eigenvalue_index: int,
        threshold: float = 0.5,
        resolution: int = 50,
    ) -> list[npt.NDArray[np.float64]]:
        """Find isoglosses as contour lines where eigenvalue gradient is steep.

        Parameters
        ----------
        eigenvalue_index : int
            Which eigenvalue to compute isoglosses for.
        threshold : float
            Gradient magnitude threshold (as fraction of max gradient).
        resolution : int
            Grid resolution.

        Returns
        -------
        list of ndarray
            Each array is an (n, 2) set of contour coordinates.
        """
        field = self.compute_field(resolution=resolution)
        surface = field.eigenvalue_surfaces[eigenvalue_index]

        # Compute gradient magnitude via finite differences
        grad_lat = np.gradient(surface, axis=0)
        grad_lon = np.gradient(surface, axis=1)
        grad_magnitude = np.sqrt(grad_lat ** 2 + grad_lon ** 2)

        # Threshold
        max_grad = grad_magnitude.max()
        if max_grad < 1e-15:
            return []

        # Find contour at threshold level
        mask = grad_magnitude > threshold * max_grad
        contour_points = np.argwhere(mask)

        if len(contour_points) == 0:
            return []

        # Convert grid indices back to geographic coordinates
        lat_coords = field.grid_lat[contour_points[:, 0]]
        lon_coords = field.grid_lon[contour_points[:, 1]]
        geo_contour = np.column_stack([lat_coords, lon_coords])

        return [geo_contour]
