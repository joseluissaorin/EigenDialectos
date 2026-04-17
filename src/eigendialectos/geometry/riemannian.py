"""Riemannian geometry of the dialect manifold.

Treats each dialect's eigenstructure as a point on a manifold where:
- Metric tensor g_jk(i) = Σ_ℓ |σ_ℓ^(i)| · v_ℓ^(i)_j · v_ℓ^(i)_k
- Geodesic distance uses the affine-invariant metric on SPD matrices
- Ricci curvature approximated via Ollivier's discrete curvature
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import numpy.typing as npt
from scipy.linalg import logm, sqrtm

from eigendialectos.types import EigenDecomposition, RiemannianResult

logger = logging.getLogger(__name__)


class RiemannianDialectSpace:
    """Riemannian geometry analysis of the space of dialect transformations."""

    def compute_metric_tensors(
        self,
        eigendecompositions: dict[str, EigenDecomposition],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Compute Riemannian metric tensor at each dialect point.

        g_jk(i) = Σ_ℓ |λ_ℓ| · (v_ℓ)_j · (v_ℓ)_k

        This is the eigenvalue-weighted outer product sum of eigenvectors,
        producing a symmetric positive semi-definite matrix that encodes
        how 'stretched' the dialect transformation is along each direction.

        Parameters
        ----------
        eigendecompositions : dict
            Mapping from dialect name to EigenDecomposition.

        Returns
        -------
        dict mapping dialect name to metric tensor (dim × dim, real, symmetric).
        """
        metrics: dict[str, npt.NDArray[np.float64]] = {}

        for name, eigen in eigendecompositions.items():
            eigenvalues = np.abs(eigen.eigenvalues)
            V = eigen.eigenvectors  # columns are eigenvectors

            dim = V.shape[0]
            g = np.zeros((dim, dim), dtype=np.float64)

            for ell in range(len(eigenvalues)):
                v = np.real(V[:, ell])  # take real part
                weight = float(eigenvalues[ell])
                g += weight * np.outer(v, v)

            # Ensure symmetry
            g = (g + g.T) / 2

            # Regularize to ensure positive definiteness
            min_eig = np.min(np.linalg.eigvalsh(g))
            if min_eig < 1e-10:
                g += (abs(min_eig) + 1e-8) * np.eye(dim)

            metrics[name] = g

        return metrics

    def geodesic_distance(
        self,
        g_i: npt.NDArray[np.float64],
        g_j: npt.NDArray[np.float64],
    ) -> float:
        """Compute geodesic distance between two dialect points.

        Uses the affine-invariant Riemannian metric on SPD manifold:
        d(g_i, g_j) = ||log(g_i^{-1/2} g_j g_i^{-1/2})||_F

        Parameters
        ----------
        g_i, g_j : ndarray
            Symmetric positive definite metric tensors.

        Returns
        -------
        float
            Geodesic distance.
        """
        # g_i^{-1/2}
        g_i_c = g_i.astype(np.complex128)
        sqrt_g_i = sqrtm(g_i_c)
        try:
            inv_sqrt_g_i = np.linalg.inv(sqrt_g_i)
        except np.linalg.LinAlgError:
            inv_sqrt_g_i = np.linalg.pinv(sqrt_g_i)

        # g_i^{-1/2} @ g_j @ g_i^{-1/2}
        inner = inv_sqrt_g_i @ g_j.astype(np.complex128) @ inv_sqrt_g_i

        # log of the result
        log_inner = logm(inner)

        # Frobenius norm
        distance = float(np.linalg.norm(log_inner, "fro").real)

        return distance

    def compute_geodesic_distance_matrix(
        self,
        metric_tensors: dict[str, npt.NDArray[np.float64]],
    ) -> tuple[npt.NDArray[np.float64], list[str]]:
        """Compute pairwise geodesic distances between all dialects.

        Returns
        -------
        distance_matrix : ndarray, shape (n, n)
        labels : list of dialect names
        """
        names = sorted(metric_tensors.keys())
        n = len(names)
        D = np.zeros((n, n), dtype=np.float64)

        for i, j in combinations(range(n), 2):
            d = self.geodesic_distance(
                metric_tensors[names[i]],
                metric_tensors[names[j]],
            )
            D[i, j] = d
            D[j, i] = d

        return D, names

    def ricci_curvature(
        self,
        metric_tensors: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, float]:
        """Approximate Ollivier-Ricci curvature at each dialect point.

        Uses discrete approximation: for each node i, compute the average
        geodesic distance to neighbors vs. the Wasserstein-1 distance
        between uniform distributions on neighborhoods.

        κ(i) = 1 - (1/|N(i)|) Σ_{j∈N(i)} W₁(μ_i, μ_j) / d(i,j)

        For simplicity, we use all other dialects as neighbors and
        approximate W₁ via the earth mover's distance on sorted eigenvalues.

        Returns
        -------
        dict mapping dialect name to scalar Ricci curvature estimate.
        """
        D, names = self.compute_geodesic_distance_matrix(metric_tensors)
        n = len(names)
        curvatures: dict[str, float] = {}

        for i, name in enumerate(names):
            # Use all other nodes as neighbors
            neighbors = [j for j in range(n) if j != i]
            if not neighbors:
                curvatures[name] = 0.0
                continue

            # For each neighbor j, compute curvature on edge (i,j)
            edge_curvatures = []
            for j in neighbors:
                d_ij = D[i, j]
                if d_ij < 1e-15:
                    continue

                # Wasserstein-1 between neighborhood distributions of i and j
                # Approximate: average distance from i's neighbors to j's neighbors
                # μ_i = uniform over neighbors of i, μ_j = uniform over neighbors of j
                dists_from_i_neighbors = []
                for k in neighbors:
                    # Distance from k (neighbor of i) to closest neighbor of j
                    j_neighbors = [m for m in range(n) if m != j]
                    min_dist = min(D[k, m] for m in j_neighbors) if j_neighbors else 0
                    dists_from_i_neighbors.append(min_dist)

                w1_approx = float(np.mean(dists_from_i_neighbors))
                kappa = 1.0 - w1_approx / d_ij
                edge_curvatures.append(kappa)

            curvatures[name] = float(np.mean(edge_curvatures)) if edge_curvatures else 0.0

        return curvatures

    def full_analysis(
        self,
        eigendecompositions: dict[str, EigenDecomposition],
    ) -> RiemannianResult:
        """Run complete Riemannian analysis.

        Parameters
        ----------
        eigendecompositions : dict
            Mapping from dialect name to EigenDecomposition.

        Returns
        -------
        RiemannianResult
        """
        metric_tensors = self.compute_metric_tensors(eigendecompositions)
        D, labels = self.compute_geodesic_distance_matrix(metric_tensors)
        curvatures = self.ricci_curvature(metric_tensors)

        logger.info(
            "Riemannian analysis: %d dialects, distance range [%.4f, %.4f], "
            "curvature range [%.4f, %.4f]",
            len(labels),
            float(D[D > 0].min()) if np.any(D > 0) else 0.0,
            float(D.max()),
            min(curvatures.values()) if curvatures else 0.0,
            max(curvatures.values()) if curvatures else 0.0,
        )

        return RiemannianResult(
            metric_tensors=metric_tensors,
            geodesic_distances=D,
            ricci_curvatures=curvatures,
            dialect_labels=labels,
        )
