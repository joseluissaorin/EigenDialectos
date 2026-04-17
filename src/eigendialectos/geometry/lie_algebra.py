"""Lie group/algebra analysis of dialectal transformations.

Models W_i = exp(A_i) where A_i is the dialect generator in the Lie algebra.
Enables:
- Correct interpolation: W_mix = exp(β·A_i + (1-β)·A_j)
- Commutator analysis: [A_i, A_j] measures how much two dialects interfere
- Baker-Campbell-Hausdorff expansion for composition
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm, logm

from eigendialectos.types import LieAlgebraResult

logger = logging.getLogger(__name__)


class LieAlgebraAnalysis:
    """Lie group/algebra analysis of dialectal transformations.

    Each dialect's transformation matrix W_i is an element of GL(n,R).
    Its Lie algebra generator A_i = log(W_i) lives in gl(n,R).
    """

    def compute_generators(
        self,
        W_matrices: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.complex128]]:
        """Compute Lie algebra generators A_i = logm(W_i) for each dialect.

        Parameters
        ----------
        W_matrices : dict
            Mapping from dialect name to square transformation matrix.

        Returns
        -------
        dict mapping dialect name to generator matrix A_i (may be complex
        if W has negative eigenvalues).
        """
        generators: dict[str, npt.NDArray[np.complex128]] = {}
        for name, W in W_matrices.items():
            W_c = W.astype(np.complex128)
            A = logm(W_c)

            # Check if imaginary part is negligible
            if np.allclose(A.imag, 0.0, atol=1e-10):
                A = A.real.astype(np.complex128)

            # Verify: expm(A) ≈ W
            W_reconstructed = expm(A)
            error = float(np.linalg.norm(W_reconstructed - W_c, "fro"))
            if error > 1e-6:
                logger.warning(
                    "logm/expm roundtrip error for %s: %.2e", name, error
                )

            generators[name] = A
            logger.debug(
                "Generator %s: ||A||_F = %.4f, max|A| = %.4f",
                name,
                float(np.linalg.norm(A, "fro")),
                float(np.max(np.abs(A))),
            )

        return generators

    def compute_commutators(
        self,
        generators: dict[str, npt.NDArray[np.complex128]],
    ) -> LieAlgebraResult:
        """Compute Lie brackets [A_i, A_j] = A_i A_j - A_j A_i for all pairs.

        A non-zero commutator means that the order in which you apply
        dialectal transformations matters — linguistically, this indicates
        that the two varieties' features interact non-trivially.

        Parameters
        ----------
        generators : dict
            Output of compute_generators().

        Returns
        -------
        LieAlgebraResult
        """
        names = sorted(generators.keys())
        commutators: dict[tuple[str, str], npt.NDArray[np.complex128]] = {}
        commutator_norms: dict[tuple[str, str], float] = {}

        for i, name_i in enumerate(names):
            A_i = generators[name_i]
            for j, name_j in enumerate(names):
                if j <= i:
                    continue  # commutator is antisymmetric: [A_i,A_j] = -[A_j,A_i]
                A_j = generators[name_j]

                bracket = A_i @ A_j - A_j @ A_i
                norm = float(np.linalg.norm(bracket, "fro"))

                commutators[(name_i, name_j)] = bracket
                commutator_norms[(name_i, name_j)] = norm

        return LieAlgebraResult(
            generators=generators,
            commutators=commutators,
            commutator_norms=commutator_norms,
        )

    def interpolate(
        self,
        A_source: npt.NDArray[np.complex128],
        A_target: npt.NDArray[np.complex128],
        beta: float,
    ) -> npt.NDArray[np.float64]:
        """Correct Lie algebra interpolation: W_mix = exp(β·A_s + (1-β)·A_t).

        This is superior to linear matrix interpolation because it
        stays on the matrix manifold (result is always invertible).

        Parameters
        ----------
        A_source : ndarray
            Lie algebra generator for source dialect.
        A_target : ndarray
            Lie algebra generator for target dialect.
        beta : float
            Mixing weight, 0 → target, 1 → source.

        Returns
        -------
        ndarray
            Interpolated transformation matrix (real).
        """
        A_mix = beta * A_source + (1 - beta) * A_target
        W_mix = expm(A_mix)

        # Discard negligible imaginary parts
        if np.allclose(W_mix.imag, 0.0, atol=1e-10):
            return W_mix.real.astype(np.float64)
        return W_mix.real.astype(np.float64)

    def bracket_magnitude_matrix(
        self,
        generators: dict[str, npt.NDArray[np.complex128]],
    ) -> tuple[npt.NDArray[np.float64], list[str]]:
        """Build matrix of ||[A_i, A_j]||_F for all dialect pairs.

        This matrix reveals which dialect pairs have the most
        non-commutative (order-dependent) interactions.

        Returns
        -------
        matrix : ndarray, shape (n_dialects, n_dialects)
        labels : list of dialect names
        """
        names = sorted(generators.keys())
        n = len(names)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i, name_i in enumerate(names):
            A_i = generators[name_i]
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                A_j = generators[name_j]
                bracket = A_i @ A_j - A_j @ A_i
                matrix[i, j] = float(np.linalg.norm(bracket, "fro"))

        return matrix, names

    def full_analysis(
        self,
        W_matrices: dict[str, npt.NDArray[np.float64]],
    ) -> LieAlgebraResult:
        """Run complete Lie algebra analysis.

        Convenience method that chains compute_generators + compute_commutators.
        """
        generators = self.compute_generators(W_matrices)
        return self.compute_commutators(generators)
