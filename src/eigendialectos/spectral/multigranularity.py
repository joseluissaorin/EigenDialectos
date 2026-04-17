"""Multi-granularity eigendecomposition: macro → zonal → dialect.

Decomposes dialectal variation into hierarchical levels:
- Level 0 (Macro): Pan-Hispanic variation shared by ALL varieties
- Level 1 (Zonal): Family-level variation (peninsular, caribbean, etc.)
- Level 2 (Dialect): Variety-specific residual variation

Λ_total^(i) = Λ_macro ⊕ Λ_zonal(i) ⊕ Λ_dialect(i)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import DIALECT_FAMILIES, DialectCode

logger = logging.getLogger(__name__)


class MultiGranularityDecomposition:
    """Hierarchical eigendecomposition across granularity levels.

    Given per-dialect transformation matrices W_i, decomposes each into
    three additive components: macro (pan-Hispanic), zonal (family), and
    dialect (unique).
    """

    def __init__(
        self,
        families: Optional[dict[str, list[DialectCode]]] = None,
    ) -> None:
        self.families = families or DIALECT_FAMILIES
        self._results: dict[str, dict] = {}

    def _dialect_to_family(self) -> dict[str, str]:
        """Build reverse mapping: dialect_code.value -> family name."""
        mapping: dict[str, str] = {}
        for family_name, codes in self.families.items():
            for code in codes:
                mapping[code.value if isinstance(code, DialectCode) else code] = family_name
        return mapping

    def decompose(
        self,
        W_matrices: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, dict[str, npt.NDArray]]:
        """Perform 3-level decomposition.

        Parameters
        ----------
        W_matrices : dict
            Mapping from dialect name/code (str) to square W matrix.

        Returns
        -------
        dict with structure:
            {
                'macro': {'eigenvalues': ..., 'eigenvectors': ..., 'W_mean': ...},
                'zonal': {family_name: {'eigenvalues': ..., 'eigenvectors': ..., 'W_mean': ..., 'W_residual': ...}},
                'dialect': {dialect_name: {'eigenvalues': ..., 'eigenvectors': ..., 'W_residual': ...}},
                'reconstruction_errors': {dialect_name: float},
            }
        """
        if not W_matrices:
            raise ValueError("W_matrices is empty")

        dialect_names = sorted(W_matrices.keys())
        dim = next(iter(W_matrices.values())).shape[0]
        dialect_to_family = self._dialect_to_family()

        # === Level 0: Macro (pan-Hispanic mean) ===
        W_stack = np.stack([W_matrices[d] for d in dialect_names], axis=0)  # (n_dialects, dim, dim)
        W_mean = W_stack.mean(axis=0)  # (dim, dim)
        macro_eigenvalues, macro_eigenvectors = np.linalg.eig(W_mean.astype(np.complex128))

        # Sort by magnitude
        order = np.argsort(-np.abs(macro_eigenvalues))
        macro_eigenvalues = macro_eigenvalues[order]
        macro_eigenvectors = macro_eigenvectors[:, order]

        macro_result = {
            "eigenvalues": macro_eigenvalues,
            "eigenvectors": macro_eigenvectors,
            "W_mean": W_mean,
        }

        # === Level 1: Zonal (family-level residuals) ===
        # Group dialects by family
        family_members: dict[str, list[str]] = {}
        for d in dialect_names:
            family = dialect_to_family.get(d, "other")
            family_members.setdefault(family, []).append(d)

        zonal_results: dict[str, dict] = {}
        family_means: dict[str, npt.NDArray] = {}

        for family_name, members in family_members.items():
            if not members:
                continue
            # Compute family mean (residual from macro)
            family_W_stack = np.stack([W_matrices[d] for d in members], axis=0)
            family_W_mean = family_W_stack.mean(axis=0)

            # Zonal residual = family mean - macro mean
            W_zonal = family_W_mean - W_mean
            zonal_eigenvalues, zonal_eigenvectors = np.linalg.eig(
                W_zonal.astype(np.complex128)
            )

            order = np.argsort(-np.abs(zonal_eigenvalues))
            zonal_eigenvalues = zonal_eigenvalues[order]
            zonal_eigenvectors = zonal_eigenvectors[:, order]

            family_means[family_name] = family_W_mean
            zonal_results[family_name] = {
                "eigenvalues": zonal_eigenvalues,
                "eigenvectors": zonal_eigenvectors,
                "W_mean": family_W_mean,
                "W_residual": W_zonal,
                "members": members,
            }

        # === Level 2: Dialect (variety-specific residuals) ===
        dialect_results: dict[str, dict] = {}
        reconstruction_errors: dict[str, float] = {}

        for d in dialect_names:
            family = dialect_to_family.get(d, "other")
            family_mean = family_means.get(family, W_mean)

            # Dialect residual = W_i - family_mean
            W_dialect = W_matrices[d] - family_mean
            dialect_eigenvalues, dialect_eigenvectors = np.linalg.eig(
                W_dialect.astype(np.complex128)
            )

            order = np.argsort(-np.abs(dialect_eigenvalues))
            dialect_eigenvalues = dialect_eigenvalues[order]
            dialect_eigenvectors = dialect_eigenvectors[:, order]

            dialect_results[d] = {
                "eigenvalues": dialect_eigenvalues,
                "eigenvectors": dialect_eigenvectors,
                "W_residual": W_dialect,
                "family": family,
            }

            # Verify reconstruction: W_i ≈ W_mean + W_zonal + W_dialect
            W_reconstructed = W_mean + (family_mean - W_mean) + W_dialect
            error = float(
                np.linalg.norm(W_matrices[d] - W_reconstructed, "fro")
                / max(np.linalg.norm(W_matrices[d], "fro"), 1e-15)
            )
            reconstruction_errors[d] = error

        self._results = {
            "macro": macro_result,
            "zonal": zonal_results,
            "dialect": dialect_results,
            "reconstruction_errors": reconstruction_errors,
        }

        # Log summary
        max_err = max(reconstruction_errors.values()) if reconstruction_errors else 0
        logger.info(
            "Multi-granularity decomposition: %d dialects, %d families, "
            "max reconstruction error = %.2e",
            len(dialect_names),
            len(family_members),
            max_err,
        )

        return self._results

    def get_hierarchical_spectrum(
        self,
        dialect: str,
    ) -> dict[str, npt.NDArray[np.complex128]]:
        """Get the full hierarchical eigenspectrum for a dialect.

        Returns
        -------
        dict with keys 'macro', 'zonal', 'dialect', each containing eigenvalues.
        """
        if not self._results:
            raise RuntimeError("Call decompose() first")

        family = self._results["dialect"][dialect]["family"]

        return {
            "macro": self._results["macro"]["eigenvalues"],
            "zonal": self._results["zonal"].get(family, {}).get(
                "eigenvalues", np.array([])
            ),
            "dialect": self._results["dialect"][dialect]["eigenvalues"],
        }

    def explained_variance_ratio(self) -> dict[str, dict[str, float]]:
        """Compute what fraction of total Frobenius norm each level explains.

        Returns
        -------
        dict mapping dialect_name -> {'macro': float, 'zonal': float, 'dialect': float}
        """
        if not self._results:
            raise RuntimeError("Call decompose() first")

        ratios: dict[str, dict[str, float]] = {}
        W_mean = self._results["macro"]["W_mean"]
        macro_norm = float(np.linalg.norm(W_mean, "fro"))

        for d, d_result in self._results["dialect"].items():
            family = d_result["family"]
            zonal_W = self._results["zonal"].get(family, {}).get(
                "W_residual", np.zeros_like(W_mean)
            )
            dialect_W = d_result["W_residual"]

            total_norm = macro_norm + float(np.linalg.norm(zonal_W, "fro")) + float(
                np.linalg.norm(dialect_W, "fro")
            )
            if total_norm < 1e-15:
                ratios[d] = {"macro": 0.0, "zonal": 0.0, "dialect": 0.0}
                continue

            ratios[d] = {
                "macro": macro_norm / total_norm,
                "zonal": float(np.linalg.norm(zonal_W, "fro")) / total_norm,
                "dialect": float(np.linalg.norm(dialect_W, "fro")) / total_norm,
            }

        return ratios
