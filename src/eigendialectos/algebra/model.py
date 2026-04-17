"""Unified algebraic model for dialect transformations.

Models the space of dialect transformations as an approximate algebraic
structure, supporting composition, inversion, interpolation, and
subspace projection -- along with group-axiom verification.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, logm

from eigendialectos.constants import DialectCode
from eigendialectos.types import EigenDecomposition, TransformationMatrix


class DialectAlgebra:
    """Algebraic framework over per-dialect transformation matrices.

    Parameters
    ----------
    transforms : dict[DialectCode, TransformationMatrix]
        Transformation matrix W_d for each dialect.
    eigendecomps : dict[DialectCode, EigenDecomposition]
        Pre-computed eigendecomposition of each W_d.
    """

    def __init__(
        self,
        transforms: dict[DialectCode, TransformationMatrix],
        eigendecomps: dict[DialectCode, EigenDecomposition],
    ) -> None:
        self.transforms = transforms
        self.eigendecomps = eigendecomps

        # Validate consistency
        dims = {code: t.data.shape for code, t in transforms.items()}
        shapes = set(dims.values())
        if len(shapes) > 1:
            raise ValueError(f"Inconsistent matrix shapes: {dims}")
        if shapes:
            s = shapes.pop()
            if len(s) != 2 or s[0] != s[1]:
                raise ValueError(f"Matrices must be square, got shape {s}")
            self._dim = s[0]
        else:
            self._dim = 0

    @property
    def dim(self) -> int:
        """Dimensionality of the transformation matrices."""
        return self._dim

    @property
    def dialects(self) -> list[DialectCode]:
        """Available dialect codes, sorted."""
        return sorted(self.transforms.keys(), key=lambda c: c.value)

    def _get_matrix(self, d: DialectCode) -> np.ndarray:
        """Retrieve the raw matrix for a dialect, raising on missing."""
        if d not in self.transforms:
            raise KeyError(f"Dialect {d.value} not in algebra")
        return self.transforms[d].data

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def compose(
        self, d1: DialectCode, d2: DialectCode
    ) -> TransformationMatrix:
        """Compose two dialect transformations: W_d1 @ W_d2.

        Interpretation: apply d2's transformation first, then d1's.

        Parameters
        ----------
        d1, d2 : DialectCode
            The two dialects to compose.

        Returns
        -------
        TransformationMatrix
            Result of W_{d1} @ W_{d2}.
        """
        W1 = self._get_matrix(d1)
        W2 = self._get_matrix(d2)
        composed = W1 @ W2

        return TransformationMatrix(
            data=composed,
            source_dialect=d2,
            target_dialect=d1,
            regularization=0.0,
        )

    def invert(self, d: DialectCode) -> TransformationMatrix:
        """Compute the pseudo-inverse of a dialect transformation.

        Uses the eigendecomposition when available for numerical
        stability; falls back to numpy.linalg.pinv otherwise.

        Parameters
        ----------
        d : DialectCode
            Dialect whose inverse to compute.

        Returns
        -------
        TransformationMatrix
            W_d^{-1} (or pseudo-inverse if singular).
        """
        if d in self.eigendecomps:
            eig = self.eigendecomps[d]
            # W = P @ diag(lambda) @ P_inv  =>  W^-1 = P @ diag(1/lambda) @ P_inv
            eigenvalues = eig.eigenvalues.copy()
            # Safe reciprocal: zero out negligible eigenvalues
            mask = np.abs(eigenvalues) > 1e-12
            inv_eigenvalues = np.zeros_like(eigenvalues)
            inv_eigenvalues[mask] = 1.0 / eigenvalues[mask]

            inv_data = (
                eig.eigenvectors
                @ np.diag(inv_eigenvalues)
                @ eig.eigenvectors_inv
            )
            inv_data = np.real(inv_data)
        else:
            W = self._get_matrix(d)
            inv_data = np.linalg.pinv(W)

        return TransformationMatrix(
            data=inv_data,
            source_dialect=d,
            target_dialect=d,
            regularization=0.0,
        )

    def interpolate(
        self, d: DialectCode, alpha: float
    ) -> TransformationMatrix:
        """Continuous interpolation W_d(alpha) via matrix logarithm.

        At alpha=0 returns identity, at alpha=1 returns W_d.
        Uses W_d(alpha) = expm(alpha * logm(W_d)).

        Parameters
        ----------
        d : DialectCode
            Dialect to interpolate.
        alpha : float
            Interpolation parameter.  0 -> I, 1 -> W_d.

        Returns
        -------
        TransformationMatrix
            Interpolated transformation matrix.
        """
        if alpha == 0.0:
            return TransformationMatrix(
                data=np.eye(self._dim),
                source_dialect=d,
                target_dialect=d,
                regularization=0.0,
            )

        W = self._get_matrix(d).astype(np.complex128)

        # Use eigendecomposition for a more stable log when available
        if d in self.eigendecomps:
            eig = self.eigendecomps[d]
            eigenvalues = eig.eigenvalues
            # Log of eigenvalues (handle non-positive via complex log)
            log_eigenvalues = np.log(eigenvalues.astype(np.complex128))
            log_W = (
                eig.eigenvectors
                @ np.diag(log_eigenvalues)
                @ eig.eigenvectors_inv
            )
        else:
            log_W = logm(W)

        result = expm(alpha * log_W)
        result = np.real(result).astype(np.float64)

        return TransformationMatrix(
            data=result,
            source_dialect=d,
            target_dialect=d,
            regularization=0.0,
        )

    def project_onto_subspace(
        self, d: DialectCode, subspace_eigenvectors: np.ndarray
    ) -> TransformationMatrix:
        """Project a dialect's transformation onto a given eigensubspace.

        Given a matrix V whose columns span the target subspace,
        computes the projection  P = V @ V^+  and returns  P @ W_d @ P.

        This is idempotent: projecting twice gives the same result.

        Parameters
        ----------
        d : DialectCode
            Dialect to project.
        subspace_eigenvectors : np.ndarray
            Matrix whose columns define the subspace (d x k).

        Returns
        -------
        TransformationMatrix
            The projected transformation.
        """
        W = self._get_matrix(d)
        V = subspace_eigenvectors

        # Orthogonal projector onto column space of V
        # P = V @ (V^T V)^{-1} @ V^T  (for full-rank V)
        # Use pseudoinverse for robustness
        V_pinv = np.linalg.pinv(V)  # shape (k, d)
        P = V @ V_pinv  # shape (d, d)

        projected = P @ W @ P

        return TransformationMatrix(
            data=projected,
            source_dialect=d,
            target_dialect=d,
            regularization=0.0,
        )

    # ------------------------------------------------------------------
    # Group axiom testing
    # ------------------------------------------------------------------

    def is_approximate_group(
        self, tol: float = 1e-6
    ) -> dict:
        """Test approximate group axioms on the set of transformations.

        Checks:
        - **Closure**: W_i @ W_j is close to some W_k for all pairs.
        - **Associativity**: (W_i @ W_j) @ W_k ~ W_i @ (W_j @ W_k).
        - **Identity**: There exists a W_e ~ I.
        - **Inverse**: For each W_i, W_i^{-1} @ W_i ~ I.

        Parameters
        ----------
        tol : float
            Frobenius-norm tolerance for approximate equality.

        Returns
        -------
        dict
            Keys: ``closure`` (bool), ``associativity`` (bool),
            ``identity`` (dict), ``inverse`` (dict), ``details`` (str).
        """
        codes = self.dialects
        matrices = {c: self._get_matrix(c) for c in codes}
        identity = np.eye(self._dim)

        # -- Identity --
        identity_info: dict = {"exists": False, "candidate": None, "error": float("inf")}
        for c in codes:
            err = float(np.linalg.norm(matrices[c] - identity, "fro"))
            if err < identity_info["error"]:
                identity_info["error"] = err
                identity_info["candidate"] = c.value

        identity_info["exists"] = identity_info["error"] < tol

        # -- Inverse --
        inverse_info: dict[str, float] = {}
        inverse_ok = True
        for c in codes:
            inv_tm = self.invert(c)
            product = inv_tm.data @ matrices[c]
            err = float(np.linalg.norm(product - identity, "fro"))
            inverse_info[c.value] = err
            if err >= tol:
                inverse_ok = False

        # -- Associativity --
        assoc_ok = True
        max_assoc_err = 0.0
        for ci in codes:
            for cj in codes:
                for ck in codes:
                    lhs = (matrices[ci] @ matrices[cj]) @ matrices[ck]
                    rhs = matrices[ci] @ (matrices[cj] @ matrices[ck])
                    err = float(np.linalg.norm(lhs - rhs, "fro"))
                    max_assoc_err = max(max_assoc_err, err)
                    if err >= tol:
                        assoc_ok = False

        # -- Closure --
        closure_ok = True
        max_closure_err = 0.0
        for ci in codes:
            for cj in codes:
                product = matrices[ci] @ matrices[cj]
                # Find closest existing matrix
                min_dist = float("inf")
                for ck in codes:
                    dist = float(np.linalg.norm(product - matrices[ck], "fro"))
                    min_dist = min(min_dist, dist)
                max_closure_err = max(max_closure_err, min_dist)
                if min_dist >= tol:
                    closure_ok = False

        details_parts = [
            f"Closure: {'PASS' if closure_ok else 'FAIL'} "
            f"(max err={max_closure_err:.2e})",
            f"Associativity: {'PASS' if assoc_ok else 'FAIL'} "
            f"(max err={max_assoc_err:.2e})",
            f"Identity: {'PASS' if identity_info['exists'] else 'FAIL'} "
            f"(candidate={identity_info['candidate']}, err={identity_info['error']:.2e})",
            f"Inverse: {'PASS' if inverse_ok else 'FAIL'}",
        ]

        return {
            "closure": closure_ok,
            "associativity": assoc_ok,
            "identity": identity_info,
            "inverse": {"ok": inverse_ok, "errors": inverse_info},
            "details": "; ".join(details_parts),
        }
