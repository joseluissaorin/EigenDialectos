"""Dialect arithmetic on eigenvalue spectra and W matrices.

Provides two levels of algebraic operations:

1. **Spectrum-level**: linear arithmetic on eigenvalue magnitude vectors.
   Useful for fast analogy queries (e.g. CAN:CAR :: MEX:?) and
   leave-one-out predictions.

2. **Matrix-level**: operations in the matrix Lie group GL(n).
   Composition via matrix multiplication, inversion, and geodesic
   interpolation via the exponential/logarithm maps.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm, logm

from eigen3.types import ComposeResult, EigenDecomp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_spectrum(spec: np.ndarray, name: str = "spectrum") -> None:
    """Raise ValueError if *spec* is not a 1-D real array."""
    if spec.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got shape {spec.shape}"
        )


def _validate_same_length(*spectra: np.ndarray) -> None:
    """Raise ValueError unless all spectra share the same length."""
    lengths = {s.shape[0] for s in spectra}
    if len(lengths) != 1:
        raise ValueError(
            f"All spectra must have the same length, got lengths {sorted(lengths)}"
        )


def _validate_square(M: np.ndarray, name: str = "matrix") -> None:
    """Raise ValueError if *M* is not a 2-D square matrix."""
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            f"{name} must be square, got shape {M.shape}"
        )


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Normalize a weight vector so it sums to 1.

    Parameters
    ----------
    weights : sequence of float
        Raw (non-negative) weights.

    Returns
    -------
    np.ndarray
        Weights that sum to 1.0.

    Raises
    ------
    ValueError
        If all weights are zero or any weight is negative.
    """
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative")
    total = w.sum()
    if total == 0.0:
        raise ValueError("Weights sum to zero; cannot normalize")
    return w / total


# ===================================================================
# Spectrum-level operations
# ===================================================================


def interpolate_spectrum(
    spec_a: np.ndarray,
    spec_b: np.ndarray,
    t: float,
) -> np.ndarray:
    """Linearly interpolate between two eigenvalue-magnitude spectra.

    result = (1 - t) * spec_a + t * spec_b

    When *t* = 0 the result equals *spec_a*; when *t* = 1 it equals
    *spec_b*.  Values outside [0, 1] extrapolate.

    Parameters
    ----------
    spec_a : np.ndarray
        (n,) eigenvalue magnitudes of dialect A.
    spec_b : np.ndarray
        (n,) eigenvalue magnitudes of dialect B.
    t : float
        Interpolation parameter.

    Returns
    -------
    np.ndarray
        (n,) interpolated spectrum.

    Raises
    ------
    ValueError
        If the two spectra differ in length or are not 1-D.
    """
    _validate_spectrum(spec_a, "spec_a")
    _validate_spectrum(spec_b, "spec_b")
    _validate_same_length(spec_a, spec_b)

    result = (1.0 - t) * spec_a + t * spec_b
    logger.debug(
        "interpolate_spectrum: t=%.3f, ||result||=%.4f",
        t, float(np.linalg.norm(result)),
    )
    return result


def analogy_spectrum(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Spectrum-level analogy: "a is to b as c is to ?".

    result = c + (b - a)

    Typical usage: ``analogy_spectrum(CAN, CAR, MEX)`` answers the
    question *"Canarian is to Caribbean as Mexican is to ...?"*.

    Parameters
    ----------
    a : np.ndarray
        (n,) source spectrum (e.g. CAN).
    b : np.ndarray
        (n,) target spectrum (e.g. CAR).
    c : np.ndarray
        (n,) query spectrum (e.g. MEX).

    Returns
    -------
    np.ndarray
        (n,) predicted spectrum for the analogy answer.

    Raises
    ------
    ValueError
        If spectra differ in length or are not 1-D.
    """
    _validate_spectrum(a, "a")
    _validate_spectrum(b, "b")
    _validate_spectrum(c, "c")
    _validate_same_length(a, b, c)

    result = c + (b - a)
    logger.debug(
        "analogy_spectrum: ||b-a||=%.4f, ||result||=%.4f",
        float(np.linalg.norm(b - a)),
        float(np.linalg.norm(result)),
    )
    return result


def compose_spectra(
    spectra: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Weighted sum of eigenvalue-magnitude spectra.

    result = sum_i  w_i * spectra[i]

    Weights are normalized to sum to 1 before the combination so that
    the result lives in the convex hull of the inputs (assuming all
    weights are non-negative).

    Parameters
    ----------
    spectra : list of np.ndarray
        Each element is an (n,) spectrum.
    weights : list of float
        Non-negative weights; need not sum to 1.

    Returns
    -------
    np.ndarray
        (n,) composed spectrum.

    Raises
    ------
    ValueError
        If lists are empty, differ in length, spectra have inconsistent
        shapes, or weights are invalid.
    """
    if len(spectra) == 0:
        raise ValueError("spectra list must be non-empty")
    if len(spectra) != len(weights):
        raise ValueError(
            f"spectra ({len(spectra)}) and weights ({len(weights)}) "
            f"must have the same length"
        )

    for i, s in enumerate(spectra):
        _validate_spectrum(s, f"spectra[{i}]")
    _validate_same_length(*spectra)

    w = _normalize_weights(weights)
    n = spectra[0].shape[0]
    result = np.zeros(n, dtype=np.float64)
    for wi, si in zip(w, spectra):
        result += wi * si

    logger.debug(
        "compose_spectra: %d spectra, weights=%s, ||result||=%.4f",
        len(spectra),
        np.array2string(w, precision=3),
        float(np.linalg.norm(result)),
    )
    return result


def centroid_spectrum(spectra: list[np.ndarray]) -> np.ndarray:
    """Mean (centroid) of a collection of eigenvalue-magnitude spectra.

    Equivalent to ``compose_spectra(spectra, [1]*len(spectra))``.

    Parameters
    ----------
    spectra : list of np.ndarray
        Each element is an (n,) spectrum.

    Returns
    -------
    np.ndarray
        (n,) mean spectrum.

    Raises
    ------
    ValueError
        If the list is empty or spectra have inconsistent shapes.
    """
    if len(spectra) == 0:
        raise ValueError("spectra list must be non-empty")
    uniform = [1.0] * len(spectra)
    return compose_spectra(spectra, uniform)


def predict_leave_one_out(
    spectra_dict: dict[str, np.ndarray],
    left_out: str,
) -> np.ndarray:
    """Predict one dialect's spectrum from the mean of the remaining dialects.

    This is useful for leave-one-out cross-validation: the prediction
    is the centroid of every other dialect, and the error is the
    distance to the true held-out spectrum.

    Parameters
    ----------
    spectra_dict : dict[str, np.ndarray]
        Mapping from dialect label to (n,) eigenvalue-magnitude spectrum.
    left_out : str
        Label of the dialect to hold out.

    Returns
    -------
    np.ndarray
        (n,) predicted spectrum for the held-out dialect.

    Raises
    ------
    KeyError
        If *left_out* is not in *spectra_dict*.
    ValueError
        If fewer than two dialects are provided (nothing to average).
    """
    if left_out not in spectra_dict:
        raise KeyError(f"Dialect '{left_out}' not found in spectra_dict")
    if len(spectra_dict) < 2:
        raise ValueError(
            "Need at least 2 dialects for leave-one-out prediction"
        )

    others = [s for label, s in spectra_dict.items() if label != left_out]
    prediction = centroid_spectrum(others)
    logger.info(
        "predict_leave_one_out: held out '%s', averaged %d others",
        left_out, len(others),
    )
    return prediction


def spectrum_to_W(
    spectrum: np.ndarray,
    P: np.ndarray,
    P_inv: np.ndarray,
) -> np.ndarray:
    """Reconstruct a W matrix from a (possibly modified) eigenvalue spectrum.

    W_new = P @ diag(spectrum) @ P_inv

    The eigenvector basis (P, P_inv) is kept fixed while the eigenvalue
    magnitudes are replaced.  The result is real-valued (imaginary
    residuals from floating-point arithmetic are discarded).

    Parameters
    ----------
    spectrum : np.ndarray
        (n,) eigenvalue magnitudes to place on the diagonal.
    P : np.ndarray
        (n, n) eigenvector matrix.
    P_inv : np.ndarray
        (n, n) inverse eigenvector matrix.

    Returns
    -------
    np.ndarray
        (n, n) reconstructed W matrix (real part only).

    Raises
    ------
    ValueError
        If dimensions are inconsistent.
    """
    _validate_spectrum(spectrum, "spectrum")
    _validate_square(P, "P")
    _validate_square(P_inv, "P_inv")

    n = spectrum.shape[0]
    if P.shape[0] != n or P_inv.shape[0] != n:
        raise ValueError(
            f"Dimension mismatch: spectrum has {n} entries but "
            f"P is {P.shape} and P_inv is {P_inv.shape}"
        )

    Lambda = np.diag(spectrum.astype(np.complex128))
    W_new = (P @ Lambda @ P_inv).real

    cond = float(np.linalg.cond(W_new))
    logger.debug(
        "spectrum_to_W: n=%d, condition=%.2f, ||W||_F=%.4f",
        n, cond, float(np.linalg.norm(W_new, "fro")),
    )
    return W_new


# ===================================================================
# Matrix-level operations
# ===================================================================


def compose_W(W_a: np.ndarray, W_b: np.ndarray) -> np.ndarray:
    """Compose two transformation matrices via matrix multiplication.

    result = W_a @ W_b

    In the dialect-mapping interpretation this first applies the
    transformation encoded by W_b, then the one encoded by W_a
    (standard right-to-left composition).

    Parameters
    ----------
    W_a : np.ndarray
        (n, n) first transformation matrix (applied second).
    W_b : np.ndarray
        (n, n) second transformation matrix (applied first).

    Returns
    -------
    np.ndarray
        (n, n) composed transformation matrix.

    Raises
    ------
    ValueError
        If matrices are not square or have incompatible shapes.
    """
    _validate_square(W_a, "W_a")
    _validate_square(W_b, "W_b")
    if W_a.shape != W_b.shape:
        raise ValueError(
            f"Shape mismatch: W_a is {W_a.shape}, W_b is {W_b.shape}"
        )

    result = W_a @ W_b
    logger.debug(
        "compose_W: ||W_a||_F=%.4f, ||W_b||_F=%.4f, ||result||_F=%.4f",
        float(np.linalg.norm(W_a, "fro")),
        float(np.linalg.norm(W_b, "fro")),
        float(np.linalg.norm(result, "fro")),
    )
    return result


def invert_W(W: np.ndarray) -> np.ndarray:
    """Compute the matrix inverse of a transformation matrix.

    result = W^{-1}

    In the dialect-mapping interpretation this reverses the direction
    of the transformation (target -> source instead of source -> target).

    Parameters
    ----------
    W : np.ndarray
        (n, n) invertible transformation matrix.

    Returns
    -------
    np.ndarray
        (n, n) inverse transformation matrix.

    Raises
    ------
    ValueError
        If W is not square.
    numpy.linalg.LinAlgError
        If W is singular.
    """
    _validate_square(W, "W")

    cond_before = float(np.linalg.cond(W))
    result = inv(W)
    cond_after = float(np.linalg.cond(result))

    logger.debug(
        "invert_W: condition(W)=%.2f, condition(W_inv)=%.2f",
        cond_before, cond_after,
    )
    return result


def interpolate_W(
    W_a: np.ndarray,
    W_b: np.ndarray,
    t: float,
) -> np.ndarray:
    """Geodesic interpolation between two matrices in the Lie group GL(n).

    result = expm((1 - t) * logm(W_a) + t * logm(W_b))

    This traces the geodesic on the matrix Lie group connecting W_a
    (at t=0) and W_b (at t=1).  Unlike naive linear interpolation of
    entries, this respects the multiplicative structure of the group
    and preserves invertibility along the path.

    Parameters
    ----------
    W_a : np.ndarray
        (n, n) starting transformation matrix.
    W_b : np.ndarray
        (n, n) ending transformation matrix.
    t : float
        Interpolation parameter.  t=0 gives W_a, t=1 gives W_b.
        Values outside [0, 1] extrapolate along the geodesic.

    Returns
    -------
    np.ndarray
        (n, n) interpolated transformation matrix (real part only).

    Raises
    ------
    ValueError
        If matrices are not square or have incompatible shapes.

    Notes
    -----
    Both matrices must be invertible (non-singular) for the matrix
    logarithm to be well-defined.  Poorly conditioned matrices may
    lead to large imaginary residuals that are discarded.
    """
    _validate_square(W_a, "W_a")
    _validate_square(W_b, "W_b")
    if W_a.shape != W_b.shape:
        raise ValueError(
            f"Shape mismatch: W_a is {W_a.shape}, W_b is {W_b.shape}"
        )

    log_a = logm(W_a.astype(np.complex128))
    log_b = logm(W_b.astype(np.complex128))

    log_interp = (1.0 - t) * log_a + t * log_b
    result = expm(log_interp).real

    logger.debug(
        "interpolate_W: t=%.3f, ||result||_F=%.4f, condition=%.2f",
        t,
        float(np.linalg.norm(result, "fro")),
        float(np.linalg.cond(result)),
    )
    return result


# ===================================================================
# Convenience: full algebra pipeline (spectrum -> ComposeResult)
# ===================================================================


def compose_dialects(
    decomps: dict[str, EigenDecomp],
    weights: dict[str, float],
    reference_decomp: EigenDecomp,
) -> ComposeResult:
    """Compose a synthetic dialect from weighted eigenvalue spectra.

    This is the main entry point for the dialect algebra pipeline:

    1. Extract eigenvalue magnitudes from each decomposition.
    2. Weighted-sum the spectra (normalizing weights).
    3. Reconstruct a W matrix from the composed spectrum using the
       eigenvector basis of *reference_decomp*.

    Parameters
    ----------
    decomps : dict[str, EigenDecomp]
        Mapping from dialect label to its eigendecomposition.
    weights : dict[str, float]
        Non-negative weight for each dialect.  Keys must be a subset
        of *decomps*.
    reference_decomp : EigenDecomp
        Eigendecomposition whose eigenvector basis (P, P_inv) is used
        for spectrum-to-matrix reconstruction.

    Returns
    -------
    ComposeResult
        Contains the composed spectrum, reconstructed W, and its
        condition number.

    Raises
    ------
    KeyError
        If a key in *weights* is missing from *decomps*.
    ValueError
        If weights are empty or spectra have inconsistent shapes.
    """
    if not weights:
        raise ValueError("weights must be non-empty")

    missing = set(weights) - set(decomps)
    if missing:
        raise KeyError(f"Dialects in weights but not in decomps: {missing}")

    spectra = [decomps[label].magnitudes for label in weights]
    w_vals = [weights[label] for label in weights]

    composed = compose_spectra(spectra, w_vals)

    W_new = spectrum_to_W(
        composed,
        reference_decomp.P,
        reference_decomp.P_inv,
    )
    cond = float(np.linalg.cond(W_new))

    labels_str = ", ".join(
        f"{label}={weights[label]:.2f}" for label in weights
    )
    logger.info(
        "compose_dialects: [%s] -> condition=%.2f", labels_str, cond
    )

    return ComposeResult(spectrum=composed, W=W_new, condition=cond)


def analogy_dialects(
    decomps: dict[str, EigenDecomp],
    a: str,
    b: str,
    c: str,
    reference_decomp: EigenDecomp,
) -> ComposeResult:
    """Dialect analogy: "a is to b as c is to ?".

    Computes the analogy at the spectrum level and reconstructs the
    resulting W matrix using the eigenvector basis from
    *reference_decomp*.

    Parameters
    ----------
    decomps : dict[str, EigenDecomp]
        Mapping from dialect label to its eigendecomposition.
    a : str
        Source dialect label (e.g. "CAN").
    b : str
        Target dialect label (e.g. "CAR").
    c : str
        Query dialect label (e.g. "MEX").
    reference_decomp : EigenDecomp
        Eigendecomposition whose eigenvector basis is used for
        spectrum-to-matrix reconstruction.

    Returns
    -------
    ComposeResult
        Contains the predicted spectrum, reconstructed W, and its
        condition number.

    Raises
    ------
    KeyError
        If any of a, b, c is not in *decomps*.
    """
    for label in (a, b, c):
        if label not in decomps:
            raise KeyError(f"Dialect '{label}' not found in decomps")

    predicted = analogy_spectrum(
        decomps[a].magnitudes,
        decomps[b].magnitudes,
        decomps[c].magnitudes,
    )

    W_new = spectrum_to_W(
        predicted,
        reference_decomp.P,
        reference_decomp.P_inv,
    )
    cond = float(np.linalg.cond(W_new))

    logger.info(
        "analogy_dialects: %s:%s :: %s:? -> condition=%.2f",
        a, b, c, cond,
    )

    return ComposeResult(spectrum=predicted, W=W_new, condition=cond)
