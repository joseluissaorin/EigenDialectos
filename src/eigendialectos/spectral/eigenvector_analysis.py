"""Eigenvector interpretation and cross-dialect comparison."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import DialectCode
from eigendialectos.types import EigenDecomposition


def interpret_eigenvector(
    v: npt.NDArray,
    vocab: list[str],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Project vocabulary words onto an eigenvector and return the most aligned.

    For each word *w* in the vocabulary, compute the magnitude of its
    projection onto *v* (i.e. ``|v_i|`` where *i* is the word index).
    Return the *top_k* words with the largest projections.

    Parameters
    ----------
    v : ndarray, shape (d,)
        Eigenvector (may be complex; magnitudes are used).
    vocab : list of str
        Vocabulary list of length ``d``.
    top_k : int
        Number of words to return.

    Returns
    -------
    list of (str, float)
        ``(word, projection_magnitude)`` pairs sorted by descending magnitude.

    Raises
    ------
    ValueError
        If *v* and *vocab* have incompatible lengths.
    """
    v = np.asarray(v).flatten()
    if len(v) != len(vocab):
        raise ValueError(
            f"Eigenvector length {len(v)} does not match vocab size {len(vocab)}"
        )

    magnitudes = np.abs(v).astype(np.float64)
    top_indices = np.argsort(magnitudes)[::-1][:top_k]

    return [(vocab[i], float(magnitudes[i])) for i in top_indices]


def compare_eigenvectors(
    P_a: npt.NDArray,
    P_b: npt.NDArray,
    top_k: int = 10,
) -> npt.NDArray[np.float64]:
    """Compute the cosine-similarity matrix between eigenvectors of two dialects.

    Parameters
    ----------
    P_a : ndarray, shape (d, n_a)
        Eigenvector matrix of dialect A (columns are eigenvectors).
    P_b : ndarray, shape (d, n_b)
        Eigenvector matrix of dialect B (columns are eigenvectors).
    top_k : int
        Use only the first *top_k* eigenvectors from each matrix.

    Returns
    -------
    ndarray, shape (k_a, k_b)
        Cosine-similarity matrix where entry ``(i, j)`` is the absolute
        cosine similarity between the *i*-th eigenvector of A and the
        *j*-th eigenvector of B.
    """
    P_a = np.asarray(P_a, dtype=np.complex128)
    P_b = np.asarray(P_b, dtype=np.complex128)

    k_a = min(top_k, P_a.shape[1])
    k_b = min(top_k, P_b.shape[1])

    # Extract top-k columns
    Va = P_a[:, :k_a]
    Vb = P_b[:, :k_b]

    # Normalise columns
    norms_a = np.linalg.norm(Va, axis=0, keepdims=True)
    norms_b = np.linalg.norm(Vb, axis=0, keepdims=True)
    norms_a = np.maximum(norms_a, 1e-15)
    norms_b = np.maximum(norms_b, 1e-15)

    Va_normed = Va / norms_a
    Vb_normed = Vb / norms_b

    # Cosine similarity matrix (absolute, since eigenvectors have sign ambiguity)
    sim_matrix = np.abs(Va_normed.conj().T @ Vb_normed)

    return sim_matrix.real.astype(np.float64)


def find_shared_axes(
    decompositions: dict[DialectCode, EigenDecomposition],
    threshold: float = 0.8,
) -> list[dict]:
    """Find eigenvectors that are shared across multiple dialect varieties.

    An eigenvector is considered "shared" if it has high cosine similarity
    (above *threshold*) with an eigenvector in another dialect.

    Parameters
    ----------
    decompositions : dict
        Mapping from ``DialectCode`` to ``EigenDecomposition``.
    threshold : float
        Minimum absolute cosine similarity to consider a match.

    Returns
    -------
    list of dict
        Each entry contains:

        * ``'dialects'`` -- set of ``DialectCode`` sharing the axis.
        * ``'indices'`` -- dict mapping ``DialectCode`` to eigenvector index.
        * ``'avg_similarity'`` -- mean pairwise similarity.
    """
    codes = list(decompositions.keys())
    if len(codes) < 2:
        return []

    shared_axes: list[dict] = []

    # Compare each pair of dialects
    for i, code_a in enumerate(codes):
        P_a = decompositions[code_a].eigenvectors
        n_a = P_a.shape[1]
        for j in range(i + 1, len(codes)):
            code_b = codes[j]
            P_b = decompositions[code_b].eigenvectors
            n_b = P_b.shape[1]

            sim = compare_eigenvectors(P_a, P_b, top_k=max(n_a, n_b))

            # Find pairs above threshold
            for idx_a in range(sim.shape[0]):
                for idx_b in range(sim.shape[1]):
                    if sim[idx_a, idx_b] >= threshold:
                        # Check if this axis already exists in our results
                        merged = False
                        for axis in shared_axes:
                            # If code_a with same index already in an axis, merge code_b
                            if code_a in axis["dialects"] and axis["indices"].get(code_a) == idx_a:
                                axis["dialects"].add(code_b)
                                axis["indices"][code_b] = idx_b
                                merged = True
                                break
                            if code_b in axis["dialects"] and axis["indices"].get(code_b) == idx_b:
                                axis["dialects"].add(code_a)
                                axis["indices"][code_a] = idx_a
                                merged = True
                                break
                        if not merged:
                            shared_axes.append({
                                "dialects": {code_a, code_b},
                                "indices": {code_a: idx_a, code_b: idx_b},
                                "avg_similarity": float(sim[idx_a, idx_b]),
                            })

    # Recompute avg_similarity for merged axes
    for axis in shared_axes:
        if len(axis["dialects"]) > 2:
            sims = []
            axis_codes = list(axis["dialects"])
            for ii in range(len(axis_codes)):
                for jj in range(ii + 1, len(axis_codes)):
                    ca, cb = axis_codes[ii], axis_codes[jj]
                    if ca in axis["indices"] and cb in axis["indices"]:
                        va = decompositions[ca].eigenvectors[:, axis["indices"][ca]]
                        vb = decompositions[cb].eigenvectors[:, axis["indices"][cb]]
                        va_n = va / max(np.linalg.norm(va), 1e-15)
                        vb_n = vb / max(np.linalg.norm(vb), 1e-15)
                        sims.append(float(np.abs(np.dot(va_n.conj(), vb_n))))
            if sims:
                axis["avg_similarity"] = float(np.mean(sims))

    return shared_axes


def find_unique_axes(
    decompositions: dict[DialectCode, EigenDecomposition],
    threshold: float = 0.3,
) -> dict[DialectCode, list[int]]:
    """Find eigenvectors that are unique to specific dialect varieties.

    An eigenvector is considered "unique" to a dialect if its maximum
    cosine similarity with any eigenvector in any other dialect is below
    *threshold*.

    Parameters
    ----------
    decompositions : dict
        Mapping from ``DialectCode`` to ``EigenDecomposition``.
    threshold : float
        Maximum cosine similarity for an eigenvector to be considered unique.

    Returns
    -------
    dict[DialectCode, list[int]]
        Mapping from dialect code to list of unique eigenvector indices.
    """
    codes = list(decompositions.keys())
    unique: dict[DialectCode, list[int]] = {code: [] for code in codes}

    for code_a in codes:
        P_a = decompositions[code_a].eigenvectors
        n_a = P_a.shape[1]

        for idx_a in range(n_a):
            is_unique = True
            va = P_a[:, idx_a]
            va_norm = np.linalg.norm(va)
            if va_norm < 1e-15:
                continue

            va_normed = va / va_norm

            for code_b in codes:
                if code_b == code_a:
                    continue
                P_b = decompositions[code_b].eigenvectors
                n_b = P_b.shape[1]

                for idx_b in range(n_b):
                    vb = P_b[:, idx_b]
                    vb_norm = np.linalg.norm(vb)
                    if vb_norm < 1e-15:
                        continue

                    vb_normed = vb / vb_norm
                    sim = float(np.abs(np.dot(va_normed.conj(), vb_normed)))

                    if sim >= threshold:
                        is_unique = False
                        break

                if not is_unique:
                    break

            if is_unique:
                unique[code_a].append(idx_a)

    return unique
