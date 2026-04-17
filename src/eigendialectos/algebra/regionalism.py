"""Regionalism decomposition of dialect transformations.

Decomposes a transformation matrix W into additive or multiplicative
components corresponding to different linguistic feature categories
(lexical, morphosyntactic, pragmatic, phonological).
"""

from __future__ import annotations

import numpy as np

from eigendialectos.constants import FeatureCategory
from eigendialectos.types import TransformationMatrix


def decompose_regionalism(
    W: TransformationMatrix,
    feature_subspaces: dict[FeatureCategory, np.ndarray],
) -> dict[FeatureCategory, TransformationMatrix]:
    """Additive regionalism decomposition.

    Decomposes the deviation from identity into per-feature components:

        W ~ I + Delta_lex + Delta_morph + Delta_prag + Delta_phon

    Each Delta_cat is obtained by projecting the rows of (W - I)
    onto the subspace defined by feature_subspaces[cat]:

        Delta_cat = P_cat @ (W - I)

    where P_cat is the orthogonal projector onto the column space
    of the subspace basis.  When the subspaces are orthogonal and
    span the full row space, the deltas sum exactly to (W - I).

    Parameters
    ----------
    W : TransformationMatrix
        The full dialect transformation matrix.
    feature_subspaces : dict[FeatureCategory, np.ndarray]
        Mapping from feature category to a matrix whose columns span
        the relevant subspace.  Each value has shape (d, k_cat).

    Returns
    -------
    dict[FeatureCategory, TransformationMatrix]
        Per-category deviation matrices Delta_cat.  The identity
        component is not included; to reconstruct W, sum all deltas
        and add I.
    """
    data = W.data.astype(np.float64)
    d = data.shape[0]
    deviation = data - np.eye(d)

    result: dict[FeatureCategory, TransformationMatrix] = {}

    for cat, V in feature_subspaces.items():
        V = V.astype(np.float64)

        # Orthogonal projector: P = V @ (V^T V)^{-1} @ V^T
        V_pinv = np.linalg.pinv(V)
        P = V @ V_pinv

        # Project deviation rows onto subspace:
        # Delta_cat = P @ deviation
        # This ensures that when subspaces partition the row space,
        # sum of deltas = sum of P_i @ deviation = I @ deviation = deviation.
        delta = P @ deviation

        result[cat] = TransformationMatrix(
            data=delta,
            source_dialect=W.source_dialect,
            target_dialect=W.target_dialect,
            regularization=0.0,
        )

    return result


def multiplicative_decomposition(
    W: TransformationMatrix,
    feature_subspaces: dict[FeatureCategory, np.ndarray],
) -> dict[FeatureCategory, TransformationMatrix]:
    """Multiplicative regionalism decomposition via iterative projection.

    Decomposes W ~ W_lex @ W_morph @ W_prag @ W_phon where each factor
    captures the variation within its feature subspace.

    Uses an iterative approach: at each step, extract the component
    living in a subspace and divide it out of the residual.

    Parameters
    ----------
    W : TransformationMatrix
        The full dialect transformation matrix.
    feature_subspaces : dict[FeatureCategory, np.ndarray]
        Per-category subspace bases (d, k_cat).

    Returns
    -------
    dict[FeatureCategory, TransformationMatrix]
        Per-category factor matrices W_cat such that their product
        approximates W.
    """
    data = W.data.astype(np.float64)
    d = data.shape[0]
    residual = data.copy()

    # Sort categories for deterministic ordering
    sorted_cats = sorted(feature_subspaces.keys(), key=lambda c: c.value)

    result: dict[FeatureCategory, TransformationMatrix] = {}

    for cat in sorted_cats:
        V = feature_subspaces[cat].astype(np.float64)
        V_pinv = np.linalg.pinv(V)
        P = V @ V_pinv  # projector onto subspace
        P_perp = np.eye(d) - P  # complement projector

        # Extract the component: W_cat = I + P @ (residual - I) @ P
        deviation = residual - np.eye(d)
        delta = P @ deviation @ P
        W_cat = np.eye(d) + delta

        result[cat] = TransformationMatrix(
            data=W_cat,
            source_dialect=W.source_dialect,
            target_dialect=W.target_dialect,
            regularization=0.0,
        )

        # Divide out this component from residual
        # residual <- W_cat^{-1} @ residual
        try:
            W_cat_inv = np.linalg.inv(W_cat)
        except np.linalg.LinAlgError:
            W_cat_inv = np.linalg.pinv(W_cat)

        residual = W_cat_inv @ residual

    return result
