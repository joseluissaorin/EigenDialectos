"""Hierarchical decomposition: W = W_macro + W_zonal + W_dialect."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from eigen3.constants import DIALECT_FAMILIES

logger = logging.getLogger(__name__)


def decompose(
    W_dict: dict[str, np.ndarray],
    families: Optional[dict[str, list[str]]] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Multi-granularity decomposition of transformation matrices.

    W_v = W_macro + W_zonal_f(v) + W_dialect_v

    Parameters
    ----------
    W_dict : dict
        Variety -> W matrix.
    families : dict, optional
        Family name -> list of variety names. Defaults to DIALECT_FAMILIES.

    Returns
    -------
    W_macro : np.ndarray
        Shared component (mean of all W).
    W_zonal : dict[str, np.ndarray]
        Per-family zonal component (family mean - macro).
    W_dialect : dict[str, np.ndarray]
        Per-variety individual component (W - macro - zonal).
    variance_ratios : dict[str, float]
        Variance explained by each level.
    """
    if families is None:
        families = DIALECT_FAMILIES

    # Build variety -> family mapping
    variety_to_family: dict[str, str] = {}
    for fam_name, members in families.items():
        for v in members:
            if v in W_dict:
                variety_to_family[v] = fam_name

    # Macro: mean of all W
    all_W = list(W_dict.values())
    W_macro = np.mean(all_W, axis=0)

    # Zonal: per-family mean minus macro
    family_means: dict[str, np.ndarray] = {}
    for fam_name, members in families.items():
        fam_W = [W_dict[v] for v in members if v in W_dict]
        if fam_W:
            family_means[fam_name] = np.mean(fam_W, axis=0)

    W_zonal: dict[str, np.ndarray] = {}
    for fam_name, fam_mean in family_means.items():
        W_zonal[fam_name] = fam_mean - W_macro

    # Dialect: individual residual
    W_dialect: dict[str, np.ndarray] = {}
    for v, W in W_dict.items():
        fam = variety_to_family.get(v)
        if fam and fam in W_zonal:
            W_dialect[v] = W - W_macro - W_zonal[fam]
        else:
            W_dialect[v] = W - W_macro

    # Variance ratios
    total_var = sum(np.linalg.norm(W - np.eye(W.shape[0]), "fro") ** 2 for W in all_W)
    if total_var > 0:
        macro_var = len(all_W) * np.linalg.norm(W_macro - np.eye(W_macro.shape[0]), "fro") ** 2
        zonal_var = sum(
            sum(1 for v in members if v in W_dict) * np.linalg.norm(wz, "fro") ** 2
            for fam_name, wz in W_zonal.items()
            for members in [families[fam_name]]
        )
        dialect_var = sum(np.linalg.norm(wd, "fro") ** 2 for wd in W_dialect.values())

        variance_ratios = {
            "macro": float(macro_var / total_var),
            "zonal": float(zonal_var / total_var),
            "dialect": float(dialect_var / total_var),
        }
    else:
        variance_ratios = {"macro": 0.0, "zonal": 0.0, "dialect": 0.0}

    logger.info(
        "Multi-granularity: macro=%.1f%%, zonal=%.1f%%, dialect=%.1f%%",
        variance_ratios["macro"] * 100,
        variance_ratios["zonal"] * 100,
        variance_ratios["dialect"] * 100,
    )

    return W_macro, W_zonal, W_dialect, variance_ratios
