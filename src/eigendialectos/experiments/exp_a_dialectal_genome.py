"""Experiment A: The Dialectal Genome.

Collect eigenvalue signatures across all linguistic levels for each dialect,
compute pairwise spectral distances to build a distance matrix, then infer
a phylogenetic tree via Neighbor-Joining (Ward linkage).  The inferred tree
is compared with a reference tree encoding known historical relationships
(e.g.  PEN close to AND, CAN bridging to CAR, RIO close to CHI) using the
cophenetic correlation coefficient.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import squareform

from eigendialectos.constants import DialectCode, DIALECT_NAMES
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import (
    EigenDecomposition,
    ExperimentResult,
    TransformationMatrix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference phylogenetic distances (normalised expert judgement).
# Lower value = historically / dialectologically closer.
# Symmetric: only upper-triangle entries are listed; the matrix is completed
# programmatically.
# ---------------------------------------------------------------------------
_DIALECT_ORDER: list[DialectCode] = sorted(DialectCode, key=lambda c: c.value)

_REFERENCE_DISTANCES: dict[tuple[DialectCode, DialectCode], float] = {
    # Peninsular cluster
    (DialectCode.ES_PEN, DialectCode.ES_AND): 0.15,
    (DialectCode.ES_PEN, DialectCode.ES_CAN): 0.25,
    (DialectCode.ES_AND, DialectCode.ES_CAN): 0.20,
    # Canarian bridges to Caribbean
    (DialectCode.ES_CAN, DialectCode.ES_CAR): 0.35,
    # Southern-cone cluster
    (DialectCode.ES_RIO, DialectCode.ES_CHI): 0.30,
    # Mexican - Andean moderate
    (DialectCode.ES_MEX, DialectCode.ES_AND_BO): 0.40,
    # Inter-continental defaults filled as 0.70 below.
}


def _build_reference_matrix(codes: list[DialectCode]) -> np.ndarray:
    """Build a full symmetric reference distance matrix from expert judgement."""
    n = len(codes)
    idx = {c: i for i, c in enumerate(codes)}
    D = np.full((n, n), 0.70, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for (ca, cb), dist in _REFERENCE_DISTANCES.items():
        if ca in idx and cb in idx:
            D[idx[ca], idx[cb]] = dist
            D[idx[cb], idx[ca]] = dist

    return D


class DialectalGenomeExperiment(Experiment):
    experiment_id = "exp_a_dialectal_genome"
    name = "The Dialectal Genome"
    description = (
        "Extract the full eigenvalue signature of every dialect, compute "
        "pairwise spectral distances, infer a phylogenetic tree via "
        "hierarchical clustering, and measure cophenetic correlation with "
        "known historical relationships."
    )
    dependencies = [
        "eigendialectos.spectral.eigendecomposition",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._W_matrices: dict[DialectCode, np.ndarray] = {}
        self._eigendecomps: dict[DialectCode, EigenDecomposition] = {}
        self._dialect_order: list[DialectCode] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 50)
        rng = np.random.default_rng(seed)

        data_dir = config.get("data_dir")
        loaded = False

        if data_dir:
            data_path = Path(data_dir)
            if data_path.exists():
                # Try to load precomputed W matrices
                for code in DialectCode:
                    w_path = data_path / f"W_{code.value}.npy"
                    if w_path.exists():
                        self._W_matrices[code] = np.load(str(w_path))
                if len(self._W_matrices) == len(DialectCode):
                    loaded = True
                    logger.info(
                        "Loaded %d W matrices from %s",
                        len(self._W_matrices),
                        data_dir,
                    )
                else:
                    self._W_matrices.clear()

        if not loaded:
            logger.info(
                "Generating synthetic W matrices (dim=%d) for %d dialects.",
                dim,
                len(DialectCode),
            )
            # Synthetic W: identity + dialect-specific perturbation
            for code in DialectCode:
                perturbation = rng.standard_normal((dim, dim)) * 0.05
                self._W_matrices[code] = np.eye(dim) + perturbation

        # Eigendecompose each W
        for code, W in self._W_matrices.items():
            eigenvalues, P = np.linalg.eig(W)
            try:
                P_inv = np.linalg.inv(P)
            except np.linalg.LinAlgError:
                P_inv = np.linalg.pinv(P)
            self._eigendecomps[code] = EigenDecomposition(
                eigenvalues=eigenvalues.astype(np.complex128),
                eigenvectors=P.astype(np.complex128),
                eigenvectors_inv=P_inv.astype(np.complex128),
                dialect_code=code,
            )

        self._dialect_order = sorted(self._eigendecomps.keys(), key=lambda c: c.value)
        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        codes = self._dialect_order
        n = len(codes)

        # 1. Collect eigenvalue signatures (sorted absolute values)
        signatures: dict[str, list[float]] = {}
        for code in codes:
            ev = np.abs(self._eigendecomps[code].eigenvalues)
            ev_sorted = np.sort(ev.real)[::-1]
            signatures[code.value] = ev_sorted.tolist()

        # 2. Pairwise spectral distances (L2 between sorted eigenvalue vectors)
        dim = max(len(v) for v in signatures.values())
        spec_vecs = np.zeros((n, dim), dtype=np.float64)
        for i, code in enumerate(codes):
            ev = signatures[code.value]
            spec_vecs[i, : len(ev)] = ev

        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(spec_vecs[i] - spec_vecs[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # 3. Build tree via Ward linkage
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method="ward")

        # 4. Reference tree
        ref_matrix = _build_reference_matrix(codes)
        ref_condensed = squareform(ref_matrix)
        Z_ref = linkage(ref_condensed, method="ward")

        # 5. Cophenetic correlation
        coph_inferred, _ = cophenet(Z, condensed)
        coph_reference, _ = cophenet(Z_ref, ref_condensed)

        # Cross-cophenetic: correlation between inferred cophenetic dists
        # and reference cophenetic dists
        coph_dists_inferred = squareform(
            np.zeros((n, n))  # placeholder, will compute from Z
        )
        # Recompute cophenetic distance matrices
        from scipy.cluster.hierarchy import cophenet as coph_fn
        _, coph_dists_inf = coph_fn(Z, condensed)
        _, coph_dists_ref = coph_fn(Z_ref, ref_condensed)

        # Correlation between the two cophenetic distance vectors
        if len(coph_dists_inf) > 0 and len(coph_dists_ref) > 0:
            correlation = float(np.corrcoef(coph_dists_inf, coph_dists_ref)[0, 1])
        else:
            correlation = 0.0

        # Handle NaN from constant arrays
        if np.isnan(correlation):
            correlation = 0.0

        metrics: dict = {
            "distance_matrix": dist_matrix.tolist(),
            "dialect_order": [c.value for c in codes],
            "eigenvalue_signatures": signatures,
            "cophenetic_correlation_inferred": float(coph_inferred),
            "cophenetic_correlation_reference": float(coph_reference),
            "cophenetic_correlation_cross": correlation,
            "linkage_matrix": Z.tolist(),
            "linkage_matrix_ref": Z_ref.tolist(),
            "tree_topology": {
                "method": "ward",
                "n_dialects": n,
                "n_merges": len(Z),
            },
        }

        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        coph_inferred = result.metrics.get("cophenetic_correlation_inferred", 0.0)
        coph_cross = result.metrics.get("cophenetic_correlation_cross", 0.0)

        # Tree fidelity: cross-cophenetic > 0.5 means reasonable match
        fidelity_good = coph_cross > 0.5

        # Check close pairs are closer than distant pairs in inferred tree
        dist = np.array(result.metrics["distance_matrix"])
        order = [DialectCode(c) for c in result.metrics["dialect_order"]]
        idx = {c: i for i, c in enumerate(order)}

        close_pairs = [
            (DialectCode.ES_PEN, DialectCode.ES_AND),
            (DialectCode.ES_AND, DialectCode.ES_CAN),
            (DialectCode.ES_RIO, DialectCode.ES_CHI),
        ]
        distant_pairs = [
            (DialectCode.ES_PEN, DialectCode.ES_RIO),
            (DialectCode.ES_PEN, DialectCode.ES_CHI),
            (DialectCode.ES_MEX, DialectCode.ES_AND),
        ]

        close_dists = [
            dist[idx[a], idx[b]]
            for a, b in close_pairs
            if a in idx and b in idx
        ]
        far_dists = [
            dist[idx[a], idx[b]]
            for a, b in distant_pairs
            if a in idx and b in idx
        ]

        mean_close = float(np.mean(close_dists)) if close_dists else 0.0
        mean_far = float(np.mean(far_dists)) if far_dists else 0.0
        ordering_correct = mean_close < mean_far if (close_dists and far_dists) else False

        return {
            "cophenetic_inferred": float(coph_inferred),
            "cophenetic_cross": float(coph_cross),
            "tree_fidelity_good": fidelity_good,
            "mean_close_distance": mean_close,
            "mean_far_distance": mean_far,
            "ordering_correct": ordering_correct,
        }

    def visualize(self, result: ExperimentResult) -> list[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping visualisation.")
            return []

        output_dir = Path(result.config.get("output_dir", ".")) / self.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        order = result.metrics["dialect_order"]
        dist = np.array(result.metrics["distance_matrix"])
        Z = np.array(result.metrics["linkage_matrix"])
        Z_ref = np.array(result.metrics["linkage_matrix_ref"])

        # --- 1. Inferred dendrogram ---
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(Z, labels=order, ax=ax, leaf_rotation=45)
        ax.set_title("Inferred Dialectal Phylogenetic Tree (Ward)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        p = output_dir / "inferred_dendrogram.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # --- 2. Reference dendrogram ---
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(Z_ref, labels=order, ax=ax, leaf_rotation=45)
        ax.set_title("Reference Historical Phylogenetic Tree (Ward)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        p = output_dir / "reference_dendrogram.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # --- 3. Distance heatmap ---
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(dist, cmap="YlOrRd")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        fig.colorbar(im, ax=ax, label="Spectral Distance")
        ax.set_title("Pairwise Eigenvalue Signature Distance")
        plt.tight_layout()
        p = output_dir / "eigenvalue_distance_heatmap.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # --- 4. Cophenetic comparison scatter ---
        coph_cross = result.metrics.get("cophenetic_correlation_cross", 0.0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ref_matrix = _build_reference_matrix(
            [DialectCode(c) for c in order]
        )
        upper_inferred = dist[np.triu_indices(len(order), k=1)]
        upper_ref = ref_matrix[np.triu_indices(len(order), k=1)]
        ax.scatter(upper_ref, upper_inferred, alpha=0.7, edgecolors="k", linewidths=0.5)
        ax.set_xlabel("Reference Distance")
        ax.set_ylabel("Inferred Spectral Distance")
        ax.set_title(f"Distance Correlation (r = {coph_cross:.3f})")
        # Fit line
        if len(upper_ref) > 1:
            coeffs = np.polyfit(upper_ref, upper_inferred, 1)
            x_line = np.linspace(upper_ref.min(), upper_ref.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), "r--", alpha=0.7)
        plt.tight_layout()
        p = output_dir / "cophenetic_scatter.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        return paths
