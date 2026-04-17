"""Experiment 6: Dialectal Evolution (Phylogenetic Analysis).

Compare eigenvectors across all dialect pairs to identify shared vs unique
variation axes.  Construct a phylogenetic-like tree from eigenvector
similarities and link shared eigenvectors to shared historical origins.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from eigendialectos.constants import DialectCode, DIALECT_NAMES
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import (
    EigenDecomposition,
    EmbeddingMatrix,
    ExperimentResult,
    TransformationMatrix,
)

logger = logging.getLogger(__name__)

# Known historical relationships for validation
_HISTORICAL_RELATIONS: dict[str, list[tuple[str, str]]] = {
    "iberian_cluster": [
        (DialectCode.ES_PEN.value, DialectCode.ES_AND.value),
        (DialectCode.ES_PEN.value, DialectCode.ES_CAN.value),
        (DialectCode.ES_AND.value, DialectCode.ES_CAN.value),
    ],
    "seseo_cluster": [
        (DialectCode.ES_CAN.value, DialectCode.ES_CAR.value),
        (DialectCode.ES_CAN.value, DialectCode.ES_MEX.value),
    ],
    "southern_cone": [
        (DialectCode.ES_RIO.value, DialectCode.ES_CHI.value),
    ],
}


class EvolutionExperiment(Experiment):
    experiment_id = "exp6_evolution"
    name = "Dialectal Evolution (Phylogenetic Analysis)"
    description = (
        "Compare eigenvectors across all dialect variety pairs via cosine "
        "similarity, identify shared and unique spectral axes, and build a "
        "phylogenetic-like tree from eigenvector similarities."
    )
    dependencies = [
        "eigendialectos.spectral.transformation",
        "eigendialectos.spectral.eigendecomposition",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._embeddings: dict[DialectCode, EmbeddingMatrix] = {}
        self._eigendecomps: dict[DialectCode, EigenDecomposition] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 50)
        vocab_size = config.get("vocab_size", 200)
        rng = np.random.default_rng(seed)

        vocab = [f"w{i}" for i in range(vocab_size)]
        base = rng.standard_normal((dim, vocab_size)).astype(np.float64)

        # Create dialect embeddings with structure that reflects known clusters
        # Iberian dialects share a direction
        iberian_direction = rng.standard_normal((dim, vocab_size))
        iberian_direction /= np.linalg.norm(iberian_direction)
        # American dialects share a different direction
        american_direction = rng.standard_normal((dim, vocab_size))
        american_direction /= np.linalg.norm(american_direction)

        _cluster_map = {
            DialectCode.ES_PEN: ("iberian", 0.0),
            DialectCode.ES_AND: ("iberian", 0.15),
            DialectCode.ES_CAN: ("iberian", 0.12),
            DialectCode.ES_RIO: ("american", 0.20),
            DialectCode.ES_MEX: ("american", 0.18),
            DialectCode.ES_CAR: ("american", 0.22),
            DialectCode.ES_CHI: ("american", 0.25),
            DialectCode.ES_AND_BO: ("american", 0.16),
        }

        for code in DialectCode:
            cluster, scale = _cluster_map[code]
            if code == DialectCode.ES_PEN:
                data = base.copy()
            else:
                shared = iberian_direction if cluster == "iberian" else american_direction
                individual_noise = rng.standard_normal((dim, vocab_size)) * 0.05
                data = base + shared * scale + individual_noise
            self._embeddings[code] = EmbeddingMatrix(
                data=data, vocab=vocab, dialect_code=code,
            )

        from eigendialectos.spectral.transformation import compute_transformation_matrix
        from eigendialectos.spectral.eigendecomposition import eigendecompose

        ref = self._embeddings[DialectCode.ES_PEN]
        for code, emb in self._embeddings.items():
            W = compute_transformation_matrix(
                source=ref, target=emb, method="lstsq",
                regularization=config.get("regularization", 0.01),
            )
            self._eigendecomps[code] = eigendecompose(W)

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        codes = sorted(self._eigendecomps.keys(), key=lambda c: c.value)
        n = len(codes)
        k_top = self._config.get("k_top", 5)  # number of top eigenvectors to compare

        # 1. Pairwise eigenvector similarity matrix
        sim_matrix = np.zeros((n, n))
        shared_axes: dict[str, list] = {}

        for i, ci in enumerate(codes):
            for j, cj in enumerate(codes):
                if i <= j:
                    sim, shared = self._eigenvector_similarity(
                        self._eigendecomps[ci],
                        self._eigendecomps[cj],
                        k_top=k_top,
                    )
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
                    if i != j and shared:
                        pair_key = f"{ci.value}|{cj.value}"
                        shared_axes[pair_key] = shared

        # 2. Identify unique axes per dialect
        unique_axes: dict[str, int] = {}
        for i, code in enumerate(codes):
            # An axis is unique if it has low similarity with all other dialects
            max_sims = []
            for j, other in enumerate(codes):
                if i != j:
                    max_sims.append(sim_matrix[i, j])
            avg_sim = float(np.mean(max_sims)) if max_sims else 1.0
            unique_count = max(0, int(k_top * (1.0 - avg_sim)))
            unique_axes[code.value] = unique_count

        # 3. Build distance from similarity for phylogenetic tree
        distance_from_sim = 1.0 - sim_matrix
        np.fill_diagonal(distance_from_sim, 0.0)

        metrics: dict = {
            "similarity_matrix": sim_matrix.tolist(),
            "distance_matrix": distance_from_sim.tolist(),
            "dialect_order": [c.value for c in codes],
            "shared_axes_summary": {
                k: len(v) for k, v in shared_axes.items()
            },
            "unique_axes": unique_axes,
            "k_top": k_top,
        }
        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        sim = np.array(result.metrics["similarity_matrix"])
        order = [DialectCode(c) for c in result.metrics["dialect_order"]]
        idx = {c.value: i for i, c in enumerate(order)}

        # Check whether known historical clusters show higher intra-similarity
        cluster_scores: dict[str, float] = {}
        for cluster_name, pairs in _HISTORICAL_RELATIONS.items():
            sims = []
            for a, b in pairs:
                if a in idx and b in idx:
                    sims.append(sim[idx[a], idx[b]])
            if sims:
                cluster_scores[cluster_name] = float(np.mean(sims))

        # Inter-cluster similarity (pairs not in any known cluster)
        known_pairs_set = set()
        for pairs in _HISTORICAL_RELATIONS.values():
            for a, b in pairs:
                known_pairs_set.add((a, b))
                known_pairs_set.add((b, a))

        inter_sims = []
        for i, ci in enumerate(order):
            for j, cj in enumerate(order):
                if i < j and (ci.value, cj.value) not in known_pairs_set:
                    inter_sims.append(sim[i, j])

        mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
        mean_intra = float(np.mean(list(cluster_scores.values()))) if cluster_scores else 0.0

        return {
            "cluster_similarities": cluster_scores,
            "mean_intra_cluster_similarity": mean_intra,
            "mean_inter_cluster_similarity": mean_inter,
            "intra_gt_inter": mean_intra > mean_inter,
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

        sim_matrix = np.array(result.metrics["similarity_matrix"])
        dist_matrix = np.array(result.metrics["distance_matrix"])
        order = result.metrics["dialect_order"]

        # --- 1. Similarity heatmap ---
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim_matrix, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        fig.colorbar(im, ax=ax, label="Eigenvector Similarity")
        ax.set_title("Pairwise Eigenvector Similarity")
        plt.tight_layout()
        p = output_dir / "eigenvector_similarity.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # --- 2. Phylogenetic dendrogram ---
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            condensed = dist_matrix[np.triu_indices(len(order), k=1)]
            # Ensure non-negative
            condensed = np.maximum(condensed, 0.0)
            Z = linkage(condensed, method="average")
            fig, ax = plt.subplots(figsize=(10, 6))
            dendrogram(Z, labels=order, ax=ax, leaf_rotation=45)
            ax.set_title("Phylogenetic Tree (UPGMA from Eigenvector Distance)")
            ax.set_ylabel("Distance (1 - similarity)")
            plt.tight_layout()
            p = output_dir / "phylogenetic_tree.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)
        except ImportError:
            logger.warning("scipy not available for dendrogram.")

        return paths

    def report(self, result: ExperimentResult) -> str:
        base = super().report(result)
        lines = [
            base, "",
            "## Evolutionary Narrative", "",
            "The eigenvector similarity matrix reveals the following structure:", "",
        ]

        shared = result.metrics.get("shared_axes_summary", {})
        unique = result.metrics.get("unique_axes", {})

        if shared:
            lines.append("### Shared Spectral Axes")
            lines.append("")
            for pair, count in sorted(shared.items(), key=lambda x: -x[1]):
                a, b = pair.split("|", 1)
                name_a = DIALECT_NAMES.get(DialectCode(a), a)
                name_b = DIALECT_NAMES.get(DialectCode(b), b)
                lines.append(f"- **{name_a}** / **{name_b}**: {count} shared axes")
            lines.append("")

        if unique:
            lines.append("### Unique Variation Axes")
            lines.append("")
            for code, count in sorted(unique.items()):
                name = DIALECT_NAMES.get(DialectCode(code), code)
                lines.append(f"- **{name}** ({code}): ~{count} unique axes")
            lines.append("")

        lines.extend([
            "### Interpretation", "",
            "Shared eigenvectors between dialect pairs indicate common historical "
            "origins -- the spectral axes these dialects share correspond to "
            "phonological and morphosyntactic innovations they underwent together "
            "before diverging.  Unique axes represent post-divergence innovations "
            "specific to each variety.", "",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eigenvector_similarity(
        eigen_a: EigenDecomposition,
        eigen_b: EigenDecomposition,
        k_top: int = 5,
    ) -> tuple[float, list[tuple[int, int, float]]]:
        """Compute similarity between the top-k eigenvectors of two decompositions.

        Returns
        -------
        avg_max_similarity : float
            Average of the maximum cosine similarity each top eigenvector of A
            achieves with any top eigenvector of B.
        shared_pairs : list[tuple[int, int, float]]
            Pairs (idx_a, idx_b, cosine_sim) where similarity > 0.7.
        """
        Pa = eigen_a.eigenvectors.real
        Pb = eigen_b.eigenvectors.real

        # Sort eigenvectors by eigenvalue magnitude
        order_a = np.argsort(np.abs(eigen_a.eigenvalues))[::-1]
        order_b = np.argsort(np.abs(eigen_b.eigenvalues))[::-1]

        k_a = min(k_top, Pa.shape[1])
        k_b = min(k_top, Pb.shape[1])

        top_a = Pa[:, order_a[:k_a]]
        top_b = Pb[:, order_b[:k_b]]

        # Truncate to common dimension
        d = min(top_a.shape[0], top_b.shape[0])
        top_a = top_a[:d, :]
        top_b = top_b[:d, :]

        # Cosine similarity matrix
        norms_a = np.linalg.norm(top_a, axis=0, keepdims=True)
        norms_b = np.linalg.norm(top_b, axis=0, keepdims=True)
        norms_a = np.maximum(norms_a, 1e-12)
        norms_b = np.maximum(norms_b, 1e-12)

        cos_matrix = (top_a / norms_a).T @ (top_b / norms_b)  # (k_a, k_b)
        cos_matrix = np.abs(cos_matrix)  # eigenvectors can be sign-flipped

        # Average max similarity
        max_per_a = np.max(cos_matrix, axis=1)
        avg_max_sim = float(np.mean(max_per_a))

        # Find shared pairs (high similarity)
        shared_pairs: list[tuple[int, int, float]] = []
        threshold = 0.7
        for i in range(cos_matrix.shape[0]):
            for j in range(cos_matrix.shape[1]):
                if cos_matrix[i, j] > threshold:
                    shared_pairs.append((
                        int(order_a[i]),
                        int(order_b[j]),
                        float(cos_matrix[i, j]),
                    ))

        return avg_max_sim, shared_pairs
