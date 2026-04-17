"""Experiment 1: Spectral Map of Spanish Dialect Varieties.

Compute transformation matrices W_i for all 8 dialect varieties relative to
a reference (Peninsular Standard), eigendecompose them, compute spectra, and
build a pairwise distance matrix.  The distance matrix is compared against
known dialectological groupings.
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
    DialectalSpectrum,
)

logger = logging.getLogger(__name__)

# Known high-level dialectological groupings for evaluation.
# Families: Iberian (PEN, AND, CAN), Rioplatense (RIO), Mexican (MEX),
#           Caribbean (CAR), Chilean (CHI), Andean (AND_BO).
# Closer pairs within families are expected to have smaller spectral distance.
_KNOWN_CLOSE_PAIRS: list[tuple[DialectCode, DialectCode]] = [
    (DialectCode.ES_PEN, DialectCode.ES_AND),
    (DialectCode.ES_PEN, DialectCode.ES_CAN),
    (DialectCode.ES_AND, DialectCode.ES_CAN),
    (DialectCode.ES_CAR, DialectCode.ES_AND),  # seseo link
]

_KNOWN_DISTANT_PAIRS: list[tuple[DialectCode, DialectCode]] = [
    (DialectCode.ES_PEN, DialectCode.ES_RIO),
    (DialectCode.ES_PEN, DialectCode.ES_CHI),
    (DialectCode.ES_AND, DialectCode.ES_MEX),
]


class SpectralMapExperiment(Experiment):
    experiment_id = "exp1_spectral_map"
    name = "Spectral Map of Spanish Dialect Varieties"
    description = (
        "Compute the eigenspectrum of every dialect transformation matrix "
        "W_i (relative to Peninsular Standard), build a pairwise spectral "
        "distance matrix, and compare it with known dialectological "
        "classifications."
    )
    dependencies = [
        "eigendialectos.spectral.transformation",
        "eigendialectos.spectral.eigendecomposition",
        "eigendialectos.spectral.eigenspectrum",
        "eigendialectos.spectral.entropy",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._embeddings: dict[DialectCode, EmbeddingMatrix] = {}
        self._transforms: dict[DialectCode, TransformationMatrix] = {}
        self._eigendecomps: dict[DialectCode, EigenDecomposition] = {}
        self._spectra: dict[DialectCode, DialectalSpectrum] = {}
        self._distance_matrix: np.ndarray | None = None
        self._dialect_order: list[DialectCode] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 50)
        vocab_size = config.get("vocab_size", 200)
        rng = np.random.default_rng(seed)

        # Generate synthetic embeddings when no real data dir is provided
        data_dir = config.get("data_dir")
        if data_dir and Path(data_dir).exists() and any(Path(data_dir).iterdir()):
            logger.info("Loading embeddings from %s", data_dir)
            self._load_embeddings_from_dir(Path(data_dir), dim)
        else:
            logger.info(
                "No data directory found; generating synthetic embeddings "
                "(dim=%d, vocab=%d).",
                dim,
                vocab_size,
            )
            self._generate_synthetic_embeddings(rng, dim, vocab_size)

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        from eigendialectos.spectral.transformation import compute_transformation_matrix
        from eigendialectos.spectral.eigendecomposition import eigendecompose
        from eigendialectos.spectral.eigenspectrum import compute_eigenspectrum

        reference = DialectCode.ES_PEN
        ref_emb = self._embeddings[reference]

        # 1. Compute W_i for every variety (including identity for reference)
        for code, emb in self._embeddings.items():
            W = compute_transformation_matrix(
                source=ref_emb,
                target=emb,
                method="lstsq",
                regularization=self._config.get("regularization", 0.01),
            )
            self._transforms[code] = W

        # 2. Eigendecompose each W_i
        for code, W in self._transforms.items():
            eigen = eigendecompose(W)
            self._eigendecomps[code] = eigen

        # 3. Compute spectra
        for code, eigen in self._eigendecomps.items():
            self._spectra[code] = compute_eigenspectrum(eigen)

        # 4. Pairwise distance matrix
        self._dialect_order = sorted(self._spectra.keys(), key=lambda c: c.value)
        n = len(self._dialect_order)
        dist = np.zeros((n, n), dtype=np.float64)

        for i, ci in enumerate(self._dialect_order):
            for j, cj in enumerate(self._dialect_order):
                if i < j:
                    d_ij = self._spectral_distance(
                        self._spectra[ci], self._spectra[cj]
                    )
                    dist[i, j] = d_ij
                    dist[j, i] = d_ij

        self._distance_matrix = dist

        # Pack metrics
        entropies = {
            code.value: float(spec.entropy)
            for code, spec in self._spectra.items()
        }
        metrics: dict = {
            "entropies": entropies,
            "distance_matrix": dist.tolist(),
            "dialect_order": [c.value for c in self._dialect_order],
            "mean_distance": float(np.mean(dist[np.triu_indices(n, k=1)])),
            "max_distance": float(np.max(dist)),
        }

        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        """Check whether close pairs have smaller distance than distant pairs."""
        dist = np.array(result.metrics["distance_matrix"])
        order = [DialectCode(c) for c in result.metrics["dialect_order"]]
        idx = {c: i for i, c in enumerate(order)}

        close_dists = []
        for a, b in _KNOWN_CLOSE_PAIRS:
            if a in idx and b in idx:
                close_dists.append(dist[idx[a], idx[b]])

        far_dists = []
        for a, b in _KNOWN_DISTANT_PAIRS:
            if a in idx and b in idx:
                far_dists.append(dist[idx[a], idx[b]])

        mean_close = float(np.mean(close_dists)) if close_dists else 0.0
        mean_far = float(np.mean(far_dists)) if far_dists else 0.0

        # Success criterion: mean close distance < mean far distance
        ordering_correct = mean_close < mean_far if (close_dists and far_dists) else False

        return {
            "mean_close_distance": mean_close,
            "mean_far_distance": mean_far,
            "ordering_correct": ordering_correct,
            "n_close_pairs": len(close_dists),
            "n_far_pairs": len(far_dists),
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
        entropies = result.metrics["entropies"]

        # --- 1. Eigenspectrum bar chart ---
        fig, ax = plt.subplots(figsize=(10, 5))
        codes = sorted(entropies.keys())
        vals = [entropies[c] for c in codes]
        ax.bar(range(len(codes)), vals, tick_label=codes)
        ax.set_ylabel("Spectral Entropy")
        ax.set_title("Dialectal Spectral Entropy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        p = output_dir / "eigenspectrum_bars.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # --- 2. Distance heatmap ---
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(dist, cmap="YlOrRd")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        fig.colorbar(im, ax=ax, label="Spectral Distance")
        ax.set_title("Pairwise Spectral Distance")
        plt.tight_layout()
        p = output_dir / "distance_heatmap.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # --- 3. Dendrogram ---
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            condensed = dist[np.triu_indices(len(order), k=1)]
            Z = linkage(condensed, method="ward")
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(Z, labels=order, ax=ax, leaf_rotation=45)
            ax.set_title("Dialectal Dendrogram (Ward)")
            ax.set_ylabel("Distance")
            plt.tight_layout()
            p = output_dir / "dendrogram.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)
        except ImportError:
            logger.warning("scipy not available for dendrogram.")

        return paths

    def report(self, result: ExperimentResult) -> str:
        base = super().report(result)
        entropies = result.metrics.get("entropies", {})
        lines = [
            base,
            "",
            "## Spectral Entropy by Dialect",
            "",
        ]
        for code in sorted(entropies.keys()):
            name = DIALECT_NAMES.get(DialectCode(code), code)
            lines.append(f"- **{name}** ({code}): H = {entropies[code]:.4f}")
        lines.append("")
        lines.append(
            f"Mean pairwise spectral distance: "
            f"{result.metrics.get('mean_distance', 'N/A')}"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_synthetic_embeddings(
        self,
        rng: np.random.Generator,
        dim: int,
        vocab_size: int,
    ) -> None:
        """Build synthetic embeddings with controlled inter-dialect structure."""
        vocab = [f"w{i}" for i in range(vocab_size)]

        # Base embedding (Peninsular)
        base = rng.standard_normal((dim, vocab_size)).astype(np.float64)

        for code in DialectCode:
            if code == DialectCode.ES_PEN:
                data = base.copy()
            else:
                # Add dialect-specific perturbation scaled by a "distance" factor
                noise_scale = 0.1 + 0.1 * rng.random()
                perturbation = rng.standard_normal((dim, vocab_size)) * noise_scale
                data = base + perturbation
            self._embeddings[code] = EmbeddingMatrix(
                data=data, vocab=vocab, dialect_code=code
            )

    def _load_embeddings_from_dir(self, data_dir: Path, dim: int) -> None:
        """Attempt to load .npy embedding files from *data_dir*."""
        vocab: list[str] = []
        for code in DialectCode:
            path = data_dir / f"{code.value}.npy"
            if path.exists():
                data = np.load(str(path)).astype(np.float64)
                if not vocab:
                    vocab = [f"w{i}" for i in range(data.shape[1])]
                self._embeddings[code] = EmbeddingMatrix(
                    data=data, vocab=vocab, dialect_code=code
                )
            else:
                logger.warning("Missing embedding file %s; skipping.", path)

    @staticmethod
    def _spectral_distance(
        a: DialectalSpectrum, b: DialectalSpectrum
    ) -> float:
        """L2 distance between sorted eigenvalue vectors (padded to same length)."""
        ev_a = a.eigenvalues_sorted
        ev_b = b.eigenvalues_sorted
        max_len = max(len(ev_a), len(ev_b))
        va = np.zeros(max_len)
        vb = np.zeros(max_len)
        va[: len(ev_a)] = ev_a
        vb[: len(ev_b)] = ev_b
        return float(np.linalg.norm(va - vb))
