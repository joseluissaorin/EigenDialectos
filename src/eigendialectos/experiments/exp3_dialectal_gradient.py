"""Experiment 3: Dialectal Gradient -- alpha sweep.

For each dialect variety, sweep the dialectal intensity parameter alpha from
0.0 to 1.5 and measure a dialect-classifier-confidence proxy at each step.
Identify the *recognition threshold* (alpha where confidence > 0.5) and the
*naturalness threshold* (alpha where embedding norms start diverging).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from eigendialectos.constants import ALPHA_RANGE, DialectCode, DIALECT_NAMES
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import (
    EigenDecomposition,
    EmbeddingMatrix,
    ExperimentResult,
    TransformationMatrix,
)

logger = logging.getLogger(__name__)


class DialectalGradientExperiment(Experiment):
    experiment_id = "exp3_dialectal_gradient"
    name = "Dialectal Gradient (alpha sweep)"
    description = (
        "Sweep dialectal intensity alpha from 0.0 to 1.5 for every dialect "
        "variety, measure a dialect-classifier confidence proxy at each step, "
        "and identify recognition and naturalness thresholds."
    )
    dependencies = [
        "eigendialectos.spectral.transformation",
        "eigendialectos.spectral.eigendecomposition",
        "eigendialectos.generative.dial",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._embeddings: dict[DialectCode, EmbeddingMatrix] = {}
        self._eigendecomps: dict[DialectCode, EigenDecomposition] = {}
        self._neutral_embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 50)
        vocab_size = config.get("vocab_size", 200)
        n_sentences = config.get("n_sentences", 20)
        rng = np.random.default_rng(seed)

        vocab = [f"w{i}" for i in range(vocab_size)]
        base = rng.standard_normal((dim, vocab_size)).astype(np.float64)

        # Build dialect embeddings with controlled perturbation
        dialect_directions: dict[DialectCode, np.ndarray] = {}
        for code in DialectCode:
            if code == DialectCode.ES_PEN:
                self._embeddings[code] = EmbeddingMatrix(
                    data=base.copy(), vocab=vocab, dialect_code=code,
                )
                dialect_directions[code] = np.zeros((dim, vocab_size))
            else:
                direction = rng.standard_normal((dim, vocab_size))
                direction /= np.linalg.norm(direction)
                scale = 0.15 + 0.15 * rng.random()
                dialect_directions[code] = direction * scale
                self._embeddings[code] = EmbeddingMatrix(
                    data=base + dialect_directions[code],
                    vocab=vocab,
                    dialect_code=code,
                )

        # Neutral input
        self._neutral_embeddings = rng.standard_normal(
            (n_sentences, dim)
        ).astype(np.float64)

        # Eigendecompose
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

        from eigendialectos.generative.dial import dial_transform_embedding

        alpha_start, alpha_stop, alpha_step = (
            self._config.get("alpha_start", ALPHA_RANGE[0]),
            self._config.get("alpha_stop", ALPHA_RANGE[1]),
            self._config.get("alpha_step", ALPHA_RANGE[2]),
        )
        alphas = np.arange(alpha_start, alpha_stop + 1e-9, alpha_step)
        alphas = np.round(alphas, 4)

        curves: dict[str, dict] = {}

        for code, eigen in self._eigendecomps.items():
            confidences: list[float] = []
            norms: list[float] = []

            for alpha in alphas:
                transformed = dial_transform_embedding(
                    self._neutral_embeddings, eigen, alpha=float(alpha)
                )
                conf = self._classifier_confidence_proxy(
                    self._neutral_embeddings, transformed, float(alpha)
                )
                mean_norm = float(np.mean(np.linalg.norm(transformed, axis=1)))
                confidences.append(conf)
                norms.append(mean_norm)

            # Find recognition threshold: first alpha where confidence > 0.5
            recog_threshold = None
            for a, c in zip(alphas, confidences):
                if c > 0.5:
                    recog_threshold = float(a)
                    break

            # Find naturalness threshold: alpha where norm diverges by >50%
            # from the alpha=0 norm
            base_norm = norms[0] if norms else 1.0
            natural_threshold = None
            for a, n in zip(alphas, norms):
                if base_norm > 0 and abs(n - base_norm) / base_norm > 0.5:
                    natural_threshold = float(a)
                    break

            curves[code.value] = {
                "alphas": alphas.tolist(),
                "confidences": confidences,
                "norms": norms,
                "recognition_threshold": recog_threshold,
                "naturalness_threshold": natural_threshold,
            }

        metrics: dict = {
            "curves": curves,
            "alphas": alphas.tolist(),
        }
        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        curves = result.metrics.get("curves", {})
        recog = {}
        natural = {}
        for code, data in curves.items():
            recog[code] = data.get("recognition_threshold")
            natural[code] = data.get("naturalness_threshold")

        recog_vals = [v for v in recog.values() if v is not None]
        natural_vals = [v for v in natural.values() if v is not None]

        return {
            "recognition_thresholds": recog,
            "naturalness_thresholds": natural,
            "mean_recognition_threshold": (
                float(np.mean(recog_vals)) if recog_vals else None
            ),
            "mean_naturalness_threshold": (
                float(np.mean(natural_vals)) if natural_vals else None
            ),
            "n_dialects_with_recognition": len(recog_vals),
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

        curves = result.metrics.get("curves", {})
        if not curves:
            return paths

        # --- Alpha vs confidence curves ---
        fig, ax = plt.subplots(figsize=(12, 6))
        for code in sorted(curves.keys()):
            data = curves[code]
            ax.plot(data["alphas"], data["confidences"], label=code, linewidth=1.5)
            # Annotate recognition threshold
            rt = data.get("recognition_threshold")
            if rt is not None:
                idx = min(
                    range(len(data["alphas"])),
                    key=lambda i: abs(data["alphas"][i] - rt),
                )
                ax.annotate(
                    f"{rt:.1f}",
                    xy=(rt, data["confidences"][idx]),
                    fontsize=7,
                    textcoords="offset points",
                    xytext=(5, 5),
                )

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.6, label="Confidence=0.5")
        ax.set_xlabel("alpha (dialectal intensity)")
        ax.set_ylabel("Classifier Confidence (proxy)")
        ax.set_title("Dialectal Gradient: alpha vs Classifier Confidence")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        p = output_dir / "gradient_curves.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        return paths

    def report(self, result: ExperimentResult) -> str:
        base = super().report(result)
        curves = result.metrics.get("curves", {})
        lines = [base, "", "## Thresholds", ""]
        for code in sorted(curves.keys()):
            data = curves[code]
            name = DIALECT_NAMES.get(DialectCode(code), code)
            rt = data.get("recognition_threshold", "N/A")
            nt = data.get("naturalness_threshold", "N/A")
            lines.append(
                f"- **{name}**: recognition alpha={rt}, naturalness alpha={nt}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Proxy classifier
    # ------------------------------------------------------------------

    @staticmethod
    def _classifier_confidence_proxy(
        neutral: np.ndarray,
        transformed: np.ndarray,
        alpha: float,
    ) -> float:
        """Proxy for dialect classifier confidence.

        Uses the cosine distance between neutral and transformed embeddings
        as a proxy for how "dialectal" the output is.  Maps through a
        sigmoid so the output is in [0, 1].

        At alpha=0, transformed == neutral => distance ~0 => confidence ~0.
        As alpha grows, distance grows => confidence approaches 1.
        """
        if alpha == 0.0:
            return 0.0

        delta = transformed - neutral
        mean_dist = float(np.mean(np.linalg.norm(delta, axis=1)))
        neutral_norm = float(np.mean(np.linalg.norm(neutral, axis=1)))
        if neutral_norm < 1e-12:
            return 0.0

        # Normalise distance by neutral norm to make scale-invariant
        relative_dist = mean_dist / neutral_norm

        # Sigmoid mapping: confidence = 1 / (1 + exp(-k*(x - x0)))
        # Tuned so that relative_dist ~ 0.3 gives confidence ~ 0.5
        k = 10.0
        x0 = 0.3
        confidence = 1.0 / (1.0 + np.exp(-k * (relative_dist - x0)))
        return float(confidence)
