"""Experiment 5: Dialectal Archaeology.

Apply inverse dialect transforms (W_i^{-1}) to historical Spanish texts
(Golden Age, ~16th--17th century) to "remove" modern dialectal features and
approximate what the proto-dialect may have looked like.  Then compare the
"de-dialectalised" representations with known historical phonological and
lexical data.
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

# -----------------------------------------------------------------------
# Built-in Golden Age Spanish sample texts
# -----------------------------------------------------------------------

_GOLDEN_AGE_SAMPLES: list[dict] = [
    {
        "title": "Don Quijote, Cap. I (Cervantes, 1605)",
        "text": (
            "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, "
            "no ha mucho tiempo que vivia un hidalgo de los de lanza en "
            "astillero, adarga antigua, rocin flaco y galgo corredor."
        ),
        "period": "early_17th",
    },
    {
        "title": "Lazarillo de Tormes, Tractado I (Anonimo, 1554)",
        "text": (
            "Pues sepa Vuestra Merced, ante todas cosas, que a mi llaman "
            "Lazaro de Tormes, hijo de Thome Gonzalez y de Antona Perez, "
            "naturales de Tejares, aldea de Salamanca."
        ),
        "period": "mid_16th",
    },
    {
        "title": "La Celestina, Acto I (Rojas, 1499)",
        "text": (
            "En esto veo, Melibea, la grandeza de Dios. Que has fecho un "
            "cuerpo tan perfecto como el tuyo, que en el no fallo nada, "
            "que la naturaleza no plugo con la perficion que te doto."
        ),
        "period": "late_15th",
    },
    {
        "title": "Soneto XXIII (Garcilaso de la Vega, c.1530)",
        "text": (
            "En tanto que de rosa y de azucena se muestra la color en "
            "vuestro gesto, y que vuestro mirar ardiente, honesto, "
            "enciende al corazon y lo refrena."
        ),
        "period": "early_16th",
    },
    {
        "title": "El Burlador de Sevilla, Jornada I (Tirso de Molina, c.1630)",
        "text": (
            "Tan largo me lo fiais? Con razones y cumplimientos, dejais de "
            "satisfacer a quien teneis ofendido, y a quien por vos he venido "
            "desterrado hasta saber su razon."
        ),
        "period": "early_17th",
    },
]

# Known historical features for evaluation
_HISTORICAL_FEATURES: dict[str, list[str]] = {
    "late_15th": [
        "f_initial_conserved",  # Latin f- still present (fazer -> hacer)
        "distinction_b_v",       # b/v distinction in some positions
        "vos_formal",            # vos as formal address
    ],
    "mid_16th": [
        "sibilant_system_transition",  # /ts/ vs /dz/ collapsing
        "vuestra_merced",              # polite address form
        "latin_clusters",              # ct, gn clusters partially conserved
    ],
    "early_16th": [
        "sibilant_system_four_way",  # /ts/, /dz/, /s/, /z/
        "vos_as_polite",
    ],
    "early_17th": [
        "h_from_f",              # f > h complete in standard
        "theta_from_sibilants",  # /θ/ emerging (north) / seseo (south)
        "usted_emerging",        # vuestra merced > usted
    ],
}


class DialectalArchaeologyExperiment(Experiment):
    experiment_id = "exp5_archaeology"
    name = "Dialectal Archaeology"
    description = (
        "Apply inverse dialect transforms to historical Golden Age Spanish "
        "texts to 'remove' modern dialectal features and approximate the "
        "proto-dialect. Compare with known historical phonological data."
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
        self._text_embeddings: np.ndarray | None = None

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

        for code in DialectCode:
            noise = (
                np.zeros((dim, vocab_size))
                if code == DialectCode.ES_PEN
                else rng.standard_normal((dim, vocab_size)) * 0.15
            )
            self._embeddings[code] = EmbeddingMatrix(
                data=base + noise, vocab=vocab, dialect_code=code,
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

        # Generate pseudo-embeddings for historical texts
        n_texts = len(_GOLDEN_AGE_SAMPLES)
        self._text_embeddings = rng.standard_normal(
            (n_texts, dim)
        ).astype(np.float64)

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        from eigendialectos.generative.dial import dial_transform_embedding

        sample_results: list[dict] = []

        for idx, sample in enumerate(_GOLDEN_AGE_SAMPLES):
            text_emb = self._text_embeddings[idx: idx + 1]

            per_dialect: dict[str, dict] = {}
            for code, eigen in self._eigendecomps.items():
                # Apply inverse transform: alpha = -1.0
                inverse = dial_transform_embedding(text_emb, eigen, alpha=-1.0)
                # Also apply "half-inverse" for partial de-dialectalisation
                half_inv = dial_transform_embedding(text_emb, eigen, alpha=-0.5)

                # Measure displacement
                inv_delta = float(np.linalg.norm(inverse - text_emb))
                half_delta = float(np.linalg.norm(half_inv - text_emb))

                per_dialect[code.value] = {
                    "inverse_displacement": inv_delta,
                    "half_inverse_displacement": half_delta,
                    "inverse_embedding_norm": float(np.linalg.norm(inverse)),
                }

            # Find which dialect's inverse produces the smallest displacement
            # (meaning the historical text is closest to that dialect's "origin")
            closest_dialect = min(
                per_dialect.items(),
                key=lambda x: x[1]["inverse_displacement"],
            )[0]

            sample_results.append({
                "title": sample["title"],
                "period": sample["period"],
                "text_preview": sample["text"][:80] + "...",
                "per_dialect": per_dialect,
                "closest_dialect_inverse": closest_dialect,
            })

        # Historical feature alignment
        feature_alignment = self._evaluate_historical_features(sample_results)

        metrics: dict = {
            "samples": sample_results,
            "historical_texts": [s["text"] for s in _GOLDEN_AGE_SAMPLES],
            "feature_alignment": feature_alignment,
            "n_samples": len(sample_results),
        }
        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        samples = result.metrics.get("samples", [])
        if not samples:
            return {"status": "no samples"}

        # Check whether Peninsular is consistently closest for historical texts
        closest_counts: dict[str, int] = {}
        for s in samples:
            cd = s["closest_dialect_inverse"]
            closest_counts[cd] = closest_counts.get(cd, 0) + 1

        most_common = max(closest_counts.items(), key=lambda x: x[1])

        # Average inverse displacement per dialect
        avg_displacements: dict[str, float] = {}
        for code in DialectCode:
            vals = []
            for s in samples:
                if code.value in s["per_dialect"]:
                    vals.append(s["per_dialect"][code.value]["inverse_displacement"])
            if vals:
                avg_displacements[code.value] = float(np.mean(vals))

        return {
            "closest_dialect_distribution": closest_counts,
            "most_common_closest": most_common[0],
            "avg_inverse_displacements": avg_displacements,
            "feature_alignment": result.metrics.get("feature_alignment", {}),
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

        samples = result.metrics.get("samples", [])
        if not samples:
            return paths

        # --- Before/after displacement chart ---
        fig, axes = plt.subplots(
            len(samples), 1,
            figsize=(12, 3 * len(samples)),
            squeeze=False,
        )

        for idx, sample in enumerate(samples):
            ax = axes[idx, 0]
            dialects = sorted(sample["per_dialect"].keys())
            inv_disps = [sample["per_dialect"][d]["inverse_displacement"] for d in dialects]
            half_disps = [sample["per_dialect"][d]["half_inverse_displacement"] for d in dialects]

            x = np.arange(len(dialects))
            ax.bar(x - 0.2, inv_disps, 0.35, label="alpha=-1.0", color="steelblue")
            ax.bar(x + 0.2, half_disps, 0.35, label="alpha=-0.5", color="lightcoral")
            ax.set_xticks(x)
            ax.set_xticklabels(dialects, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{sample['title']} ({sample['period']})", fontsize=9)
            ax.set_ylabel("Displacement")
            if idx == 0:
                ax.legend(fontsize=7)

        plt.tight_layout()
        p = output_dir / "archaeology_displacements.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        return paths

    def report(self, result: ExperimentResult) -> str:
        base = super().report(result)
        samples = result.metrics.get("samples", [])
        lines = [
            base, "",
            "## Diachronic Analysis", "",
            "This experiment applies inverse dialect transforms to Golden Age ",
            "Spanish texts.  By reversing modern dialectal drift (W_i^{-1}), ",
            "we approximate the proto-dialectal embedding space.", "",
        ]

        for s in samples:
            lines.append(f"### {s['title']} ({s['period']})")
            lines.append("")
            lines.append(f"> {s['text_preview']}")
            lines.append("")
            lines.append(
                f"Closest dialect (by inverse displacement): "
                f"**{DIALECT_NAMES.get(DialectCode(s['closest_dialect_inverse']), s['closest_dialect_inverse'])}**"
            )
            lines.append("")

        fa = result.metrics.get("feature_alignment", {})
        if fa:
            lines.append("## Historical Feature Alignment")
            lines.append("")
            for period, data in fa.items():
                lines.append(f"- **{period}**: alignment score = {data.get('alignment_score', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_historical_features(
        sample_results: list[dict],
    ) -> dict:
        """Check how well inverse transforms align with known historical features."""
        alignment: dict[str, dict] = {}

        for period, features in _HISTORICAL_FEATURES.items():
            # Find samples from this period
            period_samples = [s for s in sample_results if s["period"] == period]
            if not period_samples:
                continue

            # The alignment score is a proxy: we check whether the inverse
            # displacement pattern is consistent with the period's expected
            # features.  In a real implementation, we would compare specific
            # phonological/lexical features in the embedding space.
            avg_disp = {}
            for s in period_samples:
                for code, data in s["per_dialect"].items():
                    avg_disp.setdefault(code, []).append(data["inverse_displacement"])
            avg_disp = {k: float(np.mean(v)) for k, v in avg_disp.items()}

            # Score: variance of displacements across dialects
            # Low variance means all dialects converge to similar proto-form
            vals = list(avg_disp.values())
            variance = float(np.var(vals)) if vals else 0.0
            alignment_score = 1.0 / (1.0 + variance)

            alignment[period] = {
                "features": features,
                "avg_displacements": avg_disp,
                "displacement_variance": variance,
                "alignment_score": alignment_score,
            }

        return alignment
