"""Experiment 4: Impossible Dialects.

Create mixed transforms that combine contradictory dialectal features and
analyse their coherence.  Specific impossible combinations:

- **voseo + vosotros**: The Rioplatense second-person singular "vos" cannot
  co-exist with the Peninsular second-person plural "vosotros" -- they
  occupy the same morphosyntactic slot.
- **seseo + ceceo**: Seseo (s=c/z) is incompatible with ceceo (c/z=s);
  both cannot hold simultaneously.
- **yeismo + sheismo**: Argentinian "sheismo" (ll/y -> sh) presupposes
  "yeismo" (ll=y), but combining with distinction (ll != y) is
  contradictory.
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
# Feature definitions for impossible combinations
# -----------------------------------------------------------------------

_IMPOSSIBLE_COMBOS: list[dict] = [
    {
        "name": "voseo + vosotros",
        "description": (
            "Rioplatense voseo (vos hablás) combined with Peninsular "
            "vosotros (vosotros habláis). These occupy the same "
            "morphosyntactic paradigm slot and cannot co-occur."
        ),
        "dialects": (DialectCode.ES_RIO, DialectCode.ES_PEN),
        "weights": (0.5, 0.5),
        "feature_category": "MORPHOSYNTACTIC",
    },
    {
        "name": "seseo + ceceo",
        "description": (
            "Caribbean/Mexican seseo (/s/ for /θ/) combined with Andalusian "
            "ceceo (/θ/ for /s/). These are opposite phonological mappings "
            "and cannot hold simultaneously."
        ),
        "dialects": (DialectCode.ES_MEX, DialectCode.ES_AND),
        "weights": (0.5, 0.5),
        "feature_category": "PHONOLOGICAL",
    },
    {
        "name": "sheismo + distincion",
        "description": (
            "Rioplatense sheismo (ll/y -> [ʃ]) combined with Andean "
            "distinction (ll != y). Sheismo presupposes yeismo, which is "
            "contradicted by maintaining the distinction."
        ),
        "dialects": (DialectCode.ES_RIO, DialectCode.ES_AND_BO),
        "weights": (0.5, 0.5),
        "feature_category": "PHONOLOGICAL",
    },
    {
        "name": "aspiracion + conservacion de s",
        "description": (
            "Andalusian/Caribbean aspiration of syllable-final /s/ combined "
            "with Andean/Mexican conservation of /s/. These are mutually "
            "exclusive treatments of the same phoneme."
        ),
        "dialects": (DialectCode.ES_CAR, DialectCode.ES_AND_BO),
        "weights": (0.5, 0.5),
        "feature_category": "PHONOLOGICAL",
    },
]


class ImpossibleDialectsExperiment(Experiment):
    experiment_id = "exp4_impossible_dialects"
    name = "Impossible Dialects"
    description = (
        "Create mixed dialect transforms that combine contradictory features "
        "(voseo+vosotros, seseo+ceceo, etc.), check their coherence, and "
        "analyse why these combinations are linguistically impossible."
    )
    dependencies = [
        "eigendialectos.spectral.transformation",
        "eigendialectos.spectral.eigendecomposition",
        "eigendialectos.generative.mixing",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._embeddings: dict[DialectCode, EmbeddingMatrix] = {}
        self._transforms: dict[DialectCode, TransformationMatrix] = {}
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

        for code in DialectCode:
            noise = rng.standard_normal((dim, vocab_size)) * (0.0 if code == DialectCode.ES_PEN else 0.15)
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
            self._transforms[code] = W
            self._eigendecomps[code] = eigendecompose(W)

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        from eigendialectos.generative.mixing import mix_dialects

        combo_results: list[dict] = []

        for combo in _IMPOSSIBLE_COMBOS:
            d1, d2 = combo["dialects"]
            w1, w2 = combo["weights"]

            if d1 not in self._transforms or d2 not in self._transforms:
                logger.warning(
                    "Missing transforms for %s or %s; skipping combo %s",
                    d1.value, d2.value, combo["name"],
                )
                continue

            # Mix
            W_mix = mix_dialects([
                (self._transforms[d1], w1),
                (self._transforms[d2], w2),
            ])

            # Coherence analysis
            coherence = self._coherence_check(
                self._transforms[d1],
                self._transforms[d2],
                W_mix,
            )

            # Feature conflict score: based on eigenvalue structure divergence
            conflict_score = self._feature_conflict_score(
                self._eigendecomps[d1],
                self._eigendecomps[d2],
            )

            combo_results.append({
                "name": combo["name"],
                "description": combo["description"],
                "dialect_a": d1.value,
                "dialect_b": d2.value,
                "weights": [w1, w2],
                "feature_category": combo["feature_category"],
                "coherence": coherence,
                "conflict_score": conflict_score,
                "mixed_matrix_norm": float(np.linalg.norm(W_mix.data, "fro")),
            })

        # Build conflict matrix (all pairs)
        codes = sorted(self._eigendecomps.keys(), key=lambda c: c.value)
        n = len(codes)
        conflict_matrix = np.zeros((n, n))
        for i, ci in enumerate(codes):
            for j, cj in enumerate(codes):
                if i < j:
                    score = self._feature_conflict_score(
                        self._eigendecomps[ci], self._eigendecomps[cj]
                    )
                    conflict_matrix[i, j] = score
                    conflict_matrix[j, i] = score

        metrics: dict = {
            "impossible_combinations": combo_results,
            "conflict_matrix": conflict_matrix.tolist(),
            "dialect_order": [c.value for c in codes],
        }
        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        combos = result.metrics.get("impossible_combinations", [])
        if not combos:
            return {"status": "no combinations evaluated"}

        conflict_scores = [c["conflict_score"] for c in combos]
        coherence_scores = [c["coherence"]["score"] for c in combos]

        return {
            "mean_conflict_score": float(np.mean(conflict_scores)),
            "max_conflict_score": float(np.max(conflict_scores)),
            "mean_coherence": float(np.mean(coherence_scores)),
            "n_incoherent": sum(1 for c in coherence_scores if c < 0.5),
            "all_combos_incoherent": all(c < 0.7 for c in coherence_scores),
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

        # --- Feature conflict matrix heatmap ---
        conflict_matrix = np.array(result.metrics.get("conflict_matrix", []))
        order = result.metrics.get("dialect_order", [])

        if conflict_matrix.size > 0:
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(conflict_matrix, cmap="Reds")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order)
            fig.colorbar(im, ax=ax, label="Feature Conflict Score")
            ax.set_title("Pairwise Feature Conflict Matrix")
            plt.tight_layout()
            p = output_dir / "conflict_matrix.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        # --- Bar chart of impossible combination conflict scores ---
        combos = result.metrics.get("impossible_combinations", [])
        if combos:
            fig, ax = plt.subplots(figsize=(10, 5))
            names = [c["name"] for c in combos]
            scores = [c["conflict_score"] for c in combos]
            coherence = [c["coherence"]["score"] for c in combos]

            x = np.arange(len(names))
            ax.bar(x - 0.2, scores, 0.35, label="Conflict Score", color="red", alpha=0.7)
            ax.bar(x + 0.2, coherence, 0.35, label="Coherence", color="blue", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.legend()
            ax.set_title("Impossible Dialect Combinations")
            ax.set_ylabel("Score")
            plt.tight_layout()
            p = output_dir / "impossible_combos.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        return paths

    def report(self, result: ExperimentResult) -> str:
        base = super().report(result)
        combos = result.metrics.get("impossible_combinations", [])
        lines = [base, "", "## Impossible Combinations Analysis", ""]

        for c in combos:
            lines.append(f"### {c['name']}")
            lines.append("")
            lines.append(f"**Dialects:** {c['dialect_a']} + {c['dialect_b']}")
            lines.append(f"**Category:** {c['feature_category']}")
            lines.append(f"**Conflict score:** {c['conflict_score']:.4f}")
            lines.append(f"**Coherence:** {c['coherence']['score']:.4f}")
            lines.append("")
            lines.append(c["description"])
            lines.append("")
            lines.append(f"**Why impossible:** {c['coherence']['interpretation']}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coherence_check(
        W1: TransformationMatrix,
        W2: TransformationMatrix,
        W_mix: TransformationMatrix,
    ) -> dict:
        """Check whether the mixed transform is "coherent".

        A coherent mix should approximately satisfy:
        - Rank is preserved (not collapsed).
        - The mixed matrix is close to the convex hull of the originals.
        - Condition number is reasonable.
        """
        M = W_mix.data
        rank = int(np.linalg.matrix_rank(M, tol=1e-8))
        cond = float(np.linalg.cond(M))
        frob = float(np.linalg.norm(M, "fro"))
        frob1 = float(np.linalg.norm(W1.data, "fro"))
        frob2 = float(np.linalg.norm(W2.data, "fro"))

        # Coherence score: penalise high condition number and rank loss
        max_rank = M.shape[0]
        rank_ratio = rank / max(max_rank, 1)
        cond_penalty = min(1.0, 10.0 / max(cond, 1.0))
        score = rank_ratio * cond_penalty

        # Interpretation
        if score > 0.7:
            interp = (
                "The mixed transform is numerically coherent but may be "
                "linguistically invalid due to contradictory feature expectations."
            )
        elif score > 0.4:
            interp = (
                "Partial coherence loss: the mixture collapses some dimensions, "
                "suggesting the feature spaces are incompatible."
            )
        else:
            interp = (
                "Severe coherence loss: the mixture is near-singular, confirming "
                "that the combined features cannot co-exist in a consistent transform."
            )

        return {
            "score": score,
            "rank": rank,
            "max_rank": max_rank,
            "condition_number": cond,
            "frobenius_norm": frob,
            "interpretation": interp,
        }

    @staticmethod
    def _feature_conflict_score(
        eigen_a: EigenDecomposition,
        eigen_b: EigenDecomposition,
    ) -> float:
        """Quantify feature conflict between two dialect eigendecompositions.

        Uses the angle between the top eigenvectors and the divergence of
        eigenvalue distributions as a proxy for feature incompatibility.
        """
        ev_a = np.abs(eigen_a.eigenvalues)
        ev_b = np.abs(eigen_b.eigenvalues)

        # Pad to same length
        max_len = max(len(ev_a), len(ev_b))
        pa = np.zeros(max_len)
        pb = np.zeros(max_len)
        pa[:len(ev_a)] = np.sort(ev_a)[::-1]
        pb[:len(ev_b)] = np.sort(ev_b)[::-1]

        # KL-like divergence between normalised spectra
        eps = 1e-10
        pa_norm = pa / (np.sum(pa) + eps)
        pb_norm = pb / (np.sum(pb) + eps)
        kl = float(np.sum(pa_norm * np.log((pa_norm + eps) / (pb_norm + eps))))

        # Top eigenvector alignment
        va = eigen_a.eigenvectors[:, 0].real
        vb = eigen_b.eigenvectors[:, 0].real
        n = min(len(va), len(vb))
        va, vb = va[:n], vb[:n]
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a > eps and norm_b > eps:
            cos_sim = abs(float(np.dot(va, vb) / (norm_a * norm_b)))
        else:
            cos_sim = 0.0

        # High KL + low alignment = high conflict
        conflict = kl * (1.0 - cos_sim)
        # Normalise to [0, 1] via sigmoid
        conflict_score = float(1.0 / (1.0 + np.exp(-conflict + 1.0)))
        return conflict_score
