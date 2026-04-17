"""Experiment 7: Zero-shot Dialect Transfer via Tensor Completion.

Build a tensor from all 8 dialect transformation matrices, perform
leave-2-out cross-validation: remove 2 varieties, reconstruct the
tensor via Tucker or CP decomposition on the remaining, and compare
the reconstructed W_i with the true holdout matrices.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import (
    EmbeddingMatrix,
    ExperimentResult,
    TensorDialectal,
    TransformationMatrix,
)

logger = logging.getLogger(__name__)


class ZeroshotTransferExperiment(Experiment):
    experiment_id = "exp7_zeroshot"
    name = "Zero-shot Dialect Transfer (Leave-2-Out Tensor Completion)"
    description = (
        "Build a multi-dialect tensor T from all 8 transformation matrices, "
        "perform leave-2-out reconstruction: hold out 2 varieties, decompose "
        "the remaining tensor via Tucker/CP, reconstruct the held-out "
        "matrices, and measure Frobenius reconstruction error."
    )
    dependencies = [
        "eigendialectos.spectral.transformation",
        "eigendialectos.tensor.construction",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._embeddings: dict[DialectCode, EmbeddingMatrix] = {}
        self._transforms: dict[DialectCode, TransformationMatrix] = {}
        self._full_tensor: TensorDialectal | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 10)  # small for speed in leave-2-out
        vocab_size = config.get("vocab_size", 50)
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

        ref = self._embeddings[DialectCode.ES_PEN]
        for code, emb in self._embeddings.items():
            self._transforms[code] = compute_transformation_matrix(
                source=ref, target=emb, method="lstsq",
                regularization=config.get("regularization", 0.01),
            )

        from eigendialectos.tensor.construction import build_dialect_tensor

        self._full_tensor = build_dialect_tensor(self._transforms)

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        codes = sorted(self._transforms.keys(), key=lambda c: c.value)
        max_holdout_pairs = self._config.get("max_holdout_pairs", 10)
        decomposition_method = self._config.get("decomposition_method", "svd")

        all_pairs = list(itertools.combinations(codes, 2))
        # Limit pairs if there are too many
        rng = np.random.default_rng(self._config.get("seed", 42))
        if len(all_pairs) > max_holdout_pairs:
            indices = rng.choice(len(all_pairs), size=max_holdout_pairs, replace=False)
            pairs_to_test = [all_pairs[i] for i in sorted(indices)]
        else:
            pairs_to_test = all_pairs

        holdout_results: list[dict] = []

        for held_a, held_b in pairs_to_test:
            # Build reduced tensor (exclude held-out)
            remaining = {
                c: self._transforms[c]
                for c in codes if c not in (held_a, held_b)
            }

            # Reconstruct held-out matrices
            reconstructed_a, reconstructed_b = self._reconstruct_holdout(
                remaining, held_a, held_b, codes, decomposition_method
            )

            # Measure error
            true_a = self._transforms[held_a].data
            true_b = self._transforms[held_b].data
            err_a = float(np.linalg.norm(reconstructed_a - true_a, "fro"))
            err_b = float(np.linalg.norm(reconstructed_b - true_b, "fro"))
            norm_a = float(np.linalg.norm(true_a, "fro"))
            norm_b = float(np.linalg.norm(true_b, "fro"))
            rel_a = err_a / max(norm_a, 1e-12)
            rel_b = err_b / max(norm_b, 1e-12)

            holdout_results.append({
                "held_a": held_a.value,
                "held_b": held_b.value,
                "frobenius_error_a": err_a,
                "frobenius_error_b": err_b,
                "relative_error_a": rel_a,
                "relative_error_b": rel_b,
            })

        errors_a = [r["frobenius_error_a"] for r in holdout_results]
        errors_b = [r["frobenius_error_b"] for r in holdout_results]
        all_errors = errors_a + errors_b
        all_rel = [r["relative_error_a"] for r in holdout_results] + [
            r["relative_error_b"] for r in holdout_results
        ]

        metrics: dict = {
            "holdout_results": holdout_results,
            "mean_frobenius_error": float(np.mean(all_errors)),
            "std_frobenius_error": float(np.std(all_errors)),
            "mean_relative_error": float(np.mean(all_rel)),
            "max_relative_error": float(np.max(all_rel)),
            "n_holdout_pairs": len(holdout_results),
            "decomposition_method": decomposition_method,
        }
        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        holdout = result.metrics.get("holdout_results", [])
        if not holdout:
            return {"status": "no holdout results"}

        # Per-dialect average error (when that dialect was held out)
        per_dialect_errors: dict[str, list[float]] = {}
        for r in holdout:
            per_dialect_errors.setdefault(r["held_a"], []).append(r["relative_error_a"])
            per_dialect_errors.setdefault(r["held_b"], []).append(r["relative_error_b"])

        avg_per_dialect = {
            code: float(np.mean(errs)) for code, errs in per_dialect_errors.items()
        }

        # Identify easiest and hardest dialects to reconstruct
        easiest = min(avg_per_dialect.items(), key=lambda x: x[1])[0]
        hardest = max(avg_per_dialect.items(), key=lambda x: x[1])[0]

        return {
            "avg_relative_error_per_dialect": avg_per_dialect,
            "easiest_to_reconstruct": easiest,
            "hardest_to_reconstruct": hardest,
            "overall_mean_relative_error": result.metrics.get("mean_relative_error", 0.0),
            "good_generalisation": result.metrics.get("mean_relative_error", 1.0) < 0.5,
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

        holdout = result.metrics.get("holdout_results", [])
        if not holdout:
            return paths

        # --- Error bars per holdout pair ---
        fig, ax = plt.subplots(figsize=(14, 5))
        labels = [f"{r['held_a']}\n{r['held_b']}" for r in holdout]
        err_a = [r["relative_error_a"] for r in holdout]
        err_b = [r["relative_error_b"] for r in holdout]
        x = np.arange(len(labels))

        ax.bar(x - 0.2, err_a, 0.35, label="Held-out A", color="steelblue")
        ax.bar(x + 0.2, err_b, 0.35, label="Held-out B", color="lightcoral")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Relative Frobenius Error")
        ax.set_title("Leave-2-Out Reconstruction Error")
        ax.legend()
        plt.tight_layout()
        p = output_dir / "holdout_errors.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # --- Per-dialect average error ---
        eval_data = result.metrics.get("_evaluation", {})
        avg_per = eval_data.get("avg_relative_error_per_dialect", {})
        if avg_per:
            fig, ax = plt.subplots(figsize=(10, 5))
            codes = sorted(avg_per.keys())
            vals = [avg_per[c] for c in codes]
            ax.bar(codes, vals, color="teal")
            ax.set_ylabel("Mean Relative Reconstruction Error")
            ax.set_title("Per-Dialect Average Reconstruction Error")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            p = output_dir / "per_dialect_error.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        return paths

    def report(self, result: ExperimentResult) -> str:
        base = super().report(result)
        lines = [
            base, "",
            "## Generalisation Analysis", "",
            f"Decomposition method: **{result.metrics.get('decomposition_method', 'N/A')}**", "",
            f"Mean relative error: **{result.metrics.get('mean_relative_error', 0):.4f}**",
            f"Max relative error: **{result.metrics.get('max_relative_error', 0):.4f}**", "",
        ]

        eval_data = result.metrics.get("_evaluation", {})
        if eval_data:
            lines.append(
                f"Easiest to reconstruct: **{eval_data.get('easiest_to_reconstruct', 'N/A')}**"
            )
            lines.append(
                f"Hardest to reconstruct: **{eval_data.get('hardest_to_reconstruct', 'N/A')}**"
            )
            lines.append("")
            lines.append(
                "Good generalisation (<50% error): "
                f"**{'YES' if eval_data.get('good_generalisation') else 'NO'}**"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tensor completion
    # ------------------------------------------------------------------

    def _reconstruct_holdout(
        self,
        remaining: dict[DialectCode, TransformationMatrix],
        held_a: DialectCode,
        held_b: DialectCode,
        all_codes: list[DialectCode],
        method: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct held-out matrices from the remaining tensor.

        Uses truncated SVD on the mode-3 unfolding of the reduced tensor,
        then projects back to approximate the held-out slices.
        Falls back to mean imputation if tensor decomposition libraries
        are unavailable.
        """
        d = list(remaining.values())[0].data.shape[0]
        sorted_remaining = sorted(remaining.keys(), key=lambda c: c.value)

        # Stack remaining matrices: (d, d, m_remaining)
        stacked = np.stack(
            [remaining[c].data for c in sorted_remaining], axis=2
        )

        if method == "tucker":
            return self._tucker_completion(
                stacked, remaining, sorted_remaining, held_a, held_b, all_codes, d
            )
        else:
            # SVD-based completion on mode-3 unfolding
            return self._svd_completion(stacked, d)

    @staticmethod
    def _svd_completion(
        stacked: np.ndarray,
        d: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SVD-based reconstruction: low-rank approximation of mode-3 unfolding.

        Mode-3 unfolding: shape (m_remaining, d*d).
        Truncated SVD gives us a basis; we project the mean onto it to
        approximate unseen slices.
        """
        m = stacked.shape[2]
        # Mode-3 unfolding: each slice becomes a row
        unfolded = stacked.reshape(d * d, m).T  # (m, d*d)

        # SVD
        k = min(m, d * d, max(2, m - 1))
        U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
        # Low-rank basis
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

        # Reconstruct: mean of the training set in the low-rank subspace
        mean_coeff = np.mean(U_k * S_k, axis=0)
        reconstructed_vec = mean_coeff @ Vt_k  # (d*d,)

        recon_a = reconstructed_vec.reshape(d, d)
        recon_b = reconstructed_vec.reshape(d, d)  # same approximation

        return recon_a, recon_b

    def _tucker_completion(
        self,
        stacked: np.ndarray,
        remaining: dict[DialectCode, TransformationMatrix],
        sorted_remaining: list[DialectCode],
        held_a: DialectCode,
        held_b: DialectCode,
        all_codes: list[DialectCode],
        d: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tucker-decomposition-based tensor completion."""
        try:
            from eigendialectos.tensor.construction import build_dialect_tensor
            from eigendialectos.tensor.tucker import tucker_decompose, tucker_reconstruct

            tensor = build_dialect_tensor(remaining)
            ranks = (min(d, 5), min(d, 5), min(len(remaining), 4))
            result = tucker_decompose(tensor, ranks=ranks)
            core = result["core_tensor"]
            factors = result["factor_matrices"]
            reconstructed = tucker_reconstruct(core, factors)

            # The reconstructed tensor only has slices for remaining dialects.
            # Extrapolate held-out via factor projection (use mean of mode-3 factor).
            C = factors[2]  # (m_remaining, r3)
            mean_c = np.mean(C, axis=0, keepdims=True)  # (1, r3)

            # Approximate held-out slice
            import tensorly as tl
            tl.set_backend("numpy")
            # mode-3 product with mean coefficient
            partial = np.tensordot(core, mean_c.T, axes=([2], [0]))  # (r1, r2, 1)
            partial = partial[:, :, 0]  # (r1, r2)
            recon_slice = factors[0] @ partial @ factors[1].T  # (d, d)

            return recon_slice, recon_slice

        except (ImportError, Exception) as exc:
            logger.warning(
                "Tucker completion failed (%s); falling back to SVD.", exc,
            )
            return self._svd_completion(stacked, d)
