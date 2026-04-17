"""Experiment F: The Eigenvalue Microscope.

Surgically edit single features by manipulating specific eigenvalues.
For each dialect's transformation matrix W = P diag(lambda) P^{-1},
selectively neutralise individual eigenvalues (set to 1.0) and measure
how much dialectal strength is lost -- revealing which eigenvectors
encode the most important linguistic features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import ExperimentResult

logger = logging.getLogger(__name__)


class EigenvalueMicroscopeExperiment(Experiment):
    experiment_id = "exp_f_eigenvalue_microscope"
    name = "The Eigenvalue Microscope"
    description = (
        "Surgically edit single features by manipulating specific eigenvalues. "
        "For each dialect, neutralise individual eigenvalues (set to 1.0) and "
        "measure feature importance via Frobenius norm change.  Find the "
        "minimum set of eigenvalues whose neutralisation reduces dialectal "
        "strength by 90%%."
    )
    dependencies = ["scipy", "numpy"]

    def __init__(self) -> None:
        super().__init__()
        self._eigendecomps: dict[str, dict] = {}
        # Each entry: {"eigenvalues": ndarray, "P": ndarray, "P_inv": ndarray, "W": ndarray}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 10)
        rng = np.random.default_rng(seed)

        # Try loading real eigendecompositions
        data_dir = config.get("data_dir")
        loaded = False
        if data_dir and Path(data_dir).exists():
            for code in DialectCode:
                w_path = Path(data_dir) / f"W_{code.value}.npy"
                if w_path.exists():
                    W = np.load(str(w_path)).astype(np.complex128)
                    eigenvalues, P = np.linalg.eig(W)
                    try:
                        P_inv = np.linalg.inv(P)
                    except np.linalg.LinAlgError:
                        P_inv = np.linalg.pinv(P)
                    self._eigendecomps[code.value] = {
                        "eigenvalues": eigenvalues,
                        "P": P,
                        "P_inv": P_inv,
                        "W": W,
                    }
                    loaded = True

        if not loaded or len(self._eigendecomps) == 0:
            logger.info(
                "Generating synthetic eigendecompositions for %d dialects (dim=%d).",
                len(DialectCode),
                dim,
            )
            for code in DialectCode:
                # Build W = I + noise, so eigenvalues are near 1
                W = np.eye(dim, dtype=np.complex128) + 0.1 * rng.standard_normal(
                    (dim, dim)
                ).astype(np.complex128)
                eigenvalues, P = np.linalg.eig(W)
                try:
                    P_inv = np.linalg.inv(P)
                except np.linalg.LinAlgError:
                    P_inv = np.linalg.pinv(P)
                self._eigendecomps[code.value] = {
                    "eigenvalues": eigenvalues,
                    "P": P,
                    "P_inv": P_inv,
                    "W": W,
                }

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        dialect_names = sorted(self._eigendecomps.keys())
        n_dialects = len(dialect_names)

        # Determine eigenvalue dimension (may vary; use first one as reference)
        first = self._eigendecomps[dialect_names[0]]
        n_eigs = len(first["eigenvalues"])

        # ---- 1. Feature importance matrix: dialects x eigenvalues ----
        # feature_importance[d, k] = ||W - W_modified_k||_F
        # where W_modified_k has lambda_k set to 1.0
        feature_importance = np.zeros((n_dialects, n_eigs), dtype=np.float64)

        # remaining_strength[d, k] = ||W_modified_k - I||_F
        remaining_strength = np.zeros((n_dialects, n_eigs), dtype=np.float64)

        # Full dialectal strength = ||W - I||_F for each dialect
        full_strength = np.zeros(n_dialects, dtype=np.float64)

        for d_idx, dname in enumerate(dialect_names):
            decomp = self._eigendecomps[dname]
            eigenvalues = decomp["eigenvalues"].copy()
            P = decomp["P"]
            P_inv = decomp["P_inv"]
            W = decomp["W"]
            dim = W.shape[0]
            I = np.eye(dim, dtype=np.complex128)

            full_strength[d_idx] = float(np.linalg.norm(W - I, "fro").real)

            for k in range(n_eigs):
                # Create modified eigenvalues: set lambda_k to 1.0
                lam_mod = eigenvalues.copy()
                lam_mod[k] = 1.0 + 0j

                # Reconstruct modified W
                W_mod = P @ np.diag(lam_mod) @ P_inv

                # Feature importance: how much W changed when we neutralised k
                feature_importance[d_idx, k] = float(
                    np.linalg.norm(W - W_mod, "fro").real
                )

                # Remaining dialectal strength after neutralising k
                remaining_strength[d_idx, k] = float(
                    np.linalg.norm(W_mod - I, "fro").real
                )

        # ---- 2. Minimum neutralisation set: greedy approach ----
        # For each dialect, greedily pick eigenvalues to neutralise until
        # dialectal strength is reduced by 90%.
        min_neutralisation_sets: dict[str, list[int]] = {}
        neutralisation_curves: dict[str, list[float]] = {}

        for d_idx, dname in enumerate(dialect_names):
            decomp = self._eigendecomps[dname]
            eigenvalues = decomp["eigenvalues"].copy()
            P = decomp["P"]
            P_inv = decomp["P_inv"]
            dim = P.shape[0]
            I = np.eye(dim, dtype=np.complex128)

            target_strength = full_strength[d_idx] * 0.10  # 90% reduction

            # Rank eigenvalues by individual feature importance (descending)
            importances = feature_importance[d_idx, :]
            order = np.argsort(-importances)

            current_lam = eigenvalues.copy()
            neutralised: list[int] = []
            curve: list[float] = [full_strength[d_idx]]

            for k in order:
                current_lam[k] = 1.0 + 0j
                neutralised.append(int(k))
                W_mod = P @ np.diag(current_lam) @ P_inv
                strength_now = float(np.linalg.norm(W_mod - I, "fro").real)
                curve.append(strength_now)
                if strength_now <= target_strength:
                    break

            min_neutralisation_sets[dname] = neutralised
            neutralisation_curves[dname] = curve

        # ---- 3. Pack metrics ----
        metrics: dict = {
            "feature_importance_matrix": feature_importance.tolist(),
            "remaining_strength_matrix": remaining_strength.tolist(),
            "full_dialectal_strength": {
                dname: float(full_strength[i]) for i, dname in enumerate(dialect_names)
            },
            "min_neutralisation_sets": min_neutralisation_sets,
            "neutralisation_curves": neutralisation_curves,
            "dialect_names": dialect_names,
            "n_eigenvalues": n_eigs,
            "eigenvalue_magnitudes": {
                dname: np.abs(self._eigendecomps[dname]["eigenvalues"]).tolist()
                for dname in dialect_names
            },
        }

        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        """How many eigenvalues needed to reduce dialectal strength by 90%?"""
        min_sets = result.metrics["min_neutralisation_sets"]
        n_eigs = result.metrics["n_eigenvalues"]
        dialect_names = result.metrics["dialect_names"]

        counts = {dname: len(indices) for dname, indices in min_sets.items()}
        mean_count = float(np.mean(list(counts.values())))
        fraction_needed = {
            dname: c / n_eigs for dname, c in counts.items()
        }

        # Feature importance concentration: how much of total importance
        # is in the top-k eigenvalues?
        fi_matrix = np.array(result.metrics["feature_importance_matrix"])
        concentration: dict[str, float] = {}
        for d_idx, dname in enumerate(dialect_names):
            row = fi_matrix[d_idx, :]
            total = float(np.sum(row))
            if total > 1e-12:
                sorted_row = np.sort(row)[::-1]
                top3 = float(np.sum(sorted_row[:3]))
                concentration[dname] = top3 / total
            else:
                concentration[dname] = 0.0

        return {
            "eigenvalues_for_90pct_reduction": counts,
            "mean_eigenvalues_for_90pct": mean_count,
            "fraction_needed_for_90pct": fraction_needed,
            "top3_importance_concentration": concentration,
            "mean_top3_concentration": float(np.mean(list(concentration.values()))),
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

        fi_matrix = np.array(result.metrics["feature_importance_matrix"])
        dialect_names = result.metrics["dialect_names"]
        n_eigs = result.metrics["n_eigenvalues"]
        neutralisation_curves = result.metrics["neutralisation_curves"]

        # ---- 1. Heatmap: feature importance (dialects x eigenvalues) ----
        fig, ax = plt.subplots(figsize=(max(10, n_eigs * 0.5), max(6, len(dialect_names) * 0.6)))
        im = ax.imshow(fi_matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(n_eigs))
        ax.set_xticklabels([f"$\\lambda_{{{k}}}$" for k in range(n_eigs)], fontsize=7)
        ax.set_yticks(range(len(dialect_names)))
        ax.set_yticklabels(dialect_names, fontsize=8)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Dialect")
        ax.set_title("Feature Importance: $\\|W - W_{\\lambda_k \\to 1}\\|_F$")
        fig.colorbar(im, ax=ax, label="Frobenius norm change")
        plt.tight_layout()
        p = output_dir / "feature_importance_heatmap.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # ---- 2. Bar chart: neutralisation progression for each dialect ----
        n_dialects = len(dialect_names)
        n_cols = min(3, n_dialects)
        n_rows = (n_dialects + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        for d_idx, dname in enumerate(dialect_names):
            r, c = divmod(d_idx, n_cols)
            ax = axes[r, c]
            curve = neutralisation_curves[dname]
            ax.bar(range(len(curve)), curve, color="steelblue", edgecolor="navy", linewidth=0.3)
            # Draw 10% threshold line
            if len(curve) > 0:
                threshold = curve[0] * 0.10
                ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8, label="10% remaining")
            ax.set_xlabel("# eigenvalues neutralised")
            ax.set_ylabel("$\\|W_{mod} - I\\|_F$")
            ax.set_title(dname, fontsize=9)
            ax.legend(fontsize=7)
        # Hide empty subplots
        for d_idx in range(n_dialects, n_rows * n_cols):
            r, c = divmod(d_idx, n_cols)
            axes[r, c].set_visible(False)
        plt.suptitle("Neutralisation Progression", fontsize=12, y=1.02)
        plt.tight_layout()
        p = output_dir / "neutralisation_progression.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # ---- 3. Eigenvalue magnitude overview ----
        eig_mags = result.metrics["eigenvalue_magnitudes"]
        fig, ax = plt.subplots(figsize=(max(10, n_eigs * 0.5), 6))
        x = np.arange(n_eigs)
        bar_width = 0.8 / max(n_dialects, 1)
        for d_idx, dname in enumerate(dialect_names):
            mags = np.array(eig_mags[dname])
            ax.bar(x + d_idx * bar_width, mags, bar_width, label=dname, alpha=0.8)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.5, label="neutral (1.0)")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("$|\\lambda_k|$")
        ax.set_title("Eigenvalue Magnitudes by Dialect")
        ax.legend(fontsize=6, ncol=2, loc="upper right")
        plt.tight_layout()
        p = output_dir / "eigenvalue_magnitudes.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        return paths
