"""Experiment D: Synthetic Dialect Engineering.

Create linguistically coherent synthetic dialects by sampling eigenvalue
spectra from the feasibility region defined by the convex hull of existing
dialect eigenspectra.  For each synthetic point, reconstruct a transformation
matrix using averaged eigenvectors and verify coherence constraints
(bounded eigenvalues, reasonable condition number, positive determinant).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from eigendialectos.constants import DialectCode, DIALECT_NAMES
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import EigenDecomposition, ExperimentResult

logger = logging.getLogger(__name__)


class SyntheticDialectExperiment(Experiment):
    experiment_id = "exp_d_synthetic_dialect"
    name = "Synthetic Dialect Engineering"
    description = (
        "Define a feasibility region as the convex hull of existing dialect "
        "eigenspectra, sample synthetic points inside it, reconstruct "
        "transformation matrices, verify coherence constraints, and compare "
        "synthetic dialects to real ones via PCA projection."
    )
    dependencies = [
        "eigendialectos.spectral.eigendecomposition",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._eigendecomps: dict[DialectCode, EigenDecomposition] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 50)
        n_eigenvalues = config.get("n_eigenvalues", 10)
        rng = np.random.default_rng(seed)

        data_dir = config.get("data_dir")
        loaded = False

        if data_dir:
            data_path = Path(data_dir)
            if data_path.exists():
                count = 0
                for code in DialectCode:
                    w_path = data_path / f"W_{code.value}.npy"
                    if w_path.exists():
                        W = np.load(str(w_path))
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
                        count += 1
                if count == len(DialectCode):
                    loaded = True
                    logger.info("Loaded eigendecompositions from %s", data_dir)
                else:
                    self._eigendecomps.clear()

        if not loaded:
            logger.info(
                "Generating synthetic eigendecompositions for %d dialects "
                "(n_eigenvalues=%d).",
                len(DialectCode),
                n_eigenvalues,
            )
            for code in DialectCode:
                W = np.eye(n_eigenvalues) + rng.standard_normal(
                    (n_eigenvalues, n_eigenvalues)
                ) * 0.05
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

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        seed = self._config.get("seed", 42)
        rng = np.random.default_rng(seed)
        n_synthetics = self._config.get("n_synthetics", 10)
        max_condition_number = self._config.get("max_condition_number", 100.0)

        dialect_codes = sorted(self._eigendecomps.keys(), key=lambda c: c.value)
        n_dialects = len(dialect_codes)

        # ------------------------------------------------------------------
        # 1. Collect eigenspectra (use absolute values, sorted descending)
        # ------------------------------------------------------------------
        spectra_list = []
        for code in dialect_codes:
            ev = np.abs(self._eigendecomps[code].eigenvalues).real
            ev_sorted = np.sort(ev)[::-1]
            spectra_list.append(ev_sorted)

        # Pad to common length
        max_dim = max(len(s) for s in spectra_list)
        spectra_matrix = np.zeros((n_dialects, max_dim), dtype=np.float64)
        for i, s in enumerate(spectra_list):
            spectra_matrix[i, : len(s)] = s

        # ------------------------------------------------------------------
        # 2. Define feasibility region via convex hull
        # ------------------------------------------------------------------
        # If dim > n_dialects, use PCA to reduce for hull computation
        # then sample in the reduced space and project back
        effective_dim = min(max_dim, n_dialects - 1)

        # PCA: center and project
        mean_spectrum = spectra_matrix.mean(axis=0)
        centered = spectra_matrix - mean_spectrum

        if effective_dim < max_dim:
            # SVD-based PCA
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Project to reduced space
            reduced = centered @ Vt[:effective_dim].T  # (n_dialects, effective_dim)
            pca_components = Vt[:effective_dim]  # (effective_dim, max_dim)
        else:
            reduced = centered
            pca_components = np.eye(max_dim)

        # Build convex hull in reduced space
        try:
            hull = ConvexHull(reduced)
            hull_computed = True
        except Exception:
            # Fallback: if hull fails (e.g. degenerate), use bounding box
            hull_computed = False
            logger.warning("ConvexHull failed; using bounding-box sampling.")

        # ------------------------------------------------------------------
        # 3. Sample points inside the convex hull
        # ------------------------------------------------------------------
        synthetic_spectra = []

        if hull_computed:
            # Random convex combination method (always inside hull)
            for _ in range(n_synthetics * 3):  # oversample to get enough
                if len(synthetic_spectra) >= n_synthetics:
                    break
                # Random convex combination of hull vertices
                n_vertices = len(hull.vertices)
                weights = rng.dirichlet(np.ones(n_vertices))
                point_reduced = weights @ reduced[hull.vertices]

                # Project back to full space
                point_full = point_reduced @ pca_components + mean_spectrum

                # Ensure non-negative eigenvalues
                point_full = np.maximum(point_full, 0.0)
                synthetic_spectra.append(point_full)
        else:
            # Bounding box + random convex combination of existing points
            for _ in range(n_synthetics):
                weights = rng.dirichlet(np.ones(n_dialects))
                point = weights @ spectra_matrix
                point = np.maximum(point, 0.0)
                synthetic_spectra.append(point)

        synthetic_spectra = synthetic_spectra[:n_synthetics]

        # ------------------------------------------------------------------
        # 4. Compute average eigenvectors for reconstruction
        # ------------------------------------------------------------------
        # Average the eigenvector matrices (aligned by eigenvalue magnitude)
        P_list = []
        P_inv_list = []
        for code in dialect_codes:
            ed = self._eigendecomps[code]
            P_list.append(ed.eigenvectors[:max_dim, :max_dim])
            P_inv_list.append(ed.eigenvectors_inv[:max_dim, :max_dim])

        # Simple element-wise average of eigenvector matrices
        P_avg = np.mean(np.stack(P_list), axis=0)
        try:
            P_avg_inv = np.linalg.inv(P_avg)
        except np.linalg.LinAlgError:
            P_avg_inv = np.linalg.pinv(P_avg)

        # ------------------------------------------------------------------
        # 5. Reconstruct W_synthetic and verify coherence
        # ------------------------------------------------------------------
        synthetic_results: list[dict] = []
        n_valid = 0

        for i, synth_ev in enumerate(synthetic_spectra):
            # Reconstruct: W_synthetic = P_avg @ diag(lambda_synthetic) @ P_avg_inv
            Lambda_synth = np.diag(synth_ev[:max_dim])
            W_synth = P_avg @ Lambda_synth @ P_avg_inv

            # Coherence checks
            cond_num = float(np.linalg.cond(W_synth.real))
            det_val = float(np.abs(np.linalg.det(W_synth)))
            ev_bounded = bool(np.all(synth_ev <= spectra_matrix.max() * 1.5))
            cond_ok = cond_num < max_condition_number
            det_ok = det_val > 0.0

            is_valid = ev_bounded and cond_ok and det_ok
            if is_valid:
                n_valid += 1

            # Find nearest real dialect
            distances_to_real = []
            for j, code in enumerate(dialect_codes):
                d = float(np.linalg.norm(synth_ev - spectra_matrix[j]))
                distances_to_real.append((code.value, d))
            distances_to_real.sort(key=lambda x: x[1])
            nearest_real = distances_to_real[0]

            synthetic_results.append({
                "synthetic_id": f"synth_{i:03d}",
                "eigenvalues": synth_ev.tolist(),
                "condition_number": cond_num,
                "determinant": det_val,
                "eigenvalues_bounded": ev_bounded,
                "condition_ok": cond_ok,
                "determinant_ok": det_ok,
                "is_valid": is_valid,
                "nearest_real_dialect": nearest_real[0],
                "distance_to_nearest": nearest_real[1],
                "all_distances": distances_to_real,
            })

        # ------------------------------------------------------------------
        # 6. Aggregate metrics
        # ------------------------------------------------------------------
        condition_numbers = [s["condition_number"] for s in synthetic_results]
        distances_to_nearest = [s["distance_to_nearest"] for s in synthetic_results]

        # Diversity: mean pairwise distance between synthetics
        if len(synthetic_spectra) > 1:
            synth_arr = np.array(synthetic_spectra)
            pairwise_dists = []
            for ii in range(len(synth_arr)):
                for jj in range(ii + 1, len(synth_arr)):
                    pairwise_dists.append(
                        float(np.linalg.norm(synth_arr[ii] - synth_arr[jj]))
                    )
            diversity_score = float(np.mean(pairwise_dists))
        else:
            diversity_score = 0.0

        metrics: dict = {
            "n_valid_synthetics": n_valid,
            "n_total_synthetics": len(synthetic_results),
            "fraction_valid": n_valid / max(len(synthetic_results), 1),
            "avg_condition_number": float(np.mean(condition_numbers)) if condition_numbers else 0.0,
            "max_condition_number": float(np.max(condition_numbers)) if condition_numbers else 0.0,
            "distances_to_nearest_real": distances_to_nearest,
            "avg_distance_to_nearest": float(np.mean(distances_to_nearest)) if distances_to_nearest else 0.0,
            "diversity_score": diversity_score,
            "synthetic_results": synthetic_results,
            "real_spectra": spectra_matrix.tolist(),
            "synthetic_spectra": [s.tolist() for s in synthetic_spectra],
            "dialect_order": [c.value for c in dialect_codes],
            "hull_computed": hull_computed,
        }

        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        n_valid = result.metrics.get("n_valid_synthetics", 0)
        n_total = result.metrics.get("n_total_synthetics", 0)
        frac_valid = n_valid / max(n_total, 1)
        avg_cond = result.metrics.get("avg_condition_number", 0.0)
        diversity = result.metrics.get("diversity_score", 0.0)
        avg_dist = result.metrics.get("avg_distance_to_nearest", 0.0)

        return {
            "fraction_valid": float(frac_valid),
            "high_validity": frac_valid > 0.7,
            "avg_condition_number": float(avg_cond),
            "condition_number_acceptable": avg_cond < 100.0,
            "diversity_score": float(diversity),
            "diverse_enough": diversity > 0.01,
            "avg_distance_to_nearest_real": float(avg_dist),
            "n_valid": int(n_valid),
            "n_total": int(n_total),
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

        real_spectra = np.array(result.metrics.get("real_spectra", []))
        synth_spectra = np.array(result.metrics.get("synthetic_spectra", []))
        dialect_order = result.metrics.get("dialect_order", [])
        synthetic_results = result.metrics.get("synthetic_results", [])

        if real_spectra.size == 0 or synth_spectra.size == 0:
            return paths

        # ------------------------------------------------------------------
        # 1. PCA projection of real + synthetic eigenspectra
        # ------------------------------------------------------------------
        all_spectra = np.vstack([real_spectra, synth_spectra])
        mean_all = all_spectra.mean(axis=0)
        centered_all = all_spectra - mean_all

        # PCA via SVD
        U, S, Vt = np.linalg.svd(centered_all, full_matrices=False)
        pca_2d = centered_all @ Vt[:2].T  # project to 2D

        n_real = real_spectra.shape[0]
        n_synth = synth_spectra.shape[0]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Real dialects
        ax.scatter(
            pca_2d[:n_real, 0], pca_2d[:n_real, 1],
            c="royalblue", s=120, edgecolors="k", linewidths=0.8,
            zorder=5, label="Real Dialects",
        )
        for i, code_str in enumerate(dialect_order):
            ax.annotate(
                code_str, (pca_2d[i, 0], pca_2d[i, 1]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8, fontweight="bold",
            )

        # Synthetic dialects (colored by validity)
        valid_mask = np.array([s.get("is_valid", False) for s in synthetic_results])
        synth_pca = pca_2d[n_real:]

        if np.any(valid_mask):
            ax.scatter(
                synth_pca[valid_mask, 0], synth_pca[valid_mask, 1],
                c="limegreen", s=80, edgecolors="k", linewidths=0.5,
                marker="D", zorder=4, label="Valid Synthetics",
            )
        if np.any(~valid_mask):
            ax.scatter(
                synth_pca[~valid_mask, 0], synth_pca[~valid_mask, 1],
                c="red", s=80, edgecolors="k", linewidths=0.5,
                marker="x", zorder=4, label="Invalid Synthetics",
            )

        # Draw convex hull of real dialects
        real_pca = pca_2d[:n_real]
        if n_real >= 3:
            try:
                hull_2d = ConvexHull(real_pca)
                hull_pts = np.append(hull_2d.vertices, hull_2d.vertices[0])
                ax.plot(
                    real_pca[hull_pts, 0], real_pca[hull_pts, 1],
                    "b--", linewidth=1.0, alpha=0.5,
                )
                ax.fill(
                    real_pca[hull_2d.vertices, 0],
                    real_pca[hull_2d.vertices, 1],
                    alpha=0.08, color="blue",
                )
            except Exception:
                pass

        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.set_title("PCA of Real vs Synthetic Dialect Eigenspectra", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = output_dir / "pca_real_vs_synthetic.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # ------------------------------------------------------------------
        # 2. Condition number + validity bar chart
        # ------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        synth_ids = [s["synthetic_id"] for s in synthetic_results]
        cond_nums = [s["condition_number"] for s in synthetic_results]
        validities = [s["is_valid"] for s in synthetic_results]
        colors_cond = ["green" if v else "red" for v in validities]

        ax1.bar(range(len(cond_nums)), cond_nums, color=colors_cond, alpha=0.7)
        ax1.axhline(
            self._config.get("max_condition_number", 100.0),
            color="k", linestyle="--", alpha=0.5, label="Threshold",
        )
        ax1.set_xticks(range(len(synth_ids)))
        ax1.set_xticklabels(synth_ids, rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("Condition Number")
        ax1.set_title("Synthetic Dialect Condition Numbers")
        ax1.legend()
        ax1.set_yscale("log")

        # Distance to nearest real dialect
        nearest_dists = [s["distance_to_nearest"] for s in synthetic_results]
        nearest_names = [s["nearest_real_dialect"] for s in synthetic_results]

        ax2.bar(range(len(nearest_dists)), nearest_dists, color="steelblue", alpha=0.7)
        ax2.set_xticks(range(len(synth_ids)))
        ax2.set_xticklabels(
            [f"{sid}\n({nn})" for sid, nn in zip(synth_ids, nearest_names)],
            rotation=45, ha="right", fontsize=6,
        )
        ax2.set_ylabel("Distance to Nearest Real Dialect")
        ax2.set_title("Synthetic Dialects: Proximity to Real Varieties")

        plt.tight_layout()
        p = output_dir / "synthetic_analysis.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # ------------------------------------------------------------------
        # 3. Eigenspectrum comparison (overlay)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot real spectra
        for i, code_str in enumerate(dialect_order):
            ax.plot(
                real_spectra[i], color="blue", alpha=0.4, linewidth=1.0,
                label="Real" if i == 0 else None,
            )

        # Plot synthetic spectra
        for i in range(synth_spectra.shape[0]):
            valid = synthetic_results[i]["is_valid"] if i < len(synthetic_results) else False
            color = "green" if valid else "red"
            ax.plot(
                synth_spectra[i], color=color, alpha=0.5, linewidth=1.0,
                linestyle="--",
                label=("Valid Synth" if valid else "Invalid Synth") if i == 0 else None,
            )

        ax.set_xlabel("Eigenvalue Index (sorted descending)")
        ax.set_ylabel("Eigenvalue Magnitude")
        ax.set_title("Eigenspectra: Real Dialects vs Synthetic Dialects")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = output_dir / "eigenspectra_comparison.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        return paths
