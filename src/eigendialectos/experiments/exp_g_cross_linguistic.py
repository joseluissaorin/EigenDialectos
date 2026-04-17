"""Experiment G: Cross-Linguistic Spectral Transfer.

Test whether dialectal eigenvectors generalise across languages, suggesting
universal structure in how dialects deform a base language.  Since we lack
real Portuguese/Catalan data, we simulate these by applying known Procrustes-
like rotations to existing Spanish eigenstructures and then test whether
alignment recovery is significantly better than random.

The key insight: if dialect structure is universal, the *subspace* spanned by
the top-k eigenvectors (those carrying most dialectal energy) should be
similar across related languages.  We measure this via principal angles
between eigensubspaces of rank k < dim, which are not trivially solved by
full-rank Procrustes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import ExperimentResult

logger = logging.getLogger(__name__)


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a uniformly random orthogonal matrix via QR of Gaussian."""
    H = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(H)
    # Ensure det(Q)=+1 (proper rotation)
    d = np.diag(R)
    sign = np.sign(d)
    sign[sign == 0] = 1.0
    Q = Q * sign[np.newaxis, :]
    return Q


def _principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Compute principal angles between subspaces spanned by columns of V1, V2.

    Parameters
    ----------
    V1, V2 : ndarray, shape (dim, k)
        Column-orthonormal bases for two subspaces.

    Returns
    -------
    ndarray, shape (k,)
        Principal angles in ascending order (radians).
    """
    # QR to guarantee orthonormality even after slicing
    Q1, _ = np.linalg.qr(V1)
    Q2, _ = np.linalg.qr(V2)
    k = min(Q1.shape[1], Q2.shape[1])
    Q1 = Q1[:, :k]
    Q2 = Q2[:, :k]
    M = Q1.T @ Q2
    _, s, _ = np.linalg.svd(M)
    s_clamped = np.clip(s[:k], 0.0, 1.0)
    return np.arccos(s_clamped)


class CrossLinguisticExperiment(Experiment):
    experiment_id = "exp_g_cross_linguistic"
    name = "Cross-Linguistic Spectral Transfer"
    description = (
        "Test if dialectal eigenvectors generalise across languages, "
        "suggesting universal dialect structure.  Simulate Portuguese and "
        "Catalan eigenstructures via known rotations of Spanish eigenvectors, "
        "then recover alignments with Procrustes and compare against random "
        "rotations (null hypothesis).  Uses subspace principal angles on "
        "top-k eigensubspaces to detect shared structure."
    )
    dependencies = ["scipy", "numpy"]

    def __init__(self) -> None:
        super().__init__()
        # Full eigenvector matrices: language -> (dim, dim) orthonormal
        self._eigenvectors: dict[str, np.ndarray] = {}
        # Eigenvalue spectra: language -> (dim,) real values
        self._eigenvalues: dict[str, np.ndarray] = {}
        # Ground-truth rotation angles used to generate simulated languages
        self._true_rotations: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 10)
        rng = np.random.default_rng(seed)

        # Step 1: obtain Spanish eigenvectors + eigenvalues
        spanish_V, spanish_lam = self._load_or_generate_spanish(config, rng, dim)
        self._eigenvectors["Spanish"] = spanish_V
        self._eigenvalues["Spanish"] = spanish_lam

        # Step 2: simulate Portuguese by rotating eigenvectors + adding
        # small perturbation to eigenvalues
        pt_angle = config.get("portuguese_rotation_angle", 0.3)
        R_pt = self._small_rotation(dim, pt_angle, rng)
        V_pt = spanish_V @ R_pt
        # Perturb eigenvalues slightly (shared structure => similar spectra)
        lam_pt = spanish_lam + 0.05 * rng.standard_normal(dim)
        self._eigenvectors["Portuguese"] = V_pt
        self._eigenvalues["Portuguese"] = lam_pt
        self._true_rotations["Portuguese"] = R_pt

        # Step 3: Catalan -- closer to Spanish than Portuguese
        ca_angle = config.get("catalan_rotation_angle", 0.15)
        R_ca = self._small_rotation(dim, ca_angle, rng)
        V_ca = spanish_V @ R_ca
        lam_ca = spanish_lam + 0.03 * rng.standard_normal(dim)
        self._eigenvectors["Catalan"] = V_ca
        self._eigenvalues["Catalan"] = lam_ca
        self._true_rotations["Catalan"] = R_ca

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        seed = self._config.get("seed", 42)
        rng = np.random.default_rng(seed)
        n_random = self._config.get("n_random_baselines", 100)
        languages = sorted(self._eigenvectors.keys())
        dim = self._eigenvectors[languages[0]].shape[1]
        # Test subspace ranks: top-k for k in [2, dim//2, dim-1]
        subspace_ranks = sorted(set([
            2,
            max(2, dim // 4),
            max(2, dim // 2),
            max(2, dim - 1),
        ]))
        subspace_ranks = [k for k in subspace_ranks if k < dim]

        pairs = []
        for i, lang_i in enumerate(languages):
            for j, lang_j in enumerate(languages):
                if j <= i:
                    continue
                pairs.append((lang_i, lang_j))

        # ---- 1. Spectral alignment score ----
        # Compare the reconstruction W = V diag(lam) V^T across languages
        # after Procrustes alignment of eigenvector bases.
        # Alignment score = 1 - normalised(||W_a - W_b||_F) measures how
        # similar the overall eigenstructures are.
        alignment_scores: dict[str, float] = {}
        procrustes_errors: dict[str, float] = {}

        for lang_a, lang_b in pairs:
            pair_key = f"{lang_a}-{lang_b}"
            V_a = self._eigenvectors[lang_a]
            V_b = self._eigenvectors[lang_b]
            lam_a = self._eigenvalues[lang_a]
            lam_b = self._eigenvalues[lang_b]

            # Reconstruct "transformation-like" matrices
            W_a = V_a @ np.diag(lam_a) @ V_a.T
            W_b = V_b @ np.diag(lam_b) @ V_b.T

            # Procrustes error (how well bases align)
            R_hat, _ = orthogonal_procrustes(V_a, V_b)
            V_a_aligned = V_a @ R_hat
            procrustes_errors[pair_key] = float(
                np.linalg.norm(V_a_aligned - V_b, "fro")
            )

            # Spectral alignment: compare W matrices
            diff_norm = float(np.linalg.norm(W_a - W_b, "fro"))
            scale = 0.5 * (np.linalg.norm(W_a, "fro") + np.linalg.norm(W_b, "fro"))
            if scale > 1e-12:
                alignment_scores[pair_key] = float(1.0 - diff_norm / scale)
            else:
                alignment_scores[pair_key] = 0.0

        # ---- 2. Principal angles between eigensubspaces of rank k ----
        principal_angles_dict: dict[str, dict[int, list[float]]] = {}
        for lang_a, lang_b in pairs:
            pair_key = f"{lang_a}-{lang_b}"
            V_a = self._eigenvectors[lang_a]
            V_b = self._eigenvectors[lang_b]
            angles_by_k: dict[int, list[float]] = {}
            for k in subspace_ranks:
                angles = _principal_angles(V_a[:, :k], V_b[:, :k])
                angles_by_k[k] = angles.tolist()
            principal_angles_dict[pair_key] = angles_by_k

        # ---- 3. Null hypothesis: random rotations ----
        random_alignment_scores: dict[str, list[float]] = {}
        random_principal_angles: dict[str, dict[int, list[list[float]]]] = {}

        for lang_a, lang_b in pairs:
            pair_key = f"{lang_a}-{lang_b}"
            V_a = self._eigenvectors[lang_a]
            V_b = self._eigenvectors[lang_b]
            lam_a = self._eigenvalues[lang_a]
            lam_b = self._eigenvalues[lang_b]

            # Precompute W_b for the spectral alignment score
            W_b = V_b @ np.diag(lam_b) @ V_b.T
            W_b_norm = float(np.linalg.norm(W_b, "fro"))

            rand_scores: list[float] = []
            rand_pa: dict[int, list[list[float]]] = {k: [] for k in subspace_ranks}

            for _ in range(n_random):
                # Fully random orthonormal basis
                Q = _random_rotation(dim, rng)
                V_rand = Q  # pure random, no relation to V_a

                # Spectral alignment: random W vs real W_b
                # Use lam_a as the eigenvalues (same spectrum, different basis)
                W_rand = V_rand @ np.diag(lam_a) @ V_rand.T
                diff_norm = float(np.linalg.norm(W_rand - W_b, "fro"))
                scale = 0.5 * (np.linalg.norm(W_rand, "fro") + W_b_norm)
                if scale > 1e-12:
                    rand_scores.append(float(1.0 - diff_norm / scale))
                else:
                    rand_scores.append(0.0)

                # Principal angles for random
                for k in subspace_ranks:
                    angles = _principal_angles(V_rand[:, :k], V_b[:, :k])
                    rand_pa[k].append(angles.tolist())

            random_alignment_scores[pair_key] = rand_scores
            random_principal_angles[pair_key] = rand_pa

        # ---- 4. Statistical test: p-values and effect sizes ----
        p_values: dict[str, float] = {}
        effect_sizes: dict[str, float] = {}

        for pair_key in alignment_scores:
            real_score = alignment_scores[pair_key]
            rand_scores_arr = np.array(random_alignment_scores[pair_key])
            # One-sided p-value: fraction of random >= real
            p_val = float(np.mean(rand_scores_arr >= real_score))
            p_values[pair_key] = p_val

            rand_mean = float(np.mean(rand_scores_arr))
            rand_std = float(np.std(rand_scores_arr))
            if rand_std > 1e-12:
                effect_sizes[pair_key] = (real_score - rand_mean) / rand_std
            else:
                effect_sizes[pair_key] = float("inf") if real_score > rand_mean else 0.0

        # p-values for principal angles (use mean angle as summary statistic)
        pa_p_values: dict[str, dict[int, float]] = {}
        pa_effect_sizes: dict[str, dict[int, float]] = {}
        for pair_key in principal_angles_dict:
            pa_p_values[pair_key] = {}
            pa_effect_sizes[pair_key] = {}
            for k in subspace_ranks:
                real_mean_angle = float(np.mean(principal_angles_dict[pair_key][k]))
                rand_mean_angles = [
                    float(np.mean(ra)) for ra in random_principal_angles[pair_key][k]
                ]
                rand_arr = np.array(rand_mean_angles)
                # Smaller angle = better alignment, so p = P(random <= real)
                p_val_pa = float(np.mean(rand_arr <= real_mean_angle))
                pa_p_values[pair_key][k] = p_val_pa

                rand_m = float(np.mean(rand_arr))
                rand_s = float(np.std(rand_arr))
                if rand_s > 1e-12:
                    pa_effect_sizes[pair_key][k] = (rand_m - real_mean_angle) / rand_s
                else:
                    pa_effect_sizes[pair_key][k] = 0.0

        # ---- 5. Summarise random principal angle means for visualisation ----
        random_pa_mean: dict[str, dict[int, list[float]]] = {}
        for pair_key in random_principal_angles:
            random_pa_mean[pair_key] = {}
            for k in subspace_ranks:
                all_angles = np.array(random_principal_angles[pair_key][k])
                random_pa_mean[pair_key][k] = np.mean(all_angles, axis=0).tolist()

        # ---- Pack metrics ----
        # Convert int keys to str for JSON serialisation
        def _intkey_to_str(d: dict) -> dict:
            return {str(k): v for k, v in d.items()}

        metrics: dict = {
            "alignment_scores": alignment_scores,
            "procrustes_errors": procrustes_errors,
            "principal_angles": {
                pk: _intkey_to_str(v) for pk, v in principal_angles_dict.items()
            },
            "p_values": p_values,
            "effect_sizes": effect_sizes,
            "pa_p_values": {
                pk: _intkey_to_str(v) for pk, v in pa_p_values.items()
            },
            "pa_effect_sizes": {
                pk: _intkey_to_str(v) for pk, v in pa_effect_sizes.items()
            },
            "random_alignment_scores": random_alignment_scores,
            "random_pa_mean": {
                pk: _intkey_to_str(v) for pk, v in random_pa_mean.items()
            },
            "language_pairs": [f"{a}-{b}" for a, b in pairs],
            "languages": languages,
            "dim": dim,
            "subspace_ranks": subspace_ranks,
            "n_random_baselines": n_random,
        }

        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        """Is alignment significantly better than random?"""
        p_values = result.metrics["p_values"]
        effect_sizes = result.metrics["effect_sizes"]
        alignment_scores = result.metrics["alignment_scores"]
        random_alignment_scores = result.metrics["random_alignment_scores"]
        pa_p_values = result.metrics["pa_p_values"]

        significant_pairs: list[str] = []
        for pair_key, p in p_values.items():
            if p < 0.05:
                significant_pairs.append(pair_key)

        # Subspace-level significance
        subspace_significant: dict[str, list[str]] = {}
        for pair_key, k_pvals in pa_p_values.items():
            sig_ks = [k for k, p in k_pvals.items() if p < 0.05]
            subspace_significant[pair_key] = sig_ks

        mean_real = float(np.mean(list(alignment_scores.values())))
        all_rand = []
        for rand_list in random_alignment_scores.values():
            all_rand.extend(rand_list)
        mean_random = float(np.mean(all_rand)) if all_rand else 0.0

        return {
            "significant_pairs_fullrank": significant_pairs,
            "n_significant_fullrank": len(significant_pairs),
            "n_total_pairs": len(p_values),
            "all_significant_fullrank": len(significant_pairs) == len(p_values),
            "subspace_significant_ranks": subspace_significant,
            "mean_real_alignment": mean_real,
            "mean_random_alignment": mean_random,
            "alignment_advantage": mean_real - mean_random,
            "mean_effect_size": float(np.mean(list(effect_sizes.values()))),
            "min_p_value": float(min(p_values.values())) if p_values else 1.0,
            "max_p_value": float(max(p_values.values())) if p_values else 1.0,
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

        language_pairs = result.metrics["language_pairs"]
        alignment_scores = result.metrics["alignment_scores"]
        random_alignment_scores = result.metrics["random_alignment_scores"]
        principal_angles = result.metrics["principal_angles"]
        random_pa_mean = result.metrics["random_pa_mean"]
        p_values = result.metrics["p_values"]
        effect_sizes = result.metrics["effect_sizes"]
        subspace_ranks = result.metrics["subspace_ranks"]

        # ---- 1. Bar chart: real vs random alignment per pair ----
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(language_pairs))
        bar_w = 0.35
        real_vals = [alignment_scores[pk] for pk in language_pairs]
        rand_means = [float(np.mean(random_alignment_scores[pk])) for pk in language_pairs]
        rand_stds = [float(np.std(random_alignment_scores[pk])) for pk in language_pairs]

        ax.bar(x - bar_w / 2, real_vals, bar_w, label="Real alignment", color="steelblue")
        ax.bar(
            x + bar_w / 2,
            rand_means,
            bar_w,
            yerr=rand_stds,
            label="Random baseline",
            color="lightcoral",
            capsize=3,
        )
        for i, pk in enumerate(language_pairs):
            p = p_values[pk]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(
                i,
                max(real_vals[i], rand_means[i] + rand_stds[i]) + 0.02,
                f"p={p:.3f} {star}",
                ha="center",
                fontsize=7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(language_pairs, rotation=20, ha="right")
        ax.set_ylabel("Mean cosine similarity (eigenvalue-weighted)")
        ax.set_title("Cross-Linguistic Eigenvector Alignment")
        ax.legend()
        ymax = max(real_vals + [m + s for m, s in zip(rand_means, rand_stds)]) + 0.12
        ax.set_ylim(0, min(1.3, ymax))
        plt.tight_layout()
        p_path = output_dir / "alignment_real_vs_random.png"
        fig.savefig(p_path, dpi=150)
        plt.close(fig)
        paths.append(p_path)

        # ---- 2. Principal angles: real vs random for each rank ----
        for pk in language_pairs:
            n_ranks = len(subspace_ranks)
            fig, axes = plt.subplots(1, n_ranks, figsize=(5 * n_ranks, 5), squeeze=False)
            for r_idx, k in enumerate(subspace_ranks):
                ax = axes[0, r_idx]
                k_str = str(k)
                real_angles = np.array(principal_angles[pk][k_str])
                rand_angles_mean = np.array(random_pa_mean[pk][k_str])
                n_angles = len(real_angles)
                x_idx = np.arange(n_angles)

                ax.bar(x_idx - 0.2, np.degrees(real_angles), 0.4, label="Real", color="steelblue")
                ax.bar(
                    x_idx + 0.2,
                    np.degrees(rand_angles_mean),
                    0.4,
                    label="Random (mean)",
                    color="lightcoral",
                )
                ax.set_xlabel("Principal angle index")
                ax.set_ylabel("Angle (degrees)")
                ax.set_title(f"{pk}, rank-{k} subspace")
                ax.legend(fontsize=7)
            plt.suptitle(f"Principal Angles: {pk}", fontsize=12)
            plt.tight_layout()
            safe_pk = pk.replace("-", "_")
            p_path = output_dir / f"principal_angles_{safe_pk}.png"
            fig.savefig(p_path, dpi=150)
            plt.close(fig)
            paths.append(p_path)

        # ---- 3. Eigenvector alignment heatmap per pair ----
        dim = result.metrics["dim"]
        for pk in language_pairs:
            lang_a, lang_b = pk.split("-")
            V_a = self._eigenvectors.get(lang_a)
            V_b = self._eigenvectors.get(lang_b)
            if V_a is None or V_b is None:
                continue

            R_hat, _ = orthogonal_procrustes(V_a, V_b)
            V_a_aligned = V_a @ R_hat

            # Absolute cosine similarity matrix
            sim_matrix = np.zeros((dim, dim), dtype=np.float64)
            for ci in range(dim):
                for cj in range(dim):
                    v1 = V_a_aligned[:, ci]
                    v2 = V_b[:, cj]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 1e-12 and n2 > 1e-12:
                        sim_matrix[ci, cj] = float(np.abs(np.dot(v1, v2)) / (n1 * n2))

            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(sim_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xlabel(f"{lang_b} eigenvector index")
            ax.set_ylabel(f"{lang_a} (aligned) eigenvector index")
            ax.set_title(f"Eigenvector Alignment: {pk}")
            fig.colorbar(im, ax=ax, label="|cos similarity|")
            plt.tight_layout()
            safe_pk = pk.replace("-", "_")
            p_path = output_dir / f"alignment_heatmap_{safe_pk}.png"
            fig.savefig(p_path, dpi=150)
            plt.close(fig)
            paths.append(p_path)

        # ---- 4. Effect size summary bar chart ----
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(language_pairs))
        es_vals = [effect_sizes[pk] for pk in language_pairs]
        colors = ["green" if e > 0.8 else "orange" if e > 0.5 else "red" for e in es_vals]
        ax.bar(x, es_vals, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(language_pairs, rotation=20, ha="right")
        ax.set_ylabel("Cohen's d")
        ax.set_title("Effect Size of Alignment vs Random")
        ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.5, label="large (0.8)")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5, label="medium (0.5)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        p_path = output_dir / "effect_sizes.png"
        fig.savefig(p_path, dpi=150)
        plt.close(fig)
        paths.append(p_path)

        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_or_generate_spanish(
        self, config: dict, rng: np.random.Generator, dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load real Spanish eigenvectors/values or generate synthetic.

        Returns
        -------
        V : ndarray, shape (dim, dim)
            Orthonormal eigenvector matrix.
        lam : ndarray, shape (dim,)
            Eigenvalue magnitudes (real, positive).
        """
        data_dir = config.get("data_dir")
        if data_dir and Path(data_dir).exists():
            w_path = Path(data_dir) / "W_ES_PEN.npy"
            if w_path.exists():
                W = np.load(str(w_path)).astype(np.float64)
                eigenvalues, V = np.linalg.eig(W)
                Q, _ = np.linalg.qr(V.real)
                logger.info("Loaded Spanish eigenvectors from %s", w_path)
                return Q, np.abs(eigenvalues)

        logger.info("Generating synthetic Spanish eigenstructure (dim=%d).", dim)
        H = rng.standard_normal((dim, dim))
        Q, _ = np.linalg.qr(H)
        # Eigenvalues with a clear spectrum: dominant few + tail
        lam = np.sort(rng.uniform(0.5, 2.0, size=dim))[::-1]
        lam[:3] *= 2.0  # amplify top-3 eigenvalues
        return Q, lam

    @staticmethod
    def _small_rotation(dim: int, angle: float, rng: np.random.Generator) -> np.ndarray:
        """Generate a rotation close to identity with controlled angular deviation.

        Constructs R = expm(angle * A) where A is a random skew-symmetric matrix
        normalised so that ||A||_F = 1.

        Parameters
        ----------
        dim : int
            Matrix dimension.
        angle : float
            Rotation magnitude (radians-like scale).
        rng : numpy Generator
            Random state.

        Returns
        -------
        ndarray, shape (dim, dim)
            Orthogonal matrix close to identity for small *angle*.
        """
        from scipy.linalg import expm as _expm

        A = rng.standard_normal((dim, dim))
        A = (A - A.T) / 2.0  # skew-symmetric
        norm = np.linalg.norm(A, "fro")
        if norm > 1e-12:
            A = A / norm
        R = _expm(angle * A)
        return R.real
