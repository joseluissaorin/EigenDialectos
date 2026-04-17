"""Experiment C: Eigenvalue Archaeology.

Simulate temporal eigenvalue trajectories using an Ornstein-Uhlenbeck process
to model how dialectal features might evolve over time.  Apply change-point
detection to classify evolutionary dynamics as gradual (smooth drift) vs
punctuated (sharp transitions), mirroring the gradualism-vs-punctuation debate
in historical linguistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from eigendialectos.constants import DialectCode, DIALECT_NAMES
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import EigenDecomposition, ExperimentResult

logger = logging.getLogger(__name__)


class EigenvalueArchaeologyExperiment(Experiment):
    experiment_id = "exp_c_eigenvalue_archaeology"
    name = "Eigenvalue Archaeology"
    description = (
        "Simulate temporal eigenvalue trajectories via the "
        "Ornstein-Uhlenbeck process, detect change points using "
        "second-derivative analysis, and classify dialectal evolution "
        "as gradual vs punctuated."
    )
    dependencies = [
        "eigendialectos.spectral.eigendecomposition",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._eigendecomps: dict[DialectCode, EigenDecomposition] = {}
        self._initial_eigenvalues: dict[DialectCode, np.ndarray] = {}

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
                # Synthetic W: identity + perturbation
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

        # Extract initial eigenvalue magnitudes for each dialect
        for code, ed in self._eigendecomps.items():
            self._initial_eigenvalues[code] = np.abs(ed.eigenvalues).real

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        seed = self._config.get("seed", 42)
        rng = np.random.default_rng(seed)

        n_steps = self._config.get("n_time_steps", 100)
        theta = self._config.get("ou_theta", 0.1)       # mean reversion rate
        sigma = self._config.get("ou_sigma", 0.02)       # volatility
        dt = self._config.get("ou_dt", 1.0)
        punctuation_threshold = self._config.get("punctuation_threshold", None)

        dialect_codes = sorted(self._initial_eigenvalues.keys(), key=lambda c: c.value)

        all_trajectories: dict[str, list[list[float]]] = {}
        all_change_points: dict[str, list[int]] = {}
        all_change_types: dict[str, list[str]] = {}
        all_autocorrelations: dict[str, list[float]] = {}
        all_max_slope_changes: dict[str, list[float]] = {}

        total_punctuated = 0
        total_gradual = 0

        for code in dialect_codes:
            ev_init = self._initial_eigenvalues[code]
            n_ev = len(ev_init)
            mu = ev_init.copy()  # long-run mean = current observed value

            # ------------------------------------------------------------------
            # 1. Simulate OU process for each eigenvalue
            # ------------------------------------------------------------------
            # dλ = -θ(λ - μ)dt + σ dW
            trajectories = np.zeros((n_ev, n_steps), dtype=np.float64)
            trajectories[:, 0] = ev_init

            for t in range(1, n_steps):
                dW = rng.standard_normal(n_ev) * np.sqrt(dt)
                drift = -theta * (trajectories[:, t - 1] - mu) * dt
                diffusion = sigma * dW
                trajectories[:, t] = trajectories[:, t - 1] + drift + diffusion

            # ------------------------------------------------------------------
            # 2. Change-point detection via second derivative
            # ------------------------------------------------------------------
            change_points_per_ev: list[int] = []
            change_types_per_ev: list[str] = []
            autocorrelations_per_ev: list[float] = []
            max_slope_changes_per_ev: list[float] = []

            for k in range(n_ev):
                traj = trajectories[k]

                # First derivative (discrete)
                first_deriv = np.diff(traj)
                # Second derivative
                second_deriv = np.diff(first_deriv)

                # Find point of maximum absolute second derivative
                if len(second_deriv) > 0:
                    abs_second = np.abs(second_deriv)
                    cp_idx = int(np.argmax(abs_second)) + 1  # offset for diff
                    max_slope_change = float(abs_second.max())
                else:
                    cp_idx = n_steps // 2
                    max_slope_change = 0.0

                change_points_per_ev.append(cp_idx)
                max_slope_changes_per_ev.append(max_slope_change)

                # Auto-determine threshold if not provided
                if punctuation_threshold is None:
                    # Use 3 * median absolute second derivative as threshold
                    threshold = 3.0 * float(np.median(np.abs(second_deriv))) if len(second_deriv) > 0 else 0.01
                else:
                    threshold = punctuation_threshold

                if max_slope_change > threshold:
                    change_types_per_ev.append("punctuated")
                    total_punctuated += 1
                else:
                    change_types_per_ev.append("gradual")
                    total_gradual += 1

                # ------------------------------------------------------------------
                # 3. Temporal autocorrelation (lag-1)
                # ------------------------------------------------------------------
                if len(traj) > 1:
                    mean_t = traj.mean()
                    var_t = traj.var()
                    if var_t > 1e-15:
                        autocorr = float(
                            np.mean((traj[:-1] - mean_t) * (traj[1:] - mean_t)) / var_t
                        )
                    else:
                        autocorr = 1.0
                else:
                    autocorr = 0.0
                autocorrelations_per_ev.append(autocorr)

            all_trajectories[code.value] = trajectories.tolist()
            all_change_points[code.value] = change_points_per_ev
            all_change_types[code.value] = change_types_per_ev
            all_autocorrelations[code.value] = autocorrelations_per_ev
            all_max_slope_changes[code.value] = max_slope_changes_per_ev

        # ------------------------------------------------------------------
        # 4. Aggregate metrics
        # ------------------------------------------------------------------
        all_autocorr_flat = [
            v for vals in all_autocorrelations.values() for v in vals
        ]

        metrics: dict = {
            "n_punctuated": total_punctuated,
            "n_gradual": total_gradual,
            "fraction_punctuated": total_punctuated / max(total_punctuated + total_gradual, 1),
            "change_points": all_change_points,
            "change_types": all_change_types,
            "max_slope_changes": all_max_slope_changes,
            "autocorrelations": all_autocorrelations,
            "mean_autocorrelation": float(np.mean(all_autocorr_flat)) if all_autocorr_flat else 0.0,
            "trajectories": all_trajectories,
            "dialect_order": [c.value for c in dialect_codes],
            "n_time_steps": n_steps,
            "ou_params": {"theta": theta, "sigma": sigma, "dt": dt},
        }

        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        n_punct = result.metrics.get("n_punctuated", 0)
        n_grad = result.metrics.get("n_gradual", 0)
        total = n_punct + n_grad
        frac_punct = n_punct / max(total, 1)
        mean_autocorr = result.metrics.get("mean_autocorrelation", 0.0)

        # With OU process and moderate sigma, expect mostly gradual changes
        # Punctuated fraction should be small (< 0.3) for well-calibrated OU
        dynamics_consistent = frac_punct < 0.5

        return {
            "n_punctuated": int(n_punct),
            "n_gradual": int(n_grad),
            "fraction_punctuated": float(frac_punct),
            "mean_autocorrelation": float(mean_autocorr),
            "dynamics_consistent_with_ou": dynamics_consistent,
            "high_autocorrelation": mean_autocorr > 0.8,
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

        dialect_order = result.metrics.get("dialect_order", [])
        trajectories = result.metrics.get("trajectories", {})
        change_points = result.metrics.get("change_points", {})
        change_types = result.metrics.get("change_types", {})
        n_steps = result.metrics.get("n_time_steps", 100)

        # --- 1. Eigenvalue trajectories per dialect (grid of subplots) ---
        n_dialects = len(dialect_order)
        if n_dialects > 0 and trajectories:
            n_cols = 2
            n_rows = (n_dialects + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            axes_flat = np.array(axes).flatten() if n_dialects > 1 else [axes]

            time_axis = np.arange(n_steps)

            for i, code_str in enumerate(dialect_order):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                traj_data = np.array(trajectories.get(code_str, []))
                cps = change_points.get(code_str, [])
                cts = change_types.get(code_str, [])

                if traj_data.size == 0:
                    ax.set_visible(False)
                    continue

                n_ev = min(traj_data.shape[0], 5)  # show at most 5 eigenvalues
                colors = plt.cm.tab10(np.linspace(0, 1, n_ev))

                for k in range(n_ev):
                    ax.plot(
                        time_axis, traj_data[k],
                        color=colors[k], linewidth=0.8, alpha=0.8,
                        label=f"lambda_{k}",
                    )
                    # Mark change point
                    if k < len(cps):
                        cp = cps[k]
                        ct = cts[k] if k < len(cts) else "unknown"
                        marker = "v" if ct == "punctuated" else "o"
                        mcolor = "red" if ct == "punctuated" else "green"
                        ax.axvline(cp, color=mcolor, linestyle=":", alpha=0.4)
                        ax.plot(
                            cp, traj_data[k, cp] if cp < traj_data.shape[1] else traj_data[k, -1],
                            marker, color=mcolor, markersize=6, zorder=5,
                        )

                name = DIALECT_NAMES.get(DialectCode(code_str), code_str)
                ax.set_title(f"{name} ({code_str})", fontsize=9)
                ax.set_xlabel("Time step", fontsize=8)
                ax.set_ylabel("Eigenvalue magnitude", fontsize=8)
                ax.tick_params(labelsize=7)

            # Hide unused subplots
            for j in range(n_dialects, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(
                "Eigenvalue Archaeology: Temporal Trajectories (OU Process)",
                fontsize=12,
            )
            plt.tight_layout()
            p = output_dir / "eigenvalue_trajectories.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        # --- 2. Change type summary bar chart ---
        if change_types:
            fig, ax = plt.subplots(figsize=(10, 5))

            dialect_labels = []
            punct_counts = []
            grad_counts = []

            for code_str in dialect_order:
                cts = change_types.get(code_str, [])
                n_p = sum(1 for t in cts if t == "punctuated")
                n_g = sum(1 for t in cts if t == "gradual")
                dialect_labels.append(code_str)
                punct_counts.append(n_p)
                grad_counts.append(n_g)

            x = np.arange(len(dialect_labels))
            width = 0.35
            ax.bar(x - width / 2, punct_counts, width, label="Punctuated", color="red", alpha=0.7)
            ax.bar(x + width / 2, grad_counts, width, label="Gradual", color="green", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(dialect_labels, rotation=45, ha="right")
            ax.set_ylabel("Number of Eigenvalues")
            ax.set_title("Change Type Classification per Dialect")
            ax.legend()
            plt.tight_layout()
            p = output_dir / "change_type_summary.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        # --- 3. Autocorrelation heatmap ---
        autocorrelations = result.metrics.get("autocorrelations", {})
        if autocorrelations:
            ac_matrix = []
            labels_y = []
            for code_str in dialect_order:
                ac = autocorrelations.get(code_str, [])
                if ac:
                    ac_matrix.append(ac)
                    labels_y.append(code_str)

            if ac_matrix:
                ac_arr = np.array(ac_matrix)
                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(ac_arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
                ax.set_yticks(range(len(labels_y)))
                ax.set_yticklabels(labels_y)
                ax.set_xlabel("Eigenvalue Index")
                ax.set_ylabel("Dialect")
                ax.set_title("Temporal Autocorrelation of Eigenvalue Trajectories")
                fig.colorbar(im, ax=ax, label="Lag-1 Autocorrelation")
                plt.tight_layout()
                p = output_dir / "autocorrelation_heatmap.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                paths.append(p)

        return paths
