"""Experiment E: Code-Switching Dynamics.

Model bidialectal speakers as oscillating eigenvalue systems.  A mixing
parameter alpha(t) interpolates between two dialect transformation matrices
on the Lie group (via matrix logarithm), and the spectral consequences of
different switching regimes (sinusoidal, random telegraph, logistic) are
analysed through power spectra and eigenvalue trajectories.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.linalg import expm, logm

from eigendialectos.constants import DialectCode
from eigendialectos.experiments.base import Experiment
from eigendialectos.types import ExperimentResult

logger = logging.getLogger(__name__)


class CodeSwitchingExperiment(Experiment):
    experiment_id = "exp_e_code_switching"
    name = "Code-Switching Dynamics"
    description = (
        "Model bidialectal speakers as oscillating eigenvalue systems.  "
        "Simulate code-switching via time-varying alpha(t) between two "
        "dialect transformation matrices using sinusoidal, random telegraph, "
        "and logistic transition models.  Analyse spectral consequences "
        "through power spectra and eigenvalue trajectories."
    )
    dependencies = ["scipy", "numpy"]

    def __init__(self) -> None:
        super().__init__()
        self._W_A: np.ndarray | None = None
        self._W_B: np.ndarray | None = None
        self._log_A: np.ndarray | None = None
        self._log_B: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 10)
        rng = np.random.default_rng(seed)

        # Try to load real W matrices for two dialects
        data_dir = config.get("data_dir")
        loaded = False
        if data_dir and Path(data_dir).exists():
            path_a = Path(data_dir) / "W_ES_PEN.npy"
            path_b = Path(data_dir) / "W_ES_AND.npy"
            if path_a.exists() and path_b.exists():
                self._W_A = np.load(str(path_a)).astype(np.float64)
                self._W_B = np.load(str(path_b)).astype(np.float64)
                loaded = True
                logger.info("Loaded real W matrices from %s", data_dir)

        if not loaded:
            logger.info(
                "Generating synthetic W matrices (dim=%d) for code-switching.", dim
            )
            self._W_A = np.eye(dim) + 0.1 * rng.standard_normal((dim, dim))
            self._W_B = np.eye(dim) + 0.1 * rng.standard_normal((dim, dim))

        # Pre-compute matrix logarithms
        self._log_A = logm(self._W_A.astype(np.complex128))
        self._log_B = logm(self._W_B.astype(np.complex128))

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()

        seed = self._config.get("seed", 42)
        rng = np.random.default_rng(seed)
        n_time = self._config.get("n_time_steps", 1000)
        n_alpha_samples = self._config.get("n_alpha_samples", 50)
        t = np.linspace(0.0, 1.0, n_time)

        # ---- 1. Generate alpha(t) signals for each switching model ----
        frequencies = [0.01, 0.05, 0.1, 0.5]
        alpha_models: dict[str, np.ndarray] = {}

        # a) Sinusoidal
        for f in frequencies:
            alpha_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * f * np.arange(n_time))
            alpha_models[f"sinusoidal_f{f}"] = alpha_sin

        # b) Random telegraph (Poisson switching between 0 and 1)
        telegraph = np.zeros(n_time)
        state = 0
        switch_rate = self._config.get("telegraph_rate", 0.05)
        for i in range(n_time):
            if rng.random() < switch_rate:
                state = 1 - state
            telegraph[i] = float(state)
        alpha_models["telegraph"] = telegraph

        # c) Smooth logistic transitions at random trigger points
        n_triggers = self._config.get("n_logistic_triggers", 5)
        trigger_points = np.sort(rng.choice(n_time, size=n_triggers, replace=False))
        logistic_alpha = np.zeros(n_time)
        current_target = 1.0
        steepness = self._config.get("logistic_steepness", 0.1)
        for tp in trigger_points:
            sigmoid = 1.0 / (1.0 + np.exp(-steepness * (np.arange(n_time) - tp)))
            if current_target == 1.0:
                logistic_alpha += sigmoid
            else:
                logistic_alpha += 1.0 - sigmoid
            current_target = 1.0 - current_target
        # Normalise to [0, 1]
        lo, hi = logistic_alpha.min(), logistic_alpha.max()
        if hi - lo > 1e-12:
            logistic_alpha = (logistic_alpha - lo) / (hi - lo)
        alpha_models["logistic"] = logistic_alpha

        # ---- 2. Compute switching rates = mean|d alpha / dt| ----
        switching_rates: dict[str, float] = {}
        for name, alpha in alpha_models.items():
            d_alpha = np.diff(alpha)
            switching_rates[name] = float(np.mean(np.abs(d_alpha)))

        # ---- 3. Compute power spectra via FFT ----
        power_spectra: dict[str, list[float]] = {}
        for name, alpha in alpha_models.items():
            fft_vals = np.fft.rfft(alpha - np.mean(alpha))
            psd = np.abs(fft_vals) ** 2
            power_spectra[name] = psd.tolist()

        # ---- 4. For each model, compute W(t) at a few t and get eigenvalues ----
        # We compute W(t) = expm(alpha * log_A + (1-alpha) * log_B)
        eig_trajectories_by_model: dict[str, list[list[complex]]] = {}
        sample_indices = np.linspace(0, n_time - 1, min(20, n_time), dtype=int)

        for name, alpha in alpha_models.items():
            trajectories: list[list[complex]] = []
            for idx in sample_indices:
                a = alpha[idx]
                W_t = expm(a * self._log_A + (1.0 - a) * self._log_B)
                evals = np.linalg.eigvals(W_t)
                # Sort by magnitude for consistency
                order = np.argsort(-np.abs(evals))
                trajectories.append([complex(evals[o]) for o in order])
            eig_trajectories_by_model[name] = trajectories

        # ---- 5. Eigenvalue spectrum as a function of alpha (50 values) ----
        alpha_sweep = np.linspace(0.0, 1.0, n_alpha_samples)
        eigenvalue_vs_alpha: list[list[complex]] = []
        for a in alpha_sweep:
            W_a = expm(a * self._log_A + (1.0 - a) * self._log_B)
            evals = np.linalg.eigvals(W_a)
            order = np.argsort(-np.abs(evals))
            eigenvalue_vs_alpha.append([complex(evals[o]) for o in order])

        # ---- Pack metrics ----
        # Convert complex eigenvalue lists to serialisable form
        def _complex_list_to_dict(clist: list[complex]) -> list[dict[str, float]]:
            return [{"real": c.real, "imag": c.imag} for c in clist]

        eig_traj_serial: dict[str, list[list[dict[str, float]]]] = {}
        for name, traj in eig_trajectories_by_model.items():
            eig_traj_serial[name] = [_complex_list_to_dict(step) for step in traj]

        eig_vs_alpha_serial = [_complex_list_to_dict(step) for step in eigenvalue_vs_alpha]

        # Store raw alpha curves for visualisation
        alpha_curves: dict[str, list[float]] = {
            name: alpha.tolist() for name, alpha in alpha_models.items()
        }

        metrics: dict = {
            "switching_rates": switching_rates,
            "power_spectra": power_spectra,
            "eigenvalue_trajectories": eig_traj_serial,
            "eigenvalue_vs_alpha": eig_vs_alpha_serial,
            "alpha_sweep": alpha_sweep.tolist(),
            "alpha_curves": alpha_curves,
            "model_names": list(alpha_models.keys()),
            "n_time_steps": n_time,
            "dim": self._W_A.shape[0],
        }

        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        """Identify dominant switching frequency and stability of intermediates."""
        power_spectra = result.metrics["power_spectra"]
        switching_rates = result.metrics["switching_rates"]
        n_time = result.metrics["n_time_steps"]

        # Dominant frequency for each model
        dominant_frequencies: dict[str, float] = {}
        for name, psd in power_spectra.items():
            psd_arr = np.array(psd)
            if len(psd_arr) > 1:
                # Skip DC component (index 0)
                peak_idx = int(np.argmax(psd_arr[1:])) + 1
                freq = peak_idx / n_time
                dominant_frequencies[name] = float(freq)
            else:
                dominant_frequencies[name] = 0.0

        # Stability of intermediate states: look at eigenvalues at alpha=0.5
        eig_vs_alpha = result.metrics["eigenvalue_vs_alpha"]
        alpha_sweep = result.metrics["alpha_sweep"]
        mid_idx = int(np.argmin(np.abs(np.array(alpha_sweep) - 0.5)))
        mid_eigs = eig_vs_alpha[mid_idx]
        mid_magnitudes = [abs(complex(e["real"], e["imag"])) for e in mid_eigs]
        intermediate_stability = float(np.std(mid_magnitudes))

        # Check if eigenvalue magnitudes stay bounded across full sweep
        max_magnitude_per_alpha: list[float] = []
        for step in eig_vs_alpha:
            mags = [abs(complex(e["real"], e["imag"])) for e in step]
            max_magnitude_per_alpha.append(max(mags))
        all_bounded = all(m < 10.0 for m in max_magnitude_per_alpha)

        return {
            "dominant_frequencies": dominant_frequencies,
            "intermediate_stability_std": intermediate_stability,
            "intermediate_max_eigenvalue_mag": float(max(mid_magnitudes)),
            "eigenvalues_bounded": all_bounded,
            "max_eigenvalue_magnitude": float(max(max_magnitude_per_alpha)),
            "mean_switching_rate": float(np.mean(list(switching_rates.values()))),
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

        alpha_curves = result.metrics["alpha_curves"]
        power_spectra = result.metrics["power_spectra"]
        eig_vs_alpha = result.metrics["eigenvalue_vs_alpha"]
        alpha_sweep = result.metrics["alpha_sweep"]
        model_names = result.metrics["model_names"]

        # ---- 1. Alpha(t) for each switching model ----
        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models), squeeze=False)
        for i, name in enumerate(model_names):
            ax = axes[i, 0]
            curve = alpha_curves[name]
            ax.plot(curve, linewidth=0.7)
            ax.set_ylabel(r"$\alpha(t)$")
            ax.set_title(f"Switching model: {name}")
            ax.set_ylim(-0.05, 1.05)
        axes[-1, 0].set_xlabel("Time step")
        plt.tight_layout()
        p = output_dir / "alpha_curves.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # ---- 2. Power spectra ----
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models), squeeze=False)
        for i, name in enumerate(model_names):
            ax = axes[i, 0]
            psd = np.array(power_spectra[name])
            freqs = np.arange(len(psd))
            ax.semilogy(freqs[1:], psd[1:], linewidth=0.7)
            ax.set_ylabel("Power")
            ax.set_title(f"Power spectrum: {name}")
        axes[-1, 0].set_xlabel("Frequency bin")
        plt.tight_layout()
        p = output_dir / "power_spectra.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # ---- 3. Eigenvalue trajectories vs alpha ----
        n_eigs = len(eig_vs_alpha[0]) if eig_vs_alpha else 0
        if n_eigs > 0:
            # Plot magnitude of each eigenvalue as alpha sweeps from 0 to 1
            fig, ax = plt.subplots(figsize=(10, 6))
            for k in range(min(n_eigs, 8)):  # cap at 8 eigenvalues for readability
                magnitudes = []
                for step in eig_vs_alpha:
                    e = step[k]
                    magnitudes.append(abs(complex(e["real"], e["imag"])))
                ax.plot(alpha_sweep, magnitudes, label=f"$|\\lambda_{{{k}}}|$", linewidth=1.2)
            ax.set_xlabel(r"$\alpha$ (0 = dialect B, 1 = dialect A)")
            ax.set_ylabel("Eigenvalue magnitude")
            ax.set_title("Eigenvalue Trajectories vs Mixing Parameter")
            ax.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            p = output_dir / "eigenvalue_trajectories.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

            # Complex plane plot
            fig, ax = plt.subplots(figsize=(8, 8))
            cmap = plt.get_cmap("viridis")
            for step_idx, step in enumerate(eig_vs_alpha):
                color = cmap(step_idx / max(len(eig_vs_alpha) - 1, 1))
                reals = [e["real"] for e in step]
                imags = [e["imag"] for e in step]
                ax.scatter(reals, imags, color=color, s=10, alpha=0.6)
            ax.set_xlabel("Re")
            ax.set_ylabel("Im")
            ax.set_title("Eigenvalues in Complex Plane (colour = alpha)")
            ax.set_aspect("equal")
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.axvline(0, color="grey", linewidth=0.5)
            # Unit circle
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.5, alpha=0.4)
            plt.tight_layout()
            p = output_dir / "eigenvalue_complex_plane.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        return paths
