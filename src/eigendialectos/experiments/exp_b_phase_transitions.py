"""Experiment B: Dialect Phase Transitions.

Detect mathematical isoglosses as discontinuities in the eigenvalue field
(high-gradient regions) and model dialect boundaries via a Potts model
simulation.  A temperature sweep reveals phase transitions -- the critical
temperature at which dialect boundaries sharpen or dissolve.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from eigendialectos.constants import DialectCode, DIALECT_COORDINATES
from eigendialectos.experiments.base import Experiment
from eigendialectos.geometry.eigenfield import EigenvalueField
from eigendialectos.types import ExperimentResult

logger = logging.getLogger(__name__)


class PhaseTransitionsExperiment(Experiment):
    experiment_id = "exp_b_phase_transitions"
    name = "Dialect Phase Transitions"
    description = (
        "Compute the continuous eigenvalue field via GP interpolation, "
        "identify isogloss lines from gradient discontinuities, and "
        "simulate a Potts model to find the critical temperature at which "
        "dialect boundaries undergo phase transitions."
    )
    dependencies = [
        "eigendialectos.geometry.eigenfield",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._coordinates: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._dialect_codes: list[DialectCode] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        self._config = config
        seed = config.get("seed", 42)
        dim = config.get("dim", 50)
        n_eigenvalues = config.get("n_eigenvalues", 5)
        rng = np.random.default_rng(seed)

        data_dir = config.get("data_dir")
        loaded = False

        if data_dir:
            data_path = Path(data_dir)
            ev_path = data_path / "eigenvalues.npy"
            coords_path = data_path / "coordinates.npy"
            if ev_path.exists() and coords_path.exists():
                self._eigenvalues = np.load(str(ev_path))
                self._coordinates = np.load(str(coords_path))
                self._dialect_codes = list(DialectCode)[:self._coordinates.shape[0]]
                loaded = True
                logger.info(
                    "Loaded eigenvalue data from %s", data_dir
                )

        if not loaded:
            logger.info(
                "Generating synthetic eigenvalue data for %d dialects.",
                len(DialectCode),
            )
            self._dialect_codes = sorted(DialectCode, key=lambda c: c.value)
            coords_list = []
            ev_list = []

            for code in self._dialect_codes:
                lat, lon = DIALECT_COORDINATES[code]
                coords_list.append([lat, lon])
                # Synthetic eigenvalues: base spectrum + dialect-specific perturbation
                base = np.linspace(1.0, 0.3, n_eigenvalues)
                perturbation = rng.standard_normal(n_eigenvalues) * 0.1
                ev_list.append(np.abs(base + perturbation))

            self._coordinates = np.array(coords_list, dtype=np.float64)
            self._eigenvalues = np.array(ev_list, dtype=np.float64)

        self._is_setup = True

    def run(self) -> ExperimentResult:
        self._check_setup()
        assert self._coordinates is not None
        assert self._eigenvalues is not None

        seed = self._config.get("seed", 42)
        rng = np.random.default_rng(seed)
        resolution = self._config.get("field_resolution", 40)

        # ------------------------------------------------------------------
        # 1. Compute eigenvalue field via GP
        # ------------------------------------------------------------------
        ef = EigenvalueField(
            kernel_lengthscale=self._config.get("gp_lengthscale", 15.0),
            kernel_variance=1.0,
            noise_variance=0.01,
        )
        ef.fit(self._coordinates, self._eigenvalues)
        field_result = ef.compute_field(resolution=resolution)

        # ------------------------------------------------------------------
        # 2. Gradient magnitude field (per eigenvalue, then average)
        # ------------------------------------------------------------------
        n_ev = field_result.eigenvalue_surfaces.shape[0]
        grad_magnitudes = np.zeros(
            (n_ev, resolution, resolution), dtype=np.float64
        )

        for k in range(n_ev):
            surface = field_result.eigenvalue_surfaces[k]
            grad_lat = np.gradient(surface, axis=0)
            grad_lon = np.gradient(surface, axis=1)
            grad_magnitudes[k] = np.sqrt(grad_lat ** 2 + grad_lon ** 2)

        avg_gradient = np.mean(grad_magnitudes, axis=0)

        # ------------------------------------------------------------------
        # 3. Identify isogloss lines (high-gradient contours)
        # ------------------------------------------------------------------
        grad_threshold = self._config.get("isogloss_threshold", 0.5)
        max_grad = avg_gradient.max()
        if max_grad > 1e-15:
            isogloss_mask = avg_gradient > grad_threshold * max_grad
            n_isogloss_pixels = int(np.sum(isogloss_mask))
        else:
            isogloss_mask = np.zeros_like(avg_gradient, dtype=bool)
            n_isogloss_pixels = 0

        # Count connected isogloss regions as proxy for number of isoglosses
        try:
            from scipy.ndimage import label as ndimage_label
            labeled, n_isoglosses = ndimage_label(isogloss_mask)
        except ImportError:
            n_isoglosses = 1 if n_isogloss_pixels > 0 else 0

        # ------------------------------------------------------------------
        # 4. Potts model simulation
        # ------------------------------------------------------------------
        n_dialects = len(self._dialect_codes)
        n_states = n_dialects  # each "spin" can be one of n dialect identities

        # Build coupling matrix J_ij from eigenvalue similarity
        J = np.zeros((n_dialects, n_dialects), dtype=np.float64)
        for i in range(n_dialects):
            for j in range(i + 1, n_dialects):
                # J_ij = exp(-||lambda_i - lambda_j||^2 / sigma^2)
                diff = self._eigenvalues[i] - self._eigenvalues[j]
                sim = float(np.exp(-np.dot(diff, diff) / 0.5))
                J[i, j] = sim
                J[j, i] = sim

        # Simulate on a small lattice with Metropolis algorithm
        lattice_size = self._config.get("lattice_size", 20)
        n_sites = lattice_size * lattice_size
        n_sweeps = self._config.get("potts_sweeps", 200)

        # Temperature sweep
        T_min = self._config.get("T_min", 0.1)
        T_max = self._config.get("T_max", 5.0)
        n_temps = self._config.get("n_temps", 30)
        temperatures = np.linspace(T_min, T_max, n_temps)

        energy_means = np.zeros(n_temps, dtype=np.float64)
        energy_sq_means = np.zeros(n_temps, dtype=np.float64)
        heat_capacities = np.zeros(n_temps, dtype=np.float64)

        for t_idx, T in enumerate(temperatures):
            # Initialize random spin configuration
            spins = rng.integers(0, n_states, size=(lattice_size, lattice_size))

            energies_at_T = []

            for sweep in range(n_sweeps):
                # One sweep = n_sites random single-spin updates
                for _ in range(n_sites):
                    x = rng.integers(0, lattice_size)
                    y = rng.integers(0, lattice_size)
                    current_spin = spins[x, y]
                    new_spin = rng.integers(0, n_states)

                    # Compute energy change
                    dE = 0.0
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = (x + dx) % lattice_size, (y + dy) % lattice_size
                        neighbor_spin = spins[nx, ny]
                        # Potts interaction: -J * delta(sigma_i, sigma_j)
                        j_old = J[current_spin, neighbor_spin] if current_spin != neighbor_spin else J[current_spin, current_spin]
                        j_new = J[new_spin, neighbor_spin] if new_spin != neighbor_spin else J[new_spin, new_spin]
                        # delta function contribution
                        old_contribution = -1.0 * (1.0 if current_spin == neighbor_spin else 0.0) * (1.0 + J[current_spin, neighbor_spin])
                        new_contribution = -1.0 * (1.0 if new_spin == neighbor_spin else 0.0) * (1.0 + J[new_spin, neighbor_spin])
                        dE += new_contribution - old_contribution

                    # Metropolis acceptance
                    if dE <= 0 or rng.random() < np.exp(-dE / T):
                        spins[x, y] = new_spin

                # Compute total energy after this sweep (last half for equilibration)
                if sweep >= n_sweeps // 2:
                    E_total = 0.0
                    for x in range(lattice_size):
                        for y in range(lattice_size):
                            s = spins[x, y]
                            # Right and down neighbors (avoid double counting)
                            for dx, dy in [(1, 0), (0, 1)]:
                                nx, ny = (x + dx) % lattice_size, (y + dy) % lattice_size
                                ns = spins[nx, ny]
                                E_total += -(1.0 if s == ns else 0.0) * (1.0 + J[s, ns])
                    energies_at_T.append(E_total)

            energies_arr = np.array(energies_at_T, dtype=np.float64)
            energy_means[t_idx] = energies_arr.mean()
            energy_sq_means[t_idx] = (energies_arr ** 2).mean()

            # Heat capacity: C(T) = (<E^2> - <E>^2) / T^2
            variance = energy_sq_means[t_idx] - energy_means[t_idx] ** 2
            heat_capacities[t_idx] = max(variance, 0.0) / (T ** 2)

        # ------------------------------------------------------------------
        # 5. Find critical temperature (heat capacity peak)
        # ------------------------------------------------------------------
        critical_idx = int(np.argmax(heat_capacities))
        critical_temperature = float(temperatures[critical_idx])
        peak_heat_capacity = float(heat_capacities[critical_idx])

        metrics: dict = {
            "critical_temperature": critical_temperature,
            "peak_heat_capacity": peak_heat_capacity,
            "n_isoglosses": int(n_isoglosses),
            "n_isogloss_pixels": n_isogloss_pixels,
            "temperatures": temperatures.tolist(),
            "energy_curve": energy_means.tolist(),
            "heat_capacity_curve": heat_capacities.tolist(),
            "coupling_matrix": J.tolist(),
            "dialect_order": [c.value for c in self._dialect_codes],
            "avg_gradient_max": float(max_grad),
            "field_grid_lat": field_result.grid_lat.tolist(),
            "field_grid_lon": field_result.grid_lon.tolist(),
            "eigenvalue_surface_0": field_result.eigenvalue_surfaces[0].tolist(),
            "avg_gradient_field": avg_gradient.tolist(),
        }

        return self._make_result(metrics)

    def evaluate(self, result: ExperimentResult) -> dict:
        Tc = result.metrics.get("critical_temperature", 0.0)
        peak_C = result.metrics.get("peak_heat_capacity", 0.0)
        heat_caps = np.array(result.metrics.get("heat_capacity_curve", []))
        n_iso = result.metrics.get("n_isoglosses", 0)

        # Check sharpness of phase transition: ratio of peak to median
        if len(heat_caps) > 0:
            median_C = float(np.median(heat_caps))
            sharpness = peak_C / max(median_C, 1e-10)
        else:
            sharpness = 0.0

        return {
            "critical_temperature": float(Tc),
            "peak_heat_capacity": float(peak_C),
            "transition_sharpness": sharpness,
            "sharp_transition": sharpness > 2.0,
            "n_isoglosses": int(n_iso),
            "has_isoglosses": n_iso > 0,
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

        temperatures = np.array(result.metrics.get("temperatures", []))
        energy_curve = np.array(result.metrics.get("energy_curve", []))
        heat_caps = np.array(result.metrics.get("heat_capacity_curve", []))
        Tc = result.metrics.get("critical_temperature", 0.0)

        # --- 1. Eigenvalue field heatmap ---
        surface = np.array(result.metrics.get("eigenvalue_surface_0", []))
        grid_lat = np.array(result.metrics.get("field_grid_lat", []))
        grid_lon = np.array(result.metrics.get("field_grid_lon", []))

        if surface.size > 0 and grid_lat.size > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            extent = [grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()]
            im = ax.imshow(
                surface, extent=extent, origin="lower", cmap="viridis", aspect="auto"
            )
            fig.colorbar(im, ax=ax, label="Eigenvalue (k=0)")
            # Overlay dialect positions
            for code in self._dialect_codes or []:
                if code in DIALECT_COORDINATES:
                    lat, lon = DIALECT_COORDINATES[code]
                    ax.plot(lon, lat, "r^", markersize=10, markeredgecolor="k")
                    ax.annotate(
                        code.value, (lon, lat),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
                    )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Eigenvalue Field (GP Interpolation, k=0)")
            plt.tight_layout()
            p = output_dir / "eigenvalue_field.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        # --- 2. Gradient magnitude heatmap ---
        avg_grad = np.array(result.metrics.get("avg_gradient_field", []))
        if avg_grad.size > 0 and grid_lat.size > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            extent = [grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()]
            im = ax.imshow(
                avg_grad, extent=extent, origin="lower", cmap="hot", aspect="auto"
            )
            fig.colorbar(im, ax=ax, label="Gradient Magnitude")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Eigenvalue Gradient Field (Isogloss Indicator)")
            plt.tight_layout()
            p = output_dir / "gradient_field.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        # --- 3. Potts energy vs temperature ---
        if len(temperatures) > 0 and len(energy_curve) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(temperatures, energy_curve, "b-", linewidth=1.5)
            ax1.axvline(Tc, color="r", linestyle="--", alpha=0.7, label=f"Tc = {Tc:.2f}")
            ax1.set_xlabel("Temperature T")
            ax1.set_ylabel("Mean Energy <E>")
            ax1.set_title("Potts Model: Energy vs Temperature")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(temperatures, heat_caps, "r-", linewidth=1.5)
            ax2.axvline(Tc, color="b", linestyle="--", alpha=0.7, label=f"Tc = {Tc:.2f}")
            ax2.set_xlabel("Temperature T")
            ax2.set_ylabel("Heat Capacity C(T)")
            ax2.set_title("Potts Model: Heat Capacity (Phase Transition)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            p = output_dir / "potts_energy_temperature.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        # --- 4. Coupling matrix ---
        J = np.array(result.metrics.get("coupling_matrix", []))
        order = result.metrics.get("dialect_order", [])
        if J.size > 0:
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(J, cmap="Blues")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order)
            fig.colorbar(im, ax=ax, label="Coupling J_ij")
            ax.set_title("Potts Model Coupling Matrix")
            plt.tight_layout()
            p = output_dir / "coupling_matrix.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)

        return paths
