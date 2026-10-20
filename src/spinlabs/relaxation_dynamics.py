import warnings
from pathlib import Path

import numpy as np
import scipy.constants as Cn
from scipy.optimize import OptimizeWarning, curve_fit
from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Initial spin states in table/column order, and their initial populations
# over (ms=+1, ms=0, ms=-1).
INIT_STATES = ("ms_1", "ms_0", "ms_-1")
INIT_POPULATIONS = {
    "ms_1": [1.0, 0.0, 0.0],
    "ms_0": [0.0, 1.0, 0.0],
    "ms_-1": [0.0, 0.0, 1.0],
}


def normalize_init_state(value: str) -> str:
    """Accept ``ms_0``, ``ms0``, ``0`` (and likewise for 1/-1/+1) and return the ``ms_<m>`` name."""
    text = str(value).strip().lower().removeprefix("ms").lstrip("_").removeprefix("+")
    return f"ms_{text}"


def t1_file_name(init_state: str) -> str:
    """T1 result file of one initial state inside a run directory."""
    return f"t1_relaxation_{init_state}.npz"


def find_t1_files(run_dir: str | Path) -> list[Path]:
    """T1 result files in a run directory, in :data:`INIT_STATES` order.

    Runs solve the dynamics for several initial states into
    ``t1_relaxation_<state>.npz``; older runs hold a single
    ``t1_relaxation.npz``, which is returned when no per-state file exists.
    """
    run_dir = Path(run_dir)
    files = [run_dir / t1_file_name(s) for s in INIT_STATES if (run_dir / t1_file_name(s)).exists()]
    legacy = run_dir / "t1_relaxation.npz"
    return files or ([legacy] if legacy.exists() else [])


class RelaxationDynamics:
    def __init__(self, init_state, rates_data):
        self.defect: str = None
        self.cell_size: int = None
        self.sim_type: str = None

        self.population: np.ndarray = np.array(init_state)
        self.rate_matrices, self.temperatures = self.read_data(rates_data)

    def read_data(self, data_file):
        data = np.load(data_file, allow_pickle=True)
        self.defect = data.get("defect")
        self.cell_size = data.get("cell_size")
        self.sim_type = "(spin corrected)" if data.get("calc_method") == "defect_band_approx" else ""

        rate_matrices = []

        LOGGER.info("Creating rate matrices...")
        for i, t in enumerate(data.get("temperatures")):
            A = [
                [-data["1_to_0"][i] - data["1_to_-1"][i], data["0_to_1"][i], data["-1_to_1"][i]],
                [data["1_to_0"][i], -data["0_to_1"][i] - data["0_to_-1"][i], data["-1_to_0"][i]],
                [data["1_to_-1"][i], data["0_to_-1"][i], -data["-1_to_0"][i] - data["-1_to_1"][i]],
            ]

            rate_matrices.append(A)

        temperatures = np.array(data.get("temperatures", []))

        return np.array(rate_matrices), temperatures

    @staticmethod
    def _decay_modes(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Eigenvalues and eigenvectors of the rate matrix, and the index of its steady state.

        Every column of a rate matrix sums to zero, so one eigenvalue is exactly 0 (the
        steady state). Round-off leaves it at about 1e-16 x the largest rate, which for
        GHz rates is far above any fixed threshold, so it is taken as the eigenvalue
        closest to zero and set to 0.
        """
        eigenvalues, eigenvectors = np.linalg.eig(A)
        steady = int(np.argmin(np.abs(eigenvalues)))
        eigenvalues = eigenvalues.copy()
        eigenvalues[steady] = 0.0
        return eigenvalues, eigenvectors, steady

    def _get_temp_index(self, T):
        if T:
            temp_index = np.argmin(np.abs(self.temperatures - T))
        else:
            raise ValueError(f"Incorrect temperature format: {T}")
        return temp_index

    def fit_T1_exp(self, population, time_points, steady_state, T1_eigenval):
        def exp_func(t, T1, c, a):
            return a * np.exp(-t / T1) + c

        a_init = population[0] - steady_state
        initial_guess = [T1_eigenval, steady_state, a_init]

        try:
            with warnings.catch_warnings(), np.errstate(over="ignore"):
                # Only the fitted T1 is used, not its covariance; trial parameters
                # may overflow exp, and the result is checked below.
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(exp_func, time_points, population, p0=initial_guess, maxfev=5000)
            extracted_T1 = float(popt[0])
            if not np.isfinite(extracted_T1) or extracted_T1 <= 0:
                return T1_eigenval
            return extracted_T1
        except Exception:
            return T1_eigenval

    def simulate_relaxation(self, time_points, T=None, temp_index=None):
        if temp_index is None:
            temp_index = self._get_temp_index(T)
        A = self.rate_matrices[temp_index]

        # dP/dt = A P with constant A: P(t) = V exp(L t) V^-1 P0, exact and free of the
        # step limits of an ODE integrator on this stiff system (rates spanning many decades).
        eigenvalues, eigenvectors, _ = self._decay_modes(A)
        coeffs = np.linalg.solve(eigenvectors, self.population.astype(complex))
        t = np.asarray(time_points, dtype=float)[:, None]
        population_evolution = np.real((np.exp(eigenvalues[None, :] * t) * coeffs[None, :]) @ eigenvectors.T)

        return population_evolution

    def get_T1_eigenval(self, T=None, temp_index=None):
        if temp_index is None:
            temp_index = self._get_temp_index(T)
        A = self.rate_matrices[temp_index]

        eigenvalues, eigenvectors, steady = self._decay_modes(A)
        decays = np.real(np.delete(eigenvalues, steady))
        decays = decays[decays < 0]

        # T1 = the slowest decay; no decaying eigenvalue (all rates zero) = no relaxation
        relaxation_time = float(-1.0 / np.max(decays)) if len(decays) else np.inf

        return relaxation_time, eigenvalues, eigenvectors

    def get_steady_state(self, T=None, temp_index=None):
        if temp_index is None:
            temp_index = self._get_temp_index(T)
        A = self.rate_matrices[temp_index]

        _, eigenvectors, zero_idx = self._decay_modes(A)
        steady_state = np.real(eigenvectors[:, zero_idx])
        sum_s = np.sum(steady_state)
        if np.abs(sum_s) > 1e-12:
            steady_state = steady_state / sum_s

        return steady_state

    def compute_T1_range(self):
        T1_range = []

        for i in range(len(self.temperatures)):
            T1_fit = self.get_T1_fit(temp_index=i)

            T1_range.append(T1_fit)

        return T1_range

    def get_time_estimate(self, T=None, temp_index=None):
        T1_eigenval, _, _ = self.get_T1_eigenval(T=T, temp_index=temp_index)
        T = T if T else self.temperatures[temp_index]
        if T == 0 or not np.isfinite(T):  # no 1/T time scale at 0 K
            return None

        t_end = float(np.real(10 * T1_eigenval + 1 / T))

        time_points = np.linspace(0, t_end, 10000, dtype=float)

        return time_points

    def get_T1_fit(self, T=None, temp_index=None):
        T1_eigenval, _, _ = self.get_T1_eigenval(T=T, temp_index=temp_index)
        steady_state = self.get_steady_state(T=T, temp_index=temp_index)[self.population.argmax()]

        if T1_eigenval > 1e9:
            return T1_eigenval

        time_points = self.get_time_estimate(T, temp_index)
        if time_points is None:
            return np.nan

        population = self.simulate_relaxation(time_points, T=T, temp_index=temp_index)[:, self.population.argmax()]

        T1_fit = self.fit_T1_exp(population, time_points, steady_state, T1_eigenval)

        return T1_fit

    def fit_rate_scaling(self, low_T, high_T, file=None):
        k_B = Cn.Boltzmann / Cn.e  # eV·K⁻¹
        hbar_eVs = Cn.hbar / Cn.e  # eV·s

        # ── 1. Obtain T₁ values ──────────────────────────────────────────────
        if file is not None:
            data = np.load(file, allow_pickle=True)
            T1_range = np.array(data["T1_range"], dtype=float)
            temperatures = np.array(data["temperatures"], dtype=float)
        else:
            T1_range = np.array(self.compute_T1_range(), dtype=float)
            temperatures = self.temperatures.astype(float)

        # Replace non-finite T₁ with NaN (they will be excluded by masks)
        T1_range = np.where(np.isfinite(T1_range) & (T1_range > 0), T1_range, np.nan)
        rate = 1.0 / T1_range  # 1/T₁ [s⁻¹]

        # ── 2. Temperature‑region masks ──────────────────────────────────────
        mask_low = temperatures <= low_T
        mask_mid = (temperatures > low_T) & (temperatures <= high_T)
        mask_high = temperatures > high_T

        results = {"temperatures": temperatures, "rate": rate}

        # ── 3. Helper: power‑law fit in log‑log space ─────────────────────────
        def _power_law_fit(T_region, rate_region, label):
            valid = np.isfinite(T_region) & (T_region > 0) & np.isfinite(rate_region) & (rate_region > 0)
            if valid.sum() < 2:
                LOGGER.warning(f"{label}: not enough valid data points – skipping.")
                return None, None

            log_T = np.log(T_region[valid])
            log_rate = np.log(rate_region[valid])

            # Degree‑1 polyfit → log(rate) = n·log(T) + log(c)
            n, log_c = np.polyfit(log_T, log_rate, 1)
            c = np.exp(log_c)

            LOGGER.info(f"{label}  |  power‑law exponent n = {n:+.4f}   (1/T₁ ∝ T^{n:.4f},  prefactor c = {c:.4e} s⁻¹)")
            return n, c

        # ── 4. Power‑law fit – Low T (0–50 K) ────────────────────────────────
        LOGGER.info("── Low‑T region (0–50 K) ─────────────────────────────────────")
        n_low, c_low = _power_law_fit(temperatures[mask_low], rate[mask_low], "Low T  (0–50 K) ")

        # ── 5. Bose‑factor fit – Mid T (50–200 K) ────────────────────────────
        LOGGER.info("── Mid‑T region (50–200 K) ───────────────────────────────────")
        T_mid = temperatures[mask_mid]
        rate_mid = rate[mask_mid]
        valid_m = np.isfinite(T_mid) & (T_mid > 0) & np.isfinite(rate_mid) & (rate_mid > 0)

        A_fit = omega_fit = None

        if valid_m.sum() >= 2:
            from scipy.optimize import curve_fit

            def mid_model(T, A, omega):
                exponent = hbar_eVs * omega / (k_B * T)
                return A * 1 / (np.exp(np.clip(exponent, -np.inf, 700)) - 1.0)

            # Initial guess – adjust if needed
            A0 = 1e-20
            omega0 = 1e12

            try:
                popt, pcov = curve_fit(
                    mid_model, T_mid[valid_m], rate_mid[valid_m], p0=[A0, omega0], bounds=([1e-30, 1e6], [1e-10, 1e15])
                )
                A_fit, omega_fit = popt
                E_fit = hbar_eVs * omega_fit  # ħω in eV

                LOGGER.info("Mid T  (50–200 K)  |  Fit: rate = A ω³/(exp(ħω/kT)-1)")
                LOGGER.info(f"  A = {A_fit:.4e} s⁻¹,  ω = {omega_fit:.4e} s⁻¹  (ħω = {E_fit:.4f} eV)")
            except Exception as e:
                LOGGER.warning(f"Mid T (50–200 K): curve_fit failed – {e}")
        else:
            LOGGER.warning("Mid T (50–200 K): not enough valid data points – skipping.")

        # ── 6. Power‑law fit – High T (> 200 K) ──────────────────────────────
        LOGGER.info("── High‑T region (>200 K) ────────────────────────────────────")
        n_high, c_high = _power_law_fit(temperatures[mask_high], rate[mask_high], "High T (>200 K)")

        # ── 7. Build callable fit functions ──────────────────────────────────
        def _make_power_law(n, c):
            if n is None or c is None:
                return None
            return lambda T: c * T**n

        func_low = _make_power_law(n_low, c_low)
        func_high = _make_power_law(n_high, c_high)

        if A_fit is not None and omega_fit is not None:
            # Capture the fitted parameters and constants in a closure
            _A, _omega = A_fit, omega_fit

            def func_mid(T):
                exponent = hbar_eVs * _omega / (k_B * T)
                return _A * 1 / (np.exp(np.clip(exponent, -np.inf, 700)) - 1.0)
        else:
            func_mid = None

        # ── 8. Assemble result dictionary ────────────────────────────────────
        results.update(
            {
                # power‑law parameters (existing keys)
                "n_low": n_low,
                "c_low": c_low,
                "n_high": n_high,
                "c_high": c_high,
                # mid‑T Bose‑factor parameters (new keys for the plotting script)
                "omega": omega_fit,
                "A_mid": A_fit,
                # keep the old Arrhenius‑named keys for backward compatibility
                "E_a": None if omega_fit is None else hbar_eVs * omega_fit,
                "A": A_fit,
                # callable functions
                "func_low": func_low,
                "func_mid": func_mid,
                "func_high": func_high,
            }
        )
        return results
