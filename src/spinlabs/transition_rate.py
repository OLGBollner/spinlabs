from typing import Union
import numpy as np
from scipy import constants as Cn
from pathlib import Path

from spinlabs.utils import MathUtils
from spinlabs.constants import CONSTANTS

# (sign_l, sign_lp) for energy conservation: omega_spin = s*omega_l + s'*omega_lp
PHONON_PROCESSES = {
    "abs_em": (+1, -1),
    "em_abs": (-1, +1),
    "abs_abs": (+1, +1),
    "em_em": (-1, -1),
}


class Phonons:
    @staticmethod
    def bose_einstein(omega: np.ndarray, T: float) -> np.ndarray:
        if T <= 0:
            return np.zeros_like(omega)
        kB = Cn.k / Cn.e * 1000
        x = omega / (kB * T)
        n = np.zeros_like(omega)
        mask = (x > 1e-5) & (x < 40)
        n[mask] = 1.0 / (np.exp(x[mask]) - 1.0)
        return n

    @staticmethod
    def _bose_factors_2ph(n_l, n_lp):
        return {
            "abs_em": n_l * (n_lp + 1),
            "em_abs": (n_l + 1) * n_lp,
            "abs_abs": n_l * n_lp,
            "em_em": (n_l + 1) * (n_lp + 1),
        }

    @staticmethod
    def bose_einstein_2d(omega: np.ndarray, T: float) -> np.ndarray:
        n = Phonons.bose_einstein(omega, T)

        # outer products: shape (N, N)
        N_l = n[None, :]
        N_lp = n[:, None]

        bose = Phonons._bose_factors_2ph(N_l, N_lp)

        return bose


class TransitionRate:
    def __init__(self, data_file: str | None = None, two_phonon_data_file: Path | None = None):
        self.transition_rate: dict[str, dict[str, float]] = {"first_order": {}, "second_order": {}, "two_phonon": {}}
        self.ms_values = [1, 0, -1]
        self.Fz_elements = {1: 1.0, 0: -2.0, -1: 1.0}
        self.data = None
        self.data_2ph = None
        if data_file:
            self.load_data(data_file)
        if two_phonon_data_file is not None:
            self.load_data_2ph(two_phonon_data_file)

    def load_data_2ph(self, filename: Union[str, Path]) -> None:
        self.data_2ph = np.load(str(filename), allow_pickle=True)

    def load_data(self, filename: Union[str, Path]) -> None:
        from spinlabs.models import SpinPhononCouplingData

        if isinstance(filename, SpinPhononCouplingData):
            # Export to dictionary representation matching npz format
            self.data = {
                "defect": filename.defect,
                "cell_size": filename.cell_size,
                "calc_method": filename.calc_method,
                "pert_scale": filename.pert_scale,
                "zfs": filename.ground_state_zfs.to_unit("J").D
                if filename.ground_state_zfs is not None
                else 0.0,  # legacy scalar D in Joules
                "V_0_0": filename.V_0_0,
                "V_p_m": filename.V_p_m,
                "V_0_pm": filename.V_0_pm,
                "freqs": filename.frequencies,
                "symmetries": filename.symmetries,
                "ipr": filename.iprs,
            }
            # Carry second-order coefficients into the dict; used by get_2ph_coupling
            if filename.has_second_order:
                self.data["V2_0_0"] = filename.V2_0_0
                self.data["V2_p_m"] = filename.V2_p_m
                self.data["V2_0_pm"] = filename.V2_0_pm
                if self.data_2ph is None:
                    self.data_2ph = self.data
        else:
            # Load through the schema-aware loader so legacy and new keys both map
            if "V_0_0" in np.load(str(filename), allow_pickle=True):
                self.load_data(SpinPhononCouplingData.load(filename))
                if self.data_2ph is None:
                    self.data_2ph = self.data
                return
            self.data = np.load(str(filename), allow_pickle=True)

    def get_spectral_density(self, res, sigma):
        V_0_0 = self.data["V_0_0"] / CONSTANTS["meV2J"]
        V_0_pm = self.data["V_0_pm"] / CONSTANTS["meV2J"]
        V_p_m = self.data["V_p_m"] / CONSTANTS["meV2J"]
        freqs = self.data["freqs"] / CONSTANTS["meV2J"]

        omega, J_0_pm = MathUtils.smear_data(freqs, V_0_pm**2, res, sigma)
        _, J_p_m = MathUtils.smear_data(freqs, V_p_m**2, res, sigma)
        _, J_0_0 = MathUtils.smear_data(freqs, V_0_0**2, res, sigma)

        return omega, J_0_pm, J_p_m, J_0_0

    def compute_transition_rates(self, T, omega, J_0_pm, J_p_m, J_0_0, omega_zfs):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        res = omega[1] - omega[0]
        n = Phonons.bose_einstein(omega, T)
        mask = omega > 0.1
        delta = MathUtils.gaussian_density(omega, omega_zfs, 1.0)

        f1_factor = (2 * np.pi / (Cn.hbar / CONSTANTS["meV2J"])) * res
        self.transition_rate["first_order"]["0_to_1"] = f1_factor * np.sum(J_0_pm[mask] * n[mask] * delta[mask])
        self.transition_rate["first_order"]["0_to_-1"] = self.transition_rate["first_order"]["0_to_1"]
        self.transition_rate["first_order"]["1_to_0"] = f1_factor * np.sum(J_0_pm[mask] * (n[mask] + 1) * delta[mask])
        self.transition_rate["first_order"]["-1_to_0"] = self.transition_rate["first_order"]["1_to_0"]

        self.transition_rate["first_order"]["1_to_-1"] = f1_factor * np.sum(J_p_m[mask] * (n[mask] + 1) * delta[mask])
        self.transition_rate["first_order"]["-1_to_1"] = f1_factor * np.sum(J_p_m[mask] * n[mask] * delta[mask])

        f2_factor = 2 * f1_factor

        def get_J_path(m1, m2):
            if m1 == m2:
                return (self.Fz_elements[m1] ** 2) * J_0_0
            diff = abs(m1 - m2)
            if diff == 1:
                return J_0_pm
            if diff == 2:
                return J_p_m
            return np.zeros_like(omega)

        for ms in self.ms_values:
            for ms_prime in self.ms_values:
                if ms == ms_prime:
                    continue
                rate_key = f"{ms}_to_{ms_prime}"
                total_integrand = np.zeros_like(omega[mask])
                for ms_double_prime in self.ms_values:
                    J_a = get_J_path(ms_prime, ms_double_prime)[mask]
                    J_b = get_J_path(ms_double_prime, ms)[mask]
                    E_sq = omega[mask] ** 2
                    total_integrand += (J_a * J_b / E_sq) * n[mask] * (n[mask] + 1)

                self.transition_rate["second_order"][rate_key] = f2_factor * np.sum(total_integrand)

    # Two phonon stuff
    def get_2ph_coupling(self, V_key: str) -> np.ndarray:
        if self.data_2ph is None:
            raise ValueError("Data not loaded. Please call load_data_2ph() first.")
        # Prioritise second-order prefix V2_* when looking up 2-phonon coupling
        # (e.g. key 'V_0_pm' -> look for 'V2_0_pm' before 'V_0_pm')
        v2_key = f"V2_{V_key[2:]}" if V_key.startswith("V_") else f"V2_{V_key}"
        if v2_key in self.data_2ph and np.asarray(self.data_2ph[v2_key]).ndim == 2:
            return self.data_2ph[v2_key]
        if V_key in self.data_2ph:
            return self.data_2ph[V_key]
        raise KeyError(f"Key '{V_key}' (or '{v2_key}') not found in data file.")

    def get_2d_spectral_density(self, res, sigma):
        V_0_pm_2ph = self.get_2ph_coupling("V_0_pm") / CONSTANTS["meV2J"]
        V_p_m_2ph = self.get_2ph_coupling("V_p_m") / CONSTANTS["meV2J"]
        V_0_0_2ph = self.get_2ph_coupling("V_0_0") / CONSTANTS["meV2J"]
        freqs = self.data["freqs"] / CONSTANTS["meV2J"]

        omega_x, omega_y, J_0_pm = MathUtils.get_2d_spectral_density(freqs, V_0_pm_2ph**2, res, sigma)
        _, _, J_p_m = MathUtils.get_2d_spectral_density(freqs, V_p_m_2ph**2, res, sigma)
        _, _, J_0_0_base = MathUtils.get_2d_spectral_density(freqs, V_0_0_2ph**2, res, sigma)

        return omega_x, omega_y, J_0_pm, J_p_m, J_0_0_base

    def _get_V2ph(self, m1, m2, V_0_pm_2ph, V_p_m_2ph, V_0_0_2ph):
        if m1 == m2:
            return (self.Fz_elements[m1] ** 2) * V_0_0_2ph
        diff = abs(m1 - m2)
        if diff == 1:
            return V_0_pm_2ph
        if diff == 2:
            return V_p_m_2ph
        return np.zeros_like(V_0_pm_2ph)

    def compute_two_phonon_rates(self, T, omega_x, omega_y, J_0_pm, J_p_m, J_0_0, omega_zfs):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        res = omega_x[1, 1] - omega_x[0, 0]

        def get_J_path(m1, m2) -> np.ndarray:
            if m1 == m2:
                return (self.Fz_elements[m1] ** 2) * J_0_0
            diff = abs(m1 - m2)
            if diff == 1:
                return J_0_pm
            if diff == 2:
                return J_p_m
            return np.zeros_like(omega_x)

        bose = Phonons.bose_einstein_2d(omega_x[0, :], T)

        f2ph = (2 * np.pi / (Cn.hbar / CONSTANTS["meV2J"])) * (res) ** 2

        for ms in self.ms_values:
            for ms_prime in self.ms_values:
                if ms == ms_prime:
                    continue

                rate_key = f"{ms}_to_{ms_prime}"
                J = get_J_path(ms_prime, ms)
                total = 0.0

                for proc, (s, sp) in PHONON_PROCESSES.items():
                    # Energy conservation is a line delta(s*w_x + s'*w_y - w_zfs); its width
                    # only needs to resolve the line on the grid, the physical broadening
                    # (sigma) is already in J.
                    delta = MathUtils.gaussian_density(s * omega_x + sp * omega_y, omega_zfs, res)
                    integrand = np.sum(J * bose[proc] * delta)
                    total += integrand

                self.transition_rate["two_phonon"][rate_key] = f2ph * total

    def get_directional_rates(self):
        directional_rates = {
            "0_to_1": 0.0,
            "0_to_-1": 0.0,
            "1_to_0": 0.0,
            "-1_to_0": 0.0,
            "1_to_-1": 0.0,
            "-1_to_1": 0.0,
        }
        for direction_dict in self.transition_rate.values():
            for direction, rate in direction_dict.items():
                directional_rates[direction] += rate

        return directional_rates

    def get_total_rates(self):
        total_rate = {
            "first_order": {
                "0_1": 0.0,
                "1_-1": 0.0,
            },
            "second_order": {
                "0_1": 0.0,
                "1_-1": 0.0,
            },
            "two_phonon": {
                "0_1": 0.0,
                "1_-1": 0.0,
            },
        }

        total_rate["first_order"]["0_1"] = (
            self.transition_rate["first_order"]["0_to_1"]
            + self.transition_rate["first_order"]["0_to_-1"]
            + self.transition_rate["first_order"]["1_to_0"]
            + self.transition_rate["first_order"]["-1_to_0"]
        ) / 4

        total_rate["first_order"]["1_-1"] = (
            self.transition_rate["first_order"]["1_to_-1"] + self.transition_rate["first_order"]["-1_to_1"]
        ) / 2

        total_rate["second_order"]["0_1"] = (
            self.transition_rate["second_order"]["0_to_1"]
            + self.transition_rate["second_order"]["0_to_-1"]
            + self.transition_rate["second_order"]["-1_to_0"]
            + self.transition_rate["second_order"]["1_to_0"]
        ) / 4

        total_rate["second_order"]["1_-1"] = (
            self.transition_rate["second_order"]["1_to_-1"] + self.transition_rate["second_order"]["-1_to_1"]
        ) / 2

        total_rate["two_phonon"]["0_1"] = (
            self.transition_rate["two_phonon"].get("0_to_1", 0)
            + self.transition_rate["two_phonon"].get("0_to_-1", 0)
            + self.transition_rate["two_phonon"].get("-1_to_0", 0)
            + self.transition_rate["two_phonon"].get("1_to_0", 0)
        ) / 4

        total_rate["two_phonon"]["1_-1"] = (
            self.transition_rate["two_phonon"].get("1_to_-1", 0) + self.transition_rate["two_phonon"].get("-1_to_1", 0)
        ) / 2

        return total_rate
