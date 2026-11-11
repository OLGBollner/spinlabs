"""Frozen spin-1 rate and T1 code, kept as the reference of the general-spin implementation.

Verbatim copies of TransitionRate (spinlabs.transition_rate) and RelaxationDynamics (spinlabs.relaxation_dynamics,
without fit_rate_scaling) as they were before the rates were generalized to any spin S. Nothing in production
uses them: tests/test_general_spin.py checks that the general code with S = 1 reproduces them. Do not edit.
"""

import warnings
from pathlib import Path
from typing import Union

import numpy as np
from scipy import constants as Cn
from scipy.optimize import OptimizeWarning, curve_fit

from spinlabs.constants import CONSTANTS
from spinlabs.relaxation_dynamics import MIN_DECAY_RATE
from spinlabs.transition_rate import Phonons
from spinlabs.utils import MathUtils


class TransitionRateSpin1:
    # How the one-phonon (direct) rate gets the spectral function at the spin splitting D:
    #   "gaussian": the smeared J(D), only the Gaussian tails of modes tens of meV up (no physics at D)
    #   "debye":    J(D) = C D^3 from the supercell's folded acoustic modes (set_debye_from_spectrum)
    #   "none":     no direct rate
    DIRECT_MODELS = ("gaussian", "debye", "none")

    def __init__(self, data_file: str | None = None, two_phonon_data_file: Path | None = None):
        self.transition_rate: dict[str, dict[str, float]] = {"first_order": {}, "second_order": {}, "two_phonon": {}}
        self.ms_values = [1, 0, -1]
        self.Fz_elements = {1: 1.0, 0: -2.0, -1: 1.0}
        self.data = None
        self.data_2ph = None
        self.direct = "gaussian"
        self.debye_fit = None  # spinlabs.debye.DebyeFit, from set_debye_from_spectrum
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

    @staticmethod
    def _bose_at(energy_mev: float, T: float) -> float:
        """Bose-Einstein occupation of one phonon energy (meV), without the grid cut-offs of bose_einstein."""
        if T <= 0 or energy_mev <= 0:
            return 0.0
        x = energy_mev / (Cn.k / Cn.e * 1000 * T)
        return 0.0 if x > 700 else float(1.0 / np.expm1(x))

    def set_debye_from_spectrum(self, spectrum, min_weight: float | None = None):
        """Fit the Debye direct-rate model (spinlabs.debye) to this coupling data's acoustic modes and use it.

        ``spectrum`` is the PhononSpectrum the couplings were computed on (same modes, same order): its
        geometry and eigenvectors identify the folded acoustic modes. Returns the DebyeFit.
        """
        from spinlabs.debye import MIN_ACOUSTIC_WEIGHT, debye_coefficient

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        freqs = np.asarray(self.data["freqs"], dtype=float) / CONSTANTS["meV2J"]
        v0pm = np.asarray(self.data["V_0_pm"], dtype=float) / CONSTANTS["meV2J"]
        self.debye_fit = debye_coefficient(
            spectrum, freqs, v0pm, MIN_ACOUSTIC_WEIGHT if min_weight is None else min_weight
        )
        self.direct = "debye"
        return self.debye_fit

    def compute_transition_rates(self, T, omega, J_0_pm, J_p_m, J_0_0, omega_zfs):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        res = omega[1] - omega[0]
        n = Phonons.bose_einstein(omega, T)
        mask = omega > 0.1

        # Direct (one-phonon) process: energy conservation puts the phonon at the spin
        # splitting, so the rate is 2pi/hbar J(dE) n(dE) (absorption, upward) or
        # 2pi/hbar J(dE) (n(dE) + 1) (emission, downward). J is already a density, so it
        # is read off at dE, not weighed with a second broadened delta.
        # At zero field m_s = +1 and -1 lie at D and m_s = 0 at 0; D < 0 swaps which is upper.
        f1 = 2 * np.pi / (Cn.hbar / CONSTANTS["meV2J"])
        split = abs(float(omega_zfs))
        if self.direct == "gaussian":
            J_split = float(np.interp(split, omega, J_0_pm))
        elif self.direct == "debye":
            if self.debye_fit is None:
                raise ValueError("direct='debye' needs the acoustic fit: call set_debye_from_spectrum(spectrum)")
            J_split = self.debye_fit.C * split**3
        elif self.direct == "none":
            J_split = 0.0
        else:
            raise ValueError(f"direct must be one of {self.DIRECT_MODELS}, got {self.direct!r}")
        n_split = self._bose_at(split, T)
        up, down = f1 * J_split * n_split, f1 * J_split * (n_split + 1)
        zero_is_lower = omega_zfs >= 0
        first = self.transition_rate["first_order"]
        first["0_to_1"] = first["0_to_-1"] = up if zero_is_lower else down
        first["1_to_0"] = first["-1_to_0"] = down if zero_is_lower else up
        # +1 and -1 are degenerate at zero field: a direct process would need a phonon of
        # zero energy, where a physical spectral function vanishes (acoustic J ~ w^3, so
        # J n ~ w^2 -> 0). The smeared J stays finite there only through its Gaussian tails.
        first["1_to_-1"] = first["-1_to_1"] = 0.0

        f2_factor = 2 * f1 * res

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

    @staticmethod
    def pair_weights(V2: np.ndarray) -> np.ndarray:
        """Half of the squared matrix-element coefficients, as an ordered-pair array.

        For the spin-flip channels, V2 includes the Taylor 1/2. For i != j, the two symmetric
        interaction terms cancel that half, so absorbing i and emitting j has amplitude
        2 V2_ij sqrt(n_i (n_j + 1)) and squared matrix element 4 |V2_ij|^2 n_i (n_j + 1).
        Each ordered golden-rule term therefore needs 4 |V2_ij|^2 with 2 pi / hbar.

        compute_two_phonon_rates instead applies 4 pi / hbar to every spectral entry.
        These weights are consequently half the matrix-element coefficients: 2 |V2_ij|^2
        off diagonal, and |V2_ii|^2 on the diagonal. The latter gives 2 |V2_ii|^2 after
        applying 4 pi / hbar, the diagonal-only convention of Cambria et al.
        (PRL 130, 256903, Eq. S12).

        Entry (i, j) denotes absorption in i and emission in j. The mirrored entry (j, i)
        is the opposite Raman process, not an extra combinatorial copy of (i, j).
        """
        w = np.abs(np.asarray(V2, dtype=float)) ** 2
        if w.ndim != 2:
            return w
        return 2.0 * w - np.diag(np.diag(w))

    def get_2d_spectral_density(self, res, sigma):
        V_0_pm_2ph = self.get_2ph_coupling("V_0_pm") / CONSTANTS["meV2J"]
        V_p_m_2ph = self.get_2ph_coupling("V_p_m") / CONSTANTS["meV2J"]
        V_0_0_2ph = self.get_2ph_coupling("V_0_0") / CONSTANTS["meV2J"]
        freqs = self.data["freqs"] / CONSTANTS["meV2J"]

        omega_x, omega_y, J_0_pm = MathUtils.get_2d_spectral_density(freqs, self.pair_weights(V_0_pm_2ph), res, sigma)
        _, _, J_p_m = MathUtils.get_2d_spectral_density(freqs, self.pair_weights(V_p_m_2ph), res, sigma)
        _, _, J_0_0_base = MathUtils.get_2d_spectral_density(freqs, self.pair_weights(V_0_0_2ph), res, sigma)

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

    @staticmethod
    def line_spectral_density(
        freqs: np.ndarray, weights: np.ndarray, sigma: float, gap: float, res: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The 2D spectral function J2(w, w') = sum_ij weights_ij G(w - w_i) G(w' - w_j) on the line w' = w - gap.

        The sum retains every ordered pair (i, j): i is absorbed and j is emitted. A mirrored
        entry (j, i) is the opposite Raman process rather than a duplicate pair, so there is
        no triangular restriction or further off-diagonal multiplicity in this sum.

        G is a Gaussian of width ``sigma``. On the line each pair's two Gaussians multiply into
        g_{sqrt2 sigma}(w_i - w_j - gap) G_{sigma/sqrt2}(w - (w_i + w_j + gap) / 2) exactly: a constant
        weight for how near the pair is to energy conservation, times a Gaussian along the line
        around the pair's mean energy. So each pair is binned at its mean energy with that weight
        (split linearly between the two nearest grid points) and the histogram is smoothed by
        G_{sigma/sqrt2}. Returns the absorbed-phonon energy w (meV, grid spacing ``res``) and
        J2 there, per meV^2 like the 2D function.
        """
        freqs = np.asarray(freqs, dtype=float)
        keep = freqs > 0  # acoustic modes at ~0 (or slightly imaginary) carry no coupling
        f = freqs[keep]
        w = np.asarray(weights, dtype=float)[np.ix_(keep, keep)]
        pair = w * MathUtils.gaussian_density(f[:, None] - f[None, :], gap, np.sqrt(2) * sigma)
        centre = (f[:, None] + f[None, :] + gap) / 2

        width = sigma / np.sqrt(2)
        omega = np.arange(0.0, f.max() + abs(gap) + 8 * width + res, res)
        pos = np.clip(centre.ravel() / res, 0.0, len(omega) - 1.0)
        lower = np.floor(pos).astype(int)
        upper = np.minimum(lower + 1, len(omega) - 1)
        frac = pos - lower
        hist = np.bincount(lower, pair.ravel() * (1 - frac), len(omega)) + np.bincount(
            upper, pair.ravel() * frac, len(omega)
        )
        half = int(np.ceil(8 * width / res))
        kernel = MathUtils.gaussian_density(np.arange(-half, half + 1) * res, 0.0, width)
        density = np.convolve(hist, kernel, mode="full")[half : half + len(omega)]
        return omega, density

    @staticmethod
    def symmetry_zero_gap_pairs(n_modes: int, degenerate_groups) -> np.ndarray:
        """Pairs (i, j) at zero frequency difference by symmetry: every (i, i), and i != j in one degenerate group.

        ``degenerate_groups`` are lists of mode indices (PhononSpectrum.deg_groups). Accidental
        near-degeneracies between modes of different groups are not among them.
        """
        pairs = np.eye(n_modes, dtype=bool)
        for group in degenerate_groups or ():
            idx = np.asarray(group, dtype=int)
            if idx.size and (idx.min() < 0 or idx.max() >= n_modes):
                raise ValueError(f"degenerate group {list(group)} is outside the {n_modes} modes of the couplings")
            pairs[np.ix_(idx, idx)] = True
        return pairs

    def get_line_spectral_density(self, sigma: float, res: float | None = None, degenerate_groups=None) -> dict:
        """Temperature-independent input of compute_two_phonon_rates: J2 on each transition's energy line.

        A transition m_s -> m_s' changes the spin energy by gap = E(m_s') - E(m_s), with
        E(+-1) = D and E(0) = 0 at zero field, so the Raman line is w_absorbed - w_emitted = gap.
        |dm| = 1 uses V2_0pm, |dm| = 2 V2_+-. ``res`` defaults to sigma/20 (at most 0.05 meV),
        fine against the sigma/sqrt2 width along the line.

        ``degenerate_groups`` (PhononSpectrum.deg_groups; [] for none) leaves out the pairs at zero
        frequency difference by symmetry: the diagonal pairs (i, i) and the partners of each
        degenerate group (symmetry_zero_gap_pairs). Their gap is zero at any sigma, so they add a
        term ~ 1/sigma that a phonon continuum does not have (there such pairs have measure zero);
        without them the rate is the supercell's continuum estimate. Accidental near-degeneracies
        are kept: they sample the continuum of pair splittings like any other pair. None keeps all.
        """
        res = min(0.05, sigma / 20) if res is None else res
        freqs = self.data["freqs"] / CONSTANTS["meV2J"]
        D = float(self.data["zfs"]) / CONSTANTS["meV2J"]
        couplings = {1: "V_0_pm", 2: "V_p_m"}
        weights = {
            dm: self.pair_weights(self.get_2ph_coupling(key) / CONSTANTS["meV2J"]) for dm, key in couplings.items()
        }
        if degenerate_groups is not None:
            zero_gap = self.symmetry_zero_gap_pairs(len(freqs), degenerate_groups)
            weights = {dm: np.where(zero_gap, 0.0, w) for dm, w in weights.items()}

        lines, cache = {}, {}
        for ms in self.ms_values:
            for ms_prime in self.ms_values:
                if ms == ms_prime:
                    continue
                dm, gap = abs(ms - ms_prime), D * (ms_prime**2 - ms**2)
                if (dm, gap) not in cache:
                    cache[(dm, gap)] = self.line_spectral_density(freqs, weights[dm], sigma, gap, res)
                lines[f"{ms}_to_{ms_prime}"] = (*cache[(dm, gap)], gap)
        return lines

    @staticmethod
    def _bose_grid(omega: np.ndarray, T: float) -> np.ndarray:
        """Bose-Einstein occupation on a grid; 0 at T = 0 and for w <= 0 (those points are left out)."""
        n = np.zeros_like(omega, dtype=float)
        if T <= 0:
            return n
        x = omega / (Cn.k / Cn.e * 1000 * T)
        ok = (omega > 0) & (x < 700)
        n[ok] = 1.0 / np.expm1(x[ok])
        return n

    def compute_two_phonon_rates(self, T: float, lines: dict) -> None:
        """Raman rates from the ordered-pair spectra returned by get_line_spectral_density.

        Gamma = 4 pi / hbar * integral dw J2(w, w - gap) n(w) (n(w - gap) + 1), absorbing w
        and emitting w - gap. J2 already contains both ordered Raman processes, (i, j) and (j, i).

        The 4 pi / hbar is the normalization paired with pair_weights, not an additional
        two-pair multiplicity. Off diagonal it gives (4 pi / hbar) * 2 |V2_ij|^2 =
        (2 pi / hbar) * 4 |V2_ij|^2, the squared matrix element of 2 V2_ij. On the diagonal,
        (4 pi / hbar) * |V2_ii|^2 = (2 pi / hbar) * 2 |V2_ii|^2, the adopted Cambria
        diagonal-only convention (PRL 130, 256903, Eq. 3).

        Double absorption and double emission (w + w' = |gap|) need two phonons below the
        splitting (0.01 meV), where only the Gaussian tails of optical modes put weight,
        so they are left out.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        prefactor = 4 * np.pi / (Cn.hbar / CONSTANTS["meV2J"])
        for rate_key, (omega, density, gap) in lines.items():
            res = omega[1] - omega[0]
            emitted = omega - gap
            ok = (omega > 0) & (emitted > 0)
            bose = self._bose_grid(omega[ok], T) * (self._bose_grid(emitted[ok], T) + 1)
            self.transition_rate["two_phonon"][rate_key] = prefactor * res * float(np.sum(density[ok] * bose))

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


class RelaxationDynamicsSpin1:
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

    @staticmethod
    def _fitted_T1(population, time_points, steady_state, T1_eigenval):
        """T1 of a * exp(-t / T1) + c fitted to the population, or None when the fit fails."""

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
        except Exception:
            return None
        extracted_T1 = float(popt[0])
        return extracted_T1 if np.isfinite(extracted_T1) and extracted_T1 > 0 else None

    def fit_T1_exp(self, population, time_points, steady_state, T1_eigenval):
        fitted = self._fitted_T1(population, time_points, steady_state, T1_eigenval)
        return T1_eigenval if fitted is None else fitted

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

        # T1 = the slowest decay; none faster than MIN_DECAY_RATE = no relaxation (T1 = inf)
        slowest = -np.max(decays) if len(decays) else 0.0
        relaxation_time = float(1.0 / slowest) if slowest >= MIN_DECAY_RATE else np.inf

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
        return self.get_T1_fit_details(T=T, temp_index=temp_index)[0]

    def get_T1_fit_details(self, T=None, temp_index=None) -> tuple[float, float, str]:
        """T1 (s) from the population fit, the eigenvalue T1 (s), and where the T1 came from.

        The source is "fit" (exponential + offset fitted to the initial state's population), "eigenvalue"
        (the fallback: no relaxation slower than 1/MIN_DECAY_RATE, or a failed fit) or "none" (0 K, NaN).
        """
        T1_eigenval, _, _ = self.get_T1_eigenval(T=T, temp_index=temp_index)
        steady_state = self.get_steady_state(T=T, temp_index=temp_index)[self.population.argmax()]

        if T1_eigenval > 1.0 / MIN_DECAY_RATE:
            return T1_eigenval, T1_eigenval, "eigenvalue"

        time_points = self.get_time_estimate(T, temp_index)
        if time_points is None:
            return np.nan, T1_eigenval, "none"

        population = self.simulate_relaxation(time_points, T=T, temp_index=temp_index)[:, self.population.argmax()]

        fitted = self._fitted_T1(population, time_points, steady_state, T1_eigenval)
        if fitted is None:
            return T1_eigenval, T1_eigenval, "eigenvalue"
        return fitted, T1_eigenval, "fit"
