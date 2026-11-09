from typing import Union
import numpy as np
from scipy import constants as Cn
from pathlib import Path

from spinlabs.utils import MathUtils
from spinlabs.constants import CONSTANTS
from spinlabs.spin import (
    channel,
    direction_key,
    directions,
    fz,
    level_energies,
    spin_factor,
    spin_projections,
    transition_classes,
)
from spinlabs.spin import gap as spin_gap


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


# Electron spin g-factor (NV ground state; isotropic to ~1e-4) and the Zeeman energy per field
G_ELECTRON = 2.0028
ZEEMAN_MEV_PER_MT = G_ELECTRON * Cn.physical_constants["Bohr magneton"][0] / Cn.e * 1e3 * 1e-3  # meV/mT


class TransitionRate:
    """Phonon-induced transition rates between the levels m = S, ..., -S of a spin S.

    The spin comes from the coupling data (SpinPhononCouplingData.spin, from NUPDOWN; 1 when not recorded) or the
    ``spin`` argument. The couplings are the channel amplitudes V_00, V_0pm, V_+- of zfs_manager, the same for
    every S; a transition m -> m' takes V^2 times spinlabs.spin.spin_factor(S, m, m') (exactly 1 for S = 1), with
    V_0pm for |dm| = 1 and V_+- for |dm| = 2. |dm| > 2 and the Kramers partners m <-> -m of a half-integer spin get
    zero from that factor. Levels at E_m = D (m^2 - S(S+1)/3) + g mu_B B m, D the ground-state ZFS and B the
    axial field b_field_mt (0 by default).

    The spin-1 code this generalizes is kept as spinlabs.legacy_spin1.TransitionRateSpin1 (regression reference).
    """

    # How the one-phonon (direct) rate gets the spectral function at a level splitting E:
    #   "gaussian": the smeared J(E), only the Gaussian tails of modes tens of meV up (no physics at E)
    #   "debye":    J(E) = C E^3 from the supercell's folded acoustic modes (set_debye_from_spectrum): C_0pm
    #               (from V_0pm) for |dm| = 1, C_+- (from V_+-) for |dm| = 2 (for S = 1 the +1 <-> -1 rate,
    #               non-zero only in a field)
    #   "none":     no direct rate
    DIRECT_MODELS = ("gaussian", "debye", "none")

    def __init__(
        self, data_file: str | None = None, two_phonon_data_file: Path | None = None, spin: float | None = None
    ):
        self.transition_rate: dict[str, dict[str, float]] = {"first_order": {}, "second_order": {}, "two_phonon": {}}
        self.spin = 1.0  # from the coupling data's spin on load; an explicit ``spin`` takes precedence
        self.data = None
        self.data_2ph = None
        self.direct = "gaussian"
        self.debye_fit = None  # spinlabs.debye.DebyeFit of V_0pm (|dm| = 1), from set_debye_from_spectrum
        self.debye_fit_pm = None  # ... of V_+- (|dm| = 2)
        # Axial magnetic field along the defect axis (mT): adds g mu_B B m to E_m (level_energies)
        self.b_field_mt = 0.0
        if data_file:
            self.load_data(data_file)
        if two_phonon_data_file is not None:
            self.load_data_2ph(two_phonon_data_file)
        if spin is not None:
            self._set_spin(spin)

    def _set_spin(self, value) -> None:
        spin_projections(value)  # validates: a positive integer or half-integer
        self.spin = float(value)

    @property
    def ms_values(self) -> list[float]:
        """m = S, S-1, ..., -S."""
        return spin_projections(self.spin)

    @property
    def Fz_elements(self) -> dict[float, float]:
        """dm = 0 element of S.d.S per m in units of V_00: 3 m^2 - S(S+1) (1, -2, 1 for S = 1)."""
        return {m: fz(self.spin, m) for m in self.ms_values}

    @property
    def directions(self) -> list[str]:
        """Every ordered transition key "m_to_m'" of this spin (spinlabs.spin.direction_key)."""
        return directions(self.spin)

    def _ordered_pairs(self):
        """(m, m', key) for every ordered pair m != m'."""
        for m in self.ms_values:
            for m_prime in self.ms_values:
                if m != m_prime:
                    yield m, m_prime, direction_key(m, m_prime)

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
                "spin": filename.spin,
            }
            if filename.spin is not None:
                self._set_spin(filename.spin)
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
            if "spin" in self.data and self.data["spin"].ndim == 0 and self.data["spin"].dtype.kind in "fiu":
                self._set_spin(float(self.data["spin"]))

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
        geometry and eigenvectors identify the folded acoustic modes. Fits V_0pm (C_0pm, the |dm| = 1 rates,
        debye_fit) and V_+- (C_+-, the |dm| = 2 rates, debye_fit_pm) the same way; returns the V_0pm fit.
        """
        from spinlabs.debye import MIN_ACOUSTIC_WEIGHT, debye_coefficient

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        freqs = np.asarray(self.data["freqs"], dtype=float) / CONSTANTS["meV2J"]
        weight = MIN_ACOUSTIC_WEIGHT if min_weight is None else min_weight
        v0pm = np.asarray(self.data["V_0_pm"], dtype=float) / CONSTANTS["meV2J"]
        vpm = np.asarray(self.data["V_p_m"], dtype=float) / CONSTANTS["meV2J"]
        self.debye_fit = debye_coefficient(spectrum, freqs, v0pm, weight)
        self.debye_fit_pm = debye_coefficient(spectrum, freqs, vpm, weight)
        self.direct = "debye"
        return self.debye_fit

    def level_energies(self, d_mev: float) -> dict[float, float]:
        """Level energies (meV) E_m = D (m^2 - S(S+1)/3) + g mu_B B m, B the axial field b_field_mt."""
        zeeman = ZEEMAN_MEV_PER_MT * self.b_field_mt
        return {m: e + zeeman * m for m, e in level_energies(self.spin, d_mev).items()}

    def _gap(self, d_mev: float, m: float, m_prime: float) -> float:
        """E_m' - E_m = D (m'^2 - m^2) + g mu_B B (m' - m) (meV), exact at zero field for half-integer m."""
        return spin_gap(d_mev, m, m_prime) + ZEEMAN_MEV_PER_MT * self.b_field_mt * (m_prime - m)

    def _direct_density(self, energy: float, dm, omega, J_0_pm, J_p_m) -> float:
        """One-phonon spectral function (meV, per unit spin factor) at a splitting ``energy`` of a |dm| transition."""
        if energy <= 0 or dm not in (1, 2):
            # degenerate levels: a phonon of zero energy, where a physical J vanishes (acoustic J ~ w^3, so
            # J n ~ w^2 -> 0); the smeared J stays finite there only through its Gaussian tails
            return 0.0
        if self.direct == "gaussian":
            return float(np.interp(energy, omega, J_0_pm if dm == 1 else J_p_m))
        if self.direct == "debye":
            fit = self.debye_fit if dm == 1 else self.debye_fit_pm
            if fit is None:
                raise ValueError("direct='debye' needs the acoustic fit: call set_debye_from_spectrum(spectrum)")
            return fit.C * energy**3
        if self.direct == "none":
            return 0.0
        raise ValueError(f"direct must be one of {self.DIRECT_MODELS}, got {self.direct!r}")

    def compute_transition_rates(self, T, omega, J_0_pm, J_p_m, J_0_0, omega_zfs):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        if self.direct not in self.DIRECT_MODELS:
            raise ValueError(f"direct must be one of {self.DIRECT_MODELS}, got {self.direct!r}")

        res = omega[1] - omega[0]
        n = Phonons.bose_einstein(omega, T)
        mask = omega > 0.1

        # Direct (one-phonon) process: energy conservation puts the phonon at the splitting of the two levels,
        # so the rate is 2pi/hbar J(dE) n(dE) (absorption, upward) or 2pi/hbar J(dE) (n(dE) + 1) (emission,
        # downward), times the transition's spin factor. J is already a density, so it is read off at dE, not
        # weighed with a second broadened delta. |dm| = 1 uses J_0pm, |dm| = 2 J_+-. Degenerate levels (+-m at
        # zero field) get no direct rate; an axial field splits them by 2 g mu_B B m.
        f1 = 2 * np.pi / (Cn.hbar / CONSTANTS["meV2J"])
        D = float(omega_zfs)
        first = self.transition_rate["first_order"]
        first.clear()
        for m, m_prime, key in self._ordered_pairs():
            gap = self._gap(D, m, m_prime)
            J_gap = self._direct_density(abs(gap), channel(m, m_prime), omega, J_0_pm, J_p_m)
            n_gap = self._bose_at(abs(gap), T)
            first[key] = f1 * J_gap * (n_gap if gap > 0 else n_gap + 1) * spin_factor(self.spin, m, m_prime)

        # First-order Raman (the first-order couplings at second order), from incoherent |V|^2 products over the
        # intermediate level. Only for S = 1: for a half-integer spin the incoherent sum gives the Kramers partners
        # (+1/2 <-> -1/2 through +-3/2) a rate that the coherent sum cancels (time reversal), so it is not
        # generalized. Production leaves this process out for every spin.
        second = self.transition_rate["second_order"]
        second.clear()
        if self.spin != 1.0:
            return

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

        for ms, ms_prime, rate_key in self._ordered_pairs():
            total_integrand = np.zeros_like(omega[mask])
            for ms_double_prime in self.ms_values:
                J_a = get_J_path(ms_prime, ms_double_prime)[mask]
                J_b = get_J_path(ms_double_prime, ms)[mask]
                E_sq = omega[mask] ** 2
                total_integrand += (J_a * J_b / E_sq) * n[mask] * (n[mask] + 1)

            second[rate_key] = f2_factor * np.sum(total_integrand)

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
        """|V2|^2 per mode pair, with the off-diagonal pairs (i != j) doubled.

        V2 carries the 1/2 of the Taylor series, so the Hamiltonian holds a pair i != j
        twice, V2_ij q_i q_j + V2_ji q_j q_i = 2 V2_ij q_i q_j, and its matrix element is
        2 V2_ij sqrt(n_i (n_j + 1)), squared 4 |V2_ij|^2 = 2 x these 2 |V2_ij|^2: compute_two_phonon_rates
        uses 4 pi / hbar, 2 pi / hbar for each of the two Raman orderings. A diagonal
        pair (i, i) appears once and keeps |V2_ii|^2, the diagonal-only convention of Cambria et al.
        (PRL 130, 256903, Eq. S12).
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
        """Second-order coupling of m1 -> m2 as an amplitude: the channel's V2 times sqrt(spin factor)."""
        if m1 == m2:
            return (self.Fz_elements[m1] ** 2) * V_0_0_2ph
        diff = abs(m1 - m2)
        if diff in (1, 2):
            return np.sqrt(spin_factor(self.spin, m1, m2)) * (V_0_pm_2ph if diff == 1 else V_p_m_2ph)
        return np.zeros_like(V_0_pm_2ph)

    @staticmethod
    def line_spectral_density(
        freqs: np.ndarray, weights: np.ndarray, sigma: float, gap: float, res: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The 2D spectral function J2(w, w') = sum_ij weights_ij G(w - w_i) G(w' - w_j) on the line w' = w - gap.

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

        A transition m -> m' changes the spin energy by gap = E_m' - E_m = D (m'^2 - m^2) + g mu_B B (m' - m)
        (level_energies; B the axial field b_field_mt), so the Raman line is w_absorbed - w_emitted = gap.
        |dm| = 1 uses V2_0pm, |dm| = 2 V2_+-; each line is computed once per (channel, gap) from the channel's
        pair weights and the transition carries its spin factor (spinlabs.spin.spin_factor, 1 for S = 1), which
        compute_two_phonon_rates multiplies in. A transition with |dm| > 2 has no channel: its line is None and
        its rate 0. ``res`` defaults to sigma/20 (at most 0.05 meV), fine against the sigma/sqrt2 width along
        the line.

        ``degenerate_groups`` (PhononSpectrum.deg_groups; [] for none) leaves out the pairs at zero
        frequency difference by symmetry: the diagonal pairs (i, i) and the partners of each
        degenerate group (symmetry_zero_gap_pairs). Their gap is zero at any sigma, so they add a
        term ~ 1/sigma that a phonon continuum does not have (there such pairs have measure zero);
        without them the rate is the supercell's continuum estimate. Accidental near-degeneracies
        are kept: they sample the continuum of pair splittings like any other pair. None keeps all.

        Returns {key: (omega, density, gap, spin_factor)}.
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
        for m, m_prime, key in self._ordered_pairs():
            dm, gap = channel(m, m_prime), self._gap(D, m, m_prime)
            factor = spin_factor(self.spin, m, m_prime)
            if dm not in weights:
                lines[key] = (None, None, gap, 0.0)
                continue
            if (dm, gap) not in cache:
                cache[(dm, gap)] = self.line_spectral_density(freqs, weights[dm], sigma, gap, res)
            lines[key] = (*cache[(dm, gap)], gap, factor)
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
        """Raman rates from the second-order coupling, from get_line_spectral_density's lines.

        Gamma = 4 pi / hbar * F * integral dw J2(w, w - gap) n(w) (n(w - gap) + 1), absorbing w and
        emitting w - gap, F the transition's spin factor (1 for S = 1). The 4 pi is 2 pi / hbar times the
        two Raman orderings, Cambria et al. Eq. 3; with pair_weights a pair i != j gets the 4 |V2_ij|^2 of
        its matrix element. Double absorption and double emission (w + w' = |gap|) need two phonons below
        the splitting (0.01 meV), where only the Gaussian tails of optical modes put weight, so
        they are left out.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        prefactor = 4 * np.pi / (Cn.hbar / CONSTANTS["meV2J"])
        two = self.transition_rate["two_phonon"]
        two.clear()
        for rate_key, line in lines.items():
            omega, density, gap = line[:3]
            factor = line[3] if len(line) > 3 else 1.0
            if omega is None or factor == 0.0:
                two[rate_key] = 0.0
                continue
            res = omega[1] - omega[0]
            emitted = omega - gap
            ok = (omega > 0) & (emitted > 0)
            bose = self._bose_grid(omega[ok], T) * (self._bose_grid(emitted[ok], T) + 1)
            two[rate_key] = prefactor * res * float(np.sum(density[ok] * bose)) * factor

    def get_directional_rates(self):
        """Total rate per direction key, summed over the processes in transition_rate."""
        directional_rates = dict.fromkeys(self.directions, 0.0)
        for direction_dict in self.transition_rate.values():
            for direction, rate in direction_dict.items():
                directional_rates[direction] += rate

        return directional_rates

    def get_total_rates(self):
        """Per process, the mean directional rate of each transition class (spinlabs.spin.transition_class).

        For S = 1 the classes are "0_1" (0 <-> +-1, the mean of four directions) and "1_-1" (+1 <-> -1, two).
        """
        classes = transition_classes(self.spin)
        return {
            process: {
                name: sum(rates.get(key, 0) for key in keys) / len(keys) for name, keys in classes.items()
            }
            for process, rates in self.transition_rate.items()
        }
