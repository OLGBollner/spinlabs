"""Debye estimate of the one-phonon (direct) spin-flip rate from the supercell's folded acoustic modes.

The direct process absorbs or emits one phonon at the spin splitting D (~0.01 meV for NV). Phonons that
soft are long-wavelength acoustic waves, which a supercell does not hold; a Gaussian-smeared spectral
function only reaches D through the tails of modes tens of meV higher, so it says nothing about them.

A supercell of side L does hold acoustic phonons at its smallest reciprocal-lattice vectors q ~ 2 pi / L
(folded to Gamma; ~35 meV in NV_512). Their couplings measure the spin-strain coupling, and the Debye
form carries it down to D:

    J(E) = C E^3,   C = sum_s n_s V a_s / (2 pi^2 (hbar v_s)^3)

for branch s (longitudinal n_s = 1, transverse n_s = 2): a_s = <|V|^2 / E> over its modes (strain
coupling: |V|^2 ~ E), hbar v_s = <E> / |q| (sound speed), V the cell volume. The direct rate is then
2 pi / hbar C D^3 (n(D) + 1) down, n(D) up. An order-of-magnitude estimate: it extrapolates a factor of
~10^3 in energy from wavelengths comparable to the defect.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

MIN_ACOUSTIC_WEIGHT = 0.5  # share of a mode in the acoustic plane waves for it to count as one


@dataclass(frozen=True)
class DebyeFit:
    """C of J(E) = C E^3 (1/meV^2 for couplings in meV) and the acoustic modes it rests on."""

    C: float
    q: float  # |q| of the shell, 1/Angstrom
    branches: dict  # "longitudinal" / "transverse": {"modes", "energy_mev", "hbar_v_mev_A", "a_mev", "C"}
    acoustic_modes: np.ndarray  # mode indices used
    weights: np.ndarray  # per mode: acoustic weight (longitudinal + transverse)


def first_shell(lattice: np.ndarray, rel_tol: float = 0.01) -> np.ndarray:
    """The smallest non-zero reciprocal-lattice vectors of a supercell (1/Angstrom), one of each +-q pair.

    Rows of ``lattice`` are the lattice vectors (Angstrom).
    """
    recip = 2 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T
    qs = [np.asarray(n) @ recip for n in itertools.product((-1, 0, 1), repeat=3) if any(n)]
    norms = np.linalg.norm(qs, axis=1)
    shell = [q for q, n in zip(qs, norms) if n <= norms.min() * (1 + rel_tol)]
    kept: list[np.ndarray] = []
    for q in shell:  # cos and sin of q.r cover -q as well
        if not any(np.allclose(q, -k) for k in kept):
            kept.append(q)
    return np.array(kept)


def acoustic_weights(spectrum, qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per mode, its weight in longitudinal and in transverse acoustic plane waves at the q vectors ``qs``.

    A uniform displacement u e^(i q.r) along a polarization is sqrt(m) u e^(i q.r) in the mass-weighted
    coordinates of the eigenvectors; the cos and sin parts, polarized along q (longitudinal) or across it
    (transverse), span the acoustic modes of the shell.
    """
    lattice = np.asarray(spectrum.lattice, dtype=float)
    r = np.asarray(spectrum.atom_frac_coords, dtype=float) @ lattice
    masses = spectrum.atomic_masses if spectrum.atomic_masses is not None else np.ones(len(r))
    sqrt_m = np.sqrt(np.asarray(masses, dtype=float))
    eig = np.asarray(spectrum.eigenvectors)
    E = eig.reshape(eig.shape[0], -1)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)

    def span(polarizations_of):
        vecs = []
        for q in qs:
            for pol in polarizations_of(q / np.linalg.norm(q)):
                for trig in (np.cos, np.sin):
                    vecs.append(((sqrt_m * trig(r @ q))[:, None] * pol[None, :]).ravel())
        Q, _ = np.linalg.qr(np.array(vecs).T)
        return Q

    def transverse(qhat):
        a = np.cross(qhat, [1.0, 0.0, 0.0] if abs(qhat[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        return [a, np.cross(qhat, a)]

    w_long = np.sum(np.abs(E @ span(lambda qhat: [qhat])) ** 2, axis=1)
    w_trans = np.sum(np.abs(E @ span(transverse)) ** 2, axis=1)
    return w_long, w_trans


def debye_coefficient(spectrum, frequencies_mev, couplings_mev, min_weight: float = MIN_ACOUSTIC_WEIGHT) -> DebyeFit:
    """Fit J(E) = C E^3 to the folded acoustic modes of the first reciprocal-lattice shell.

    ``frequencies_mev`` and ``couplings_mev`` (one coupling magnitude per mode, e.g. V_0pm in meV) are in
    the spectrum's mode order. A branch with no acoustic mode found is left out; none at all is an error.
    """
    f = np.asarray(frequencies_mev, dtype=float)
    v = np.abs(np.asarray(couplings_mev, dtype=float))
    if len(f) != spectrum.n_modes or len(v) != spectrum.n_modes:
        raise ValueError(f"{len(f)} frequencies and {len(v)} couplings for {spectrum.n_modes} modes")
    qs = first_shell(spectrum.lattice)
    q = float(np.linalg.norm(qs[0]))
    w_long, w_trans = acoustic_weights(spectrum, qs)
    weight = w_long + w_trans
    acoustic = (weight > min_weight) & (f > 0)
    if not acoustic.any():
        raise ValueError(
            f"no mode has more than {min_weight:.0%} of its weight in acoustic plane waves at |q| = {q:.3f} 1/A"
        )
    volume = abs(np.linalg.det(np.asarray(spectrum.lattice, dtype=float)))  # Angstrom^3
    branches, C = {}, 0.0
    for name, n_s, members in (
        ("longitudinal", 1, acoustic & (w_long >= w_trans)),
        ("transverse", 2, acoustic & (w_trans > w_long)),
    ):
        if not members.any():
            continue
        energy = float(f[members].mean())
        hbar_v = energy / q  # meV Angstrom
        a = float(np.mean(v[members] ** 2 / f[members]))  # meV
        C_s = n_s * volume * a / (2 * np.pi**2 * hbar_v**3)
        branches[name] = {
            "modes": np.flatnonzero(members),
            "energy_mev": energy,
            "hbar_v_mev_A": hbar_v,
            "a_mev": a,
            "C": C_s,
        }
        C += C_s
    return DebyeFit(C=C, q=q, branches=branches, acoustic_modes=np.flatnonzero(acoustic), weights=weight)
