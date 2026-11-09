"""Spin levels and spin-phonon matrix elements for any spin S.

The ZFS and its phonon derivatives d (first or second order, defect frame, z along the C3 axis) couple to the
spin as H = S.d.S, a rank-2 operator: it changes m by at most 2 for any S. With S+- = Sx +- i Sy, p = d_xz + i d_yz
and w = (d_xx - d_yy) + 2i d_xy,

    S.d.S = (3/2) d~_zz (Sz^2 - S(S+1)/3) + (1/2) [p* {S+, Sz} + p {S-, Sz}] + (1/4) [w* S+^2 + w S-^2] + const,

d~ the traceless part. The m -> m+1 and m -> m+2 elements are therefore

    <m+1|S.d.S|m> = (1/2) p* (2m+1) sqrt(S(S+1) - m(m+1))
    <m+2|S.d.S|m> = (1/4) w* sqrt((S-m)(S+m+1)(S-m-1)(S+m+2)).

The channel amplitudes of zfs_manager are the spin-1 elements: V_0pm = |p|/sqrt(2) = |<1|S.d.S|0>| = |<0|S.d.S|-1>|
and V_+- = |w|/2 = |<1|S.d.S|-1>| (second order the same with the Taylor 1/2). For spin S a transition m -> m'
then has |<m'|S.d.S|m>|^2 = V^2 x spin_factor(S, m, m'), where spin_factor is the squared ratio of its element
to the spin-1 one:

    |dm| = 1:  (2 m_l + 1)^2 (S(S+1) - m_l(m_l+1)) / 2
    |dm| = 2:  (S - m_l)(S + m_l + 1)(S - m_l - 1)(S + m_l + 2) / 4

m_l = min(m, m'). Both are exactly 1 for every S = 1 transition. |dm| > 2 gives 0, and so does m -> -m for
half-integer S (Kramers partners): |dm| = 1 only for +-1/2, where 2 m_l + 1 = 0, and |dm| >= 3 otherwise.
The diagonal (dm = 0) part is V_00 F_z(m) with V_00 = |d~_zz|/2 and F_z(m) = 3 m^2 - S(S+1) ((1, -2, 1) for S = 1).
"""

from __future__ import annotations

import numpy as np


def spin_projections(spin: float) -> list[float]:
    """The m values S, S-1, ..., -S (floats; half-integers are exact in binary)."""
    s = float(spin)
    if s <= 0 or not np.isclose(2 * s, round(2 * s)):
        raise ValueError(f"spin must be a positive integer or half-integer, got {spin!r}")
    s = round(2 * s) / 2
    return [s - k for k in range(int(round(2 * s)) + 1)]


def m_label(m: float) -> str:
    """Text form of m in rate keys: "1", "0", "-1" for S = 1, "1.5", "0.5", "-0.5", "-1.5" for S = 3/2."""
    return f"{float(m):g}" if m != 0 else "0"


def direction_key(m: float, m_prime: float) -> str:
    """Rate key of the transition m -> m', e.g. "0_to_1", "0.5_to_1.5"."""
    return f"{m_label(m)}_to_{m_label(m_prime)}"


def directions(spin: float) -> list[str]:
    """Every ordered transition key m -> m' (m != m'), m and m' in spin_projections order."""
    ms = spin_projections(spin)
    return [direction_key(m, mp) for m in ms for mp in ms if m != mp]


def transition_class(m: float, m_prime: float) -> str:
    """Label of the unordered pair {m, m'} up to time reversal (m -> -m), for averaged rates.

    The element of smaller |m| comes first (the positive one on a tie), signs flipped so that it is >= 0 (and the
    second > 0 when the first is 0). S = 1: "0_1" (0 <-> +-1) and "1_-1"; S = 3/2: "0.5_1.5" (dm = 1 between the
    doublets), "0.5_-1.5" (dm = 2), "0.5_-0.5" and "1.5_-1.5" (Kramers partners, zero).
    """
    a, b = sorted((m, m_prime), key=lambda x: (abs(x), -x))
    if a < 0 or (a == 0 and b < 0):
        a, b = -a, -b
    return f"{m_label(a)}_{m_label(b)}"


def transition_classes(spin: float) -> dict[str, list[str]]:
    """Each transition_class with the direction keys in it, in first-appearance order of directions(spin)."""
    ms = spin_projections(spin)
    classes: dict[str, list[str]] = {}
    for m in ms:
        for mp in ms:
            if m != mp:
                classes.setdefault(transition_class(m, mp), []).append(direction_key(m, mp))
    return classes


def spin_matrices(spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sx, Sy, Sz in the basis spin_projections(spin) (m descending)."""
    ms = np.array(spin_projections(spin))
    s = ms[0]
    # <m+1|S+|m> = sqrt(S(S+1) - m(m+1)); row index of m+1 is one above m
    plus = np.diag(np.sqrt(s * (s + 1) - ms[1:] * (ms[1:] + 1)), k=1)
    sx = (plus + plus.T) / 2
    sy = (plus - plus.T) / 2j
    return sx.astype(complex), sy, np.diag(ms).astype(complex)


def zfs_hamiltonian(spin: float, d: np.ndarray) -> np.ndarray:
    """S.d.S for a 3x3 tensor d, in the basis spin_projections(spin)."""
    ops = spin_matrices(spin)
    d = np.asarray(d, dtype=float)
    return sum(d[a, b] * ops[a] @ ops[b] for a in range(3) for b in range(3))


def element_dm1(spin: float, m: float) -> float:
    """<m+1| {S+, Sz} |m> = (2m+1) sqrt(S(S+1) - m(m+1))."""
    s = float(spin)
    return (2 * m + 1) * np.sqrt(max(s * (s + 1) - m * (m + 1), 0.0))


def element_dm2(spin: float, m: float) -> float:
    """<m+2| S+^2 |m> = sqrt((S-m)(S+m+1)(S-m-1)(S+m+2))."""
    s = float(spin)
    return np.sqrt(max((s - m) * (s + m + 1) * (s - m - 1) * (s + m + 2), 0.0))


def channel(m: float, m_prime: float) -> int | None:
    """Coupling channel of m -> m': 0 (V_00), 1 (V_0pm), 2 (V_+-), None for |dm| > 2."""
    dm = abs(m - m_prime)
    return int(round(dm)) if dm <= 2 else None


def spin_factor(spin: float, m: float, m_prime: float) -> float:
    """|<m'|S.d.S|m>|^2 / V^2 for m != m', V the channel amplitude (V_0pm, V_+-) of zfs_manager.

    Normalized to the spin-1 elements those amplitudes are: exactly 1 for every S = 1 transition. 0 for
    |dm| > 2 and for Kramers partners of half-integer S. See the module docstring.
    """
    s = float(spin)
    low = min(m, m_prime)
    ch = channel(m, m_prime)
    if ch == 1:
        return (2 * low + 1) ** 2 * (s * (s + 1) - low * (low + 1)) / 2
    if ch == 2:
        return (s - low) * (s + low + 1) * (s - low - 1) * (s + low + 2) / 4
    return 0.0


def fz(spin: float, m: float) -> float:
    """F_z(m) = 3 m^2 - S(S+1): the dm = 0 element of S.d.S in units of V_00 (1, -2, 1 for S = 1)."""
    s = float(spin)
    return 3 * m * m - s * (s + 1)


def level_energies(spin: float, d: float) -> dict[float, float]:
    """Zero-field level energies E_m = D (m^2 - S(S+1)/3), in the unit of ``d``.

    The gap of m -> m' is D (m'^2 - m^2); use :func:`gap` for it (exact for integer and half-integer m).
    """
    s = float(spin)
    return {m: d * (m * m - s * (s + 1) / 3) for m in spin_projections(s)}


def gap(d: float, m: float, m_prime: float) -> float:
    """E_m' - E_m = D (m'^2 - m^2)."""
    return d * (m_prime**2 - m**2)
