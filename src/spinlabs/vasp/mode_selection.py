"""Pick a varied set of phonon modes for the step-size convergence runs (``spinlabs init convergence``).

The convergence runs check the finite-difference step for the modes the trees will run, so the few
modes they run should span what sets how a mode's ZFS responds to a step: its frequency, how few
atoms it moves (IPR), how near the defect it lives (P_2bond: its weight within two bond lengths of
the defect centre) and its irrep. ``select_convergence_modes`` picks, in this order, until it has
``n``:

1. the most localized mode (largest IPR): the local vibrational mode, the largest ZFS response;
2. the lowest- and 3. the highest-frequency mode;
4. for each irrep not yet picked, its mode most concentrated near the defect (largest P_2bond);
5. the most delocalized, bulk-like mode (smallest IPR): the smallest response, the noisiest;
6. the mode most concentrated near the defect (largest P_2bond);

then fills the rest by farthest-point sampling in (frequency, log IPR, P_2bond), each scaled to
[0, 1]. Rigid translations and modes below 1 meV are never picked. Where the defect centre is not
found, P_2bond and <r> are NaN and the IPR stands in for P_2bond.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_N_MODES = 6
MIN_FREQ_MEV = 1.0  # below this a mode is numerical noise about zero, not a vibration


@dataclass(frozen=True)
class ModeInfo:
    number: int  # 1-based mode number, as the run folders name it (spinlabs.vasp.modes)
    freq_mev: float
    irrep: str
    ipr: float
    p_near: float  # P_2bond, NaN without a defect centre
    mean_r: float  # <r> in Angstrom from the defect centre, NaN without one
    translation: bool = False

    @property
    def near(self) -> float:
        """How concentrated near the defect the mode is: P_2bond, or the IPR without a defect centre."""
        return self.p_near if np.isfinite(self.p_near) else self.ipr


@dataclass(frozen=True)
class SelectedMode:
    mode: ModeInfo
    reasons: tuple[str, ...]


def mode_properties(phonon: Path) -> list[ModeInfo]:
    """Frequency, irrep, IPR, P_2bond, <r> and whether it is a translation, for every mode of ``phonon`` (.npz).

    Labels are classified when the npz has none.
    """
    from spinlabs.logging_utils import get_logger
    from spinlabs.parsers import parse_phonon_npz
    from spinlabs.symmetry import classify_and_pair
    from spinlabs.vasp.modes import mode_numbers

    numbers = mode_numbers(phonon)
    spectrum = parse_phonon_npz(phonon)
    if spectrum.symmetries is None:
        classify_and_pair(spectrum)
    labels = [str(s) for s in spectrum.symmetries]
    if len(labels) != len(numbers):
        raise ValueError(f"{len(labels)} symmetry labels for {len(numbers)} modes in {phonon}")
    try:
        p_near, mean_r = spectrum.defect_participation, spectrum.mean_defect_distance
    except (ValueError, TypeError) as exc:  # no single defect centre: rank by the IPR alone
        get_logger(__name__).warning(f"defect localisation not computed ({exc}); the IPR stands in for P_2bond")
        p_near = mean_r = np.full(len(numbers), np.nan)
    return [
        ModeInfo(n, float(f), label, float(ipr), float(p), float(r), bool(t))
        for n, f, label, ipr, p, r, t in zip(
            numbers, spectrum.frequencies_mev, labels, spectrum.get_ipr(), p_near, mean_r,
            spectrum.translation_modes(),
        )
    ]


def candidates(modes: list[ModeInfo]) -> list[ModeInfo]:
    """The modes a convergence run may use: no rigid translation, frequency at least MIN_FREQ_MEV."""
    return [m for m in modes if not m.translation and m.freq_mev >= MIN_FREQ_MEV]


def _features(modes: list[ModeInfo]) -> np.ndarray:
    """(frequency, log IPR, P_2bond) per mode, each scaled to [0, 1] over ``modes``."""
    cols = [
        np.array([m.freq_mev for m in modes]),
        np.log(np.array([m.ipr for m in modes])),
        np.array([m.p_near for m in modes]),
    ]
    out = []
    for col in cols:
        if not np.all(np.isfinite(col)):
            col = np.zeros_like(col)
        span = col.max() - col.min()
        out.append((col - col.min()) / span if span > 0 else np.zeros_like(col))
    return np.stack(out, axis=1)


def select_convergence_modes(modes: list[ModeInfo], n: int = DEFAULT_N_MODES) -> list[SelectedMode]:
    """Up to ``n`` varied modes for the convergence runs (see the module docstring), by mode number.

    Each comes with the reasons it was picked. ValueError when no mode qualifies or ``n`` < 1.
    """
    if n < 1:
        raise ValueError(f"need at least one mode, got n = {n}")
    pool = candidates(modes)
    if not pool:
        raise ValueError("no vibrational modes to choose from (only translations or modes below "
                         f"{MIN_FREQ_MEV} meV)")
    n = min(n, len(pool))
    picked: dict[int, list[str]] = {}  # mode number -> reasons, in pick order

    def add(mode: ModeInfo, why: str) -> None:
        if mode.number in picked:
            picked[mode.number].append(why)
        elif len(picked) < n:
            picked[mode.number] = [why]

    by_number = {m.number: m for m in pool}
    add(max(pool, key=lambda m: m.ipr), "most localized (largest IPR)")
    add(min(pool, key=lambda m: m.freq_mev), "lowest frequency")
    add(max(pool, key=lambda m: m.freq_mev), "highest frequency")
    for irrep in sorted({m.irrep for m in pool}):
        if not any(by_number[k].irrep == irrep for k in picked):
            add(max((m for m in pool if m.irrep == irrep), key=lambda m: m.near), f"most defect-centred {irrep}")
    add(min(pool, key=lambda m: m.ipr), "most delocalized (smallest IPR)")
    if any(np.isfinite(m.p_near) for m in pool):
        add(max(pool, key=lambda m: m.near), "most defect-centred (largest P_2bond)")

    features = _features(pool)
    index = {m.number: i for i, m in enumerate(pool)}
    while len(picked) < n:
        chosen = features[[index[k] for k in picked]]
        dist = np.min(np.linalg.norm(features[:, None, :] - chosen[None, :, :], axis=2), axis=1)
        dist[[index[k] for k in picked]] = -1.0
        add(pool[int(np.argmax(dist))], "fills the widest gap in frequency / IPR / P_2bond")
    return [SelectedMode(by_number[k], tuple(picked[k])) for k in sorted(picked)]


def given_modes(modes: list[ModeInfo], numbers: list[int]) -> list[SelectedMode]:
    """The modes ``numbers`` as a selection (reason: given). ValueError for a number not in ``modes``."""
    by_number = {m.number: m for m in modes}
    missing = [k for k in numbers if k not in by_number]
    if missing:
        raise ValueError(f"modes {missing} are not in the phonon file (its modes: {min(by_number)}..{max(by_number)})")
    return [SelectedMode(by_number[k], ("given",)) for k in dict.fromkeys(numbers)]


def spectrum_summary(modes: list[ModeInfo]) -> str:
    """One line: mode count, per irrep, frequency and IPR range of the vibrational modes."""
    pool = candidates(modes) or modes
    irreps = ", ".join(f"{k} {v}" for k, v in sorted(Counter(m.irrep for m in pool).items()))
    freqs = [m.freq_mev for m in pool]
    iprs = [m.ipr for m in pool]
    return (f"{len(pool)} vibrational modes ({irreps}); {min(freqs):.1f}-{max(freqs):.1f} meV; "
            f"IPR {min(iprs):.4f}-{max(iprs):.4f}")


def format_selection(selection: list[SelectedMode]) -> str:
    """A table of the selected modes: number, frequency, irrep, IPR, P_2bond, <r> and why each was picked."""
    lines = [f"{'mode':>6} {'E[meV]':>8} {'irrep':<6} {'IPR':>7} {'P_2bond':>8} {'<r>[A]':>7}  why"]
    for s in selection:
        m = s.mode
        lines.append(f"{m.number:>6} {m.freq_mev:>8.2f} {m.irrep:<6} {m.ipr:>7.4f} {m.p_near:>8.3f} "
                     f"{m.mean_r:>7.2f}  {'; '.join(s.reasons)}")
    return "\n".join(lines)
