"""V_Si^- (V2, k site, S = 3/2) in 4H-SiC (128 atoms) at the production settings of config.py: numbers, no figure.

Ground-state ZFS (AO, UO) against the measured 2D = 70 MHz and the UO spin correction (2/3 for S = 3/2); the
couplings and rho (first and second order, as M3); the Debye coefficients C_0pm and C_+-; the directional rates
(direct + two-phonon) between the doublets {+-3/2} and {+-1/2}, dm = 1 (+-3/2 <-> +-1/2) and dm = 2
(+-3/2 <-> -+1/2) separately; T1(T) from the {+-1/2} initial state (optical pumping polarizes V2 into it), fitted
to the doublet's summed population, and from the {+-3/2} state. Writes figures/out/numbers_vsi.md (or the path
given as the first argument). Run from the repo root: python scripts/paper/vsi_numbers.py
"""

import sys
from pathlib import Path

import numpy as np

import config as C
from common import couplings, directional, rates, t1_details
from m3_rho import KEYS, rho_sd

from spinlabs.models import ZFSTensor, spin_correction_scale
from spinlabs.spin import direction_key, level_energies, spin_factor, transition_classes

SYSTEM = "V_Si_128"
SPIN = 1.5
TEMPS = np.concatenate([[1.0], np.arange(5.0, 601.0, 5.0)])  # K
QUOTE = (1.0, 10.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0)
STATES = ("ms_pm1_2", "ms_pm3_2")
EXPERIMENT_2D_MHZ = 70.0  # V2 (k): D = 35 MHz (ivady_identification_2017, the D_table.tex row)
# The UO ground state of the raw (uncorrected) convergence-test pack, from the same ZFS_occup OUTCAR
RAW_UO_GROUND_STATE = "V_Si_128/defect_band_approx/20261103/V_Si_128_20261103_conv_test_occup.npz"
MEV_PER_MHZ = 4.135667696e-6  # h x 1 MHz in meV

# The inter-doublet directions, one representative each (time reversal gives the -m -> -m' partners)
DM1 = ("0.5_to_1.5", "1.5_to_0.5")  # +1/2 <-> +3/2
DM2 = ("-0.5_to_1.5", "1.5_to_-0.5")  # -1/2 <-> +3/2
ZERO = ("0.5_to_-0.5", "1.5_to_-1.5")  # Kramers partners: must be 0


def g(x, digits=4):
    if x is None or not np.isfinite(x):
        return str(x)
    return f"{x:.{digits}g}"


def zfs_section(lines):
    lines += ["## Ground-state ZFS", ""]
    lines += [f"Measured: 2D = {EXPERIMENT_2D_MHZ:g} MHz (D = {EXPERIMENT_2D_MHZ / 2:g} MHz). D = 3/2 D_zz, E = (D_xx - D_yy)/2 "
              "in the principal frame; the {+-3/2} and {+-1/2} doublets are split by 2D (positive D: +-3/2 above).", ""]
    lines += ["| scheme | S (package) | spin correction applied | D (MHz) | 2D (MHz) | 2D / 70 MHz | E (MHz) |",
              "|---|---|---|---|---|---|---|"]
    out = {}
    for scheme in ("AO", "UO"):
        d = np.load(C.package(SYSTEM, scheme), allow_pickle=True)
        tensor = ZFSTensor(matrix=np.asarray(d["ground_state_zfs_matrix"], dtype=float), unit=str(d["ground_state_zfs_unit"]))
        D, E = tensor.D, tensor.E
        out[scheme] = dict(D=D, E=E, matrix=tensor.matrix, spin=float(d["spin"]), applied=bool(d["spin_correction_applied"]))
        lines.append(f"| {scheme} | {float(d['spin']):g} | {bool(d['spin_correction_applied'])} | {g(D)} | {g(2 * D)} | "
                     f"{g(2 * D / EXPERIMENT_2D_MHZ, 3)} | {g(E, 3)} |")
    raw = np.load(C.DATA / RAW_UO_GROUND_STATE, allow_pickle=True)
    d0 = np.asarray(raw["d0"], dtype=float)
    stored = out["UO"]["matrix"]
    ratio = float(stored[2, 2] / d0[2, 2])
    max_dev = float(np.max(np.abs(stored - spin_correction_scale(SPIN) * d0)))
    lines += ["", "UO spin correction: the package records S = 1.5 (NUPDOWN = 3) and spin_correction_applied = True, so "
              "the loader does not scale it again; the parser scaled the ground state and every perturbed tensor by "
              f"spin_correction_scale(1.5) = {spin_correction_scale(SPIN):.6g} at intake. Check against the uncorrected "
              f"ground state of the same ZFS_occup OUTCAR ({RAW_UO_GROUND_STATE}): stored D_zz / raw D_zz = "
              f"{ratio:.8f}, max |stored - 2/3 raw| = {max_dev:.2e} MHz over the tensor. (The perturbed runs of that "
              "pack are separate calculations at other settings, so they cannot be compared one to one.)", ""]
    return out


def coupling_section(lines):
    lines += ["## Couplings (MHz, as they enter the rates)", "",
              "Per channel: modes (first order) or diagonal pairs (second order) with a nonzero value, the largest, "
              "and sqrt(sum V^2). Spin-independent channel amplitudes; a transition m -> m' takes V^2 x the spin factor "
              "(S = 3/2: 6 for dm = 1 between the doublets, 3 for dm = 2, 0 for +1/2 <-> -1/2 and +3/2 <-> -3/2).", "",
              "| order | channel | AO n | AO max | AO sqrt(sum V^2) | UO n | UO max | UO sqrt(sum V^2) |",
              "|---|---|---|---|---|---|---|---|"]
    values = {}
    for order in (1, 2):
        for ch in C.CHANNELS:
            row = [f"{order}", ch]
            for scheme in ("AO", "UO"):
                v = np.asarray(couplings(SYSTEM, scheme)[KEYS[order][ch]], dtype=float)
                v = np.abs(np.diag(v) if order == 2 else v)
                values[(order, ch, scheme)] = v
                row += [str(int(np.count_nonzero(v))), g(v.max()), g(np.sqrt((v**2).sum()))]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "### rho = sqrt(sum V_UO^2 / sum V_AO^2) (M3: modes / diagonal pairs, weighted SD of r = |V_UO|/|V_AO|, "
              "weights V_AO^2)", "", "| order | channel | rho | SD | n |", "|---|---|---|---|---|"]
    for order in (1, 2):
        for ch in C.CHANNELS:
            rho, sd, n = rho_sd(values[(order, ch, "AO")], values[(order, ch, "UO")])
            lines.append(f"| {order} | {ch} | {g(rho)} | {g(sd, 3)} | {n} |")
    c = couplings(SYSTEM, "AO")
    lines += ["", f"Modes: {len(c['frequencies'])}; second-order off-diagonal pairs from the geometric-mean fill "
              "(central-difference packages have only the diagonal (l, l) curvatures).", ""]


def rates_section(lines, zfs):
    lines += ["## Rates (1/s), production settings", "",
              f"sigma {C.SIGMA:g} meV, direct = {C.DIRECT} (C_0pm for dm = 1, C_+- for dm = 2), zero-gap pairs "
              f"excluded, pairs = {C.PAIRS} (fill), processes {' + '.join(C.PROCESSES)}. One representative per "
              "time-reversed pair (the partner -m -> -m' is equal). up/down: the gap E_m' - E_m > 0 / < 0.", ""]
    results = {}
    for scheme in ("AO", "UO"):
        r = rates(SYSTEM, scheme, temps=TEMPS)
        results[scheme] = r
        if float(r["spin"]) != SPIN:
            raise SystemExit(f"{scheme}: rates computed for S = {float(r['spin']):g}, expected {SPIN:g}")
        D = float(r["zfs_mev"])
        energies = level_energies(SPIN, D)
        lines += [f"### {scheme}", "",
                  f"D = {g(D / MEV_PER_MHZ)} MHz = {g(D)} meV; gap E(3/2) - E(1/2) = 2D = {g(2 * D)} meV "
                  f"({g(2 * D / MEV_PER_MHZ)} MHz). Debye C_0pm = {g(float(r['debye_C']))} 1/meV^2, "
                  f"C_+- = {g(float(r['debye_C_pm']))} 1/meV^2.", ""]
        lines += ["| T (K) | " + " | ".join(f"{p} {k}" for k in DM1 + DM2 for p in ("direct", "2ph")) + " |",
                  "|---|" + "---|" * 8]
        idx = {T: int(np.flatnonzero(np.isclose(TEMPS, T))[0]) for T in QUOTE}
        for T, i in idx.items():
            cells = []
            for k in DM1 + DM2:
                cells += [g(float(r[f"first_order:{k}"][i]), 3), g(float(r[f"two_phonon:{k}"][i]), 3)]
            lines.append(f"| {T:g} | " + " | ".join(cells) + " |")
        # checks: time reversal, Kramers zeros, class means
        tr = max(abs(float(r[f"{p}:{direction_key(-a, -b)}"][i]) - float(r[f"{p}:{direction_key(a, b)}"][i]))
                 / max(abs(float(r[f"{p}:{direction_key(a, b)}"][i])), 1e-300)
                 for p in ("first_order", "two_phonon") for i in range(len(TEMPS))
                 for a in (1.5, 0.5, -0.5, -1.5) for b in (1.5, 0.5, -0.5, -1.5) if a != b)
        zero = max(abs(float(r[f"{p}:{k}"][i])) for p in C.PROCESSES for k in ZERO + ("-0.5_to_0.5", "-1.5_to_1.5")
                   for i in range(len(TEMPS)))
        up = [k for k in DM1 + DM2 if energies[float(k.split("_to_")[1])] > energies[float(k.split("_to_")[0])]]
        lines += ["", f"up (absorbing) directions: {', '.join(up)}; max relative time-reversal mismatch {tr:.1e}; "
                  f"largest Kramers-pair rate (+-1/2 <-> -+1/2, +-3/2 <-> -+3/2) {zero:.1e} 1/s.", ""]
        means = {name: keys for name, keys in transition_classes(SPIN).items() if name in ("0.5_1.5", "0.5_-1.5")}
        tot = {name: sum(directional(r)[k] for k in keys) / len(keys) for name, keys in means.items()}
        lines += ["Mean rate per class (direct + two-phonon, averaged over the four directions): dm = 1 (0.5_1.5), "
                  "dm = 2 (0.5_-1.5)", "", "| T (K) | dm = 1 | dm = 2 | dm = 2 / dm = 1 |", "|---|---|---|---|"]
        for T, i in idx.items():
            a, b = tot["0.5_1.5"][i], tot["0.5_-1.5"][i]
            lines.append(f"| {T:g} | {g(a, 3)} | {g(b, 3)} | {g(b / a if a else np.nan, 3)} |")
        lines.append("")
    return results


def t1_section(lines, results):
    lines += ["## T1 (s)", "",
              "Rate matrix from the directional rates (direct + two-phonon), p(t) = exp(A t) p0, a exp(-t/T1) + c "
              "fitted to the summed population of the initial doublet (RelaxationDynamics). ms_pm1_2: p(+1/2) = "
              "p(-1/2) = 1/2; ms_pm3_2: p(+3/2) = p(-3/2) = 1/2. For time-reversal-symmetric rates the doublet "
              "population relaxes with the single rate k = G(1/2->3/2) + G(1/2->-3/2) + G(3/2->1/2) + G(3/2->-1/2) "
              "(1/k in the last column). source: fit / eigenvalue fallback / none (0 K).", ""]
    for scheme, r in results.items():
        d = directional(r)
        k = d["0.5_to_1.5"] + d["0.5_to_-1.5"] + d["1.5_to_0.5"] + d["1.5_to_-0.5"]
        t1 = {s: t1_details(TEMPS, d, s, spin=SPIN) for s in STATES}
        lines += [f"### {scheme}", "", "| T (K) | T1 ms_pm1_2 | source | T1 ms_pm3_2 | source | 1/k |",
                  "|---|---|---|---|---|---|"]
        for T in QUOTE:
            i = int(np.flatnonzero(np.isclose(TEMPS, T))[0])
            row = [f"{T:g}"]
            for s in STATES:
                row += [g(float(t1[s][0][i])), t1[s][2][i]]
            row.append(g(1 / k[i] if k[i] > 0 else np.inf))
            lines.append("| " + " | ".join(row) + " |")
        fallbacks = {s: [float(T) for T, src in zip(TEMPS, t1[s][2]) if src == "eigenvalue"] for s in STATES}
        fit = t1["ms_pm1_2"][0]
        ok = np.isfinite(fit) & (k > 0)
        dev = float(np.max(np.abs(fit[ok] * k[ok] - 1))) if ok.any() else np.nan
        lines += ["", "eigenvalue fallbacks: " + "; ".join(f"{s}: {v if v else 'none'}" for s, v in fallbacks.items())
                  + f". max |T1(ms_pm1_2) k - 1| over the grid: {dev:.1e}.", ""]
        lines += ["Full T1(T), ms_pm1_2 (s):", "", "```"]
        lines += [" ".join(f"{T:g}:{g(float(v), 4)}" for T, v in zip(TEMPS[j:j + 8], fit[j:j + 8]))
                  for j in range(0, len(TEMPS), 8)]
        lines += ["```", ""]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else C.OUT / "numbers_vsi.md"
    lines = ["# V_Si^- (V2, k site, S = 3/2), 4H-SiC 128 atoms: numbers", "",
             "Generated by scripts/paper/vsi_numbers.py (branch general-spin-rates). Packages: AO "
             f"{C.SYSTEMS[SYSTEM]['packages']['AO']}, UO {C.SYSTEMS[SYSTEM]['packages']['UO']} (central differences: "
             "diagonal second order (f(+Q) + f(-Q) - 2 f(0)) / Q^2, off-diagonal pairs from the fill). "
             f"Spin factors: dm = 1 between the doublets {spin_factor(SPIN, 0.5, 1.5):g}, dm = 2 "
             f"{spin_factor(SPIN, -0.5, 1.5):g} (x V^2; 1 for every S = 1 transition).", ""]
    zfs = zfs_section(lines)
    coupling_section(lines)
    results = rates_section(lines, zfs)
    t1_section(lines, results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
