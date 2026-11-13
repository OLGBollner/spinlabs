"""F3: level diagram of V_Si- (S = 3/2) with the transition operators, a schematic (not to scale).

Two Kramers doublets {+-3/2}, {+-1/2}, E_m = D (m^2 - 5/4), separated by 2|D|; the higher doublet on top by the
sign of the computed ground-state D (AO package). Allowed transitions as double-headed arrows coloured by channel
(dm = +-1: tilting E modes, F_x' = sqrt2 {Sx,Sz}, F_y' = sqrt2 {Sy,Sz}; dm = +-2: in-plane E modes,
F_x = Sx^2 - Sy^2, F_y = {Sx,Sy}), forbidden ones gray dashed with a cross, the A1 channel (F_z = 3 Sz^2 - S(S+1))
as a short vertical arrow beside each doublet. Next to each allowed arrow the squared matrix element relative to
the corresponding spin-1 transition (<+1|F|0> for dm = 1, <+1|F|-1> for dm = 2), checked here from explicit
spin-3/2 and spin-1 matrices against spinlabs.spin.spin_factor.
Outputs: figures/out/main/F3_VSi_levels.{pdf,png} (vector PDF), figures/out/main/numbers_vsi_levels.md, records/F3.json.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch

import config as C
from common import Record, rates

from spinlabs.spin import spin_factor

MHZ_PER_MEV = 241798.9242
COL = {"dm1": C.COLORS["V0pm"], "dm2": C.COLORS["Vpm"], "A1": C.COLORS["V00"], "forbidden": "#9a9893"}


def spin_matrices(s):
    """Sx, Sy, Sz in the basis m = s, s-1, ..., -s (hbar = 1)."""
    m = np.arange(s, -s - 1, -1)
    sp = np.zeros((len(m), len(m)), dtype=complex)
    for i in range(1, len(m)):  # <m+1|S+|m> = sqrt(s(s+1) - m(m+1))
        sp[i - 1, i] = np.sqrt(s * (s + 1) - m[i] * (m[i] + 1))
    return (sp + sp.conj().T) / 2, (sp - sp.conj().T) / 2j, np.diag(m).astype(complex), list(m)


def operators(s):
    sx, sy, sz, m = spin_matrices(s)
    ac = lambda a, b: a @ b + b @ a  # noqa: E731
    ops = {"Fx'": np.sqrt(2) * ac(sx, sz), "Fy'": np.sqrt(2) * ac(sy, sz), "Fx": sx @ sx - sy @ sy, "Fy": ac(sx, sy),
           "Fz": 3 * sz @ sz - s * (s + 1) * np.eye(len(m))}
    return ops, m


def element(ops, m, op, a, b):
    """|<a|F|b>|^2."""
    return float(abs(ops[op][m.index(a), m.index(b)]) ** 2)


def check():
    """Squared matrix elements of the spin-3/2 transitions relative to spin 1, every operator; and the forbidden ones."""
    o3, m3 = operators(1.5)
    o1, m1 = operators(1.0)
    ref = {1: (1.0, 0.0), 2: (1.0, -1.0)}  # spin-1 reference transitions <+1|F|0>, <+1|F|-1>
    rows, forbidden = [], []
    for a, b in ((1.5, 0.5), (-1.5, -0.5), (1.5, -0.5), (-1.5, 0.5)):
        dm = int(round(abs(a - b)))
        for op in ("Fx'", "Fy'", "Fx", "Fy", "Fz"):
            e3, e1 = element(o3, m3, op, a, b), element(o1, m1, op, *ref[dm])
            if e3 > 1e-12 or e1 > 1e-12:
                rows.append((f"{a:+g} <-> {b:+g}", dm, op, e3, e1, e3 / e1 if e1 > 1e-12 else float("nan"),
                             spin_factor(1.5, a, b)))
    for a, b in ((0.5, -0.5), (1.5, -1.5)):
        mx = max(element(o3, m3, op, a, b) for op in o3)
        forbidden.append((f"{a:+g} <-> {b:+g}", mx, spin_factor(1.5, a, b)))
    return rows, forbidden


def arrow(ax, p, q, color, ls="-", lw=0.9, z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>,head_length=3.2,head_width=1.6", color=color, lw=lw,
                                 ls=ls, shrinkA=0, shrinkB=0, zorder=z, mutation_scale=1))


def draw(plt, d_sign):
    fig, ax = plt.subplots(figsize=(C.SINGLE, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.98)
    ax.axis("off")
    y_hi, y_lo = 0.70, 0.22
    upper, lower = ((0.5, -0.5), (1.5, -1.5)) if d_sign < 0 else ((1.5, -1.5), (0.5, -0.5))
    xl, xr, w = 0.36, 0.70, 0.15  # level centres (+m left, -m right) and half width
    ypos = {upper[0]: (xl, y_hi), upper[1]: (xr, y_hi), lower[0]: (xl, y_lo), lower[1]: (xr, y_lo)}
    frac = {0.5: r"\frac{1}{2}", 1.5: r"\frac{3}{2}"}
    for mval, (x, y) in ypos.items():
        ax.plot([x - w / 2 * 1.0, x + w / 2 * 1.0], [y, y], color="k", lw=1.6, solid_capstyle="butt", zorder=4)
        sign = "+" if mval > 0 else "-"
        ax.text(x, y + (0.03 if y == y_hi else -0.03), rf"$m_s={sign}{frac[abs(mval)]}$", ha="center",
                va="bottom" if y == y_hi else "top", fontsize=C.FONT - 1)

    def end(mval, top_side):
        x, y = ypos[mval]
        return x, y - 0.012 if top_side else y + 0.012

    def at_top(mval):
        return ypos[mval][1] == y_hi

    # dm = 1: +3/2 <-> +1/2 and -3/2 <-> -1/2 (vertical, same side)
    for a, b in ((1.5, 0.5), (-1.5, -0.5)):
        top, bot = (a, b) if at_top(a) else (b, a)
        (x, yt), (_, yb) = end(top, True), end(bot, False)
        dx = -0.035 if x < 0.5 else 0.035
        arrow(ax, (x + dx, yb), (x + dx, yt), COL["dm1"])
        ax.text(x + dx + (-0.018 if dx < 0 else 0.018), (yt + yb) / 2, r"$6\times$", color="k", fontsize=C.FONT - 1.5,
                ha="right" if dx < 0 else "left", va="center")
    # dm = 2: +3/2 <-> -1/2 and -3/2 <-> +1/2 (diagonals)
    for a, b in ((1.5, -0.5), (-1.5, 0.5)):
        top, bot = (a, b) if at_top(a) else (b, a)
        (xt, yt), (xb, yb) = end(top, True), end(bot, False)
        xt += 0.02 if xt < 0.5 else -0.02
        xb += 0.02 if xb < 0.5 else -0.02
        arrow(ax, (xb, yb), (xt, yt), COL["dm2"])
    ax.text(0.5 * (xl + xr), (y_hi + y_lo) / 2 + 0.09, r"$3\times$", color="k", fontsize=C.FONT - 1.5, ha="center",
            va="center")
    # forbidden: Kramers partners within each doublet (horizontal, gray dashed, crossed)
    why = {1.0: "Kramers partners", 3.0: "Kramers partners;\n" + r"$\Delta m_s=3$ > rank of the ZFS"}
    for y, top in ((y_hi, upper), (y_lo, lower)):
        xa, xb = xl + w / 2 + 0.012, xr - w / 2 - 0.012
        yy = y + (0.06 if y == y_hi else -0.06)
        ax.text((xa + xb) / 2, yy + (0.035 if y == y_hi else -0.035), why[2 * abs(top[0])], ha="center",
                va="bottom" if y == y_hi else "top", fontsize=C.FONT - 3, color=C.COLORS["ref"], linespacing=1.0)
        arrow(ax, (xa, yy), (xb, yy), COL["forbidden"], ls=(0, (3, 2)), lw=0.9)
        ax.text((xa + xb) / 2, yy, r"$\times$", color="#c92a2a", ha="center", va="center", fontsize=C.FONT + 2,
                fontweight="bold", zorder=5)
    # A1: energy modulation beside each doublet
    for y in (y_hi, y_lo):
        x = xr + w / 2 + 0.06
        arrow(ax, (x, y - 0.045), (x, y + 0.045), COL["A1"], lw=1.0)
        ax.text(x + 0.02, y, "energy\nmodulation\n(no transition)", fontsize=C.FONT - 2.5, ha="left", va="center",
                linespacing=1.05)
    # 2|D| bracket
    xb = 0.12
    ax.annotate("", (xb, y_lo), (xb, y_hi), arrowprops=dict(arrowstyle="<->", lw=0.6, color=C.COLORS["ref"],
                                                             shrinkA=0, shrinkB=0))
    ax.plot([xb - 0.015, xl - w / 2 - 0.01], [y_hi, y_hi], color=C.COLORS["ref"], lw=0.4, ls=":")
    ax.plot([xb - 0.015, xl - w / 2 - 0.01], [y_lo, y_lo], color=C.COLORS["ref"], lw=0.4, ls=":")
    ax.text(xb - 0.02, (y_hi + y_lo) / 2, r"$2|D|$", ha="right", va="center", fontsize=C.FONT - 1)
    # optically pumped doublet
    yp = y_hi if 0.5 in upper else y_lo
    ax.text(0.5 * (xl + xr), yp + (0.19 if yp == y_hi else -0.21), r"optically pumped ($\pm\frac{1}{2}$)",
            ha="center", va="center", fontsize=C.FONT - 2.5, style="italic", color=C.COLORS["ref"])
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL["dm1"], lw=1.4, label=r"$\Delta m_s=\pm1$: $F_{x'},F_{y'}$, tilting $E$"),
               Line2D([], [], color=COL["dm2"], lw=1.4, label=r"$\Delta m_s=\pm2$: $F_x,F_y$, in-plane $E$"),
               Line2D([], [], color=COL["A1"], lw=1.4, label=r"$F_z$, $A_1$ modes"),
               Line2D([], [], color=COL["forbidden"], lw=0.9, ls=(0, (3, 2)), label="forbidden")]
    ax.text(0.5, 0.035, r"$n\times$: $|\langle m'|F|m\rangle|^2$ relative to the spin-1 transition",
            ha="center", va="center", fontsize=C.FONT - 2.5)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=C.FONT - 2.5,
              handlelength=1.6, columnspacing=1.0, labelspacing=0.3, frameon=False)
    return fig


def main():
    plt = C.style()
    rec = Record("F3", "V_Si- (S = 3/2) level diagram with the transition operators", "scripts/paper/f3_vsi_levels.py",
                 "main")
    d_ao = float(rates("V_Si_128", "AO")["zfs_mev"])
    d_uo = float(rates("V_Si_128", "UO")["zfs_mev"])
    rows, forbidden = check()
    for label, dm, op, e3, e1, ratio, sf in rows:
        if op != "Fz" and not np.isclose(ratio, sf):
            raise SystemExit(f"{label} {op}: ratio {ratio} != spin_factor {sf}")
    if any(mx > 1e-24 for _, mx, _ in forbidden):
        raise SystemExit(f"forbidden element not zero: {forbidden}")

    fig = draw(plt, np.sign(d_ao))
    rec.data["files"] = C.save(fig, C.MAIN, "F3_VSi_levels")
    plt.close(fig)

    L = ["# V_Si- level diagram (F3)", "", "Generated by scripts/paper/f3_vsi_levels.py.", "",
         "## Sign of D", "",
         f"AO (used): D = {d_ao * MHZ_PER_MEV:.1f} MHz ({'negative' if d_ao < 0 else 'positive'}): E(+-3/2) = D, "
         f"E(+-1/2) = -D, so the {'+-1/2' if d_ao < 0 else '+-3/2'} doublet is on top. "
         f"UO: D = {d_uo * MHZ_PER_MEV:.2f} MHz, the opposite sign.", "",
         "## Matrix elements", "",
         "Explicit spin matrices (basis m = S..-S, hbar = 1), F_x' = sqrt2 {Sx,Sz}, F_y' = sqrt2 {Sy,Sz}, "
         "F_x = Sx^2 - Sy^2, F_y = {Sx,Sy}, F_z = 3 Sz^2 - S(S+1). Spin-1 reference: <+1|F|0> (dm = 1), <+1|F|-1> "
         "(dm = 2). Non-zero elements only.", "",
         "| transition | dm | operator | abs(<a|F|b>)^2, S = 3/2 | spin 1 | ratio | spinlabs spin_factor |",
         "|---|---|---|---|---|---|---|"]
    for label, dm, op, e3, e1, ratio, sf in rows:
        L.append(f"| {label} | {dm} | {op} | {e3:.6g} | {e1:.6g} | {ratio:.6g} | {sf:g} |")
    L += ["", "Forbidden (max abs(<a|F|b>)^2 over all five operators):", "", "| transition | max | spin_factor |", "|---|---|---|"]
    for label, mx, sf in forbidden:
        L.append(f"| {label} | {mx:.1e} | {sf:g} |")
    path = C.MAIN / "numbers_vsi_levels.md"
    path.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    rec.data["settings"] = {"schematic": "not to scale", "D sign from": "AO package " + C.SYSTEMS["V_Si_128"]["packages"]["AO"],
                            "D_MHz": {"AO": d_ao * MHZ_PER_MEV, "UO": d_uo * MHZ_PER_MEV},
                            "numbers file": "main/numbers_vsi_levels.md"}
    rec.data["numbers"] = {"matrix elements": [dict(zip(("transition", "dm", "operator", "S=3/2", "S=1", "ratio",
                                                         "spin_factor"), r)) for r in rows],
                           "forbidden": [dict(zip(("transition", "max", "spin_factor"), f)) for f in forbidden]}
    rec.write()


if __name__ == "__main__":
    main()
