"""M1: LaTeX table of the zero-field splitting D (and E where nonzero), AO / UO / experiment.

Cells: ClV+ (576), VV0 (128), NV- (512, 64). The V_Si row is copied verbatim from the existing D_table.tex in
the repo root (V_Si is not recomputed here). D = 3/2 D_zz and E = (D_xx - D_yy)/2 of the packaged ground-state
tensor (UO packages carry the 3/2 spin correction).
"""

import re

import config as C
from common import Record

from spinlabs.models import RawZFSData
from spinlabs.tables import DEFECT_INFO, _fmt_sig

ROWS = [  # (defect key, LaTeX name, configuration, base crystal, cells)
    ("ClV", r"ClV$^{+}$", "kk", "4H-SiC", ["ClV_576"]),
    ("DiV", r"VV$^{0}$", "kk", "4H-SiC", ["DiV_128"]),
    ("V_Si", None, None, "4H-SiC", []),  # copied from the existing table
    ("NV", r"NV$^{-}$", "---", "Diamond", ["NV_512", "NV_64"]),
]
E_MIN_MHZ = 1.0  # |E| below this is numerical noise of an axial (C3v) defect: the E columns are left out
OLD_TABLE = C.ROOT / "D_table.tex"


def zfs(system, scheme):
    t = RawZFSData.load(str(C.package(system, scheme))).ground_state_zfs.to_unit("MHz")
    return float(t.D), float(t.E)


def main():
    rec = Record("M1", "ZFS table", "scripts/paper/m1_zfs_table.py", "main")
    old = OLD_TABLE.read_text()
    vsi = next(line for line in old.splitlines() if "V$_{\\mathrm{Si}}$" in line)
    if not re.search(r"D_\\text\{experiment\}\$ & \$D_\\text\{UE\}\$ & \$D_\\text\{AE\}\$", old):
        raise SystemExit("the old table's columns are not experiment, UE, AE: the V_Si row would land wrong")

    letters, values = {}, {}
    for _, _, _, _, systems in ROWS:
        for s in systems:
            letters.setdefault(C.SYSTEMS[s]["cell"], chr(ord("a") + len(letters)))
            values[s] = {sch: zfs(s, sch) for sch in C.SCHEMES}
    # V_Si's superscript letters must mean the same cells as before
    for letter in set(re.findall(r"\^\{\((\w)\)\}", vsi)):
        old_cell = re.search(rf"\({letter}\) cell size (\d+)", old).group(1)
        new_cell = next((str(c) for c, l in letters.items() if l == letter), None)
        if old_cell != new_cell:
            raise SystemExit(f"V_Si row uses ({letter}) = cell {old_cell}, the new table has {new_cell}")

    show_e = any(abs(v[1]) >= E_MIN_MHZ for s in values.values() for v in s.values())

    def joined(systems, scheme, k):
        return ", ".join(f"${_fmt_sig(values[s][scheme][k])}^{{({letters[C.SYSTEMS[s]['cell']]})}}$" for s in systems)

    mhz = r"(\si{\mega\hertz})"
    # UO before AO: the column order of the old table, whose V_Si row is copied verbatim
    head = ["Defect", "Configuration", r"$D_\text{experiment}$", r"$D_\text{UO}$", r"$D_\text{AO}$"]
    if show_e:
        head += [r"$E_\text{UO}$", r"$E_\text{AO}$"]
    n = len(head)
    footnote = "; ".join(f"({l}) cell size {c}" for c, l in letters.items())
    lines = [r"\begin{table}[htbp]",
             r"  \caption{Zero-field splitting parameter $D$ of the treated point defects, from all orbitals (AO) and "
             rf"from the unpaired orbitals (UO). The superscripts refer to different defect supercells: {footnote}.}}",
             r"  \centering", rf"  \begin{{tabular}}{{ll{'c' * (n - 2)}}}", r"    \toprule",
             "    " + " & ".join(head) + r" \\", "    " + " & ".join(["", ""] + [mhz] * (n - 2)) + r" \\",
             r"    \midrule"]
    base = None
    for key, name, conf, crystal, systems in ROWS:
        if crystal != base:
            if base is not None:
                lines.append(r"    \midrule")
            lines.append(rf"    \multicolumn{{{n}}}{{c}}{{\textit{{{crystal}}}}} \\")
            base = crystal
        if key == "V_Si":
            if show_e:
                raise SystemExit("E columns would need an E entry in the V_Si row, which is kept unchanged")
            lines.append(vsi)
            continue
        info = DEFECT_INFO[key]
        exp = (info["d_experiment"] + rf"~\cite{{{info['cite']}}}") if info.get("d_experiment") else "---"
        row = [name, conf, exp, joined(systems, "UO", 0), joined(systems, "AO", 0)]
        if show_e:
            row += [joined(systems, "UO", 1), joined(systems, "AO", 1)]
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"  \label{tab:zfs_d}", r"\end{table}"]
    out = C.MAIN / "D_table.tex"
    C.MAIN.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())

    rec.data["files"] = [str(out.relative_to(C.OUT))]
    rec.data["settings"] = {
        "packages": {s: {sch: C.SYSTEMS[s]["packages"][sch] for sch in C.SCHEMES} for s in values},
        "quantity": "D = 3/2 D_zz, E = (D_xx - D_yy)/2 of the ground-state tensor; UO with the 3/2 spin correction",
        "experiment": {k: f"{DEFECT_INFO[k]['d_experiment']} MHz ({DEFECT_INFO[k]['cite']})"
                       for k in ("DiV", "NV")} | {"ClV": "none"},
        "E columns": f"shown only if some |E| >= {E_MIN_MHZ} MHz: {'shown' if show_e else 'not shown'}",
        "V_Si row": "copied verbatim from D_table.tex in the repo root",
    }
    rec.data["numbers"] = {"D and E (MHz)": {s: {sch: {"D": v[0], "E": v[1]} for sch, v in d.items()}
                                              for s, d in values.items()}}
    rec.write()


if __name__ == "__main__":
    main()
