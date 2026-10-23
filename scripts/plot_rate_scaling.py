import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as Cn

from spinlabs.relaxation_dynamics import RelaxationDynamics

plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 14})

file_path = sys.argv[1:]
show_plot = False

ms_states = {"ms_1": [1.0, 0.0, 0.0], "ms_0": [0.0, 1.0, 0.0], "ms_-1": [0.0, 0.0, 1.0]}

init_state = "ms_0"

if file_path[0] == "--plot":
    show_plot = True
    file_path = file_path[1:]

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_yscale("log")
low_T = 50
high_T = 200

# Shaded temperature regions
region_cfg = [
    (0, low_T, "#3B8BD4", "0–50 K"),
    (low_T, high_T, "#EF9F27", "50–200 K"),
    (low_T, 500 + 5, "#D85A30", ">200 K"),
]
for x0, x1, col, lbl in region_cfg:
    ax.axvspan(x0, x1, color=col, alpha=0.08)
    ax.text(
        (x0 + min(x1, 500)) / 2,
        max(1, ax.get_ylim()[1]),
        lbl,
        ha="center",
        va="bottom",
        fontsize=8,
        color=col,
        alpha=0.9,
    )

# Physical constants
k_B = Cn.Boltzmann / Cn.e  # eV·K⁻¹
hbar_eVs = Cn.hbar / Cn.e  # eV·s

for file in file_path:
    print("processing file: ", file)

    # ── Fit (now returns callable functions in addition to parameters) ─────
    rd = RelaxationDynamics(init_state=ms_states[init_state], rates_data=file)
    results = rd.fit_rate_scaling(low_T=low_T, high_T=high_T)

    T = results["temperatures"]
    rate = results["rate"]  # 1/T₁ [s⁻¹]

    # Extract the pre‑fitted callable functions
    func_low = results.get("func_low")
    func_mid = results.get("func_mid")
    func_high = results.get("func_high")

    # ── Build smooth curves using the returned functions ──────────────────
    # Temperature grids for smooth plotting
    T_low = T[T <= low_T]
    T_mid = T[(T > low_T) & (T <= high_T)]
    T_high = T[T > high_T]

    T_fl = np.linspace(T_low[0], T_low[-1], 500) if len(T_low) > 1 else None
    T_fm = np.linspace(T_mid[0], T_mid[-1], 500) if len(T_mid) > 1 else None
    T_fh = np.linspace(T_high[0], T_high[-1], 500) if len(T_high) > 1 else None

    R_fl = func_low(T_fl) if func_low is not None and T_fl is not None else None
    R_fm = func_mid(T_fm) if func_mid is not None and T_fm is not None else None
    R_fh = func_high(T_fh) if func_high is not None and T_fh is not None else None

    # ── Plot data points ──────────────────────────────────────────────────
    valid = np.isfinite(rate) & (rate > 0)
    ax.scatter(T[valid], rate[valid], s=18, color="gray", alpha=0.55, zorder=3, label="$1/T_1$ data")

    # ── Build labels from fitted parameters ───────────────────────────────
    low_label = f"Low T — $T^n$   ($n={results['n_low']:.2f}$)" if results.get("n_low") is not None else None

    if results.get("omega") is not None:
        E_mid = hbar_eVs * results["omega"]
        mid_label = f"Mid T – Bose: $\\hbar\\omega={E_mid:.4f}$ meV"
    else:
        mid_label = None

    high_label = f"High T — $T^n$   ($n={results['n_high']:.2f}$)" if results.get("n_high") is not None else None

    # ── Plot smooth fit curves ────────────────────────────────────────────
    fit_cfg = [
        (T_fl, R_fl, "#185FA5", low_label),
        (T_fm, R_fm, "#BA7517", mid_label),
        (T_fh, R_fh, "#993C1D", high_label),
    ]
    for T_f, R_f, col, lbl in fit_cfg:
        if T_f is not None and R_f is not None and lbl is not None:
            ax.plot(T_f, R_f, color=col, lw=2, label=lbl)

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("$1/T_1$  (s$^{-1}$)")
ax.set_xlim(left=0)
ax.legend(fontsize=9, framealpha=0.4)
fig.tight_layout()

plt.savefig("rate_scaling.png", dpi=150)
plt.show()
