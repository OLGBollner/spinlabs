from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from spinlabs.constants import CONSTANTS
from spinlabs.models import SpinPhononCouplingData
from spinlabs.relaxation_dynamics import find_t1_files
from spinlabs.utils import MathUtils

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)


def plot_vlines_sorted_by_magnitude(
    ax: plt.Axes,
    x: np.ndarray,
    y_data_dict: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]],
    *,
    sort_metric: str = "max",
    colors: Sequence[str | None] | None = None,
    alphas: float | Sequence[float] = 1.0,
    **vlines_kwargs,
):
    """
    Plots vertical lines so that datasets with smaller values are rendered on top (drawn last).
    """
    labels = list(y_data_dict.keys())
    data = list(y_data_dict.values())

    metric_func = {"max": np.max, "mean": np.mean, "median": np.median}.get(sort_metric, np.max)
    sort_vals = [metric_func(d[1] if isinstance(d, tuple) else d) for d in data]

    order = np.argsort(sort_vals)[::-1]  # descending: large first, small last

    if colors is None:
        colors = [None] * len(labels)
    if isinstance(alphas, (int, float)):
        alphas = [float(alphas)] * len(labels)

    # Remove alpha from kwargs if already in alphas
    vlines_kwargs.pop("alpha", None)

    for idx in order:
        label = labels[idx]
        y_data = data[idx]
        ymin, ymax = y_data if isinstance(y_data, tuple) else ([0], y_data)
        ax.vlines(
            x,
            ymin,
            ymax,
            label=label,
            color=colors[idx] if idx < len(colors) else None,
            alpha=alphas[idx] if idx < len(alphas) else 1.0,
            **vlines_kwargs,
        )


def plot_1d_spectral_functions(
    frequencies_mev: np.ndarray,
    V_0_0: np.ndarray,
    V_p_m: np.ndarray,
    V_0_pm: np.ndarray,
    *,
    zfs_mev: float | None = None,
    sigma: float = 7.5,
    res: float = 1.0,
    label_prefix: str = "",
    order: int = 1,
    ax: plt.Axes | None = None,
    ax2: plt.Axes | None = None,
    vline_scale: str = "MHz",
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plots 1D or diagonal 2D spin-phonon coupling coefficients (V_00, V_pm, V_0pm in MHz)
    as discrete vertical lines, and their Gaussian-smeared spectral functions F(w) on a twin y-axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()
    ax2 = ax.twinx() if ax2 is None else ax2

    vfs = CONSTANTS.get("MHz2J", 1.0)
    if vline_scale == "J":
        V_0_0_j = V_0_0 * vfs
        V_p_m_j = V_p_m * vfs
        V_0_pm_j = V_0_pm * vfs
    else:
        V_0_0_j = V_0_0
        V_p_m_j = V_p_m
        V_0_pm_j = V_0_pm

    sup = f"({order})"
    idx_str = "ll" if order == 2 else "l"
    y_data = {
        label_prefix + rf"$V_{{+-}}^{{{idx_str}}}$": V_p_m_j,
        label_prefix + rf"$V_{{0\pm}}^{{{idx_str}}}$": V_0_pm_j,
        label_prefix + rf"$V_{{00}}^{{{idx_str}}}$": V_0_0_j,
    }

    plot_vlines_sorted_by_magnitude(
        ax,
        frequencies_mev,
        y_data,
        sort_metric="mean",
        colors=["red", "blue", "black"],
        alphas=[0.6, 0.6, 0.6],
        linewidth=1,
    )

    for V, col, lbl in [(V_p_m_j, "red", "+-"), (V_0_pm_j, "blue", r"0\pm"), (V_0_0_j, "black", "00")]:
        smooth_x, smooth_y = MathUtils.smear_data(frequencies_mev, V**2, res, sigma)
        ax2.plot(smooth_x, smooth_y, color=col, linewidth=2, label=label_prefix + f"$F_{{{lbl}}}^{{{sup}}}$")

    ax.set_ylabel(f"Coupling coefficient ({vline_scale})")
    ax.set_ylim(bottom=0)

    ax.set_xlim(0, max(200.0, float(np.max(frequencies_mev)) * 1.05 if len(frequencies_mev) > 0 else 200.0))
    ax.set_xlabel("Vibration frequency (meV)")
    ax2.set_ylabel(r"Spectral function (MHz$^2$/meV)")
    ax2.set_ylim(bottom=0)

    if zfs_mev is not None:
        ax.axvline(x=zfs_mev, color="gray", linewidth=1, linestyle="--", label=r"$\hbar\omega = D$")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    return fig, ax, ax2


def plot_transition_rates_stacked(
    rates_data: str | Path | dict[str, Any],
    output_path: str | Path | None = None,
    show: bool = False,
    log_scale: bool = True,
) -> list[tuple[plt.Figure, plt.Axes]]:
    """
    Plots stacked area charts for transition rates (0_1 and 1_-1) versus temperature.
    """
    if isinstance(rates_data, (str, Path)):
        data = np.load(str(rates_data), allow_pickle=True)
    else:
        data = rates_data

    temperatures = np.asarray(data["temperatures"], dtype=float)
    defect = str(data.get("defect", ""))
    cell_size = str(data.get("cell_size", ""))

    first = data["first_order"][()] if "first_order" in data else {}
    second = data["second_order"][()] if "second_order" in data else {}
    two_ph = data["two_phonon"][()] if "two_phonon" in data else {}

    colors = ["#1b9e77", "#d95f02", "#7570b3"]  # direct, 1st-order Raman, 2nd-order Raman
    process_labels = ["Direct (1-phonon)", "1st-order Raman", "2nd-order Raman"]

    transitions = [("0_1", r"$0 \leftrightarrow \pm 1$"), ("1_-1", r"$+1 \leftrightarrow -1$")]
    figures = []

    for trans_key, trans_title in transitions:
        rates_stack = []
        labels_stack = []

        if trans_key in first and len(first[trans_key]) == len(temperatures):
            rates_stack.append(np.asarray(first[trans_key], dtype=float))
            labels_stack.append(process_labels[0])
        else:
            rates_stack.append(np.zeros_like(temperatures))
            labels_stack.append(process_labels[0])

        if trans_key in second and len(second[trans_key]) == len(temperatures):
            rates_stack.append(np.asarray(second[trans_key], dtype=float))
            labels_stack.append(process_labels[1])
        else:
            rates_stack.append(np.zeros_like(temperatures))
            labels_stack.append(process_labels[1])

        if trans_key in two_ph and len(two_ph[trans_key]) == len(temperatures):
            rates_stack.append(np.asarray(two_ph[trans_key], dtype=float))
            labels_stack.append(process_labels[2])
        else:
            rates_stack.append(np.zeros_like(temperatures))
            labels_stack.append(process_labels[2])

        fig, ax = plt.subplots(figsize=(7, 5))
        cumulative = np.zeros_like(temperatures)

        for rate, label, col in zip(rates_stack, labels_stack, colors):
            upper = cumulative + rate
            ax.fill_between(temperatures, cumulative, upper, color=col, alpha=0.55, label=label, linewidth=0)
            ax.plot(temperatures, upper, color=col, lw=0.8, alpha=0.9)
            cumulative = upper.copy()

        # Total line
        ax.plot(temperatures, cumulative, color="black", lw=1.5, label="Total")

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            valid_cum = cumulative[cumulative > 0]
            if len(valid_cum) > 0:
                ax.set_ylim(bottom=max(1e-12, float(np.min(valid_cum)) * 0.5))

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"Transition rate $\Gamma$ (s$^{-1}$)")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        fig.tight_layout()

        if output_path is not None:
            p = Path(output_path)
            stem = p.stem
            # Metadata belongs in the filename, not a figure title
            meta = "_".join(x.replace(" ", "-") for x in (defect, cell_size) if x)
            if meta:
                out_file = p.parent / f"{stem}_{meta}_{trans_key}{p.suffix}"
            else:
                out_file = p.parent / f"{stem}_{trans_key}{p.suffix}"
            fig.savefig(out_file, dpi=300)
            LOGGER.info(f"saved figure: {out_file}")

        figures.append((fig, ax))

    if show:
        plt.show()

    return figures


def plot_ipr_spectrum(
    frequencies_mev: np.ndarray,
    ipr: np.ndarray,
    *,
    sigma: float = 7.5,
    as_bar: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots Inverse Participation Ratio (IPR) across phonon frequencies or mode indices."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if as_bar:
        ax.bar(range(len(ipr)), ipr, color="blue", alpha=0.6, label="IPR")
        ax.set_xlabel("Mode index")
    else:
        smooth_x, smooth_y = MathUtils.smear_data(frequencies_mev, ipr, 1.0, sigma)
        ax.plot(smooth_x, smooth_y, color="blue", alpha=0.8, linewidth=2, label="IPR")
        ax.set_xlabel("Vibration frequency (meV)")

    ax.set_ylabel("Phonon IPR")
    ax.legend(loc="upper right", frameon=True)
    return fig, ax


def plot_2d_spectral_density_map(
    frequencies_mev: np.ndarray,
    zfs_2nd_derivs_mhz: np.ndarray,
    *,
    sigma: float = 7.5,
    res: float = 1.0,
    ax: plt.Axes | None = None,
    zfs_2nd_derivs_unit: str = "MHz",
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a 2D Raman phonon coupling intensity heatmap."""
    if zfs_2nd_derivs_unit == "J":
        zfs_2nd_derivs_mhz = zfs_2nd_derivs_mhz / CONSTANTS["MHz2J"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    Z = np.linalg.norm(zfs_2nd_derivs_mhz, axis=(2, 3)) if zfs_2nd_derivs_mhz.ndim == 4 else zfs_2nd_derivs_mhz
    X, Y, spectral_density = MathUtils.get_2d_spectral_density(frequencies_mev, Z, sigma, res)
    allowed_transitions = MathUtils.broad_delta(X, Y, sigma)

    mesh = ax.pcolormesh(X, Y, spectral_density * allowed_transitions, cmap="viridis", shading="auto")
    ax.set_xlabel("Vibration frequency (meV)")
    ax.set_ylabel("Vibration frequency (meV)")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Spectral function intensity (MHz)")
    return fig, ax


def _finish_run_figures(figures: list[plt.Figure], title: str | None, show: bool) -> list[plt.Figure]:
    """Name each run figure's window after its run; close the figures unless they are to be shown.

    The title only labels the interactive window, not the saved image (metadata
    belongs in the filename).
    """
    for fig in figures:
        if title and fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(title)
        if not show:
            plt.close(fig)
    return figures if show else []


def plot_run_coupling(
    run_dir: Path, out_dir: Path, fmt: str, dpi: int, show: bool, title: str | None = None
) -> list[plt.Figure]:
    """Plots 1D and diagonal 2D spin-phonon couplings and spectral functions.

    Returns the figures left open for ``show`` (the caller calls ``plt.show()``,
    so several runs open together in separate windows).
    """
    coupling_file = run_dir / "spin_phonon_coupling.npz"
    if not coupling_file.exists():
        LOGGER.warning(f"no {coupling_file.name} in {run_dir}; coupling plot skipped")
        return []

    data = SpinPhononCouplingData.load(coupling_file)
    data_display = data.to_unit("MHz").frequencies_to_unit("meV")

    # 1D plot
    fig1, _, _ = plot_1d_spectral_functions(
        frequencies_mev=data_display.frequencies,
        V_0_0=data_display.V_0_0,
        V_p_m=data_display.V_p_m,
        V_0_pm=data_display.V_0_pm,
        order=1,
    )
    out_file1 = out_dir / f"coupling_spectral_1d.{fmt}"
    fig1.savefig(out_file1, dpi=dpi, bbox_inches="tight")
    LOGGER.info(f"saved figure: {out_file1}")

    # Backward compatible alias
    out_file_alias = out_dir / f"coupling_spectral.{fmt}"
    fig1.savefig(out_file_alias, dpi=dpi, bbox_inches="tight")
    figures = [fig1]

    # 2D plot (diagonal phonons i == j)
    if data_display.V2_0_0 is not None:
        v2_00 = data_display.V2_0_0
        v2_pm = data_display.V2_p_m
        v2_0pm = data_display.V2_0_pm

        v2_00_diag = np.diag(v2_00) if v2_00.ndim == 2 else v2_00
        v2_pm_diag = (
            np.diag(v2_pm)
            if (v2_pm is not None and v2_pm.ndim == 2)
            else (v2_pm if v2_pm is not None else np.zeros_like(v2_00_diag))
        )
        v2_0pm_diag = (
            np.diag(v2_0pm)
            if (v2_0pm is not None and v2_0pm.ndim == 2)
            else (v2_0pm if v2_0pm is not None else np.zeros_like(v2_00_diag))
        )

        fig2, _, _ = plot_1d_spectral_functions(
            frequencies_mev=data_display.frequencies,
            V_0_0=v2_00_diag,
            V_p_m=v2_pm_diag,
            V_0_pm=v2_0pm_diag,
            order=2,
        )
        out_file2 = out_dir / f"coupling_spectral_2d.{fmt}"
        fig2.savefig(out_file2, dpi=dpi, bbox_inches="tight")
        LOGGER.info(f"saved figure: {out_file2}")
        figures.append(fig2)

    return _finish_run_figures(figures, title and f"Coupling: {title}", show)


def plot_run_rates(
    run_dir: Path, out_dir: Path, fmt: str, dpi: int, show: bool, title: str | None = None
) -> list[plt.Figure]:
    """Plots transition rates vs temperature, one figure per transition.

    Returns the figures left open for ``show``, like :func:`plot_run_coupling`.
    """
    rates_file = run_dir / "transition_rates.npz"
    if not rates_file.exists():
        LOGGER.warning(f"no {rates_file.name} in {run_dir}; transition rate plots skipped")
        return []

    out_base = out_dir / f"transition_rates.{fmt}"
    figures = [fig for fig, _ in plot_transition_rates_stacked(rates_file, output_path=out_base)]
    return _finish_run_figures(figures, title and f"Rates: {title}", show)


def plot_run_t1(run_dir: Path, out_dir: Path, fmt: str, dpi: int, show: bool):
    """Plots T_1 relaxation times vs temperature, one curve per initial state."""
    t1_files = find_t1_files(run_dir)
    if not t1_files:
        LOGGER.warning(f"no t1_relaxation*.npz in {run_dir}; T1 plot skipped")
        return

    out_file = out_dir / f"t1_vs_temperature.{fmt}"
    plot_t1(t1_files, output_path=out_file, plain_name=True)


_CALC_METHOD_LABELS = {"all_bands": "", "defect_band_approx": "decontaminated", "defect_band": "decontaminated"}
_INIT_STATE_LABELS = {"ms_0": "$m_s=0$", "ms_1": "$m_s=+1$", "ms_-1": "$m_s=-1$"}


def _t1_label_parts(
    defect: str, cell_size: str, calc_method: str, init_state: str, pert: float | None = None
) -> tuple[str, list[str]]:
    """Return the legend head (e.g. ``NV 512``) and readable detail strings for one T1 curve."""
    head = f"{defect} {cell_size}".strip()
    details = []
    if pert is not None:
        details.append(f"pert {pert:g}")
    method = _CALC_METHOD_LABELS.get(calc_method, calc_method.replace("_", " "))
    if method:
        details.append(method)
    if init_state:
        details.append(_INIT_STATE_LABELS.get(init_state, f"$m_s={init_state.removeprefix('ms_')}$"))
    return head, details


def _t1_pert_scale(data: np.lib.npyio.NpzFile, path: Path) -> float | None:
    """Perturbation scale of a T1 file: its ``pert_scale`` key, else a ``pert_<x>`` folder in its path."""
    if "pert_scale" in data.files and np.isfinite(float(data["pert_scale"])):
        return float(data["pert_scale"])
    for part in reversed(path.resolve().parts):
        match = re.fullmatch(r"pert_(-?[0-9.]+)", part)
        if match:
            return float(match.group(1))
    return None


def plot_t1(
    inputs: Sequence[str | Path],
    output_path: str | Path | None = None,
    show: bool = False,
    plain_name: bool = False,
    init_states: Sequence[str] | None = None,
    rate: bool = False,
    log_temperature: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot T_1 curves from one or more npz files or run directories in a single figure.

    Each input is either a T1 npz file (e.g. ``t1_relaxation_ms_0.npz``) or a
    run directory, which contributes one curve per initial state it holds. Only the ODE-fit T1 is drawn; the eigenvalue
    estimate is unreliable for initial states other than ms=0. Legend labels
    are built from the npz metadata: details shared by every curve are left
    out, the rest go in each label. No title is set. The output filename gets the metadata
    appended unless ``plain_name`` is set. With ``init_states`` only curves of
    those initial states (e.g. ``["ms_0"]``) are drawn. With ``rate`` the relaxation
    rate 1/T_1 (s^-1) is drawn instead, and ``log_temperature`` puts temperature on a log
    axis, where Raman processes show up as power laws (T^5, T^7).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    meta_parts: list[str] = []
    curves: list[tuple[np.ndarray, np.ndarray, str, list[str]]] = []

    files: list[Path] = []
    for path in map(Path, inputs):
        if path.is_dir():
            found = find_t1_files(path)
            if not found:
                raise FileNotFoundError(f"No t1_relaxation*.npz in run directory: {path}")
            files.extend(found)
        else:
            files.append(path)

    for path in files:
        data = np.load(str(path), allow_pickle=True)
        temperatures = np.asarray(data["temperatures"], dtype=float)
        defect = str(data.get("defect", ""))
        cell_size = str(data.get("cell_size", ""))
        calc_method = str(data.get("calc_method", ""))
        init_state = str(data.get("init_state", ""))
        pert = _t1_pert_scale(data, path)
        if init_states is not None and init_state not in init_states:
            continue

        t1_fit = None
        for key in ("t1_fit", "T1_times", "T1_range"):
            if key in data:
                t1_fit = np.asarray(data[key], dtype=float)
                break

        meta_parts.extend(x.replace(" ", "-") for x in (defect, cell_size, calc_method, init_state) if x)
        if pert is not None:
            meta_parts.append(f"pert-{pert:g}")

        if t1_fit is None:
            continue
        valid = np.isfinite(t1_fit) & (t1_fit > 0)
        if not np.any(valid):
            continue
        head, details = _t1_label_parts(defect, cell_size, calc_method, init_state, pert)
        curves.append((temperatures[valid], t1_fit[valid], head or path.stem, details))

    if not curves:
        LOGGER.warning("no finite T1 values in the given files; nothing plotted")
        plt.close(fig)
        return fig, ax

    # Details every curve shares are left out so the labels stay short (state them in the caption).
    shared = [d for d in curves[0][3] if all(d in c[3] for c in curves)] if len(curves) > 1 else []
    for idx, (temps, t1, head, details) in enumerate(curves):
        own = [d for d in details if d not in shared]
        label = f"{head} ({', '.join(own)})" if own else head
        values = 1.0 / t1 if rate else t1
        ax.plot(temps, values, "-", color=colors[idx % len(colors)], linewidth=2, label=label)

    ax.set_xlabel("Temperature (K)", fontsize=16)
    ax.set_ylabel(r"$1/T_1$ (s$^{-1}$)" if rate else r"$T_1$ (s)", fontsize=16)
    ax.set_yscale("log")
    if log_temperature:
        ax.set_xscale("log")
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=12)
    ax.legend(loc="best", frameon=True, fontsize=14)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = "_".join(dict.fromkeys(meta_parts))
        if meta and meta not in p.stem and not plain_name:
            p = p.with_name(f"{p.stem}_{meta}{p.suffix}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        LOGGER.info(f"saved figure: {p}")

    if show:
        plt.show()

    return fig, ax


class ZFSPlotter:
    """
    High-level plotter that coordinates matplotlib rendering without mutating underlying data arrays.
    """

    def __init__(self, plot_config: dict[str, Any] | None = None):
        self.config = plot_config or {}
        plt.rcParams.update(
            {
                "axes.labelsize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 14,
            }
        )

    def plot_data(self, data_files: Sequence[str | Path | SpinPhononCouplingData], args: Any):
        """
        Processes and plots one or more derivative dataset files or SpinPhononCouplingData objects.
        Uses the internal unit conversion of SpinPhononCouplingData instead of manual conversions.
        """
        for item in data_files:
            if isinstance(item, SpinPhononCouplingData):
                raw_data = item
            else:
                raw_data = SpinPhononCouplingData.load(item)

            # Convert to display units (MHz for coupling, meV for frequencies) using dataclass methods!
            display_data = raw_data.to_unit("MHz").frequencies_to_unit("meV")

            cell_size = display_data.cell_size
            defect = display_data.defect
            pert_scale = display_data.pert_scale
            is_second_order = display_data.order == 2

            freqs_mev = display_data.frequencies
            zfs_mev = (
                display_data.ground_state_zfs.to_unit("meV").D if display_data.ground_state_zfs is not None else None
            )

            if is_second_order and display_data.V_0_0.ndim == 2:
                # Extract diagonal for 1D representation safely without mutating original data
                V_0_0 = display_data.V_0_0.diagonal()
                V_p_m = display_data.V_p_m.diagonal()
                V_0_pm = display_data.V_0_pm.diagonal()
                calc_suffix = "_2ph"
            else:
                V_0_0 = display_data.V_0_0
                V_p_m = display_data.V_p_m
                V_0_pm = display_data.V_0_pm
                calc_suffix = ""

            if getattr(args, "ipr", False) and display_data.iprs is not None:
                fig, ax = plot_ipr_spectrum(freqs_mev, display_data.iprs, as_bar=getattr(args, "bar", False))
            else:
                fig, ax, _ = plot_1d_spectral_functions(
                    freqs_mev,
                    V_0_0=V_0_0,
                    V_p_m=V_p_m,
                    V_0_pm=V_0_pm,
                    zfs_mev=zfs_mev,
                )

            plt.tight_layout()
            if getattr(args, "plot", False):
                plt.show()
            else:
                out_name = (
                    f"{args.output}{args.format}"
                    if getattr(args, "output", None)
                    else f"{defect}_{cell_size}_zfs_plot{calc_suffix}_{pert_scale}.png"
                )
                Path("figures").mkdir(exist_ok=True)
                fig.savefig(f"figures/{out_name}")
                LOGGER.info(f"saved figure: figures/{out_name}")
                plt.close(fig)
