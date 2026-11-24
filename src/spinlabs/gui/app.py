"""Tk application for the convergence viewer.

The heavy analysis lives in :mod:`spinlabs.gui.analysis` and
:mod:`spinlabs.gui.plots`; this module only wires it to widgets so it can be
tested headlessly.  Tkinter is stdlib but not guaranteed on every Python
installation, so the import failure is turned into a helpful message.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")  # must precede the pyplot import in gui.plots

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from spinlabs.convergence import SNR_TARGET, TRUNC_TOL, format_snr
from spinlabs.convergence_channels import CHANNELS, CHANNEL_NAMES, ChannelAnalysis
from spinlabs.convergence_summary import WEAK_FRACTION
from spinlabs.gui import plots, theme
from spinlabs.gui.analysis import ModeEntry, NormResult, PackSession
from spinlabs.gui.catalog import Catalog, Group, Scheme
from spinlabs.gui.state import GuiState, TabState, load_state, save_state
from spinlabs.gui.tabs import TabView
from spinlabs.gui.tooltips import Tooltip
from spinlabs.gui.widgets import ScrollableFrame
from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)
_REPORT_HEIGHT = 230


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        if value != value or value in (float("inf"), float("-inf")):
            return "—"
    except TypeError:
        return "—"
    return f"{value:.{digits}g}"


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


@dataclass
class ColumnState:
    scheme: Scheme
    date: str
    session: PackSession
    plot_label: str = ""

    @property
    def label(self) -> str:
        return self.scheme.label

    @property
    def path(self) -> Path:
        return self.session.path


@dataclass
class ReportData:
    columns: list[tuple[str, str, int]]  # key, heading, width
    rows: list[dict[str, str]]
    details: str

    @property
    def headings(self) -> list[str]:
        return [heading for _key, heading, _width in self.columns]

    def row_values(self, row: dict[str, str]) -> list[str]:
        return [row.get(key, "") for key, _heading, _width in self.columns]


class FigureHost(ttk.Frame):
    """A Matplotlib canvas that fills its page and rescales with the window."""

    def __init__(self, master):
        super().__init__(master, style="Card.TFrame")
        self.figure: Figure | None = None
        self._canvas_widget = None
        self._toolbar = None
        self._fit_after: str | None = None
        self.toolbar_frame = ttk.Frame(self, style="Card.TFrame")
        self.toolbar_frame.grid(row=0, column=0, sticky="ew")
        self.view = ttk.Frame(self, style="Card.TFrame")
        self.view.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.view.bind("<Configure>", lambda _e: self._schedule_fit())

    def _schedule_fit(self) -> None:
        if self._fit_after is not None:
            try:
                self.after_cancel(self._fit_after)
            except tk.TclError:
                pass
        self._fit_after = self.after(60, self._fit_figure)

    def target_figsize(self, dpi: float = 100.0) -> tuple[float, float] | None:
        """The figure size that fills this page now, or None while it is hidden."""
        if not self.view.winfo_exists():
            return None
        width = self.view.winfo_width()
        height = self.view.winfo_height()
        if width < 80 or height < 80:
            return None
        return (max(2.4, (width - 2) / dpi), max(2.0, (height - 2) / dpi))

    def _apply_figure_size(self, figure: Figure, *, forward: bool = True) -> bool:
        target = self.target_figsize(figure.dpi or 100.0)
        if target is None:
            return False
        dpi = figure.dpi or 100
        current = figure.get_size_inches()
        if abs(target[0] - current[0]) * dpi < 2 and abs(target[1] - current[1]) * dpi < 2:
            return False
        figure.set_size_inches(*target, forward=forward)
        return True

    def _fit_figure(self) -> None:
        self._fit_after = None
        if self.figure is None or self._canvas_widget is None:
            return
        if self._apply_figure_size(self.figure, forward=True):
            self._canvas_widget.draw_idle()

    def set_figure(self, figure: Figure) -> None:
        old = self.figure
        if self._canvas_widget is not None:
            self._canvas_widget.get_tk_widget().destroy()
            self._canvas_widget = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        self.figure = figure
        self._apply_figure_size(figure, forward=False)
        canvas = FigureCanvasTkAgg(figure, master=self.view)
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        self._canvas_widget = canvas
        self._toolbar = NavigationToolbar2Tk(canvas, self.toolbar_frame, pack_toolbar=False)
        self._toolbar.update()
        theme.style_navigation_toolbar(self._toolbar)
        self._toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        canvas.draw()
        self._schedule_fit()
        if old is not None and old is not figure:
            plt.close(old)


class PackTab(ttk.Frame):
    """One defect tab: columns (schemes) plus Summary/Mode detail views."""

    def __init__(self, master, app: "ConvergenceApp", group: Group, columns_spec: list[tuple[str, str]] | None):
        super().__init__(master, style="Card.TFrame")
        self.app = app
        self.group = group
        self.columns: list[ColumnState] = []
        self.mode_cache = app.mode_cache
        self.forward = True
        self.central = False
        self.display = "norm"
        self.phonon_preference = "auto"
        self.noise: float | None = None
        self.target = SNR_TARGET
        self.tol = TRUNC_TOL
        self.weak_fraction = WEAK_FRACTION
        self.drop_anchor = False
        self.deviation = False
        self.quad = False
        self.visible_modes: set[int] = set()
        self.active_mode: int | None = None
        self.active_column = 0
        self._mode_entries: dict[int, ModeEntry] = {}
        self._summary_columns = None
        self._mode_result = None
        self._last_render_keys: dict[int, tuple | None] = {0: None, 1: None}
        self._last_report_key: tuple | None = None

        self.nested = ttk.Notebook(self)
        # Ttk notebook pages must be children of the notebook, not of PackTab.
        self.summary_view = FigureHost(self.nested)
        self.mode_page = ttk.Frame(self.nested, style="Card.TFrame")
        mode_bar = ttk.Frame(self.mode_page, style="Card.TFrame")
        mode_bar.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(mode_bar, text="Column:", style="CardMuted.TLabel").pack(side=tk.LEFT)
        self.column_combo = ttk.Combobox(
            mode_bar, textvariable=app.active_column_var, state="readonly", width=24
        )
        self.column_combo.pack(side=tk.LEFT, padx=(4, 0))
        self.mode_view = FigureHost(self.mode_page)
        self.mode_view.pack(fill=tk.BOTH, expand=True)
        self.nested.add(self.summary_view, text="Summary")
        self.nested.add(self.mode_page, text="Mode detail")
        self.nested.pack(fill=tk.BOTH, expand=True)
        self.nested.bind("<<NotebookTabChanged>>", lambda _e: self.render_current())
        self.set_columns(columns_spec, initial=True)

    # -- columns and modes -------------------------------------------------

    def default_columns(self) -> list[tuple[str, str]]:
        return [(scheme.id, scheme.latest.date) for scheme in self.group.schemes]

    def set_columns(self, columns_spec: list[tuple[str, str]] | None, *, initial: bool = False) -> None:
        spec = list(dict(self.default_columns() if columns_spec is None else columns_spec).items())
        old_visible = set(self.visible_modes)
        old_modes = set(self._mode_entries)
        old_active = self.active_mode
        old_active_col = self.active_column
        old_sessions = {col.scheme.id: col.session for col in self.columns}
        columns: list[ColumnState] = []
        for scheme_id, date in spec:
            scheme = self.group.scheme(scheme_id)
            if scheme is None:
                continue
            item = scheme.file_for(date)
            session = old_sessions.get(scheme_id)
            if session is None or session.path != item.path:
                session = PackSession(
                    item.path, scheme.label, self.group.displacement, self.mode_cache,
                    phonon_preference=self.phonon_preference,
                )
            columns.append(
                ColumnState(
                    scheme, item.date, session,
                    f"{self.group.defect}_{self.group.cell} {scheme.label}".strip(),
                )
            )
        self.columns = columns
        entries: dict[int, ModeEntry] = {}
        for column in columns:
            for entry in column.session.modes():
                current = entries.get(entry.mode)
                if current is None or (current.symmetry is None and entry.symmetry is not None):
                    entries[entry.mode] = entry
        self._mode_entries = entries
        if initial or not old_modes:
            self.visible_modes = set(entries)
        else:
            # Keep the user's choices; newly discovered modes start visible.
            self.visible_modes = (old_visible & set(entries)) | (set(entries) - old_modes)
        if old_active in entries:
            self.active_mode = old_active
        else:
            self.active_mode = min(entries) if entries else None
        self.active_column = min(old_active_col, max(0, len(columns) - 1))
        self._last_render_keys = {0: None, 1: None}
        self._last_report_key = None
        self.sync_column_combo()
        self.render_current()

    def set_phonon_preference(self, preference: str) -> None:
        preference = preference if preference in ("auto", "aligned", "unaligned") else "auto"
        self.phonon_preference = preference
        spec = [(column.scheme.id, column.date) for column in self.columns]
        for column in self.columns:
            column.session.set_phonon_preference(preference)
        self.set_columns(spec, initial=False)

    def sync_column_combo(self) -> None:
        labels = [col.label for col in self.columns]
        self.column_combo["values"] = labels
        if labels:
            previous = self.app._syncing
            self.app._syncing = True
            try:
                self.app.active_column_var.set(labels[min(self.active_column, len(labels) - 1)])
            finally:
                self.app._syncing = previous

    @property
    def active_session(self) -> PackSession | None:
        if not self.columns:
            return None
        return self.columns[min(self.active_column, len(self.columns) - 1)].session

    @property
    def mode_entries(self) -> dict[int, ModeEntry]:
        return self._mode_entries

    def enabled_schemes(self) -> tuple[str, ...]:
        out = []
        if self.forward:
            out.append("forward")
        if self.central:
            out.append("central")
        return tuple(out)

    def set_visible(self, modes: set[int], visible: bool) -> None:
        if visible:
            self.visible_modes |= set(modes)
        else:
            self.visible_modes -= set(modes)
        self.render_current()

    def invert_visible(self, modes: set[int]) -> None:
        for mode in modes:
            if mode in self.visible_modes:
                self.visible_modes.discard(mode)
            else:
                self.visible_modes.add(mode)
        self.render_current()

    # -- analysis and rendering -------------------------------------------

    def _norm(self, column: ColumnState, central: bool) -> NormResult:
        try:
            return column.session.norm(
                self.visible_modes, noise=self.noise, target=self.target, tol=self.tol, central=central
            )
        except Exception as exc:  # one broken pack must not take down the tab
            LOGGER.exception("norm analysis failed for %s", column.path)
            return NormResult([], float("nan"), float("nan"), central, [str(exc)])

    def _column_note(self, column: ColumnState) -> str:
        availability = column.session.availability()
        if not availability.has_q0:
            return "no Q=0 anchor (no pert=0 run and no d0 key); Summary cannot measure differences"
        if not availability.has_minus:
            return "no -Q runs; Central differences are unavailable"
        return ""

    def _channels(self, column: ColumnState) -> ChannelAnalysis:
        try:
            return column.session.channels(
                self.visible_modes, schemes=self.enabled_schemes(), noise=self.noise, target=self.target,
                tol=self.tol, weak_fraction=self.weak_fraction,
            )
        except Exception as exc:
            LOGGER.exception("channel analysis failed for %s", column.path)
            return ChannelAnalysis(
                path=column.path, sigma=float("nan"), offset=float("nan"), modes=[], errors={}, common={},
                warnings=[str(exc)],
            )

    def _render_key(self):
        return (
            self.display, self.enabled_schemes(),
            tuple(sorted(self.visible_modes)), self.active_mode, self.active_column,
            self.noise, self.target, self.tol, self.weak_fraction,
            self.drop_anchor, self.deviation, self.quad,
            tuple((str(column.path), column.date) for column in self.columns),
        )

    def render_current(self) -> None:
        view = self.nested.index("current")
        key = self._render_key()
        if key != self._last_render_keys[view]:
            if view == 1:
                self.render_mode()
            else:
                self.render_summary()
            self._last_render_keys[view] = key
        report_key = (view, key)
        if report_key != self._last_report_key:
            self.update_report()
            self._last_report_key = report_key
        self.app.set_status(self.status_text())

    def status_text(self) -> str:
        if not self.columns:
            return "No columns selected"
        col = self.columns[min(self.active_column, len(self.columns) - 1)]
        bits = [self.group.title, f"{len(self.columns)} column(s)", col.label]
        if col.session.warnings():
            bits.append(f"{len(col.session.warnings())} warning(s)")
        return "  |  ".join(bits)

    def _summary_figure(self, *, canonical: bool = False) -> Figure:
        schemes = self.enabled_schemes()
        if not schemes:
            return plots.message_figure("Select Forward and/or Central to draw differences")
        if not self.visible_modes:
            return plots.message_figure("No modes selected for the Summary")
        if self.display == "channels":
            columns = [
                plots.ChannelColumnPlot(
                    col.plot_label, self._mode_entries, self._channels(col), self._column_note(col)
                )
                for col in self.columns
            ]
            self._summary_columns = columns
            return plots.build_channel_summary_figure(
                columns, schemes=schemes, figsize=None if canonical else self.summary_view.target_figsize(),
                canonical=canonical,
            )
        columns = [
            plots.NormColumnPlot(
                col.plot_label, self._mode_entries,
                self._norm(col, False) if "forward" in schemes else None,
                self._norm(col, True) if "central" in schemes else None,
                self._column_note(col),
            )
            for col in self.columns
        ]
        self._summary_columns = columns
        return plots.build_norm_summary_figure(
            columns, figsize=None if canonical else self.summary_view.target_figsize(), canonical=canonical
        )

    def render_summary(self) -> None:
        self.summary_view.set_figure(self._summary_figure())

    def _mode_figure(self, *, canonical: bool = False):
        session = self.active_session
        mode = self.active_mode
        if session is None or mode is None:
            return plots.message_figure("Select a pack column and a mode"), None
        try:
            result = plots.build_mode_figure(
                session, mode, quad=self.quad, deviation=self.deviation, drop_anchor=self.drop_anchor,
                noise=self.noise, snr_target=self.target, trunc_tol=self.tol, schemes=self.enabled_schemes(),
                figsize=None if canonical else self.mode_view.target_figsize(), canonical=canonical,
            )
        except Exception as exc:
            return plots.message_figure(str(exc)), None
        return result.figure, result

    def render_mode(self) -> None:
        figure, result = self._mode_figure()
        self._mode_result = result
        self.mode_view.set_figure(figure)

    def canonical_figure(self) -> Figure:
        """The figure as the CLI would write it, independent of the window size.

        The GUI applies its own muted Matplotlib style for the screen.  Exports
        are rebuilt under Matplotlib's default style so they match the figures
        written by ``spinlabs plot convergence``.
        """
        with matplotlib.rc_context(matplotlib.rcParamsDefault):
            if self.nested.index("current") == 1:
                figure, _result = self._mode_figure(canonical=True)
            else:
                figure = self._summary_figure(canonical=True)
            return figure

    # -- report ------------------------------------------------------------

    def _details(self) -> str:
        active_label = "—"
        if self.active_mode is not None:
            entry = self._mode_entries.get(self.active_mode)
            if entry is not None:
                active_label = entry.label
            else:
                active_label = str(self.active_mode)
        lines = [
            f"{self.group.title}  ({self.group.displacement})",
            f"tab: {len(self.columns)} column(s); view: {self.nested.tab(self.nested.select(), 'text')}",
            f"modes visible: {len(self.visible_modes)} / {len(self.mode_entries)}; "
            f"active mode: {active_label}",
        ]
        for column in self.columns:
            lines.append("")
            lines.append(f"column: {column.plot_label}  ({column.path})")
            phonon = column.session.phonon_path
            if phonon is not None:
                lines.append(f"  phonon: {phonon}")
            for warning in column.session.warnings():
                lines.append(f"  warning: {warning}")
        if self.nested.index("current") == 0:
            if self.display == "channels":
                for column in self.columns:
                    analysis = self._channels(column)
                    lines.append("")
                    lines.append(f"[{column.label}] coupling channels")
                    for warning in analysis.warnings:
                        lines.append(f"  warning: {warning}")
                    lines.extend("  " + line for line in plots.channel_report_lines(analysis))
            else:
                for column in self.columns:
                    if column.session.path != (self.active_session.path if self.active_session else None):
                        continue
                    forward = self._norm(column, False) if self.forward else None
                    central = self._norm(column, True) if self.central else None
                    for scheme, result in (("forward", forward), ("central", central)):
                        if result is None:
                            continue
                        lines.append("")
                        lines.append(f"[{column.label}] {scheme}: σ = {_fmt(result.sigma)} MHz, "
                                     f"D0 offset {_fmt(result.offset)} MHz")
                        for warning in result.warnings:
                            lines.append(f"  warning: {warning}")
                        row = next((r for r in result.rows if int(r[0]) == self.active_mode), None)
                        if row is not None:
                            lines.append(format_snr(row[4], sorted({abs(float(x)) for x in row[1] if x != 0})))
        else:
            result = self._mode_result
            if result is not None:
                lines.append("")
                lines.append(f"noise σ = {_fmt(result.noise)} MHz; D0 offset {_fmt(result.offset)} MHz")
                for warning in result.warnings:
                    lines.append(f"warning: {warning}")
                lines.extend(result.extra_lines)
                if result.report is not None and self.columns:
                    column = self.columns[min(self.active_column, len(self.columns) - 1)]
                    central = bool(self.central and not self.forward)
                    row = next((r for r in self._norm(column, central).rows
                                if int(r[0]) == self.active_mode), None)
                    if row is not None:
                        lines.append(format_snr(row[4], sorted({abs(float(x)) for x in row[1] if x != 0})))
        return "\n".join(lines)

    def report(self) -> ReportData:
        if self.nested.index("current") == 0 and self.display == "channels":
            return self._channel_report()
        if self.nested.index("current") == 1:
            return self._mode_report()
        return self._norm_report()

    def update_report(self) -> None:
        self.app.set_report(self.report())

    def _norm_report(self) -> ReportData:
        columns = [
            ("column", "Column", 90), ("scheme", "Scheme", 70), ("mode", "Mode", 55),
            ("sym", "Sym", 45), ("freq", "freq/meV", 65), ("b", "b", 70), ("c", "2c", 70),
            ("lo1", "q_lo1", 65), ("hi1", "q_hi1", 65), ("lo2", "q_lo2", 65), ("hi2", "q_hi2", 65),
            ("flags", "Flags", 120),
        ]
        rows: list[dict[str, str]] = []
        for column in self.columns:
            for scheme, enabled, central in (("forward", self.forward, False), ("central", self.central, True)):
                if not enabled:
                    continue
                result = self._norm(column, central)
                for mode, _p, _first, _second, report in result.rows:
                    entry = self._mode_entries.get(int(mode), ModeEntry(int(mode)))
                    flags = []
                    if not report.b_resolved:
                        flags.append("b~0")
                    if not report.c_resolved:
                        flags.append("c~0")
                    if not report.next1_resolved:
                        flags.append("1st upper bound")
                    if not report.next2_resolved:
                        flags.append("2nd upper bound")
                    rows.append({
                        "column": column.plot_label, "scheme": scheme, "mode": str(int(mode)),
                        "sym": entry.symmetry or "", "freq": _fmt(entry.frequency, 3), "b": _fmt(report.b),
                        "c": _fmt(2 * report.c), "lo1": _fmt(report.q_lo1), "hi1": _fmt(report.q_hi1),
                        "lo2": _fmt(report.q_lo2), "hi2": _fmt(report.q_hi2), "flags": ", ".join(flags),
                    })
        return ReportData(columns, rows, self._details())

    def _channel_report(self) -> ReportData:
        columns = [
            ("column", "Column", 90), ("mode", "Mode", 55), ("sym", "Sym", 45), ("freq", "freq/meV", 65),
            ("order", "Order", 55), ("channel", "Channel", 75), ("scheme", "Scheme", 65), ("q0", "Q→0", 75),
            ("lo", "q_lo", 65), ("hi", "q_hi", 65), ("flags", "Flags", 140),
        ]
        rows: list[dict[str, str]] = []
        for column in self.columns:
            analysis = self._channels(column)
            for mode in analysis.modes:
                entry = self._mode_entries.get(mode.mode, ModeEntry(mode.mode, mode.symmetry, mode.frequency))
                for order in (1, 2):
                    for channel in CHANNELS:
                        for scheme in self.enabled_schemes():
                            series = mode.get(order, channel, scheme)
                            if series is None:
                                continue
                            window = series.window
                            flags = []
                            if not window.allowed:
                                flags.append("forbidden")
                            if not window.resolved:
                                flags.append("unresolved")
                            if window.weak:
                                flags.append("weak")
                            if not window.upper_known:
                                flags.append("upper bound")
                            rows.append({
                                "column": column.plot_label, "mode": str(mode.mode), "sym": entry.symmetry or "",
                                "freq": _fmt(entry.frequency, 3),
                                "order": "first" if order == 1 else "second", "channel": CHANNEL_NAMES[channel],
                                "scheme": scheme, "q0": _fmt(series.q0), "lo": _fmt(window.q_lo),
                                "hi": _fmt(window.q_hi), "flags": ", ".join(flags),
                            })
        return ReportData(columns, rows, self._details())

    def _mode_report(self) -> ReportData:
        columns = [
            ("mode", "Mode", 50), ("sym", "Sym", 45), ("freq", "freq/meV", 65),
            ("b", "slope b", 75), ("c", "2c", 75), ("lo", "safe q_lo", 75), ("hi", "safe q_hi", 75),
            ("flags", "Flags", 140),
        ]
        rows: list[dict[str, str]] = []
        session = self.active_session
        if session is not None and self.active_mode is not None:
            entry = self._mode_entries.get(self.active_mode, ModeEntry(self.active_mode))
            central = bool(self.central and not self.forward)
            result = self._norm(self.columns[min(self.active_column, len(self.columns) - 1)], central)
            row = next((r for r in result.rows if int(r[0]) == self.active_mode), None)
            if row is not None:
                report = row[4]
                rows.append({
                    "mode": str(self.active_mode), "sym": entry.symmetry or "", "freq": _fmt(entry.frequency, 3),
                    "b": _fmt(report.b), "c": _fmt(2 * report.c), "lo": _fmt(report.q_lo1),
                    "hi": _fmt(report.q_hi1),
                    "flags": ", ".join(
                        flag for flag, on in (("b~0", not report.b_resolved), ("c~0", not report.c_resolved))
                        if on
                    ),
                })
        return ReportData(columns, rows, self._details())

    def save_stem(self) -> str:
        view = "mode%d" % self.active_mode if self.nested.index("current") == 1 else (
            "channels" if self.display == "channels" else "convergence"
        )
        return _safe_name(f"{self.group.title}_{view}")

    def visible_figure(self) -> Figure | None:
        return self.mode_view.figure if self.nested.index("current") == 1 else self.summary_view.figure

    def destroy(self):
        for host in (self.summary_view, self.mode_view):
            if host.figure is not None:
                plt.close(host.figure)
                host.figure = None
        super().destroy()

    def state(self) -> TabState:
        return TabState(
            group_key=self.group.key,
            columns=[{"scheme": col.scheme.id, "date": col.date} for col in self.columns],
            visible_modes=sorted(self.visible_modes),
            active_mode=self.active_mode,
            active_column=self.active_column,
            forward=self.forward,
            central=self.central,
            display=self.display,
            phonon=self.phonon_preference,
            noise=self.noise,
            target=self.target,
            tol=self.tol,
            weak_fraction=self.weak_fraction,
            drop_anchor=self.drop_anchor,
            deviation=self.deviation,
            quad=self.quad,
        )


class ConvergenceApp:
    """Main window and the registry/control/report wiring."""

    def __init__(self, root: tk.Tk, paths: list[Path] | None = None):
        self.root = root
        self.root.title("SpinLabs · Convergence viewer")
        self.state = load_state()
        try:
            self.root.geometry(self.state.geometry)
        except tk.TclError:
            self.root.geometry("1500x950")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.mode_cache: dict = {}
        self.catalog = Catalog()
        self.tabs: dict[str, PackTab] = {}
        self._tab_by_tabid: dict[str, PackTab] = {}
        self._scheme_vars: dict[str, tuple[tk.BooleanVar, tk.StringVar]] = {}
        self._syncing = False
        self._restoring = False
        self._closed = False
        self._tooltips: list[Tooltip] = []
        self._group_by_row: dict[str, Group] = {}

        self._style()
        self._build_vars()
        self._build_menu()
        self._build_header()
        self._build_toolbar()
        self._build_statusbar()
        self._build_layout()
        self._build_registry_tab()
        self._build_view_tab()
        self._build_report()
        self.refresh_registry(initial=True)
        self._restore_or_open(paths)

    # -- construction ------------------------------------------------------

    def _style(self) -> None:
        self.palette = theme.apply_theme(self.root)
        self.family = theme.font_family(self.root)
        theme.style_matplotlib()

    def _tip(self, widget, text: str) -> None:
        self._tooltips.append(Tooltip(widget, text, palette=self.palette, family=self.family))

    def _build_header(self) -> None:
        p = self.palette
        header = tk.Frame(self.root, background=p.header_bg, height=78)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        inner = tk.Frame(header, background=p.header_bg)
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)
        left = tk.Frame(inner, background=p.header_bg)
        left.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            left, text="SpinLabs", background=p.header_bg, foreground=p.header_text,
            font=(self.family, 18, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            left, text="Convergence viewer  ·  norm and V\u0394n coupling channels",
            background=p.header_bg, foreground=p.header_muted, font=(self.family, 9),
        ).pack(anchor=tk.W, pady=(2, 0))
        right = tk.Frame(inner, background=p.header_bg)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(
            right, text="REGISTRY-DRIVEN", background=p.header_bg, foreground=p.header_text,
            font=(self.family, 8, "bold"),
        ).pack(anchor=tk.E)
        tk.Label(
            right, text="TK + MATPLOTLIB", background=p.header_bg, foreground=p.header_muted,
            font=(self.family, 8),
        ).pack(anchor=tk.E, pady=(2, 0))

    def _build_vars(self) -> None:
        self.filter_defect = tk.StringVar(value="All")
        self.filter_cell = tk.StringVar(value="All")
        self.filter_method = tk.StringVar(value="All")
        self.filter_displacement = tk.StringVar(value="All")
        self.latest_only = tk.BooleanVar(value=True)
        self.forward_var = tk.BooleanVar(value=True)
        self.central_var = tk.BooleanVar(value=False)
        self.display_var = tk.StringVar(value="Norm")
        self.active_column_var = tk.StringVar(value="")
        self.phonon_var = tk.StringVar(value="Auto")
        self.mode_filter = tk.StringVar(value="")
        self.noise_auto = tk.BooleanVar(value=True)
        self.noise_var = tk.StringVar(value="")
        self.target_var = tk.StringVar(value=f"{SNR_TARGET:g}")
        self.tol_var = tk.StringVar(value=f"{TRUNC_TOL:g}")
        self.weak_var = tk.StringVar(value=f"{WEAK_FRACTION:g}")
        self.drop_anchor_var = tk.BooleanVar(value=False)
        self.deviation_var = tk.BooleanVar(value=False)
        self.quad_var = tk.BooleanVar(value=False)
        self.report_visible = tk.BooleanVar(value=self.state.report_visible)
        self.status_var = tk.StringVar(value="Ready")
        for var in (self.forward_var, self.central_var, self.display_var, self.active_column_var):
            var.trace_add("write", self._on_view_control)
        self.mode_filter.trace_add("write", lambda *_: self._populate_mode_tree())
        self.noise_auto.trace_add("write", lambda *_: self._update_noise_sensitivity())

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open file(s)…", command=self.open_files)
        file_menu.add_command(label="Open decont folder…", command=self.open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Refresh registry", command=self.refresh_registry)
        file_menu.add_separator()
        save_menu = tk.Menu(file_menu, tearoff=False)
        for fmt, label in (("png", "PNG"), ("svg", "SVG"), ("pdf", "PDF")):
            save_menu.add_command(label=f"{label}…", command=lambda f=fmt: self.save_figure(f))
        file_menu.add_cascade(label="Save figure", menu=save_menu)
        file_menu.add_command(label="Export table CSV…", command=self.export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Close tab", command=self.close_tab, accelerator="Ctrl+W")
        file_menu.add_command(label="Close other tabs", command=self.close_other_tabs)
        file_menu.add_command(label="Close all tabs", command=self.close_all_tabs)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit, accelerator="Ctrl+Q")
        menu.add_cascade(label="File", menu=file_menu)
        view_menu = tk.Menu(menu, tearoff=False)
        view_menu.add_checkbutton(label="Report pane", variable=self.report_visible, command=self._toggle_report)
        view_menu.add_command(label="Summary", command=lambda: self._select_view(0))
        view_menu.add_command(label="Mode detail", command=lambda: self._select_view(1))
        menu.add_cascade(label="View", menu=view_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menu)
        self.root.bind("<Control-w>", self._close_shortcut)
        self.root.bind("<Control-W>", self._close_shortcut)
        self.root.bind("<Control-q>", self._quit_shortcut)

    def _typing_has_focus(self) -> bool:
        focus = self.root.focus_get()
        return isinstance(focus, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox))

    def _close_shortcut(self, _event=None):
        if not self._typing_has_focus():
            self.close_tab()
        return "break"

    def _quit_shortcut(self, _event=None):
        if not self._typing_has_focus():
            self.quit()
        return "break"

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(14, 6))
        bar.pack(side=tk.TOP, fill=tk.X)
        open_button = ttk.Button(bar, text="Open…", command=self.open_files, style="Accent.TButton")
        open_button.pack(side=tk.LEFT, padx=(0, 6))
        self._tip(open_button, "Open an unregistered convergence file or decont folder.")
        refresh_button = ttk.Button(bar, text="Refresh", command=self.refresh_registry)
        refresh_button.pack(side=tk.LEFT, padx=(0, 2))
        self._tip(refresh_button, "Re-read the registry and refresh the defect groups.")
        save_button = ttk.Menubutton(bar, text="Save figure")
        save_menu = tk.Menu(save_button, tearoff=False)
        for fmt, label in (("png", "PNG"), ("svg", "SVG"), ("pdf", "PDF")):
            save_menu.add_command(label=f"{label}…", command=lambda f=fmt: self.save_figure(f))
        save_button.configure(menu=save_menu)
        self._save_menu = save_menu
        save_button.pack(side=tk.LEFT, padx=(0, 2))
        self._tip(save_button, "Save the current view as a CLI-style PNG, SVG or PDF figure.")
        csv_button = ttk.Button(bar, text="Export CSV", command=self.export_csv)
        csv_button.pack(side=tk.LEFT, padx=(0, 2))
        self._tip(csv_button, "Export the current report table as a CSV file.")
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        close_button = ttk.Button(bar, text="Close tab", command=self.close_tab)
        close_button.pack(side=tk.LEFT)
        self._tip(close_button, "Close the active pack tab (Ctrl+W).")
        ttk.Label(
            bar, text="Convergence data, fits and spin–phonon channels", style="CardMuted.TLabel",
        ).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X)

    def _build_statusbar(self) -> None:
        self.statusbar = ttk.Frame(self.root, style="Status.TFrame", padding=(14, 4))
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(
            self.statusbar, textvariable=self.status_var, style="Status.TLabel", anchor=tk.W,
        ).pack(fill=tk.X)

    def _build_layout(self) -> None:
        p = self.palette
        self.paned = tk.PanedWindow(
            self.root, orient=tk.VERTICAL, sashwidth=8, sashrelief=tk.FLAT, background=p.background,
            borderwidth=0,
        )
        self.paned.pack(fill=tk.BOTH, expand=True)
        self.main_paned = tk.PanedWindow(
            self.paned, orient=tk.HORIZONTAL, sashwidth=8, sashrelief=tk.FLAT, background=p.background,
            borderwidth=0,
        )
        self.paned.add(self.main_paned, stretch="always")
        self.sidebar = ttk.Frame(self.main_paned, width=360, style="Card.TFrame")
        self.main_paned.add(self.sidebar, minsize=300, width=360)
        self.content = ttk.Frame(self.main_paned, style="Card.TFrame")
        self.main_paned.add(self.content, stretch="always")
        self.side_notebook = ttk.Notebook(self.sidebar)
        self.side_notebook.pack(fill=tk.BOTH, expand=True, padx=(8, 4), pady=(4, 4))
        self.registry_tab = ttk.Frame(self.side_notebook, style="Card.TFrame", padding=(8, 8))
        self.view_tab = ttk.Frame(self.side_notebook, style="Card.TFrame", padding=0)
        self.side_notebook.add(self.registry_tab, text="Registry")
        self.side_notebook.add(self.view_tab, text="View")
        self.report_frame = ttk.Frame(self.paned, style="Card.TFrame")
        self.paned.add(self.report_frame, height=_REPORT_HEIGHT, stretch="never")
        self.welcome = ttk.Frame(self.content, style="Card.TFrame")
        self.welcome_label = ttk.Label(
            self.welcome,
            text="No pack open.\nPick a defect group in the Registry tab, or use File → Open.\n"
                 "Register data with `spinlabs package --conv` to fill the registry.",
            justify=tk.CENTER, style="CardMuted.TLabel", font=(self.family, 10),
        )
        self.welcome_label.pack(expand=True)
        self.notebook = TabView(
            self.content, palette=self.palette, family=self.family,
            on_change=self._on_tab_changed, on_close=self.close_tab,
        )
        self.welcome.pack(fill=tk.BOTH, expand=True)

    def _build_registry_tab(self) -> None:
        pad = {"padx": 4, "pady": 3}
        box = ttk.LabelFrame(self.registry_tab, text="Filters")
        box.pack(fill=tk.X, padx=6, pady=4)
        self._filter_row(box, "Defect", self.filter_defect, 0)
        self._filter_row(box, "Cell", self.filter_cell, 1)
        self._filter_row(box, "Method", self.filter_method, 2)
        self._filter_row(box, "Displacement", self.filter_displacement, 3)
        latest = ttk.Checkbutton(box, text="Latest only", variable=self.latest_only,
                                 command=self.refresh_registry)
        latest.grid(row=4, column=0, columnspan=2, sticky="w", **pad)
        self._tip(latest, "Show only the newest registered file of each convergence scheme.")
        refresh = ttk.Button(box, text="Refresh", command=self.refresh_registry)
        refresh.grid(row=5, column=0, columnspan=2, sticky="ew", **pad)
        self._tip(refresh, "Re-read the registry database and refresh the defect groups.")
        self.group_tree = ttk.Treeview(
            self.registry_tab, columns=("defect", "cell", "displacement", "schemes"), show="headings", height=6,
            selectmode="browse",
        )
        for key, title, width in (("defect", "Defect", 80), ("cell", "Cell", 50),
                                  ("displacement", "Displacement", 100), ("schemes", "Schemes", 70)):
            self.group_tree.heading(key, text=title)
            self.group_tree.column(key, width=width, anchor=tk.W)
        self.group_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self._tip(self.group_tree, "Select a defect group; its available ZFS schemes appear below.")
        self.group_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_group_schemes())
        self.group_tree.bind("<Double-1>", lambda _e: self.open_selected_group())
        self.schemes_frame = ttk.LabelFrame(self.registry_tab, text="Schemes")
        self.schemes_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        open_update = ttk.Button(
            self.registry_tab, text="Open / update tab", command=self.open_selected_group,
            style="Accent.TButton",
        )
        open_update.pack(fill=tk.X, padx=0, pady=(6, 6))
        self._tip(open_update, "Open the checked schemes as a new defect tab, or update its existing columns.")
        buttons = ttk.Frame(self.registry_tab)
        buttons.pack(fill=tk.X, padx=6, pady=(0, 6))
        open_file = ttk.Button(buttons, text="Open file…", command=self.open_files)
        open_file.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._tip(open_file, "Open a convergence .npz that is not in the registry (session only).")
        open_folder = ttk.Button(buttons, text="Open folder…", command=self.open_folder)
        open_folder.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        self._tip(open_folder, "Open a full-decontamination convergence folder (session only).")

    def _filter_row(self, parent, label: str, var: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        combo = ttk.Combobox(parent, textvariable=var, state="readonly", width=16)
        combo.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_registry())
        filter_tips = {
            "Defect": "Filter registered defect groups by defect name.",
            "Cell": "Filter registered defect groups by supercell size.",
            "Method": "Filter by ZFS calculation method (AO, UO or full decont).",
            "Displacement": "Filter by the perturbation displacement convention.",
        }
        self._tip(combo, filter_tips.get(label, f"Filter by {label.lower()}."))
        setattr(self, f"_combo_{label.lower()}", combo)
        parent.columnconfigure(1, weight=1)

    def _build_view_tab(self) -> None:
        self.view_scroll = ScrollableFrame(self.view_tab, palette=self.palette)
        self.view_scroll.pack(fill=tk.BOTH, expand=True)
        content = self.view_scroll.content

        box = ttk.LabelFrame(content, text="Difference scheme")
        box.pack(fill=tk.X, padx=6, pady=4)
        forward = ttk.Checkbutton(box, text="Forward", variable=self.forward_var)
        forward.pack(anchor=tk.W, padx=6, pady=2)
        self._tip(forward, "Forward differences: (D(Q) − D0)/Q and (D(2Q) − 2D(Q) + D0)/Q². This is the production scheme.")
        self.central_check = ttk.Checkbutton(box, text="Central (−Q runs)", variable=self.central_var)
        self.central_check.pack(anchor=tk.W, padx=6, pady=2)
        self._tip(self.central_check, "Central differences from +Q and −Q runs. Needs −Q data; disabled when none is selected.")

        quality = ttk.LabelFrame(content, text="Summary quantity")
        quality.pack(fill=tk.X, padx=6, pady=4)
        self.display_combo = ttk.Combobox(
            quality, textvariable=self.display_var, values=("Norm", "Coupling channels"), state="readonly"
        )
        self.display_combo.pack(fill=tk.X, padx=6, pady=4)
        self._tip(self.display_combo, "Norm shows the usual norm summary. Coupling channels shows V_Δ0, V_Δ±1 and V_Δ±2 separately for every mode.")

        columns = ttk.LabelFrame(content, text="Columns and dates")
        columns.pack(fill=tk.X, padx=6, pady=4)
        self.columns_frame = ttk.Frame(columns, style="Card.TFrame")
        self.columns_frame.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(columns, text="Active column (Mode detail)", style="Card.TLabel").pack(
            anchor=tk.W, padx=6, pady=(4, 0)
        )
        self._column_combo = ttk.Combobox(columns, textvariable=self.active_column_var, state="readonly")
        self._column_combo.pack(fill=tk.X, padx=6, pady=4)
        self._tip(self._column_combo, "Choose which scheme column Mode detail and the Details report use.")
        ttk.Label(
            columns, text="Change columns in the Registry tab: select the group, then Open / update.",
            wraplength=320, style="CardMuted.TLabel",
        ).pack(anchor=tk.W, padx=6, pady=(0, 6))

        self._build_modes_section(content)

        self.advanced_toggle = ttk.Button(
            content, text="Advanced settings  >", command=self._toggle_advanced
        )
        self.advanced_toggle.pack(fill=tk.X, padx=6, pady=(12, 4))
        self._tip(self.advanced_toggle, "Open the analysis parameters, Mode-detail flags and phonon-spectrum choice.")
        self.advanced_frame = ttk.Frame(content, style="Card.TFrame")
        self._build_advanced_section(self.advanced_frame)
        self._advanced_visible = False

    def _toggle_advanced(self) -> None:
        if self._advanced_visible:
            self.advanced_frame.pack_forget()
            self.advanced_toggle.configure(text="Advanced settings  >")
        else:
            self.advanced_frame.pack(fill=tk.X, pady=(0, 6))
            self.advanced_toggle.configure(text="Advanced settings  v")
        self._advanced_visible = not self._advanced_visible
        self.view_scroll.refresh()

    def _checkbox_image(self, checked: bool) -> tk.PhotoImage:
        """A small flat checkbox that does not depend on the UI font's glyph coverage."""
        image = tk.PhotoImage(master=self.root, width=16, height=16)
        fill = "#e8f1fb" if checked else "#ffffff"
        border = "#005fb8" if checked else "#8a94a6"
        image.put(fill, to=(0, 0, 16, 16))
        for x in range(2, 14):
            image.put(border, to=(x, 2))
            image.put(border, to=(x, 13))
        for y in range(2, 14):
            image.put(border, to=(2, y))
            image.put(border, to=(13, y))
        if checked:
            for x, y in ((4, 8), (5, 9), (6, 10), (7, 9), (8, 8), (9, 7), (10, 6), (11, 5)):
                image.put("#005fb8", to=(x, y))
        return image

    def _build_modes_section(self, parent) -> None:
        section = ttk.LabelFrame(parent, text="Modes")
        section.pack(fill=tk.X, padx=6, pady=(12, 4))
        filter_box = ttk.Frame(section, style="Card.TFrame")
        filter_box.pack(fill=tk.X, padx=4, pady=(2, 6))
        ttk.Label(filter_box, text="Filter", style="CardMuted.TLabel").pack(side=tk.LEFT)
        mode_filter = ttk.Entry(filter_box, textvariable=self.mode_filter)
        mode_filter.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._tip(mode_filter, "Filter the mode list by mode number, symmetry or frequency.")
        self._mode_icons = (self._checkbox_image(False), self._checkbox_image(True))
        self.mode_tree = ttk.Treeview(
            section, columns=("mode", "sym", "freq"), show=("tree", "headings"), height=10,
            selectmode="browse",
        )
        self.mode_tree.heading("#0", text="Show")
        self.mode_tree.column("#0", width=58, anchor=tk.W)
        for key, title, width in (("mode", "Mode", 55), ("sym", "Sym", 45), ("freq", "freq/meV", 75)):
            self.mode_tree.heading(key, text=title)
            self.mode_tree.column(key, width=width, anchor=tk.W)
        self.mode_tree.pack(fill=tk.X, padx=4, pady=4)
        self._tip(self.mode_tree, "Click the checkbox to include a mode in Summary; click another cell to open that mode in Mode detail.")
        self.mode_tree.bind("<ButtonRelease-1>", self._on_mode_click)
        buttons = ttk.Frame(section, style="Card.TFrame")
        buttons.pack(fill=tk.X, padx=4, pady=(0, 4))
        all_button = ttk.Button(buttons, text="All", command=lambda: self._mode_bulk(True))
        all_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._tip(all_button, "Include every mode currently shown by the filter in Summary.")
        none_button = ttk.Button(buttons, text="None", command=lambda: self._mode_bulk(False))
        none_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        self._tip(none_button, "Exclude every mode currently shown by the filter from Summary.")
        invert_button = ttk.Button(buttons, text="Invert", command=self._mode_invert)
        invert_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._tip(invert_button, "Reverse the Summary visibility of the filtered modes.")
        ttk.Label(
            section, text="Click a row for Mode detail; click Show to toggle Summary visibility.",
            wraplength=300, style="CardMuted.TLabel",
        ).pack(anchor=tk.W, padx=6, pady=(0, 6))

    def _build_advanced_section(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Analysis parameters")
        box.pack(fill=tk.X, padx=6, pady=4)
        rows = [
            ("Noise σ / MHz", self.noise_var, "auto"),
            ("SNR target", self.target_var, None),
            ("Truncation tol.", self.tol_var, None),
            ("Weak fraction", self.weak_var, None),
        ]
        tips = {
            "Noise σ / MHz": "Tensor-element noise used for the finite-difference error bars and safe windows.",
            "SNR target": "Required signal-to-noise ratio at the lower end of the safe |Q| window.",
            "Truncation tol.": "Largest allowed relative truncation error at the upper end of the safe |Q| window.",
            "Weak fraction": "Channels or modes below this fraction of the strongest coefficient do not narrow the common safe |Q|.",
        }
        for row, (label, var, tag) in enumerate(rows):
            ttk.Label(box, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", padx=4, pady=2)
            entry = ttk.Entry(box, textvariable=var, width=10)
            entry.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
            self._tip(entry, tips[label])
            if tag == "auto":
                auto = ttk.Checkbutton(box, text="auto", variable=self.noise_auto)
                auto.grid(row=row, column=2, sticky="w", padx=4, pady=2)
                self._tip(auto, "Estimate σ from the pooled convergence residuals; untick to enter a fixed value.")
                self.noise_entry = entry
        box.columnconfigure(1, weight=1)
        flags = ttk.LabelFrame(parent, text="Mode detail")
        flags.pack(fill=tk.X, padx=6, pady=4)
        drop = ttk.Checkbutton(flags, text="Drop Q=0 anchor from fits", variable=self.drop_anchor_var)
        drop.pack(anchor=tk.W, padx=6, pady=2)
        self._tip(drop, "Leave the Q=0 point out of the Mode-detail fits; the fitted curves still extrapolate to Q=0.")
        deviation = ttk.Checkbutton(flags, text="Plot D(Q) − D(0)", variable=self.deviation_var)
        deviation.pack(anchor=tk.W, padx=6, pady=2)
        self._tip(deviation, "Plot the change from the relaxed tensor so the response fills the y-range.")
        quad = ttk.Checkbutton(flags, text="Quadratic fit", variable=self.quad_var)
        quad.pack(anchor=tk.W, padx=6, pady=2)
        self._tip(quad, "Add the second-order fit values and central d²D/dQ² estimates to Details.")
        phonon = ttk.LabelFrame(parent, text="Phonon symmetry")
        phonon.pack(fill=tk.X, padx=0, pady=(8, 0))
        ttk.Label(
            phonon, text="Full spectrum for labels and frequencies", style="CardMuted.TLabel",
        ).pack(anchor=tk.W, padx=4)
        phonon_combo = ttk.Combobox(
            phonon, textvariable=self.phonon_var, state="readonly",
            values=("Auto", "Aligned full", "Unaligned full"),
        )
        phonon_combo.pack(fill=tk.X, padx=4, pady=(4, 4))
        self._tip(phonon_combo, "Choose the full phonon spectrum used for symmetry labels and frequencies. Auto follows the pack's aligned tag and packaged phonon files.")
        apply_button = ttk.Button(
            parent, text="Apply", command=self.apply_advanced, style="Accent.TButton"
        )
        apply_button.pack(fill=tk.X, pady=(10, 6))
        self._tip(apply_button, "Apply the advanced settings and the selected phonon spectrum.")
        ttk.Label(
            parent, text="Defaults match `spinlabs plot convergence`.",
            wraplength=320, style="CardMuted.TLabel",
        ).pack(anchor=tk.W, padx=6, pady=(0, 6))
        self._update_noise_sensitivity()

    def _build_report(self) -> None:
        self.report_notebook = ttk.Notebook(self.report_frame)
        self.report_notebook.pack(fill=tk.BOTH, expand=True)
        table_tab = ttk.Frame(self.report_notebook, style="Card.TFrame", padding=(8, 8))
        details_tab = ttk.Frame(self.report_notebook, style="Card.TFrame", padding=(8, 8))
        self.report_notebook.add(table_tab, text="Table")
        self.report_notebook.add(details_tab, text="Details")
        self.report_tree = ttk.Treeview(table_tab, show="headings", height=5)
        self.report_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(table_tab, orient=tk.VERTICAL, command=self.report_tree.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        xscroll = ttk.Scrollbar(table_tab, orient=tk.HORIZONTAL, command=self.report_tree.xview)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.report_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.details_text = tk.Text(
            details_tab, wrap="word", height=8, background="#ffffff", foreground=self.palette.text,
            relief=tk.FLAT, borderwidth=0, highlightthickness=0, padx=10, pady=8,
            insertbackground=self.palette.text, font=("DejaVu Sans Mono", 9),
        )
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dscroll = ttk.Scrollbar(details_tab, orient=tk.VERTICAL, command=self.details_text.yview)
        dscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.details_text.configure(yscrollcommand=dscroll.set, state=tk.DISABLED)
        if not self.report_visible.get():
            self.paned.forget(self.report_frame)

    # -- registry ----------------------------------------------------------

    def _set_choices(self, combo: ttk.Combobox, values: list[str], var: tk.StringVar) -> None:
        current = var.get()
        combo["values"] = ["All", *values]
        if current not in combo["values"]:
            var.set("All")

    def refresh_registry(self, *, initial: bool = False) -> None:
        selected = self.selected_group_key()
        self.catalog = Catalog()
        self._set_choices(self._combo_defect, self.catalog.defects(), self.filter_defect)
        self._set_choices(self._combo_cell, self.catalog.cells(), self.filter_cell)
        self._set_choices(self._combo_method, self.catalog.methods(), self.filter_method)
        self._set_choices(self._combo_displacement, self.catalog.displacements(), self.filter_displacement)
        groups = self._filtered_groups()
        self.group_tree.delete(*self.group_tree.get_children())
        self._group_by_row = {}
        for group in groups:
            iid = group.key
            self._group_by_row[iid] = group
            self.group_tree.insert(
                "", tk.END, iid=iid,
                values=(group.defect, group.cell, group.displacement, len(group.schemes)),
            )
        if selected and selected in self._group_by_row:
            self.group_tree.selection_set(selected)
            self.group_tree.see(selected)
        elif groups:
            first = self.group_tree.get_children()[0]
            self.group_tree.selection_set(first)
            self.group_tree.see(first)
        self._show_group_schemes()
        if initial:
            self._sync_filters_from_state()

    def _filtered_groups(self) -> list[Group]:
        def value(var: tk.StringVar) -> str | None:
            return None if var.get() in ("", "All") else var.get()

        return self.catalog.groups(
            defect=value(self.filter_defect), cell=value(self.filter_cell), method=value(self.filter_method),
            displacement=value(self.filter_displacement), latest_only=self.latest_only.get(),
        )

    def _sync_filters_from_state(self) -> None:
        filters = self.state.filters or {}
        for var, key in (
            (self.filter_defect, "defect"), (self.filter_cell, "cell"),
            (self.filter_method, "method"), (self.filter_displacement, "displacement"),
        ):
            value = filters.get(key)
            if value and value in getattr(self, f"_combo_{key}")["values"]:
                var.set(value)
        if "latest_only" in filters:
            self.latest_only.set(bool(filters["latest_only"]))
        self.refresh_registry()

    def selected_group_key(self) -> str | None:
        selection = self.group_tree.selection() if hasattr(self, "group_tree") else ()
        return selection[0] if selection else None

    def _selected_group(self) -> Group | None:
        key = self.selected_group_key()
        return self._group_by_row.get(key) if key else None

    def _show_group_schemes(self) -> None:
        for child in self.schemes_frame.winfo_children():
            child.destroy()
        group = self._selected_group()
        self._scheme_vars = {}
        if group is None:
            ttk.Label(self.schemes_frame, text="Select a defect group", style="CardMuted.TLabel").pack(
                anchor=tk.W, padx=6, pady=6
            )
            return
        # The tree filter may be "latest only"; the tab still offers every date.
        full = self.catalog.group(group.key, latest_only=False) or group
        open_tab = self.tabs.get(group.key)
        open_columns = {c.scheme.id: c.date for c in open_tab.columns} if open_tab else {}
        for row, scheme in enumerate(full.schemes):
            check = tk.BooleanVar(value=(scheme.id in open_columns) if open_columns else True)
            date = tk.StringVar(value=open_columns.get(scheme.id, scheme.latest.date))
            label = scheme.label + ("  ⚠ no Q=0" if scheme.has_q0 is False else "")
            scheme_check = ttk.Checkbutton(
                self.schemes_frame, text=label, variable=check, command=lambda g=group: self._scheme_changed(g)
            )
            scheme_check.grid(row=row, column=0, sticky="w", padx=4, pady=2)
            self._tip(scheme_check, "Include this convergence scheme as a Summary column in the tab.")
            combo = ttk.Combobox(
                self.schemes_frame, textvariable=date, state="readonly", width=12,
                values=[item.date for item in scheme.dates],
            )
            combo.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
            self._tip(combo, "Choose which registered file/date of this scheme the column uses.")
            combo.bind("<<ComboboxSelected>>", lambda _e, g=group: self._scheme_changed(g))
            self.schemes_frame.columnconfigure(1, weight=1)
            self._scheme_vars[scheme.id] = (check, date)

    def _scheme_changed(self, group: Group) -> None:
        """Live-update an already open tab when its scheme checkboxes/dates change."""
        if group.key not in self.tabs:
            return
        spec = [
            (scheme_id, date.get())
            for scheme_id, (check, date) in self._scheme_vars.items()
            if check.get()
        ]
        if spec:
            self.open_group(group, spec)

    def open_selected_group(self) -> None:
        group = self._selected_group()
        if group is None:
            messagebox.showinfo("No group", "Select a defect group first.")
            return
        spec = [
            (scheme_id, date.get())
            for scheme_id, (check, date) in self._scheme_vars.items()
            if check.get()
        ]
        if not spec:
            messagebox.showinfo("No schemes", "Select at least one ZFS scheme.")
            return
        self.open_group(group, spec)

    def open_group(self, group: Group, columns_spec: list[tuple[str, str]] | None = None) -> None:
        # The checked schemes define the opening columns; the tab then knows every date.
        if columns_spec is None:
            columns_spec = [(scheme.id, scheme.latest.date) for scheme in group.schemes]
        if group.source_path is None:
            full = self.catalog.group(group.key, latest_only=False)
            if full is not None:
                group = full
        existing = self.tabs.get(group.key)
        if existing is not None:
            if columns_spec:
                try:
                    existing.set_columns(columns_spec)
                except Exception as exc:
                    LOGGER.exception("cannot update tab %s", group.title)
                    messagebox.showerror("Cannot open pack", f"{group.title}\n\n{exc}")
                    return
            self._select_tab(existing)
            return
        try:
            tab = PackTab(self.notebook.body, self, group, columns_spec)
        except Exception as exc:
            LOGGER.exception("cannot open tab %s", group.title)
            messagebox.showerror("Cannot open pack", f"{group.title}\n\n{exc}")
            return
        self.tabs[group.key] = tab
        self._tab_by_tabid[str(tab)] = tab
        self.notebook.add(tab, text=group.title)
        self._update_empty_state()
        self.notebook.select(tab)
        self._sync_controls_to_tab(tab)
        self._save_state()

    def _select_tab(self, tab: PackTab) -> None:
        self.notebook.select(tab)
        self._sync_controls_to_tab(tab)
        tab.render_current()

    def _on_tab_changed(self, _event=None) -> None:
        if self._restoring:
            return
        tab = self.active_tab
        if tab is not None:
            self._sync_controls_to_tab(tab)
            tab.render_current()
        self._save_state()

    @property
    def active_tab(self) -> PackTab | None:
        if not self.notebook.tabs():
            return None
        current = self.notebook.select()
        return self._tab_by_tabid.get(str(current)) if current else None

    def _update_empty_state(self) -> None:
        if self.notebook.tabs():
            self.welcome.pack_forget()
            self.notebook.pack(fill=tk.BOTH, expand=True)
        else:
            self.notebook.pack_forget()
            self.welcome.pack(fill=tk.BOTH, expand=True)

    def close_tab(self, tab: PackTab | None = None) -> None:
        tab = tab or self.active_tab
        if tab is None:
            return
        self.notebook.forget(tab)
        self.tabs.pop(tab.group.key, None)
        self._tab_by_tabid.pop(str(tab), None)
        tab.destroy()
        self._update_empty_state()
        self._save_state()
        if self.active_tab is not None:
            self._sync_controls_to_tab(self.active_tab)
            self.active_tab.render_current()
        else:
            self.set_report(ReportData([], [], "No pack open."))
            self.set_status("No pack open")

    def close_other_tabs(self) -> None:
        active = self.active_tab
        for tab in list(self.tabs.values()):
            if tab is active:
                continue
            self.notebook.forget(tab)
            self.tabs.pop(tab.group.key, None)
            self._tab_by_tabid.pop(str(tab), None)
            tab.destroy()
        self._update_empty_state()
        self._save_state()

    def close_all_tabs(self) -> None:
        for tab in list(self.tabs.values()):
            self.notebook.forget(tab)
            tab.destroy()
        self.tabs.clear()
        self._tab_by_tabid.clear()
        self._update_empty_state()
        self.set_report(ReportData([], [], "No pack open."))
        self._save_state()

    # -- open paths --------------------------------------------------------

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Open convergence npz", filetypes=[("Convergence npz", "*.npz"), ("All files", "*.*")],
            initialdir=self.state.last_directory or str(Path.cwd()),
        )
        if paths:
            self.state.last_directory = str(Path(paths[0]).parent)
            self.open_paths([Path(p) for p in paths])

    def open_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Open full-decontamination convergence folder",
            initialdir=self.state.last_directory or str(Path.cwd()),
        )
        if path:
            self.state.last_directory = str(Path(path).parent)
            self.open_paths([Path(path)])

    def _spec_for_path(self, path: Path) -> tuple[Group, list[tuple[str, str]]] | None:
        group = self.catalog.group_for_path(path)
        if group is None:
            group = self.catalog.synthetic_group(path)
            return group, [(group.schemes[0].id, group.schemes[0].latest.date)]
        for scheme in group.schemes:
            for item in scheme.dates:
                if item.path == path:
                    return group, [(scheme.id, item.date)]
        # A folder or a path that only matches the scheme's folder.
        for scheme in group.schemes:
            if scheme.folder == path:
                return group, [(scheme.id, scheme.latest.date)]
        return group, [(scheme.id, scheme.latest.date) for scheme in group.schemes]

    def open_paths(self, paths: list[Path]) -> None:
        """Open one tab per group; several files of one group become its columns."""
        by_group: dict[str, tuple[Group, list[tuple[str, str]]]] = {}
        for path in paths:
            found = self._spec_for_path(path)
            if found is None:
                continue
            group, spec = found
            if group.key in by_group:
                _existing_group, existing_spec = by_group[group.key]
                merged = dict(existing_spec)
                merged.update(spec)  # one column per scheme; the last file wins
                by_group[group.key] = (group, list(merged.items()))
            else:
                by_group[group.key] = (group, spec)
        for group, spec in by_group.values():
            try:
                self.open_group(group, spec)
            except Exception as exc:
                LOGGER.exception("cannot open %s", group.title)
                messagebox.showerror("Cannot open pack", f"{group.title}\n\n{exc}")

    # -- controls ----------------------------------------------------------

    def _sync_controls_to_tab(self, tab: PackTab) -> None:
        self._syncing = True
        try:
            self.forward_var.set(tab.forward)
            has_minus = any(col.session.availability().has_minus for col in tab.columns)
            if not has_minus:
                tab.central = False
            self.central_var.set(tab.central)
            self.central_check.configure(
                state=tk.NORMAL if has_minus else tk.DISABLED,
                text="Central (−Q runs)" if has_minus else "Central (no −Q runs in this pack)",
            )
            self.display_var.set("Coupling channels" if tab.display == "channels" else "Norm")
            self.phonon_var.set({
                "aligned": "Aligned full", "unaligned": "Unaligned full",
            }.get(tab.phonon_preference, "Auto"))
            self.noise_auto.set(tab.noise is None)
            self.noise_var.set("" if tab.noise is None else f"{tab.noise:g}")
            self.target_var.set(f"{tab.target:g}")
            self.tol_var.set(f"{tab.tol:g}")
            self.weak_var.set(f"{tab.weak_fraction:g}")
            self.drop_anchor_var.set(tab.drop_anchor)
            self.deviation_var.set(tab.deviation)
            self.quad_var.set(tab.quad)
            labels = [col.label for col in tab.columns]
            self._column_combo["values"] = labels
            self.active_column_var.set(labels[tab.active_column] if labels else "")
            tab.sync_column_combo()
            self._rebuild_column_dates(tab)
            self._populate_mode_tree()
            self._update_noise_sensitivity()
        finally:
            self._syncing = False

    def _rebuild_column_dates(self, tab: PackTab) -> None:
        for child in self.columns_frame.winfo_children():
            child.destroy()
        for row, column in enumerate(tab.columns):
            ttk.Label(self.columns_frame, text=column.label, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", padx=2, pady=1
            )
            var = tk.StringVar(value=column.date)
            combo = ttk.Combobox(
                self.columns_frame, textvariable=var, state="readonly", width=12,
                values=[item.date for item in column.scheme.dates],
            )
            combo.grid(row=row, column=1, sticky="ew", padx=2, pady=1)
            self._tip(combo, "Switch this Summary column to another registered file/date of the same scheme.")
            combo.bind("<<ComboboxSelected>>", lambda _e, c=column, v=var: self._change_date(c, v.get()))
        self.columns_frame.columnconfigure(1, weight=1)

    def _change_date(self, column: ColumnState, date: str) -> None:
        tab = self.active_tab
        if tab is None or date == column.date:
            return
        spec = [(c.scheme.id, date if c is column else c.date) for c in tab.columns]
        tab.set_columns(spec)
        self._sync_controls_to_tab(tab)
        tab.render_current()

    def _on_view_control(self, *_args) -> None:
        if self._syncing:
            return
        tab = self.active_tab
        if tab is None:
            return
        tab.forward = bool(self.forward_var.get())
        tab.central = bool(self.central_var.get())
        tab.display = "channels" if self.display_var.get() == "Coupling channels" else "norm"
        labels = [col.label for col in tab.columns]
        if self.active_column_var.get() in labels:
            tab.active_column = labels.index(self.active_column_var.get())
        tab.render_current()

    def _on_mode_click(self, event) -> None:
        tab = self.active_tab
        if tab is None:
            return
        row = self.mode_tree.identify_row(event.y)
        if not row:
            return
        try:
            mode = int(row)
        except ValueError:
            return
        column = self.mode_tree.identify_column(event.x)
        if column == "#0":
            visible = mode in tab.visible_modes
            tab.set_visible({mode}, not visible)
            self._populate_mode_tree()
        else:
            tab.active_mode = mode
            if self.mode_tree.selection():
                self.mode_tree.selection_set(row)
            tab.nested.select(1)  # selecting a mode opens its Mode-detail view
            tab.render_current()

    def _populate_mode_tree(self) -> None:
        tab = self.active_tab
        self.mode_tree.delete(*self.mode_tree.get_children())
        if tab is None:
            return
        needle = self.mode_filter.get().strip().lower()
        for mode in sorted(tab.mode_entries):
            entry = tab.mode_entries[mode]
            text = f"{mode} {entry.symmetry or ''} {entry.frequency if entry.frequency is not None else ''}"
            if needle and needle not in text.lower():
                continue
            self.mode_tree.insert(
                "", tk.END, iid=str(mode), text="",
                image=self._mode_icons[1 if mode in tab.visible_modes else 0],
                values=(str(mode), entry.symmetry or "", _fmt(entry.frequency, 3)),
            )
        if tab.active_mode is not None and self.mode_tree.exists(str(tab.active_mode)):
            self.mode_tree.selection_set(str(tab.active_mode))
            self.mode_tree.see(str(tab.active_mode))

    def _visible_in_tree(self) -> set[int]:
        return {int(iid) for iid in self.mode_tree.get_children()}

    def _mode_bulk(self, visible: bool) -> None:
        tab = self.active_tab
        if tab is None:
            return
        tab.set_visible(self._visible_in_tree(), visible)
        self._populate_mode_tree()

    def _mode_invert(self) -> None:
        tab = self.active_tab
        if tab is None:
            return
        tab.invert_visible(self._visible_in_tree())
        self._populate_mode_tree()

    def _update_noise_sensitivity(self) -> None:
        if hasattr(self, "noise_entry"):
            self.noise_entry.configure(state=tk.DISABLED if self.noise_auto.get() else tk.NORMAL)

    def apply_advanced(self) -> None:
        tab = self.active_tab
        if tab is None:
            return
        try:
            target = float(self.target_var.get())
            tol = float(self.tol_var.get())
            weak = float(self.weak_var.get())
            noise = None if self.noise_auto.get() else float(self.noise_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "SNR target, tolerance, weak fraction and noise must be numbers.")
            return
        if target <= 0 or tol <= 0 or weak < 0:
            messagebox.showerror("Invalid value", "SNR target and tolerance must be positive; weak fraction ≥ 0.")
            return
        tab.target, tab.tol, tab.weak_fraction, tab.noise = target, tol, weak, noise
        tab.drop_anchor = bool(self.drop_anchor_var.get())
        tab.deviation = bool(self.deviation_var.get())
        tab.quad = bool(self.quad_var.get())
        phonon = {
            "Aligned full": "aligned", "Unaligned full": "unaligned",
        }.get(self.phonon_var.get(), "auto")
        tab.set_phonon_preference(phonon)
        tab.render_current()

    def _select_view(self, index: int) -> None:
        tab = self.active_tab
        if tab is not None:
            tab.nested.select(index)

    # -- report and status -------------------------------------------------

    def set_report(self, data: ReportData) -> None:
        self.report_tree.delete(*self.report_tree.get_children())
        if data.columns:
            self.report_tree["columns"] = [key for key, _h, _w in data.columns]
            for key, heading, width in data.columns:
                self.report_tree.heading(key, text=heading)
                self.report_tree.column(key, width=width, anchor=tk.W)
        for row in data.rows:
            self.report_tree.insert("", tk.END, values=data.row_values(row))
        self.details_text.configure(state=tk.NORMAL)
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, data.details)
        self.details_text.configure(state=tk.DISABLED)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _toggle_report(self) -> None:
        if self.report_visible.get():
            try:
                self.paned.add(self.report_frame, height=_REPORT_HEIGHT, stretch="never")
            except tk.TclError:
                pass
        else:
            try:
                self.paned.forget(self.report_frame)
            except tk.TclError:
                pass

    # -- save/export -------------------------------------------------------

    def _default_output(self, tab: PackTab, suffix: str) -> Path:
        return Path("figures") / "convergence" / f"{tab.save_stem()}{suffix}"

    def save_figure(self, fmt: str = "png") -> None:
        """Write the CLI-style figure (not the stretched on-screen one) in the chosen format."""
        tab = self.active_tab
        if tab is None:
            return
        fmt = fmt.lower() if fmt.lower() in ("png", "svg", "pdf") else "png"
        figure = tab.canonical_figure()
        default = self._default_output(tab, f".{fmt}")
        default.parent.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title=f"Save figure ({fmt.upper()})", initialdir=str(default.parent), initialfile=default.name,
            defaultextension=f".{fmt}",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            plt.close(figure)
            return
        path = Path(path)
        if path.suffix.lower() not in (".png", ".svg", ".pdf"):
            path = path.with_suffix(f".{fmt}")
        try:
            figure.savefig(path, dpi=300)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        finally:
            plt.close(figure)
        self.set_status(f"Saved figure: {path}")

    def export_csv(self) -> None:
        tab = self.active_tab
        if tab is None:
            return
        data = tab.report()
        if not data.rows:
            messagebox.showinfo("No table", "The current view has no rows to export.")
            return
        default = self._default_output(tab, ".csv")
        default.parent.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Export table", initialdir=str(default.parent), initialfile=default.name,
            defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=data.headings)
            writer.writeheader()
            for row in data.rows:
                writer.writerow({heading: row.get(key, "") for key, heading, _w in data.columns})
        self.set_status(f"Saved table: {path}")

    def about(self) -> None:
        messagebox.showinfo(
            "About",
            "spinlabs convergence viewer\n\n"
            "Norm and V_{Δn} coupling-channel convergence plots.\n"
            "Registry data: ~/.spinlabs/registry.db",
        )

    # -- persistence -------------------------------------------------------

    def _save_state(self) -> None:
        if self._restoring:
            return
        try:
            if not self.root.winfo_exists():
                return
            geometry = self.root.geometry()
        except tk.TclError:
            return
        if self.root.winfo_width() <= 1:
            geometry = self.state.geometry
        tabs = [tab.state() for tab in self.tabs.values()]
        self.state = GuiState(
            geometry=geometry,
            filters={
                "defect": self.filter_defect.get(), "cell": self.filter_cell.get(),
                "method": self.filter_method.get(), "displacement": self.filter_displacement.get(),
                "latest_only": self.latest_only.get(),
            },
            report_visible=self.report_visible.get(),
            active_tab=self.active_tab.group.key if self.active_tab is not None else None,
            tabs=tabs,
            last_directory=self.state.last_directory,
        )
        try:
            save_state(self.state)
        except OSError as exc:
            LOGGER.warning("cannot save GUI state: %s", exc)

    def _restore_or_open(self, paths: list[Path] | None) -> None:
        if paths:
            self.open_paths(paths)
            return
        self._restoring = True
        try:
            self._restore_tabs()
        finally:
            self._restoring = False
        self._save_state()

    def _restore_tabs(self) -> None:
        restored = False
        seen: set[str] = set()
        for saved in self.state.tabs:
            if saved.group_key in seen:
                continue
            seen.add(saved.group_key)
            group = self.catalog.group(saved.group_key)
            if group is None:
                continue
            spec = [
                (item.get("scheme"), item.get("date"))
                for item in saved.columns
                if item.get("scheme") and group.scheme(item["scheme"])
            ]
            if not spec:
                continue
            try:
                tab = PackTab(self.notebook.body, self, group, spec)
            except Exception:
                LOGGER.exception("cannot restore tab %s", group.title)
                continue
            tab.set_phonon_preference(saved.phonon)
            tab.forward, tab.central, tab.display = saved.forward, saved.central, saved.display
            tab.noise, tab.target, tab.tol = saved.noise, saved.target, saved.tol
            tab.weak_fraction, tab.drop_anchor = saved.weak_fraction, saved.drop_anchor
            tab.deviation, tab.quad = saved.deviation, saved.quad
            if saved.visible_modes is not None:
                tab.visible_modes = set(saved.visible_modes) & set(tab.mode_entries)
            if saved.active_mode in tab.mode_entries:
                tab.active_mode = saved.active_mode
            tab.active_column = min(saved.active_column, max(0, len(tab.columns) - 1))
            self.tabs[group.key] = tab
            self._tab_by_tabid[str(tab)] = tab
            self.notebook.add(tab, text=group.title)
            restored = True
        if restored:
            self._update_empty_state()
            active = next(
                (tab for key, tab in self.tabs.items() if key == self.state.active_tab),
                next(iter(self.tabs.values())),
            )
            self.notebook.select(active)
            self._sync_controls_to_tab(active)
            active.render_current()
        else:
            groups = self._filtered_groups() or self.catalog.groups(latest_only=True)
            if groups:
                groups.sort(key=lambda g: max((s.latest.date for s in g.schemes), default=""), reverse=True)
                self.open_group(groups[0])
            else:
                self.set_report(ReportData([], [], "The registry has no convergence packs."))
                self.set_status("Empty registry — use File → Open or `spinlabs package --conv`")

    def quit(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._save_state()
        except Exception:
            LOGGER.exception("could not save GUI state")
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main(paths: list[Path] | None = None) -> int:
    """Entry point used by ``spinlabs gui``."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"error: cannot open a Tk window ({exc}); a display is required", file=sys.stderr)
        return 1
    app = ConvergenceApp(root, paths)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.quit()
    return 0
