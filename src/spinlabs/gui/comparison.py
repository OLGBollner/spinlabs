"""The comparison tab: production couplings, spectral functions, 2D maps, rates and T1 of several packages.

Entries are production packages (registry ``packaged_zfs``, full-decontamination sets included); any mix of
defects, cells and methods is drawn in one graph, one colour per entry. The numbers come from
:mod:`spinlabs.gui.production` (the production pipeline); the work runs on the app's worker thread and the
tab polls until it is done, drawing what is ready meanwhile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from spinlabs import pipeline
from spinlabs.figure_style import paper_style, to_paper_layout
from spinlabs.gui import production_plots as pplots
from spinlabs.gui.plots import message_figure
from spinlabs.gui.production import (
    CHANNELS,
    DIRECT_MODELS,
    PAIR_TREATMENTS,
    PROCESSES,
    T1_STATES,
    ProductionPackage,
    ProductionSession,
    Settings,
    package_for_path,
    EXPERIMENT_OFF_BY_DEFAULT,
    experiment_rates,
    experiment_t1,
    experimental_series,
    rate_direction,
    version_labels,
)
from spinlabs.gui.widgets import FigureHost, ReportData, ScrollableFrame, _fmt, _safe_name

VIEWS = ("Couplings", "Spectral functions", "2D map", "Rates", "T1")
_POLL_MS = 400
_SCHEME_GROUP = "scheme:"  # iid prefix of the entry list's scheme groups (AO, UO, FD)
_SCHEME_ORDER = ("AO", "UO", "FD")


def entry_scheme(package: ProductionPackage) -> str:
    """The scheme label of a package's method: AO, UO or FD (spinlabs.methods), else the method itself."""
    from spinlabs.methods import METHODS, normalize_calc_method

    method = METHODS.get(normalize_calc_method(package.method)) if package.method else None
    return method.label if method is not None else (package.method or "other")


_PLAIN_CHANNELS = {"d0": "Δ0 (V00)", "d1": "Δ1 (V0±)", "d2": "Δ2 (V±)"}
QUOTE_T = (100.0, 300.0)


@dataclass
class Entry:
    package: ProductionPackage
    session: ProductionSession
    color: str
    hidden: bool = False  # left out of every view and the report, keeping its colour and place


class ComparisonTab(ttk.Frame):
    """One comparison: entries, view options and production settings, and the five views."""

    def __init__(self, master, app, key: str, title: str, state: dict | None = None):
        super().__init__(master, style="Card.TFrame")
        self.app = app
        self.key = key
        self.title = title
        self.entries: list[Entry] = []
        self.settings = Settings()
        self.channels = {"d1", "d2"}
        self.log = True
        self.processes = {"total"}
        self.t1_state = T1_STATES[0]
        self.ratio = False
        self.map_key: str | None = None  # the package the 2D map shows (by key: removals and moves keep it)
        self.show_experiment = True
        self.experiment_off: set[str] = set(EXPERIMENT_OFF_BY_DEFAULT)  # series (ExperimentSeries.key) left out
        self._experiment_cache: dict[str, list] = {}  # defect -> its experimental series
        self._experiment_errors: dict[str, str] = {}
        self.experiment_open: set[str] = set()  # defects whose experiment dropdown is open
        self._poll_after: str | None = None
        self._last_keys: dict[int, tuple | None] = {i: None for i in range(len(VIEWS))}
        self._build()
        if state:
            self.apply_state(state)
        self.sync_controls()

    # -- widgets ----------------------------------------------------------------------------------------

    def _build(self) -> None:
        # The entries, view options and production settings live in the app's View sidebar tab, which shows
        # this panel while this comparison is the open tab (ConvergenceApp._show_view_panel)
        self.controls = ttk.Frame(self.app.view_tab, style="Card.TFrame")
        self.nested = ttk.Notebook(self)
        self.nested.pack(fill=tk.BOTH, expand=True)
        self.hosts = []
        for name in VIEWS:
            host = FigureHost(self.nested)
            self.nested.add(host, text=name)
            self.hosts.append(host)
        self.nested.bind("<<NotebookTabChanged>>", lambda _e: self.render_current())

        scroll = self._scroll = ScrollableFrame(self.controls, palette=self.app.palette)
        scroll.pack(fill=tk.BOTH, expand=True)
        body = scroll.content
        tip = self.app._tip

        box = ttk.LabelFrame(body, text="Entries")
        box.pack(fill=tk.X, padx=6, pady=4)
        self._show_icons = (self.app._checkbox_image(False), self.app._checkbox_image(True))
        self.entry_tree = ttk.Treeview(box, columns=("entry", "status"), show=("tree", "headings"), height=6,
                                       selectmode="extended")
        self.entry_tree.heading("#0", text="Show")
        self.entry_tree.column("#0", width=110, stretch=False, anchor=tk.W)
        self.entry_tree.tag_configure("hidden", foreground="#9aa0a6")
        self.entry_tree.tag_configure("group", font=(self.app.family, 9, "bold"))
        self.entry_groups_closed: set[str] = set()  # scheme groups of the entry list left closed
        self.entry_tree.heading("entry", text="Entry")
        self.entry_tree.heading("status", text="Status")
        self.entry_tree.column("entry", width=150, anchor=tk.W)
        self.entry_tree.column("status", width=80, anchor=tk.W)
        self.entry_tree.pack(fill=tk.X, padx=4, pady=4)
        tip(self.entry_tree, "The packages compared in this tab, one colour each. Add them from the Data tab (Production); "
                             "grouped by scheme (AO, UO, FD). Click Show to hide an entry from the views and the "
                             "report (it keeps its colour and place), a scheme's checkbox for all of its entries, its "
                             "name to open or close it; drag an entry onto another to reorder (legends follow the order; the first shown entry is "
                             "the T1 ratio reference).")
        self._drag: str | None = None
        self._dragged = False
        self.entry_tree.bind("<ButtonPress-1>", self._drag_start, add="+")
        self.entry_tree.bind("<B1-Motion>", self._drag_motion, add="+")
        self.entry_tree.bind("<ButtonRelease-1>", self._drag_end, add="+")
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X, padx=4, pady=(0, 4))
        remove = ttk.Button(row, text="Remove", command=self.remove_selected)
        remove.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tip(remove, "Remove the selected entries from this comparison.")
        toggle = ttk.Button(row, text="Hide/show", command=self.toggle_selected)
        toggle.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        tip(toggle, "Hide the selected entries from the views and the report, or show them again when all of them "
                    "are hidden.")
        add_file = ttk.Button(row, text="Add file…", command=self.add_file)
        add_file.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        tip(add_file, "Add a packaged ZFS file that is not in the registry.")
        self.entry_tree.bind("<<TreeviewSelect>>", lambda _e: self._sync_entry_date())
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Label(row, text="Date", style="CardMuted.TLabel").pack(side=tk.LEFT)
        self.entry_date = tk.StringVar(value="")
        self.entry_date_box = ttk.Combobox(row, textvariable=self.entry_date, state="disabled")
        self.entry_date_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.entry_date_box.bind("<<ComboboxSelected>>", lambda _e: self._choose_entry_date())
        tip(self.entry_date_box, "Switch the selected entry to another package of the same setup (defect, cell, "
                                 "method, step), newest first; it keeps its colour and place.")

        box = ttk.LabelFrame(body, text="Channels")
        box.pack(fill=tk.X, padx=6, pady=4)
        self.channel_vars = {}
        for channel in CHANNELS:
            var = tk.BooleanVar(value=channel in self.channels)
            check = ttk.Checkbutton(box, text=_PLAIN_CHANNELS[channel], variable=var, command=self._on_option)
            check.pack(anchor=tk.W, padx=6, pady=1)
            self.channel_vars[channel] = var
        tip(box, "Δ0: V00 (A1 modes, level modulation, no transitions); Δ1: V0± (Δm = ±1); Δ2: V± (Δm = ±2). "
                 "Rates show the spin-flip channels Δ1 and Δ2.")

        box = ttk.LabelFrame(body, text="Couplings and spectra")
        box.pack(fill=tk.X, padx=6, pady=4)
        self.log_var = tk.BooleanVar(value=self.log)
        ttk.Checkbutton(box, text="Log scale", variable=self.log_var, command=self._on_option).pack(
            anchor=tk.W, padx=6, pady=1)
        ttk.Label(box, text="σ (meV)", style="Card.TLabel").pack(anchor=tk.W, padx=6)
        self.sigma_var = tk.StringVar(value=f"{self.settings.sigma:g}")
        sigma = ttk.Entry(box, textvariable=self.sigma_var, width=8)
        sigma.pack(anchor=tk.W, padx=6, pady=2)
        sigma.bind("<Return>", lambda _e: self.apply_settings())
        tip(sigma, "Gaussian broadening of the spectral functions and the rates (production: 3.5 meV). Enter applies.")
        ttk.Label(box, text="2D map entry", style="Card.TLabel").pack(anchor=tk.W, padx=6, pady=(4, 0))
        self.map_var = tk.StringVar(value="")
        self.map_combo = ttk.Combobox(box, textvariable=self.map_var, state="readonly")
        self.map_combo.pack(fill=tk.X, padx=6, pady=2)
        self.map_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_option())
        tip(self.map_combo, "The entry whose full two-phonon spectral density the 2D map view shows.")

        box = ttk.LabelFrame(body, text="Rates and T1")
        box.pack(fill=tk.X, padx=6, pady=4)
        self.process_vars = {}
        for name in PROCESSES:
            var = tk.BooleanVar(value=name in self.processes)
            ttk.Checkbutton(box, text=name, variable=var, command=self._on_option).pack(anchor=tk.W, padx=6, pady=1)
            self.process_vars[name] = var
        tip(box, "Rates: total (direct + two-phonon, the production T1 processes) or each process alone. "
                 "T1 always uses the production processes.")
        ttk.Label(box, text="T1 initial state", style="Card.TLabel").pack(anchor=tk.W, padx=6, pady=(4, 0))
        self.state_var = tk.StringVar(value=self.t1_state)
        state = ttk.Combobox(box, textvariable=self.state_var, state="readonly", values=T1_STATES)
        state.pack(fill=tk.X, padx=6, pady=2)
        state.bind("<<ComboboxSelected>>", lambda _e: self._on_option())
        tip(state, "The state whose T1 the T1 view shows:\n"
                   "lowest |m_s|: m_s = 0 (S = 1) or ±1/2 (S = 3/2), the optically pumped state of NV- and V2\n"
                   "highest |m_s|: m_s = +1 (S = 1) or ±3/2 (S = 3/2)\n"
                   "m_s = -S: m_s = -1 (S = 1; differs from +1 only in a magnetic field); for S = 3/2 the "
                   "±3/2 doublet, as highest |m_s|")
        self.ratio_var = tk.BooleanVar(value=self.ratio)
        ratio = ttk.Checkbutton(box, text="T1 relative to the first entry", variable=self.ratio_var,
                                command=self._on_option)
        ratio.pack(anchor=tk.W, padx=6, pady=1)
        ttk.Label(box, text="B (mT, axial)", style="Card.TLabel").pack(anchor=tk.W, padx=6, pady=(4, 0))
        self.b_var = tk.StringVar(value=f"{self.settings.b_field_mt:g}")
        b_field = ttk.Entry(box, textvariable=self.b_var, width=8)
        b_field.pack(anchor=tk.W, padx=6, pady=2)
        b_field.bind("<Return>", lambda _e: self.apply_settings())
        tip(b_field, "Axial magnetic field along the defect axis (mT): Zeeman-splits +m_s and −m_s, moves every "
                     "gap and opens the +m ↔ −m direct rate (spinlabs.pipeline.rate_series b_field_mt). With B ≠ 0 "
                     "the Rates view adds the −S transitions. Enter applies; the couplings are reused.")

        box = ttk.LabelFrame(body, text="Experiment")
        box.pack(fill=tk.X, padx=6, pady=4)
        self.show_experiment_var = tk.BooleanVar(value=self.show_experiment)
        show = ttk.Checkbutton(box, text="Show experimental data", variable=self.show_experiment_var,
                               command=self._on_experiment)
        show.pack(anchor=tk.W, padx=6, pady=1)
        tip(show, "Measured points of the compared defects (experimental_data/<defect>/*.csv) in the Rates and T1 "
                  "views. Rates: Ω in the Δ1 panels (m_s 1 → 0), γ in the Δ2 panels (1 → −1). T1: the measured "
                  "T1 / 1/T1 data (an optically pumped polarization: drawn for the lowest |m_s|) and T1 of the "
                  "chosen initial state solved from a source's Ω and γ.")
        row = ttk.Frame(box, style="Card.TFrame")
        row.pack(fill=tk.X, padx=6, pady=(2, 0))
        folder = ttk.Button(row, text="Folder…", command=self.choose_experiment_dir)
        folder.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tip(folder, "Choose the experimental-data folder: one subfolder per defect (NV/, V_Si/, …) with the CSV "
                    "files. Applies to every comparison and is remembered.")
        default = ttk.Button(row, text="Default", command=lambda: self.app.set_experiment_dir(None))
        default.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        tip(default, "Find experimental_data/<defect> again in the working directory or next to the packages.")
        self.experiment_dir_label = ttk.Label(box, style="CardMuted.TLabel", wraplength=260)
        self.experiment_dir_label.pack(anchor=tk.W, padx=6, pady=(2, 2))
        # one group row per defect (its checkbox: all of its series; "n/N on"), a checkbox row per series
        self.experiment_tree = ttk.Treeview(box, show="tree", selectmode="none", height=2)
        self.experiment_tree.column("#0", width=260, anchor=tk.W)
        self.experiment_tree.tag_configure("group", font=(self.app.family, 9, "bold"))
        self.experiment_tree.tag_configure("off", foreground="#9aa0a6")
        self.experiment_tree.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.experiment_tree.bind("<ButtonRelease-1>", self._on_experiment_click)
        self.experiment_tree.bind("<<TreeviewOpen>>", lambda _e: self.after_idle(self._experiment_opened))
        self.experiment_tree.bind("<<TreeviewClose>>", lambda _e: self.after_idle(self._experiment_opened))
        tip(self.experiment_tree, "One group per defect: click a series to draw it or not, a defect's checkbox for "
                                  "all of its series, its name to open or close it.")
        self._experiment_items: dict[str, list] = {}  # defect -> its series

        box = ttk.LabelFrame(body, text="Temperatures (K)")
        box.pack(fill=tk.X, padx=6, pady=4)
        self.t_vars = {}
        for col, (name, value) in enumerate((("min", self.settings.t_min), ("max", self.settings.t_max),
                                             ("step", self.settings.t_step))):
            ttk.Label(box, text=name, style="Card.TLabel").grid(row=0, column=col, padx=4, sticky="w")
            var = tk.StringVar(value=f"{value:g}")
            ttk.Entry(box, textvariable=var, width=6).grid(row=1, column=col, padx=4, pady=2, sticky="ew")
            self.t_vars[name] = var

        box = ttk.LabelFrame(body, text="Production settings")
        box.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(box, text="Second-order pairs", style="Card.TLabel").pack(anchor=tk.W, padx=6)
        self.pairs_var = tk.StringVar(value=self.settings.pairs)
        pairs = ttk.Combobox(box, textvariable=self.pairs_var, state="readonly", values=PAIR_TREATMENTS)
        pairs.pack(fill=tk.X, padx=6, pady=2)
        tip(pairs, "Which second-order pairs (l, l') the couplings use:\n" + "\n".join(
            f"{name}: {text}" for name, text in pipeline.PAIR_MODES.items()))  # every option, not just one
        ttk.Label(box, text="Direct (one-phonon) rate", style="Card.TLabel").pack(anchor=tk.W, padx=6)
        self.direct_var = tk.StringVar(value=self.settings.direct)
        direct = ttk.Combobox(box, textvariable=self.direct_var, state="readonly", values=DIRECT_MODELS)
        direct.pack(fill=tk.X, padx=6, pady=2)
        tip(direct, "The one-phonon (direct) rate at the spin gap, far below the supercell's phonons:\n"
                    "debye: J(E) = C E^3, C fitted to the folded acoustic modes (production)\n"
                    "gaussian: the Gaussian-smeared one-phonon spectral function at the gap (depends on σ)")
        self.mirror_var = tk.BooleanVar(value=self.settings.mirror_projection)
        mirror = ttk.Checkbutton(box, text="Mirror projection", variable=self.mirror_var)
        mirror.pack(anchor=tk.W, padx=6, pady=1)
        tip(mirror, "The mirror projection of the diagonal second differences, as `spinlabs run` (production: on).")
        apply = ttk.Button(body, text="Apply", command=self.apply_settings, style="Accent.TButton")
        apply.pack(fill=tk.X, padx=6, pady=(8, 4))
        tip(apply, "Apply σ, the temperatures and the production settings. New couplings or rates are computed "
                   "in the background and cached on disk.")
        ttk.Label(body, text="Defaults are the production settings of `spinlabs run`.", wraplength=260,
                  style="CardMuted.TLabel").pack(anchor=tk.W, padx=8, pady=(0, 8))

    # -- entries ----------------------------------------------------------------------------------------

    def add_packages(self, packages: list[ProductionPackage]) -> None:
        known = {e.package.key for e in self.entries}
        for package in packages:
            if package.key in known:
                continue
            session = self.app.production_session(package)
            self.entries.append(Entry(package, session, self._free_color()))
            known.add(package.key)
        self.sync_controls()
        self.invalidate()
        self.render_current()

    def add_file(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Add a packaged ZFS file", filetypes=[("ZFS package", "*.npz"), ("All files", "*.*")],
            initialdir=self.app.state.last_directory or str(Path.cwd()),
        )
        if not path:
            return
        try:
            self.add_packages([package_for_path(Path(path))])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot add package", f"{path}\n\n{exc}")

    def remove_selected(self) -> None:
        selected = set(self.entry_tree.selection())
        self.entries = [e for e in self.entries if e.package.key not in selected]
        self.sync_controls()
        self.invalidate()
        self.render_current()

    @property
    def shown(self) -> list[Entry]:
        """The entries the views and the report draw, in entry order."""
        return [e for e in self.entries if not e.hidden]

    def set_hidden(self, keys, hidden: bool) -> None:
        keys = set(keys)
        changed = False
        for entry in self.entries:
            if entry.package.key in keys and entry.hidden != hidden:
                entry.hidden, changed = hidden, True
        if changed:
            self.update_entry_tree()
            self.invalidate()
            self.render_current()
            self.app._save_state()

    def toggle_selected(self) -> None:
        """Hide the selected entries, or show them if all of them are hidden."""
        chosen = set(self.entry_tree.selection())
        groups = {iid[len(_SCHEME_GROUP):] for iid in chosen if iid.startswith(_SCHEME_GROUP)}
        selected = [e for e in self.entries if e.package.key in chosen or entry_scheme(e.package) in groups]
        if selected:
            self.set_hidden([e.package.key for e in selected], not all(e.hidden for e in selected))

    def _selected_entry(self) -> Entry | None:
        selection = self.entry_tree.selection()
        if len(selection) != 1:
            return None
        return next((e for e in self.entries if e.package.key == selection[0]), None)

    def _sync_entry_date(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            self.entry_date_box.configure(values=[], state="disabled")
            self.entry_date.set("")
            return
        versions = self.app.production_versions(entry.package)
        labels = version_labels(versions)
        self.entry_date_box.configure(values=labels, state="readonly" if len(versions) > 1 else "disabled")
        index = next(i for i, p in enumerate(versions) if p.key == entry.package.key)
        self.entry_date.set(labels[index])

    def _choose_entry_date(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        versions = self.app.production_versions(entry.package)
        labels = version_labels(versions)
        if self.entry_date.get() not in labels:
            return
        package = versions[labels.index(self.entry_date.get())]
        if package.key == entry.package.key:
            return
        if any(e.package.key == package.key for e in self.entries):
            messagebox.showinfo("Already compared", f"{package.long_label} is already an entry of this comparison.")
            self._sync_entry_date()
            return
        self.switch_entry(entry, package)

    def switch_entry(self, entry: Entry, package: ProductionPackage) -> None:
        """Replace an entry's package by another version in place (same colour and position)."""
        if self.map_key == entry.package.key:
            self.map_key = package.key
        entry.package = package
        entry.session = self.app.production_session(package)
        self.sync_controls()
        self.entry_tree.selection_set([package.key])
        self.invalidate()
        self.render_current()
        self.app._save_state()

    def _free_color(self) -> str:
        """The first palette colour no entry uses: an entry keeps its colour when others are removed or moved."""
        used = {e.color for e in self.entries}
        index = 0
        while pplots.entry_color(index) in used and index < len(pplots.ENTRY_COLORS):
            index += 1
        return pplots.entry_color(index if index < len(pplots.ENTRY_COLORS) else len(self.entries))

    # -- drag to reorder --------------------------------------------------------------------------------

    def _drag_start(self, event):
        tree = self.entry_tree
        row = tree.identify_row(event.y)
        if row.startswith(_SCHEME_GROUP):
            self._drag, self._dragged = None, False
            element = tree.identify_element(event.x, event.y)
            if "indicator" in element:
                self.after_idle(self._entry_groups_changed)
                return None  # the tree itself opens or closes the group
            if tree.identify_column(event.x) == "#0" and "image" in element:  # Show for the whole scheme
                members = [e for e in self.entries if entry_scheme(e.package) == row[len(_SCHEME_GROUP):]]
                self.set_hidden([e.package.key for e in members], not any(e.hidden for e in members))
            else:
                tree.item(row, open=not tree.item(row, "open"))
                self._entry_groups_changed()
            return "break"
        if row and tree.identify_column(event.x) == "#0":  # the Show checkbox: toggle, no drag
            entry = next((e for e in self.entries if e.package.key == row), None)
            if entry is not None:
                self.set_hidden([row], not entry.hidden)
            self._drag, self._dragged = None, False
            return "break"
        self._drag = row or None
        self._dragged = False
        return None

    def _drag_motion(self, event) -> None:
        if self._drag is None:
            return
        target = self.entry_tree.identify_row(event.y)
        if not target or target == self._drag or target.startswith(_SCHEME_GROUP):
            return
        index = next((i for i, e in enumerate(self.entries) if e.package.key == target), None)
        if index is not None:  # to the target's place in the entry order (legends follow it)
            self.move_entry(self._drag, index, redraw=False)
            self._dragged = True

    def _drag_end(self, _event) -> None:
        dragged, self._drag, self._dragged = self._dragged, None, False
        if dragged:
            self.invalidate()
            self.render_current()
            self.app._save_state()

    @property
    def map_entry(self) -> int:
        """Index of the entry the 2D map shows: the chosen package, else the first entry."""
        keys = [e.package.key for e in self.entries]
        return keys.index(self.map_key) if self.map_key in keys else 0

    def move_entry(self, key: str, index: int, *, redraw: bool = True) -> None:
        """Move an entry to ``index`` (its colour stays; the 2D-map choice follows its entry)."""
        position = next((i for i, e in enumerate(self.entries) if e.package.key == key), None)
        if position is None:
            return
        index = max(0, min(index, len(self.entries) - 1))
        entry = self.entries.pop(position)
        self.entries.insert(index, entry)
        self.update_entry_tree()
        labels = [e.package.long_label for e in self.entries]
        self.map_combo["values"] = labels
        self.map_var.set(labels[self.map_entry] if labels else "")
        if redraw:
            self.invalidate()
            self.render_current()

    # -- options and settings ---------------------------------------------------------------------------

    def _on_option(self) -> None:
        self.channels = {c for c, var in self.channel_vars.items() if var.get()}
        self.log = bool(self.log_var.get())
        self.processes = {name for name, var in self.process_vars.items() if var.get()}
        self.t1_state = self.state_var.get() or T1_STATES[0]
        self.ratio = bool(self.ratio_var.get())
        chosen = self.map_combo.current()  # by position: identical labels stay distinct
        if 0 <= chosen < len(self.entries):
            self.map_key = self.entries[chosen].package.key
        self.render_current()

    # -- experiment -------------------------------------------------------------------------------------

    def experiment_series(self) -> list:
        """Every experimental series of the compared defects, in a stable order (markers follow it)."""
        out = []
        for entry in self.entries:
            defect = entry.package.defect
            if defect not in self._experiment_cache:
                try:
                    self._experiment_cache[defect] = experimental_series(defect, near=entry.package.path,
                                                                         root=self.app.state.experiment_dir)
                except Exception as exc:  # noqa: BLE001 -- an unreadable CSV: reported in Details
                    self._experiment_cache[defect] = []
                    self._experiment_errors[defect] = f"{type(exc).__name__}: {exc}"
        for defect in dict.fromkeys(e.package.defect for e in self.entries):
            out.extend(self._experiment_cache.get(defect, []))
        return out

    def _experiment_markers(self) -> dict[str, str]:
        return {s.key: pplots.experiment_marker(i) for i, s in enumerate(self.experiment_series())}

    def _shown_experiment(self) -> list:
        if not self.show_experiment:
            return []
        return [s for s in self.experiment_series() if s.key not in self.experiment_off]

    def _experiment_style(self, s) -> tuple[str, str | None]:
        """Legend and colour of a series: with several defects compared, the defect's first entry's colour
        and the defect in the legend; else dark grey."""
        defects = list(dict.fromkeys(e.package.defect for e in self.entries))
        if len(defects) < 2:
            return s.legend, None
        color = next((e.color for e in self.entries if e.package.defect == s.defect), None)
        return f"{s.defect}: {s.legend}", color

    def _experiment_rates(self, channels) -> dict:
        markers, shown = self._experiment_markers(), self._shown_experiment()
        return {c: [(*self._experiment_style(s)[:1], markers[s.key], self._experiment_style(s)[1], s.x, s.y)
                    for s in experiment_rates(shown, c)] for c in channels}

    def _experiment_t1(self) -> list:
        markers, shown = self._experiment_markers(), self._shown_experiment()
        out = []
        for s, x, y in experiment_t1(shown, self.t1_state):
            omega_key = f"{s.defect}/{s.source}/omega"
            marker = markers.get(s.key) or next((m for k, m in markers.items() if k.lower() == omega_key.lower()), "o")
            legend, color = self._experiment_style(s)
            out.append((legend, marker, color, x, y))
        return out

    def choose_experiment_dir(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(title="Experimental-data folder (one subfolder per defect)",
                                       initialdir=self.app.state.experiment_dir or str(Path.cwd()), mustexist=True)
        if path:
            self.app.set_experiment_dir(path)

    def reload_experiment(self) -> None:
        """Read the experimental data again (another folder chosen) and redraw."""
        self._experiment_cache.clear()
        self._experiment_errors.clear()
        self._rebuild_experiment_list()
        self.invalidate()
        self.render_current()

    def _rebuild_experiment_list(self) -> None:
        """The experiment tree: a group row per defect, a checkbox row per series (iid: the series key)."""
        tree = self.experiment_tree
        tree.delete(*tree.get_children())
        chosen = self.app.state.experiment_dir
        self.experiment_dir_label.configure(
            text=f"Folder: {chosen}" if chosen else "Folder: experimental_data/ next to the packages (default)")
        series = self.experiment_series()
        markers = self._experiment_markers()
        self._experiment_items = {}
        if not series:
            text = "No experimental data for these defects." if self.entries else "Add entries first."
            tree.insert("", tk.END, text=text, tags=("off",))
            tree.configure(height=1)
            return
        for s in series:
            self._experiment_items.setdefault(s.defect, []).append(s)
        for defect, items in self._experiment_items.items():
            tree.insert("", tk.END, iid=f"defect:{defect}", open=defect in self.experiment_open, tags=("group",))
            for s in items:
                kind = {"omega": "rate, Δ1", "gamma": "rate, Δ2"}.get(s.kind, "T1")
                tree.insert(f"defect:{defect}", tk.END, iid=s.key, text=f"{markers[s.key]}  {s.legend} ({kind})")
        self._refresh_experiment_controls()

    def _refresh_experiment_controls(self) -> None:
        """Checkboxes, counts and the greyed-out look (Show experimental data off) of the experiment tree."""
        tree = self.experiment_tree
        on_icon, off_icon = self._show_icons[1], self._show_icons[0]
        dim = () if self.show_experiment else ("off",)
        for defect, items in self._experiment_items.items():
            on = sum(s.key not in self.experiment_off for s in items)
            tree.item(f"defect:{defect}", text=f"{defect}  ({on}/{len(items)} on)",
                      image=on_icon if on == len(items) else off_icon, tags=("group", *dim))
            for s in items:
                tree.item(s.key, image=off_icon if s.key in self.experiment_off else on_icon, tags=dim)
        self._fit_experiment_tree()

    def _fit_experiment_tree(self) -> None:
        """As many rows as are shown (no inner scrollbar; the panel scrolls)."""
        tree = self.experiment_tree
        rows = sum(1 + (len(tree.get_children(g)) if tree.item(g, "open") else 0) for g in tree.get_children())
        tree.configure(height=max(1, min(rows, 16)))
        self._scroll.refresh()

    def _experiment_opened(self) -> None:
        self.experiment_open = {g.split(":", 1)[1] for g in self.experiment_tree.get_children()
                                if g.startswith("defect:") and self.experiment_tree.item(g, "open")}
        self._fit_experiment_tree()
        self.app._save_state()

    def _on_experiment_click(self, event) -> None:
        tree = self.experiment_tree
        row = tree.identify_row(event.y)
        if not row or not self.show_experiment or not self._experiment_items:
            return
        element = tree.identify_element(event.x, event.y)
        if row.startswith("defect:"):
            if "indicator" in element:
                return  # the tree itself opens or closes the group
            if "image" in element:  # the defect's checkbox: all of its series on, or all off when they are
                keys = {s.key for s in self._experiment_items[row.split(":", 1)[1]]}
                if keys & self.experiment_off:
                    self.experiment_off -= keys
                else:
                    self.experiment_off |= keys
            else:
                tree.item(row, open=not tree.item(row, "open"))
                self._experiment_opened()
                return
        else:
            self.experiment_off ^= {row}
        self._refresh_experiment_controls()
        self.render_current()

    def _on_experiment(self) -> None:
        self.show_experiment = bool(self.show_experiment_var.get())
        self._refresh_experiment_controls()
        self.render_current()

    def apply_settings(self) -> None:
        try:
            settings = Settings(
                pairs=self.pairs_var.get(), mirror_projection=bool(self.mirror_var.get()),
                direct=self.direct_var.get(), sigma=float(self.sigma_var.get()),
                t_min=float(self.t_vars["min"].get()), t_max=float(self.t_vars["max"].get()),
                t_step=float(self.t_vars["step"].get()), b_field_mt=float(self.b_var.get() or 0),
            )
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc) or "σ, B and the temperatures must be numbers.")
            return
        self.settings = settings
        self.invalidate()
        self.render_current()

    def sync_controls(self) -> None:
        """Write the tab's state into its widgets (after a restore or an entry change)."""
        for channel, var in self.channel_vars.items():
            var.set(channel in self.channels)
        self.log_var.set(self.log)
        for name, var in self.process_vars.items():
            var.set(name in self.processes)
        self.state_var.set(self.t1_state)
        self.ratio_var.set(self.ratio)
        s = self.settings
        self.sigma_var.set(f"{s.sigma:g}")
        self.b_var.set(f"{s.b_field_mt:g}")
        for name, value in (("min", s.t_min), ("max", s.t_max), ("step", s.t_step)):
            self.t_vars[name].set(f"{value:g}")
        self.pairs_var.set(s.pairs)
        self.direct_var.set(s.direct)
        self.mirror_var.set(s.mirror_projection)
        labels = [e.package.long_label for e in self.entries]
        self.map_combo["values"] = labels
        self.map_var.set(labels[self.map_entry] if labels else "")
        self.show_experiment_var.set(self.show_experiment)
        self._rebuild_experiment_list()
        self.update_entry_tree()

    def _entry_groups_changed(self) -> None:
        self.entry_groups_closed = {g[len(_SCHEME_GROUP):] for g in self.entry_tree.get_children()
                                    if not self.entry_tree.item(g, "open")}
        self.app._save_state()

    def update_entry_tree(self) -> None:
        """A group row per scheme (AO, UO, FD: Show for all of its entries, "n/N shown"), its entries in entry
        order below; updated in place (no rebuild: the selection and a drag survive the polls)."""
        tree = self.entry_tree
        keys = {e.package.key for e in self.entries}
        members: dict[str, list[Entry]] = {}
        for entry in self.entries:
            members.setdefault(entry_scheme(entry.package), []).append(entry)
        order = sorted(members, key=lambda name: (_SCHEME_ORDER.index(name) if name in _SCHEME_ORDER
                                                  else len(_SCHEME_ORDER), name))
        for group in tree.get_children():
            for iid in tree.get_children(group):
                if iid not in keys:
                    tree.delete(iid)
            if group[len(_SCHEME_GROUP):] not in members:
                tree.delete(group)
        for position, name in enumerate(order):
            group = _SCHEME_GROUP + name
            shown = sum(not e.hidden for e in members[name])
            text = f"{name}  ({shown}/{len(members[name])} shown)"
            image = self._show_icons[1 if shown == len(members[name]) else 0]
            if not tree.exists(group):
                tree.insert("", position, iid=group, open=name not in self.entry_groups_closed, tags=("group",))
            elif tree.index(group) != position:
                tree.move(group, "", position)
            tree.item(group, text=text, image=image)
        for entry in self.entries:
            group = _SCHEME_GROUP + entry_scheme(entry.package)
            index = members[entry_scheme(entry.package)].index(entry)
            tag = f"c{entry.color}"
            tree.tag_configure(tag, foreground=entry.color)
            tags = ("hidden",) if entry.hidden else (tag,)
            image = self._show_icons[0 if entry.hidden else 1]
            if entry.hidden:
                status = "hidden"
            elif entry.session.errors(self.settings):
                status = "failed"
            elif entry.session.busy():
                status = "computing…"
            else:
                status = "ready"
            values = (entry.package.long_label, status)
            if tree.exists(entry.package.key):
                if tuple(tree.item(entry.package.key, "values")) != values or \
                        tuple(tree.item(entry.package.key, "tags")) != tags:
                    tree.item(entry.package.key, values=values, tags=tags, image=image)
                if tree.parent(entry.package.key) != group or tree.index(entry.package.key) != index:
                    tree.move(entry.package.key, group, index)
            else:
                tree.insert(group, index, iid=entry.package.key, text="", image=image, values=values, tags=tags)

    # -- data -------------------------------------------------------------------------------------------

    def _fetch(self, kind: str, *args):
        """[(entry, result)] of the entries whose result is ready; queues the others. Failures are skipped
        (listed in the report)."""
        ready = []
        for entry in self.shown:
            try:
                result = entry.session.get(kind, self.settings, *args, executor=self.app.executor)
            except RuntimeError:
                continue
            if result is not None:
                ready.append((entry, result))
        return ready

    def _pending(self) -> bool:
        return any(entry.session.busy() for entry in self.shown)

    # -- rendering --------------------------------------------------------------------------------------

    def invalidate(self) -> None:
        self._last_keys = {i: None for i in range(len(VIEWS))}
        self._last_report = None

    def _view_key(self, view: int) -> tuple:
        return (view, tuple(e.package.key for e in self.shown), tuple(sorted(self.channels)),
                self.log, tuple(sorted(self.processes)), self.t1_state, self.ratio, self.map_entry,
                self.settings.rate_key(), self.show_experiment, tuple(sorted(self.experiment_off)))

    def _figure(self, view: int, *, canonical: bool = False) -> Figure:
        host = self.hosts[view]
        figsize = None if canonical else host.target_figsize()
        if not self.entries:
            return message_figure("No entries: add production packages from the Data tab", figsize=figsize or (8, 5))
        if not self.shown:
            return message_figure("Every entry is hidden: tick Show in the entry list", figsize=figsize or (8, 5))
        channels = [c for c in CHANNELS if c in self.channels]
        if view == 0:
            data = [(e.package.long_label, e.color, c) for e, c in self._fetch("couplings")]
            return pplots.build_coupling_figure(data, channels, log=self.log, figsize=figsize)
        if view == 1:
            data = [(e.package.long_label, e.color, s) for e, s in self._fetch("spectral")]
            return pplots.build_spectral_figure(data, channels, self.settings.sigma, log=self.log, figsize=figsize)
        if view == 2:
            entry = self._map_entry()
            try:
                maps = entry.session.get("spectral_map", self.settings, executor=self.app.executor)
            except RuntimeError as exc:
                return message_figure(f"{entry.package.long_label}: {exc}", figsize=figsize or (8, 5))
            if maps is None:
                return message_figure(f"Computing the 2D spectral density of {entry.package.long_label}…",
                                      figsize=figsize or (8, 5))
            return pplots.build_map_figure(entry.package.long_label, maps, channels, self.settings.sigma,
                                           figsize=figsize)
        if view == 3:
            processes = {name: PROCESSES[name] for name in PROCESSES if name in self.processes}
            data = [(e.package.long_label, e.color, r) for e, r in self._fetch("rates")]
            return pplots.build_rates_figure(data, channels, processes, figsize=figsize,
                                             mirror=bool(self.settings.b_field_mt),
                                             experiment=self._experiment_rates(channels))
        data = [(e.package.long_label, e.color, t) for e, t in self._fetch("t1", self.t1_state)]
        return pplots.build_t1_figure(data, ratio=self.ratio, experiment=self._experiment_t1(), figsize=figsize)

    def _needs(self, view: int) -> list[tuple]:
        """(session, kind, args) of the results a view draws."""
        if not self.shown:
            return []
        if view == 2:
            return [(self._map_entry().session, "spectral_map", ())]
        kind, args = {0: ("couplings", ()), 1: ("spectral", ()), 3: ("rates", ()), 4: ("t1", (self.t1_state,))}[view]
        return [(e.session, kind, args) for e in self.shown]

    def _map_entry(self) -> Entry:
        """The entry the 2D map draws: the chosen one, else (it is hidden) the first shown entry."""
        entry = self.entries[min(self.map_entry, len(self.entries) - 1)]
        return self.shown[0] if entry.hidden and self.shown else entry

    def _signature(self, needs) -> tuple:
        return tuple(session.status(kind, self.settings, *args) for session, kind, args in needs)

    def _report_signature(self) -> tuple:
        needs = [(e.session, k, a) for e in self.shown
                 for k, a in (("couplings", ()), ("rates", ()), ("t1", (self.t1_state,)))]
        return (self._signature(needs), tuple(len(e.session.errors(self.settings)) for e in self.shown),
                tuple(e.package.key for e in self.shown), self.settings.rate_key(), self.t1_state)

    def render_current(self) -> None:
        """Draw the current view when what it shows changed: new inputs, or one of its results became ready
        or failed. Polls while results are computed redraw nothing until one arrives (no flicker)."""
        view = self.nested.index("current")
        needs = self._needs(view)
        for session, kind, args in needs:  # queue what is missing (the worker process computes it)
            if session.status(kind, self.settings, *args) == "missing":
                try:
                    session.get(kind, self.settings, *args, executor=self.app.executor)
                except RuntimeError:
                    pass
        key = (self._view_key(view), self._signature(needs))
        if key != self._last_keys[view]:
            self.app.set_busy(True)
            try:
                self.hosts[view].set_figure(self._figure(view))
            finally:
                self.app.set_busy(False)
            self._last_keys[view] = key
        self.update_entry_tree()
        report_key = self._report_signature()
        if report_key != getattr(self, "_last_report", None):
            self._last_report = report_key
            self.update_report()
        self.app.set_status(self.status_text())
        self._schedule_poll()

    def _schedule_poll(self) -> None:
        if self._poll_after is not None:
            try:
                self.after_cancel(self._poll_after)
            except tk.TclError:
                pass
            self._poll_after = None
        if self._pending():
            self._poll_after = self.after(_POLL_MS, self._poll)

    def _poll(self) -> None:
        self._poll_after = None
        if not self.winfo_exists():
            return
        if self.app.active_tab is self:
            self.render_current()
        elif self._pending():
            self._schedule_poll()

    def status_text(self) -> str:
        busy = [e.package.label for e in self.entries if e.session.busy()]
        failed = [e.package.label for e in self.entries if e.session.errors(self.settings)]
        bits = [self.title, f"{len(self.entries)} entr{'y' if len(self.entries) == 1 else 'ies'}",
                VIEWS[self.nested.index("current")]]
        if busy:
            bits.append("computing: " + ", ".join(busy))
        if failed:
            bits.append(f"{len(failed)} failed (see Details)")
        return "  |  ".join(bits)

    def canonical_figure(self, paper: bool = False) -> Figure:
        """The current view rebuilt for a file: Matplotlib's defaults (the CLI's look), or with ``paper`` the
        paper figures' style and width (spinlabs.figure_style)."""
        if paper:
            with paper_style():
                figure = self._figure(self.nested.index("current"), canonical=True)
                to_paper_layout(figure)
                return figure
        with matplotlib.rc_context(matplotlib.rcParamsDefault):
            return self._figure(self.nested.index("current"), canonical=True)

    def visible_figure(self) -> Figure | None:
        return self.hosts[self.nested.index("current")].figure

    def save_stem(self) -> str:
        return _safe_name(f"{self.title}_{VIEWS[self.nested.index('current')]}")

    # -- report -----------------------------------------------------------------------------------------

    def report(self) -> ReportData:
        columns = [("entry", "Entry", 190), ("spin", "S", 40), ("modes", "Modes", 55), ("direct", "Direct", 70)]
        columns += [(f"t1_{t:g}", f"T1 {t:g} K (s)", 90) for t in QUOTE_T]
        columns += [(f"{c}_{t:g}", f"Δ{c[1]} rate {t:g} K (1/s)", 115) for c in ("d1", "d2") for t in QUOTE_T]
        # only what is already computed: the report never queues work of other views
        peek = lambda kind, *args: {  # noqa: E731
            e.package.key: v for e in self.shown if (v := e.session.peek(kind, self.settings, *args)) is not None
        }
        couplings, rates, t1 = peek("couplings"), peek("rates"), peek("t1", self.t1_state)
        rows = []
        for entry in self.shown:
            key = entry.package.key
            row = {"entry": entry.package.long_label}
            c, r = couplings.get(key), rates.get(key)
            if c is not None:
                row["spin"] = f"{c.spin:g}"
                row["modes"] = str(len(c.frequencies))
            if r is not None:
                row["direct"] = r.direct
                total = r.total(PROCESSES["total"])
                for channel in ("d1", "d2"):
                    series = total[rate_direction(r.spin, channel)]
                    for t in QUOTE_T:
                        row[f"{channel}_{t:g}"] = _at(r.temperatures, series, t)
            if key in t1:
                temps, values, _state = t1[key]
                for t in QUOTE_T:
                    row[f"t1_{t:g}"] = _at(temps, values, t)
            rows.append(row)
        return ReportData(columns, rows, self._details(couplings, rates))

    def _details(self, couplings, rates) -> str:
        s = self.settings
        hidden = len(self.entries) - len(self.shown)
        lines = [f"{self.title}: {len(self.entries)} entries" + (f" ({hidden} hidden)" if hidden else "")
                 + f"; view {VIEWS[self.nested.index('current')]}",
                 f"settings: pairs {s.pairs}, mirror projection {'on' if s.mirror_projection else 'off'}, direct "
                 f"{s.direct}, σ {s.sigma:g} meV, T {s.t_min:g}-{s.t_max:g} K step {s.t_step:g}, "
                 f"B {s.b_field_mt:g} mT (axial)",
                 "rates: Δ1 = m_s S → S−1, Δ2 = S → S−2 (equal to their reverse transitions at these "
                 "temperatures, and without a field to their −S mirrors); T1 from direct + two-phonon (production)"]
        for entry in self.shown:
            key = entry.package.key
            lines.append("")
            lines.append(f"{entry.package.long_label}: {entry.package.path}")
            c = couplings.get(key)
            if c is not None:
                lines.append(f"  {c.convention}; phonon {c.phonon_file}")
            r = rates.get(key)
            if r is not None:
                lines.append(f"  D = {r.zfs_mev * 241798.9242:.1f} MHz, S = {r.spin:g}")
                lines.extend(f"  warning: {w}" for w in r.warnings)
            if entry.session.busy():
                lines.append("  computing…")
            lines.extend(f"  error: {e}" for e in entry.session.errors(self.settings))
        series = self.experiment_series()
        if series:
            shown = {s.key for s in self._shown_experiment()}
            lines += ["", "experiment (Rates: Ω = m_s 1 → 0 with Δ1, γ = 1 → −1 with Δ2; T1: measured T1 / 1/T1 for "
                      "the lowest |m_s|, and T1 of the chosen state solved from each source's Ω and γ):"]
            for s in series:
                span = f"{s.x.min():g}-{s.x.max():g} K" if len(s.x) else "no points"
                lines.append(f"  {'on ' if s.key in shown else 'off'} {s.defect}: {s.legend} ({s.kind}, "
                             f"{len(s.x)} points, {span})")
        for defect, error in self._experiment_errors.items():
            lines.append(f"  experiment of {defect} not read: {error}")
        return "\n".join(lines)

    def update_report(self) -> None:
        self.app.set_report(self.report())

    # -- persistence ------------------------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "key": self.key, "title": self.title, "entries": [str(e.package.path) for e in self.entries],
            "colors": [e.color for e in self.entries], "hidden": [str(e.package.path) for e in self.entries if e.hidden],
            "settings": self.settings.to_dict(), "channels": sorted(self.channels),
            "log": self.log, "processes": sorted(self.processes), "t1_state": self.t1_state, "ratio": self.ratio,
            "map_entry": self.map_entry, "map_key": self.map_key, "show_experiment": self.show_experiment,
            "experiment_off": sorted(self.experiment_off), "experiment_open": sorted(self.experiment_open),
            "entry_groups_closed": sorted(self.entry_groups_closed),
            "view": self.nested.index("current"),
        }

    def apply_state(self, state: dict) -> None:
        self.settings = Settings.from_dict(state.get("settings"))
        self.channels = set(state.get("channels") or self.channels) & set(CHANNELS)
        self.log = bool(state.get("log", True))
        self.processes = set(state.get("processes") or self.processes) & set(PROCESSES)
        self.t1_state = state.get("t1_state") if state.get("t1_state") in T1_STATES else T1_STATES[0]
        self.ratio = bool(state.get("ratio", False))
        self.map_key = state.get("map_key")  # older sessions saved an index: resolved below
        saved_index = int(state.get("map_entry") or 0)
        self.show_experiment = bool(state.get("show_experiment", True))
        if "experiment_off" in state:
            self.experiment_off = {str(k) for k in state.get("experiment_off") or []}
        self.experiment_open = {str(d) for d in state.get("experiment_open") or []}
        self.entry_groups_closed = {str(g) for g in state.get("entry_groups_closed") or []}
        packages = []
        colors = list(state.get("colors") or [])
        for index, path in enumerate(state.get("entries") or []):
            try:
                packages.append((package_for_path(Path(path)), colors[index] if index < len(colors) else None))
            except Exception:  # noqa: BLE001 -- a package that is gone is left out
                continue
        known = {e.package.key for e in self.entries}
        for package, color in packages:
            if package.key not in known:
                used = {e.color for e in self.entries}
                color = color if isinstance(color, str) and color not in used else self._free_color()
                self.entries.append(Entry(package, self.app.production_session(package), color))
                known.add(package.key)
        hidden = {str(p) for p in state.get("hidden") or []}
        for entry in self.entries:
            entry.hidden = str(entry.package.path) in hidden
        if self.map_key not in known and self.entries:
            self.map_key = self.entries[min(saved_index, len(self.entries) - 1)].package.key
        view = int(state.get("view") or 0)
        if 0 <= view < len(VIEWS):
            self.nested.select(view)

    def destroy(self):
        if self._poll_after is not None:
            try:
                self.after_cancel(self._poll_after)
            except tk.TclError:
                pass
        for host in self.hosts:
            if host.figure is not None:
                plt.close(host.figure)
                host.figure = None
        self.controls.destroy()
        super().destroy()


def _at(temps, values, t: float) -> str:
    import numpy as np

    idx = np.flatnonzero(np.isclose(temps, t))
    return _fmt(float(values[idx[0]])) if len(idx) else "—"
