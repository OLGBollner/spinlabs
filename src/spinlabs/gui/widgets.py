"""Small reusable Tk widgets for the convergence GUI: the scrollable sidebar frame, the figure host and the
report table data shared by the pack and comparison tabs."""

from __future__ import annotations

from dataclasses import dataclass
import re
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from spinlabs.gui import theme
from spinlabs.gui.theme import Palette


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
        old_figure = self.figure
        old_canvas = self._canvas_widget
        old_toolbar = self._toolbar

        # Prepare the replacement before touching the visible canvas: the old
        # plot stays on screen until the new FigureCanvas has been drawn.
        self._apply_figure_size(figure, forward=False)
        canvas = FigureCanvasTkAgg(figure, master=self.view)
        widget = canvas.get_tk_widget()
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.toolbar_frame, pack_toolbar=False)
        toolbar.update()
        theme.style_navigation_toolbar(toolbar)

        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()
        if old_toolbar is not None:
            old_toolbar.destroy()
        widget.pack(fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.figure = figure
        self._canvas_widget = canvas
        self._toolbar = toolbar
        self._schedule_fit()
        if old_figure is not None and old_figure is not figure:
            plt.close(old_figure)


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable frame with a themed scrollbar."""

    def __init__(self, master, *, palette: Palette, padding: tuple[int, int] = (8, 8)):
        super().__init__(master, style="Card.TFrame")
        self.palette = palette
        self.canvas = tk.Canvas(
            self, background=palette.background, highlightthickness=0, borderwidth=0,
        )
        self.vbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.content = ttk.Frame(self.canvas, style="Card.TFrame", padding=padding)
        self._window = self.canvas.create_window((0, 0), anchor="nw", window=self.content)
        self.content.bind("<Configure>", self._update_region)
        self.canvas.bind("<Configure>", self._resize_content)
        # Wheel over any part of this pane scrolls the pane, not the whole app.
        self.bind_all("<MouseWheel>", self._on_wheel, add="+")
        self.bind_all("<Button-4>", self._on_wheel, add="+")
        self.bind_all("<Button-5>", self._on_wheel, add="+")

    def refresh(self) -> None:
        self.content.update_idletasks()
        self._update_region()

    def _update_region(self, _event=None) -> None:
        try:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except tk.TclError:
            pass

    def _resize_content(self, event) -> None:
        try:
            self.canvas.itemconfigure(self._window, width=event.width)
        except tk.TclError:
            pass

    def _on_wheel(self, event) -> None:
        # A bounds test avoids winfo_containing(): Tk's file dialogs are
        # internal toplevels whose widget names are not in Tkinter's cache and
        # nametowidget() raises KeyError when the wheel is used over them.
        try:
            left, top = self.winfo_rootx(), self.winfo_rooty()
            right, bottom = left + self.winfo_width(), top + self.winfo_height()
        except tk.TclError:
            return
        if not (left <= event.x_root < right and top <= event.y_root < bottom):
            return
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(1, "units")
        elif getattr(event, "delta", 0):
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
