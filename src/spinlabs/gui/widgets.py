"""Small reusable Tk widgets for the convergence GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from spinlabs.gui.theme import Palette


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
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except (tk.TclError, AttributeError):
            return
        if widget is None:
            return
        # Only scroll when the pointer is inside this pane.
        path = str(widget)
        if path == str(self) or path.startswith(str(self) + "."):
            if getattr(event, "num", None) == 4:
                self.canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                self.canvas.yview_scroll(1, "units")
            elif getattr(event, "delta", 0):
                self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
