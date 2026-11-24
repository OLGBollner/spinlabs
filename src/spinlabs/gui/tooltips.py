"""Hover help for the convergence viewer's option controls."""

from __future__ import annotations

import tkinter as tk

from spinlabs.gui.theme import Palette


class Tooltip:
    """A small dark hover label, shown after a short deliberate pause."""

    def __init__(
        self,
        widget: tk.Misc,
        text: str,
        *,
        palette: Palette,
        family: str,
        delay: int = 450,
        wraplength: int = 340,
    ):
        self.widget = widget
        self.text = text
        self.palette = palette
        self.family = family
        self.delay = delay
        self.wraplength = wraplength
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None or not self.widget.winfo_exists():
            return
        try:
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 7
        except tk.TclError:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        try:
            tip.attributes("-topmost", True)
        except tk.TclError:
            pass
        label = tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            background="#1c1c1c",
            foreground="#f8fafc",
            padx=10,
            pady=7,
            wraplength=self.wraplength,
            font=(self.family, 9),
        )
        label.pack()
        tip.wm_geometry(f"+{x}+{y}")
        self._tip = tip

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None
