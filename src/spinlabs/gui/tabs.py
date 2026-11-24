"""A small flat tab bar with a close button on every tab.

ttk.Notebook cannot put a close control inside its tab, so the pack tabs use
this lightweight widget instead.  Each tab is a text label plus an ``×`` that
turns red when the tab or the close control is highlighted.
"""

from __future__ import annotations

from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk

from spinlabs.gui.theme import Palette


@dataclass
class _Tab:
    frame: ttk.Frame
    cell: tk.Frame
    inner: tk.Frame
    label: tk.Label
    close: tk.Label
    line: tk.Frame
    title: str


class TabView(ttk.Frame):
    """Tab bar plus a page area, with a close button on each tab."""

    def __init__(
        self,
        master,
        *,
        palette: Palette,
        family: str,
        on_change=None,
        on_close=None,
    ):
        super().__init__(master, style="Card.TFrame")
        self.palette = palette
        self.family = family
        self.on_change = on_change
        self.on_close = on_close
        self.tabs_list: list[_Tab] = []
        self.current: ttk.Frame | None = None

        self.tabbar = tk.Frame(self, background=palette.card)
        self.tabbar.pack(side=tk.TOP, fill=tk.X, padx=(8, 8), pady=(6, 0))
        tk.Frame(self, background=palette.border, height=1).pack(side=tk.TOP, fill=tk.X)
        self.body = ttk.Frame(self, style="Card.TFrame")
        self.body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # -- public API used by the application -------------------------------

    def add(self, frame: ttk.Frame, text: str) -> None:
        cell = tk.Frame(self.tabbar, background=self.palette.card)
        inner = tk.Frame(cell, background=self.palette.card)
        inner.pack(fill=tk.X)
        label = tk.Label(
            inner, text=text, background=self.palette.card, foreground=self.palette.text,
            font=(self.family, 10), padx=2, cursor="hand2",
        )
        close = tk.Label(
            inner, text="×", background=self.palette.card, foreground=self.palette.muted,
            font=(self.family, 12), padx=4, cursor="hand2",
        )
        label.pack(side=tk.LEFT, padx=(10, 3), pady=(6, 5))
        close.pack(side=tk.LEFT, padx=(0, 8), pady=(4, 3))
        line = tk.Frame(cell, background=self.palette.card, height=2)
        line.pack(fill=tk.X)
        cell.pack(side=tk.LEFT, padx=(0, 2))

        entry = _Tab(frame, cell, inner, label, close, line, text)
        self.tabs_list.append(entry)
        for widget in (cell, inner, label):
            widget.bind("<Button-1>", lambda _e, f=frame: self.select(f))
        close.bind("<Button-1>", lambda e, f=frame: self._close_event(e, f))
        # Only the close glyph itself reacts; hovering the title must not colour it.
        close.bind("<Enter>", lambda _e, tab=entry: self._set_close_colour(tab, True))
        close.bind("<Leave>", lambda _e, tab=entry: self._set_close_colour(tab, False))
        self._style(entry, False)
        if self.current is None:
            self.select(frame)
        else:
            frame.pack_forget()

    def select(self, frame: ttk.Frame | None = None):
        """Select a page (or return the current page when called with no argument)."""
        if frame is None:
            return self.current
        entry = next((tab for tab in self.tabs_list if tab.frame is frame), None)
        if entry is None:
            return
        for tab in self.tabs_list:
            if tab.frame is frame:
                tab.frame.pack(fill=tk.BOTH, expand=True)
                self._style(tab, True)
            else:
                tab.frame.pack_forget()
                self._style(tab, False)
        self.current = frame
        if self.on_change is not None:
            self.on_change()

    def forget(self, frame: ttk.Frame) -> None:
        """Remove a tab and destroy its tab cell; the page itself is left to the caller."""
        entry = next((tab for tab in self.tabs_list if tab.frame is frame), None)
        if entry is None:
            return
        was_current = self.current is frame
        index = self.tabs_list.index(entry)
        self.tabs_list.remove(entry)
        entry.cell.destroy()
        frame.pack_forget()
        if was_current:
            self.current = None
            if self.tabs_list:
                self.select(self.tabs_list[min(index, len(self.tabs_list) - 1)].frame)

    def tabs(self) -> list[ttk.Frame]:
        return [tab.frame for tab in self.tabs_list]

    # -- internals ---------------------------------------------------------

    def _close_event(self, _event, frame: ttk.Frame):
        if self.on_close is not None:
            self.on_close(frame)
        return "break"

    def _set_close_colour(self, tab: _Tab, highlighted: bool) -> None:
        colour = self.palette.danger if highlighted else self.palette.muted
        try:
            tab.close.configure(foreground=colour)
        except tk.TclError:
            pass

    def _style(self, tab: _Tab, active: bool) -> None:
        background = self.palette.accent_soft if active else self.palette.card
        foreground = self.palette.accent if active else self.palette.text
        try:
            tab.cell.configure(background=background)
            tab.inner.configure(background=background)
            tab.label.configure(
                background=background, foreground=foreground,
                font=(self.family, 10, "bold" if active else "normal"),
            )
            tab.close.configure(background=background, foreground=self.palette.muted)
            tab.line.configure(background=self.palette.accent if active else self.palette.card)
        except tk.TclError:
            pass
