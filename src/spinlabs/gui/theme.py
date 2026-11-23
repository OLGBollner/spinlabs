"""Visual theme for the Tk convergence viewer.

Tk's stock ``clam`` theme is functional but looks dated.  This module applies a
flat, light "card" theme with a dark header, a single blue accent, and clear
type hierarchy.  It is intentionally dependency-free: no extra pip package is
needed, and the style names are used by ``spinlabs.gui.app``.
"""

from __future__ import annotations

from dataclasses import dataclass

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Palette:
    background: str = "#f5f7fa"
    card: str = "#ffffff"
    border: str = "#d9e0e8"
    text: str = "#1f2937"
    muted: str = "#667085"
    accent: str = "#2563eb"
    accent_hover: str = "#1d4ed8"
    accent_soft: str = "#e8f0fe"
    accent_soft_hover: str = "#dbeafe"
    header_bg: str = "#0f172a"
    header_text: str = "#f8fafc"
    header_muted: str = "#94a3b8"
    selected: str = "#dbeafe"
    row_alt: str = "#f8fafc"
    danger: str = "#b42318"


PALETTE = Palette()


def _pick_family(root: tk.Misc) -> str:
    available = set(tkfont.families(root))
    for family in ("Segoe UI", "Inter", "SF Pro Text", "Ubuntu", "Cantarell", "DejaVu Sans", "Helvetica"):
        if family in available:
            return family
    return "TkDefaultFont"


def _set_named_font(root: tk.Misc, name: str, family: str, size: int, weight: str = "normal") -> None:
    try:
        font = tkfont.nametofont(name)
    except tk.TclError:
        return
    font.configure(family=family, size=size, weight=weight)


def apply_theme(root: tk.Misc) -> Palette:
    """Apply the light theme and return its palette."""
    palette = PALETTE
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    root.configure(background=palette.background)
    family = _pick_family(root)
    _set_named_font(root, "TkDefaultFont", family, 10)
    _set_named_font(root, "TkTextFont", family, 10)
    _set_named_font(root, "TkMenuFont", family, 10)
    _set_named_font(root, "TkHeadingFont", family, 9, "bold")
    _set_named_font(root, "TkCaptionFont", family, 11, "bold")
    _set_named_font(root, "TkSmallCaptionFont", family, 9)
    _set_named_font(root, "TkFixedFont", "DejaVu Sans Mono", 9)
    root.option_add("*tearOff", False)
    root.option_add("*TCombobox*Listbox.background", palette.card)
    root.option_add("*TCombobox*Listbox.foreground", palette.text)
    root.option_add("*TCombobox*Listbox.selectBackground", palette.accent_soft)
    root.option_add("*TCombobox*Listbox.selectForeground", palette.text)
    root.option_add("*Menu.background", palette.card)
    root.option_add("*Menu.foreground", palette.text)
    root.option_add("*Menu.activeBackground", palette.accent_soft)
    root.option_add("*Menu.activeForeground", palette.text)
    root.option_add("*Menu.borderWidth", 0)

    # Base
    style.configure(
        ".", background=palette.background, foreground=palette.text, font=(family, 10), borderwidth=0,
    )
    style.configure("TFrame", background=palette.background)
    style.configure("Card.TFrame", background=palette.card)
    style.configure("Toolbar.TFrame", background=palette.card)
    style.configure("Status.TFrame", background=palette.background)
    style.configure("Header.TFrame", background=palette.header_bg)
    style.configure("TLabel", background=palette.card, foreground=palette.text)
    style.configure("Card.TLabel", background=palette.card, foreground=palette.text)
    style.configure("Muted.TLabel", background=palette.background, foreground=palette.muted)
    style.configure("CardMuted.TLabel", background=palette.card, foreground=palette.muted, font=(family, 9))
    style.configure("Status.TLabel", background=palette.background, foreground=palette.muted, font=(family, 9))
    style.configure("HeaderTitle.TLabel", background=palette.header_bg, foreground=palette.header_text,
                    font=(family, 18, "bold"))
    style.configure("HeaderSubtitle.TLabel", background=palette.header_bg, foreground=palette.header_muted,
                    font=(family, 9))
    style.configure("Section.TLabel", background=palette.card, foreground=palette.muted,
                    font=(family, 9, "bold"))

    # Cards
    style.configure(
        "TLabelframe", background=palette.card, bordercolor=palette.border, relief="solid", borderwidth=1,
        padding=8,
    )
    style.configure("TLabelframe.Label", background=palette.card, foreground=palette.muted,
                    font=(family, 9, "bold"))
    style.configure("Card.TLabelframe", background=palette.card, bordercolor=palette.border,
                    relief="solid", borderwidth=1, padding=10)
    style.configure("Card.TLabelframe.Label", background=palette.card, foreground=palette.muted,
                    font=(family, 9, "bold"))

    # Buttons
    style.configure(
        "TButton", background=palette.card, foreground=palette.text, bordercolor=palette.border,
        focuscolor=palette.accent, relief="flat", padding=(10, 5), font=(family, 10),
    )
    style.map(
        "TButton",
        background=[("active", palette.accent_soft), ("pressed", palette.accent_soft_hover)],
        bordercolor=[("focus", palette.accent)],
    )
    style.configure(
        "Accent.TButton", background=palette.accent, foreground="#ffffff", bordercolor=palette.accent,
        relief="flat", padding=(12, 6), font=(family, 10, "bold"),
    )
    style.map(
        "Accent.TButton",
        background=[("active", palette.accent_hover), ("pressed", "#1e40af"), ("disabled", "#93c5fd")],
        foreground=[("disabled", "#eff6ff")],
    )
    style.configure(
        "Toolbar.TButton", background=palette.card, foreground=palette.text, bordercolor=palette.card,
        relief="flat", padding=(8, 4), font=(family, 9),
    )
    style.map(
        "Toolbar.TButton",
        background=[("active", palette.accent_soft), ("pressed", palette.accent_soft_hover)],
        bordercolor=[("active", palette.accent_soft)],
    )
    style.configure(
        "Ghost.TButton", background=palette.card, foreground=palette.muted, bordercolor=palette.card,
        relief="flat", padding=(6, 3), font=(family, 9),
    )
    style.map("Ghost.TButton", background=[("active", palette.accent_soft)],
              foreground=[("active", palette.accent)])

    # Notebooks
    style.configure("TNotebook", background=palette.card, borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("TNotebook.Tab", background=palette.card, foreground=palette.muted,
                    padding=(14, 7), font=(family, 10), borderwidth=0)
    style.map(
        "TNotebook.Tab",
        background=[("selected", palette.accent_soft)],
        foreground=[("selected", palette.accent)],
        expand=[("selected", (0, 0, 0, 2))],
    )
    style.configure("Main.TNotebook", background=palette.card, borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("Main.TNotebook.Tab", padding=(16, 8), font=(family, 10))
    style.configure("Side.TNotebook", background=palette.card, borderwidth=0, tabmargins=(0, 2, 0, 0))
    style.configure("Side.TNotebook.Tab", padding=(8, 5), font=(family, 9))
    style.map(
        "Side.TNotebook.Tab",
        background=[("selected", palette.accent_soft)],
        foreground=[("selected", palette.accent)],
        expand=[("selected", (0, 0, 0, 1))],
    )
    style.configure("Sub.TNotebook", background=palette.card, borderwidth=0, tabmargins=(0, 2, 0, 0))
    style.configure("Sub.TNotebook.Tab", padding=(12, 6), font=(family, 9))
    style.map(
        "Sub.TNotebook.Tab",
        background=[("selected", palette.accent_soft)],
        foreground=[("selected", palette.accent)],
    )

    # Data views
    style.configure(
        "Treeview", background=palette.card, fieldbackground=palette.card, foreground=palette.text,
        rowheight=26, borderwidth=0, font=(family, 9),
    )
    style.configure(
        "Treeview.Heading", background=palette.accent_soft, foreground=palette.muted, relief="flat",
        borderwidth=0, padding=(6, 5), font=(family, 9, "bold"),
    )
    style.map(
        "Treeview",
        background=[("selected", palette.selected)],
        foreground=[("selected", palette.text)],
    )
    style.map("Treeview.Heading", background=[("active", palette.accent_soft_hover)])
    style.configure("TEntry", fieldbackground=palette.card, foreground=palette.text,
                    bordercolor=palette.border, lightcolor=palette.border, darkcolor=palette.border,
                    padding=(8, 5), insertcolor=palette.text)
    style.configure(
        "TCombobox", fieldbackground=palette.card, background=palette.card, foreground=palette.text,
        bordercolor=palette.border, lightcolor=palette.border, darkcolor=palette.border,
        arrowcolor=palette.muted, padding=(6, 4),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette.card), ("disabled", palette.background)],
        foreground=[("readonly", palette.text), ("disabled", palette.muted)],
    )
    style.configure("TCheckbutton", background=palette.card, foreground=palette.text,
                    padding=(2, 3), font=(family, 10), focuscolor=palette.accent)
    style.configure("Card.TCheckbutton", background=palette.card, foreground=palette.text,
                    padding=(2, 3), font=(family, 10), focuscolor=palette.accent)
    style.configure("TRadiobutton", background=palette.card, foreground=palette.text,
                    padding=(2, 3), font=(family, 10), focuscolor=palette.accent)
    style.map("TCheckbutton", background=[("active", palette.card)])
    style.map("Card.TCheckbutton", background=[("active", palette.card)])
    style.configure("TScrollbar", background=palette.card, troughcolor=palette.card,
                    bordercolor=palette.card, arrowcolor=palette.muted, relief="flat", arrowsize=14)
    style.configure("Vertical.TScrollbar", background=palette.card, troughcolor=palette.card,
                    bordercolor=palette.card, arrowcolor=palette.muted)
    style.configure("Horizontal.TScrollbar", background=palette.card, troughcolor=palette.card,
                    bordercolor=palette.card, arrowcolor=palette.muted)
    style.configure("TSeparator", background=palette.border)

    style.configure("TProgressbar", background=palette.accent, troughcolor=palette.background, borderwidth=0)
    return palette


def font_family(root: tk.Misc) -> str:
    """The selected UI font family (also useful for the header)."""
    return _pick_family(root)


def style_matplotlib() -> None:
    """Give the embedded figures the same typographic and surface language."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d9e0e8",
        "axes.labelcolor": "#1f2937",
        "text.color": "#1f2937",
        "xtick.color": "#667085",
        "ytick.color": "#667085",
        "grid.color": "#e8edf3",
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.4,
    })


def style_navigation_toolbar(toolbar: tk.Misc, background: str = "#ffffff") -> None:
    """Make Matplotlib's stock Tk toolbar blend into the card surface."""
    try:
        toolbar.configure(background=background, borderwidth=0, highlightthickness=0)
    except tk.TclError:
        pass
    for child in toolbar.winfo_children():
        try:
            if isinstance(child, (tk.Button, ttk.Button)):
                child.configure(
                    background=background, activebackground="#e8f0fe", relief="flat", borderwidth=0,
                    highlightthickness=0, padx=4, pady=2,
                )
            elif isinstance(child, tk.Frame):
                child.configure(background=background, borderwidth=0, highlightthickness=0)
            elif isinstance(child, tk.Label):
                child.configure(background=background, foreground="#667085")
        except tk.TclError:
            continue
