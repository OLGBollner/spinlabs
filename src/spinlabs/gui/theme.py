"""Modern Tk appearance for the convergence viewer.

The base theme is Sun Valley (``sv-ttk``), which provides modern flat buttons,
tabs, treeviews, comboboxes and scrollbars.  The small custom layer here only
adds the dark header, light cards, typography and status surfaces.  If
``sv-ttk`` is unavailable the previous dependency-free clam theme is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import matplotlib.pyplot as plt

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)

try:  # optional at runtime so headless CLI use never needs the GUI theme
    import sv_ttk
except ImportError:  # pragma: no cover - exercised only on installations without the GUI extra
    sv_ttk = None


@dataclass(frozen=True)
class Palette:
    background: str = "#fafafa"
    card: str = "#ffffff"
    border: str = "#e5e7eb"
    text: str = "#1c1c1c"
    muted: str = "#6b7280"
    accent: str = "#005fb8"
    accent_hover: str = "#0a4f9e"
    accent_soft: str = "#e8f1fb"
    accent_soft_hover: str = "#dbeafe"
    header_bg: str = "#0b1220"
    header_text: str = "#f8fafc"
    header_muted: str = "#94a3b8"
    selected: str = "#cfe4fb"
    row_alt: str = "#f8fafc"
    danger: str = "#d92d20"


PALETTE = Palette()


def _pick_family(root: tk.Misc) -> str:
    available = set(tkfont.families(root))
    for family in ("Segoe UI Variable", "Segoe UI", "Inter", "SF Pro Text", "Ubuntu", "Cantarell", "DejaVu Sans"):
        if family in available:
            return family
    return "TkDefaultFont"


def _set_named_font(root: tk.Misc, name: str, family: str, size: int, weight: str = "normal") -> None:
    try:
        font = tkfont.nametofont(name)
    except tk.TclError:
        return
    font.configure(family=family, size=size, weight=weight)


def _set_fonts(root: tk.Misc, family: str) -> None:
    _set_named_font(root, "TkDefaultFont", family, 10)
    _set_named_font(root, "TkTextFont", family, 10)
    _set_named_font(root, "TkMenuFont", family, 10)
    _set_named_font(root, "TkHeadingFont", family, 9, "bold")
    _set_named_font(root, "TkCaptionFont", family, 11, "bold")
    _set_named_font(root, "TkSmallCaptionFont", family, 9)
    _set_named_font(root, "TkFixedFont", "DejaVu Sans Mono", 9)


def font_family(root: tk.Misc) -> str:
    """The selected UI font family (also used by the header)."""
    return _pick_family(root)


def _option_colours(root: tk.Misc, palette: Palette) -> None:
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


def _apply_modern(style: ttk.Style, family: str, palette: Palette) -> None:
    """Only the layers the app owns; Sun Valley owns the core controls."""
    style.configure("Card.TFrame", background=palette.card)
    style.configure("Toolbar.TFrame", background=palette.card)
    style.configure("Status.TFrame", background=palette.background)
    style.configure("Card.TLabel", background=palette.card, foreground=palette.text)
    style.configure("CardMuted.TLabel", background=palette.card, foreground=palette.muted, font=(family, 9))
    style.configure("Status.TLabel", background=palette.background, foreground=palette.muted, font=(family, 9))


def _apply_fallback(style: ttk.Style, family: str, palette: Palette) -> None:
    """Dependency-free approximation when sv-ttk is not installed."""
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(".", background=palette.background, foreground=palette.text, font=(family, 10), borderwidth=0)
    style.configure("TFrame", background=palette.background)
    style.configure("Card.TFrame", background=palette.card)
    style.configure("Toolbar.TFrame", background=palette.card)
    style.configure("Status.TFrame", background=palette.background)
    style.configure("TLabel", background=palette.card, foreground=palette.text)
    style.configure("Card.TLabel", background=palette.card, foreground=palette.text)
    style.configure("CardMuted.TLabel", background=palette.card, foreground=palette.muted, font=(family, 9))
    style.configure("Status.TLabel", background=palette.background, foreground=palette.muted, font=(family, 9))
    style.configure(
        "TLabelframe", background=palette.card, bordercolor=palette.border, relief="solid", borderwidth=1,
        padding=8,
    )
    style.configure("TLabelframe.Label", background=palette.card, foreground=palette.muted,
                    font=(family, 9, "bold"))
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
        background=[("active", palette.accent_hover), ("pressed", "#083d78"), ("disabled", "#93c5fd")],
        foreground=[("disabled", "#eff6ff")],
    )
    style.configure("TNotebook", background=palette.card, borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("TNotebook.Tab", background=palette.card, foreground=palette.muted,
                    padding=(14, 7), font=(family, 10), borderwidth=0)
    style.map(
        "TNotebook.Tab",
        background=[("selected", palette.accent_soft)],
        foreground=[("selected", palette.accent)],
        expand=[("selected", (0, 0, 0, 2))],
    )
    style.configure(
        "Treeview", background=palette.card, fieldbackground=palette.card, foreground=palette.text,
        rowheight=28, borderwidth=0, font=(family, 9),
    )
    style.configure(
        "Treeview.Heading", background=palette.accent_soft, foreground=palette.muted, relief="flat",
        borderwidth=0, padding=(6, 5), font=(family, 9, "bold"),
    )
    style.map("Treeview", background=[("selected", palette.selected)], foreground=[("selected", palette.text)])
    style.map("Treeview.Heading", background=[("active", palette.accent_soft_hover)])
    style.configure("TEntry", fieldbackground=palette.card, foreground=palette.text, bordercolor=palette.border,
                    lightcolor=palette.border, darkcolor=palette.border, padding=(8, 5), insertcolor=palette.text)
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
    style.configure("TCheckbutton", background=palette.card, foreground=palette.text, padding=(2, 3),
                    font=(family, 10), focuscolor=palette.accent)
    style.map("TCheckbutton", background=[("active", palette.card)])
    style.configure("TScrollbar", background=palette.card, troughcolor=palette.card, bordercolor=palette.card,
                    arrowcolor=palette.muted, relief="flat", arrowsize=14)
    style.configure("Vertical.TScrollbar", background=palette.card, troughcolor=palette.card,
                    bordercolor=palette.card, arrowcolor=palette.muted)
    style.configure("Horizontal.TScrollbar", background=palette.card, troughcolor=palette.card,
                    bordercolor=palette.card, arrowcolor=palette.muted)
    style.configure("TSeparator", background=palette.border)


def apply_theme(root: tk.Misc) -> Palette:
    """Apply the modern theme (Sun Valley when available) and return its palette."""
    palette = PALETTE
    family = _pick_family(root)
    _set_fonts(root, family)
    root.configure(background=palette.background)
    _option_colours(root, palette)

    style = ttk.Style(root)
    modern = False
    if sv_ttk is not None:
        try:
            sv_ttk.set_theme("light", root)
            modern = True
        except Exception:  # a broken Tk theme package must not stop the app
            modern = False
    if modern:
        _apply_modern(style, family, palette)
    else:
        LOGGER.warning("sv-ttk is not available; using the fallback Tk theme (pip install 'sv-ttk')")
        _apply_fallback(style, family, palette)
    return palette


def style_matplotlib() -> None:
    """Give embedded figures the same typographic and surface language."""
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
        "axes.edgecolor": "#e5e7eb",
        "axes.labelcolor": "#1c1c1c",
        "text.color": "#1c1c1c",
        "xtick.color": "#6b7280",
        "ytick.color": "#6b7280",
        "grid.color": "#eef2f6",
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
                    background=background, activebackground="#e8f1fb", relief="flat", borderwidth=0,
                    highlightthickness=0, padx=4, pady=2,
                )
            elif isinstance(child, tk.Frame):
                child.configure(background=background, borderwidth=0, highlightthickness=0)
            elif isinstance(child, tk.Label):
                child.configure(background=background, foreground="#6b7280")
        except tk.TclError:
            continue
