"""The paper's figure style: fonts, line widths, column widths and panel letters.

One place for the style of the paper figures (scripts/paper, through its config) and of the figures the GUI
exports in paper style, so the two cannot drift apart.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib

SINGLE, DOUBLE = 3.4, 7.0  # inch: one and two journal columns
FONT = 8  # pt

PAPER_RC = {
    "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT, "legend.fontsize": FONT - 1,
    "xtick.labelsize": FONT - 1, "ytick.labelsize": FONT - 1, "lines.linewidth": 1.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
}


def apply_paper_style() -> None:
    """Set the paper style for the rest of the process (the paper scripts)."""
    matplotlib.rcParams.update(PAPER_RC)


@contextmanager
def paper_style():
    """Matplotlib's defaults with the paper style on top, for building one figure."""
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        with matplotlib.rc_context(PAPER_RC):
            yield


def panel_letter(ax, k: int, x: float = -0.02, y: float = 1.02, **kw) -> None:
    """(a), (b), ... above the top-left corner of the axes."""
    ax.text(x, y, f"({'abcdefghijklmnopqrstuvwxyz'[k]})", transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", **kw)


def panels(fig) -> list:
    """The figure's panels in reading order (rows, then columns): the axes on its grid, not colour bars or
    twin axes."""
    found, seen = [], set()
    for ax in fig.axes:
        spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
        if spec is None or ax.get_label() == "<colorbar>":
            continue
        key = (spec.rowspan.start, spec.colspan.start)
        if key in seen:  # a twin axis shares its panel's grid cell
            continue
        seen.add(key)
        found.append((key, ax))
    return [ax for _key, ax in sorted(found, key=lambda item: item[0])]


SINGLE_MIN_ASPECT = 0.8  # height / width of a one-column figure at least (a wide screen layout is too flat)
LEGEND_INSIDE_MAX = 4  # entries a one-column panel's legend may have inside the axes; more go below them


def to_paper_layout(fig, *, letters: bool = True) -> None:
    """Size a figure built under :func:`paper_style` as the paper's: one journal column for a single column
    of panels (at least SINGLE_MIN_ASPECT tall, a long legend moved below the axes), two for more, its
    aspect ratio kept; (a), (b), ... on the panels when there are several."""
    axes = panels(fig)
    columns = max((ax.get_subplotspec().get_gridspec().ncols for ax in axes), default=1)
    width = SINGLE if columns == 1 else DOUBLE
    old_width, old_height = fig.get_size_inches()
    aspect = old_height / old_width
    if columns == 1:
        aspect = max(aspect, SINGLE_MIN_ASPECT)
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None and len(legend.texts) > LEGEND_INSIDE_MAX:
                handles, labels = legend.legend_handles, [t.get_text() for t in legend.texts]
                ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1,
                          frameon=False, fontsize=FONT - 1)
    fig.set_size_inches(width, width * aspect)
    if letters and len(axes) > 1:
        for k, ax in enumerate(axes):
            panel_letter(ax, k)
