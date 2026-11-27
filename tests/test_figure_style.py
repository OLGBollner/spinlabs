"""The paper's figure style, shared by scripts/paper and the GUI's paper-style figure export."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from spinlabs.figure_style import (  # noqa: E402
    DOUBLE,
    FONT,
    LEGEND_INSIDE_MAX,
    SINGLE,
    SINGLE_MIN_ASPECT,
    panels,
    paper_style,
    to_paper_layout,
)


def test_paper_style_is_scoped_to_the_figure_it_builds():
    before = matplotlib.rcParams["font.size"]
    with paper_style():
        assert matplotlib.rcParams["font.size"] == FONT and matplotlib.rcParams["pdf.fonttype"] == 42
        assert not matplotlib.rcParams["legend.frameon"]
    assert matplotlib.rcParams["font.size"] == before


def test_several_columns_take_two_journal_columns_with_panel_letters():
    with paper_style():
        fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), layout="constrained")
        axes[0, 0].twinx()  # a twin axis is not a panel of its own
        to_paper_layout(fig)
    assert fig.get_size_inches() == pytest.approx((DOUBLE, DOUBLE * 7.2 / 12.6))
    assert len(panels(fig)) == 6 and panels(fig)[1] is axes[0, 1]  # reading order: rows, then columns
    letters = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert letters == ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    plt.close(fig)


def test_one_panel_takes_one_column_and_a_long_legend_goes_below_it():
    with paper_style():
        fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
        for k in range(LEGEND_INSIDE_MAX + 1):
            ax.plot([0, 1], [k, k], label=f"series {k}")
        ax.legend()
        to_paper_layout(fig)
    width, height = fig.get_size_inches()
    assert width == SINGLE and height == pytest.approx(SINGLE * SINGLE_MIN_ASPECT)  # not as flat as the screen
    assert not any(t.get_text().startswith("(") for t in ax.texts)  # no letter on a single panel
    legend = ax.get_legend()
    assert len(legend.texts) == LEGEND_INSIDE_MAX + 1 and legend.get_bbox_to_anchor().y0 < ax.bbox.y0
    plt.close(fig)
