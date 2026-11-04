"""Reading digitized experimental data (WebPlotDigitizer CSV export)."""

import numpy as np
import pytest

from spinlabs.experimental import ExperimentalSeries, load_experimental_data, read_experimental_csv

# The layout of experimental_data/Jarmola_2012.csv: shared units, three ragged, unsorted series.
JARMOLA_LIKE = """K,Hz
S8,,S3,,S2,
X,Y,X,Y,X,Y
59.98,0.106,6.556,0.00744,5.095,18.58
80.01,0.320,9.945,0.00513,10,20.14
10,0.0468,20.06,0.0101,19.95,21.83
400.3,539.1,395.9,464.1,,
325.0,249.8,,,325.0,398.9
"""


def _write(tmp_path, text, name="Jarmola_2012.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_reads_ragged_unsorted_series_with_shared_units(tmp_path):
    series = read_experimental_csv(_write(tmp_path, JARMOLA_LIKE))
    assert [s.label for s in series] == ["S8", "S3", "S2"]
    assert all(s.source == "Jarmola_2012" and s.x_unit == "K" and s.y_unit == "Hz" for s in series)
    s8, s3, s2 = series
    np.testing.assert_allclose(s8.x, [10, 59.98, 80.01, 325.0, 400.3])  # sorted by x
    np.testing.assert_allclose(s8.y, [0.0468, 0.106, 0.320, 249.8, 539.1])  # y follows its x
    assert len(s3) == 4  # empty cells of the last row skipped
    np.testing.assert_allclose(s2.x, [5.095, 10, 19.95, 325.0])
    np.testing.assert_allclose(s2.y, [18.58, 20.14, 21.83, 398.9])


def test_units_per_series_and_unnamed_series(tmp_path):
    text = "K,Hz,K,ms\nA,,,\nX,Y,X,Y\n1,2,3,4\n"
    a, b = read_experimental_csv(_write(tmp_path, text, "Someone_2020.csv"))
    assert (a.label, a.x_unit, a.y_unit) == ("A", "K", "Hz")
    assert (b.label, b.x_unit, b.y_unit) == ("series 2", "K", "ms")


def test_unit_conversion_within_a_dimension():
    s = ExperimentalSeries("S", np.array([300.0]), np.array([2.5]), "K", "ms", "X_2000")
    assert s.to_units(y_unit="s").y[0] == pytest.approx(2.5e-3)
    assert s.to_units(y_unit="s").y_unit == "s"
    rate = ExperimentalSeries("S", np.array([300.0]), np.array([2.0]), "K", "kHz", "X_2000")
    assert rate.to_units(y_unit="Hz").y[0] == pytest.approx(2000.0)
    with pytest.raises(ValueError, match="cannot convert"):
        rate.to_units(y_unit="K")  # a rate is not a temperature
    with pytest.raises(ValueError, match="unknown unit"):
        rate.to_units(y_unit="furlong")


def test_time_and_rate_convert_by_inverting():
    """T1 = 0.34 ms (Toyli et al. 2012 at 600 K) is 1/T1 = 2941 Hz, and back."""
    t1 = ExperimentalSeries("Toyli", np.array([600.0]), np.array([0.340]), "K", "ms", "Toyli_2012")
    rate = t1.to_units(y_unit="Hz")
    assert rate.y_unit == "Hz" and rate.y[0] == pytest.approx(1 / 0.340e-3)
    assert rate.to_units(y_unit="kHz").y[0] == pytest.approx(1 / 0.340)
    assert rate.to_units(y_unit="ms").y[0] == pytest.approx(0.340)
    s_rate = ExperimentalSeries("S", np.array([10.0, 300.0]), np.array([0.05, 200.0]), "K", "Hz", "X")
    np.testing.assert_allclose(s_rate.to_units(y_unit="s").y, [20.0, 5e-3])


def test_inverting_x_keeps_the_series_sorted_by_x():
    s = ExperimentalSeries("S", np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0]), "ms", "K", "X")
    hz = s.to_units(x_unit="Hz")
    np.testing.assert_allclose(hz.x, [250.0, 500.0, 1000.0])
    np.testing.assert_allclose(hz.y, [3.0, 2.0, 1.0])  # each y stays with its x


@pytest.mark.parametrize(
    "text, message",
    [
        ("K,Hz\nS8,\n", "header row"),
        ("K,Hz\nS8,\nX,Z\n1,2\n", "X,Y pairs"),
        ("K\nS8,\nX,Y\n1,2\n", "unit"),
        ("K,Hz\nS8,\nX,Y\n1,\n", "only one of X and Y"),
        ("K,Hz\nS8,\nX,Y\n1,abc\n", "not a number"),
        ("K,Hz\nS8,\nX,Y\n", "no data"),
    ],
)
def test_malformed_files_are_reported(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_experimental_csv(_write(tmp_path, text, "Bad_2000.csv"))


def test_load_folder_keyed_by_source(tmp_path):
    _write(tmp_path, JARMOLA_LIKE)
    _write(tmp_path, "K,s\nA,\nX,Y\n1,2\n", "Other_2021.csv")
    data = load_experimental_data(tmp_path)
    assert list(data) == ["Jarmola_2012", "Other_2021"]
    assert [s.label for s in data["Jarmola_2012"]] == ["S8", "S3", "S2"]
    with pytest.raises(FileNotFoundError):
        load_experimental_data(tmp_path / "missing")
