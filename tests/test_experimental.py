"""Reading digitized experimental data (WebPlotDigitizer CSV export)."""

import numpy as np
import pytest

from spinlabs.experimental import (
    ExperimentalSeries,
    load_experimental_data,
    read_experimental_csv,
    relaxation_series,
    t1_from_rates,
)

# The layout of experimental_data/Lin_2021.csv: Omega and gamma in kHz on different temperature grids.
LIN_LIKE = """K,kHz
omega,,gamma,
X,Y,X,Y
300,0.1,300,0.2
350,0.15,400,0.3
400,0.2,500,0.5
450,0.3,,
500,0.4,,
"""

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


def test_omega_and_gamma_labels_mark_single_transition_rates(tmp_path):
    omega, gamma = read_experimental_csv(_write(tmp_path, LIN_LIKE, "Lin_2021.csv"))
    assert (omega.kind, gamma.kind) == ("omega", "gamma")
    greek = read_experimental_csv(_write(tmp_path, "K,Hz\nΩ,,γ,\nX,Y,X,Y\n1,2,1,3\n", "Greek_2000.csv"))
    assert [s.kind for s in greek] == ["omega", "gamma"]
    assert all(s.kind == "relaxation" for s in read_experimental_csv(_write(tmp_path, JARMOLA_LIKE)))


def test_t1_of_ms0_from_omega_and_gamma_is_one_over_three_omega(tmp_path):
    """An m_s = 0 polarization decays at Gamma(+-1 -> 0) + 2 Gamma(0 -> +-1) = 3 Omega, whatever gamma is."""
    omega, gamma = read_experimental_csv(_write(tmp_path, LIN_LIKE, "Lin_2021.csv"))
    t1 = t1_from_rates(omega, gamma, "ms_0")
    assert (t1.kind, t1.x_unit, t1.y_unit, t1.source) == ("relaxation", "K", "s", "Lin_2021")
    np.testing.assert_allclose(t1.x, [300, 350, 400, 450, 500])  # union of both grids within the overlap
    np.testing.assert_allclose(t1.y, 1 / (3 * np.array([100, 150, 200, 300, 400.0])), rtol=1e-6)
    ms1 = t1_from_rates(omega, gamma, "ms_1")
    faster = ExperimentalSeries("gamma", gamma.x, gamma.y * 10, "K", "kHz", "Lin_2021", "gamma")
    assert np.all(t1_from_rates(omega, faster, "ms_1").y < ms1.y)  # an m_s = +1 polarization does feel gamma


def test_rates_between_measured_points_are_interpolated_in_log(tmp_path):
    """gamma at 350 K, between 200 Hz (300 K) and 300 Hz (400 K): the geometric mean, sqrt(200 * 300) Hz."""
    from spinlabs.experimental import _log_interp

    assert _log_interp(np.array([350.0]), np.array([300.0, 400.0]), np.array([200.0, 300.0]))[0] == pytest.approx(
        np.sqrt(200 * 300)
    )


def test_relaxation_series_keeps_measured_t1_and_adds_t1_from_rates(tmp_path):
    lin = read_experimental_csv(_write(tmp_path, LIN_LIKE, "Lin_2021.csv"))
    jarmola = read_experimental_csv(_write(tmp_path, JARMOLA_LIKE))
    out = relaxation_series(jarmola + lin, "ms_0")
    assert [s.label for s in out] == ["S8", "S3", "S2", "T1 ms_0 from omega and gamma"]
    assert relaxation_series([lin[0]]) == []  # Omega alone gives no T1
    with pytest.raises(ValueError, match="share no temperature range"):
        late = ExperimentalSeries("gamma", np.array([600.0, 700.0]), np.array([1.0, 2.0]), "K", "kHz", "X", "gamma")
        t1_from_rates(lin[0], late)
    with pytest.raises(ValueError, match="init_state"):
        t1_from_rates(lin[0], lin[1], "ms_2")
