import numpy as np
import matplotlib.pyplot as plt
from spinlabs.plotter import (
    plot_t1,
    plot_1d_spectral_functions,
    plot_ipr_spectrum,
    plot_2d_spectral_density_map,
)


class TestPlotter:
    def test_plot_1d_spectral_functions(self):
        freqs = np.linspace(10, 100, 10)
        V_00 = np.random.rand(10) * 10
        V_pm = np.random.rand(10) * 10
        V_0pm = np.random.rand(10) * 10

        fig, ax, ax2 = plot_1d_spectral_functions(freqs, V_00, V_pm, V_0pm, zfs_mev=1.2, order=1)
        assert fig is not None
        assert ax is not None
        assert ax2 is not None
        plt.close(fig)

        # Test order=2 (diagonal of 2-phonon coupling)
        fig2, ax_2, ax2_2 = plot_1d_spectral_functions(freqs, V_00, V_pm, V_0pm, order=2)
        assert fig2 is not None
        plt.close(fig2)

    def test_plot_ipr_spectrum(self):
        freqs = np.linspace(10, 100, 10)
        ipr = np.random.rand(10)

        fig, ax = plot_ipr_spectrum(freqs, ipr, as_bar=False)
        assert fig is not None
        plt.close(fig)

        fig_bar, ax_bar = plot_ipr_spectrum(freqs, ipr, as_bar=True)
        assert fig_bar is not None
        plt.close(fig_bar)

    def test_plot_2d_spectral_density_map(self):
        freqs = np.linspace(10, 100, 8)
        zfs_2nd = np.random.rand(8, 8, 3, 3)

        fig, ax = plot_2d_spectral_density_map(freqs, zfs_2nd)
        assert fig is not None
        plt.close(fig)

    def _make_t1_npz(
        self,
        tmp_path,
        name,
        defect="NV",
        cell_size="512",
        calc_method="all_bands",
        init_state="0",
        t1_fit=None,
        t1_eigenval=None,
        legacy_fit=None,
    ):
        """Write a synthetic T1 npz file for overlay tests."""
        if t1_fit is None:
            t1_fit = np.array([1e-3, 2e-3, 4e-3])
        data = {
            "temperatures": np.array([100.0, 200.0, 300.0]),
            "defect": defect,
            "cell_size": cell_size,
            "calc_method": calc_method,
            "init_state": init_state,
        }
        if t1_fit is not None:
            data["t1_fit"] = t1_fit
        if t1_eigenval is not None:
            data["t1_eigenval"] = t1_eigenval
        if legacy_fit is not None:
            data["T1_times"] = legacy_fit
        path = tmp_path / name
        np.savez(path, **data)
        return path

    def test_plot_t1_two_files_labels_and_legend(self, tmp_path):
        """Two files produce two colours and metadata-built labels."""
        p1 = self._make_t1_npz(
            tmp_path, "nv.npz", defect="NV", cell_size="512", calc_method="all_bands", init_state="0"
        )
        p2 = self._make_t1_npz(
            tmp_path, "clv.npz", defect="ClV", cell_size="128", calc_method="defect_band", init_state="+1"
        )

        fig, ax = plot_t1([p1, p2])
        labels = [line.get_label() for line in ax.get_lines()]
        assert labels == ["NV 512 ($m_s=0$)", "ClV 128 (decontaminated, $m_s=+1$)"]
        # one colour per file
        assert ax.get_lines()[0].get_color() != ax.get_lines()[1].get_color()
        plt.close(fig)

    def test_plot_t1_rate_draws_inverse_t1(self, tmp_path):
        """--rate draws 1/T1 in s^-1; --log-temperature puts temperature on a log axis."""
        p = self._make_t1_npz(tmp_path, "nv.npz", t1_fit=np.array([1e-3, 2e-3, 4e-3]))
        fig, ax = plot_t1([p], rate=True, log_temperature=True)
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [1000.0, 500.0, 250.0])
        assert ax.get_ylabel() == r"$1/T_1$ (s$^{-1}$)"
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
        plt.close(fig)

    def test_plot_t1_eigenval_not_plotted(self, tmp_path):
        """Only the ODE-fit T1 is drawn, even when t1_eigenval is present."""
        p = self._make_t1_npz(tmp_path, "both.npz", t1_eigenval=np.array([1e-4, 3e-4, 9e-4]))

        fig, ax = plot_t1([p])
        assert len(ax.get_lines()) == 1
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [1e-3, 2e-3, 4e-3])
        plt.close(fig)

    def test_plot_t1_shared_details_left_out(self, tmp_path):
        """Details common to every curve are left out of the labels and no title is set."""
        p1 = self._make_t1_npz(tmp_path, "a.npz", cell_size="512", init_state="ms_0")
        p2 = self._make_t1_npz(tmp_path, "b.npz", cell_size="64", init_state="ms_0")
        for p in (p1, p2):
            data = dict(np.load(p))
            data["order"] = np.array([1, 2])
            np.savez(p, **data)

        fig, ax = plot_t1([p1, p2])
        assert [line.get_label() for line in ax.get_lines()] == ["NV 512", "NV 64"]
        assert ax.get_title() == ""
        plt.close(fig)

    def test_plot_t1_labels_tell_pert_scales_apart(self, tmp_path):
        """The pert-scale comes from the file's pert_scale key, else a pert_<x> folder."""
        (tmp_path / "pert_0.025").mkdir()
        from_path = self._make_t1_npz(tmp_path / "pert_0.025", "t1.npz", cell_size="512", init_state="ms_0")
        from_key = self._make_t1_npz(tmp_path, "t1_key.npz", cell_size="512", init_state="ms_0")
        data = dict(np.load(from_key))
        data["pert_scale"] = np.array(0.1)
        np.savez(from_key, **data)

        fig, ax = plot_t1([from_path, from_key])
        assert [line.get_label() for line in ax.get_lines()] == ["NV 512 (pert 0.025)", "NV 512 (pert 0.1)"]
        plt.close(fig)

    def test_plot_t1_legacy_keys_fallback(self, tmp_path):
        """Legacy npz keys (T1_times / T1_range) are used when t1_fit is absent."""
        p1 = self._make_t1_npz(tmp_path, "legacy_times.npz", t1_fit=None, legacy_fit=np.array([5e-4, 6e-4, 7e-4]))
        p2 = tmp_path / "legacy_range.npz"
        np.savez(p2, temperatures=np.array([100.0, 200.0]), T1_range=np.array([1e-3, 2e-3]))

        fig, ax = plot_t1([p1, p2])
        assert len(ax.get_lines()) >= 2
        assert all(np.isfinite(line.get_xdata()).any() for line in ax.get_lines())
        plt.close(fig)

    def test_plot_t1_skips_nan_and_nonpositive(self, tmp_path):
        """NaN and non-positive T1 values are not plotted."""
        p = self._make_t1_npz(tmp_path, "dirty.npz", t1_fit=np.array([1e-3, np.nan, -5.0]))

        fig, ax = plot_t1([p])
        plotted = ax.get_lines()[0]
        assert len(plotted.get_xdata()) == 1
        plt.close(fig)

    def test_plot_t1_no_valid_data_returns_empty(self, tmp_path, caplog):
        """Files with no usable curves produce no lines and a message."""
        p = tmp_path / "empty.npz"
        np.savez(p, temperatures=np.array([1.0]), t1_fit=np.array([np.nan]))

        fig, ax = plot_t1([p])
        assert len(ax.get_lines()) == 0
        assert "no finite T1 values" in caplog.text
        plt.close(fig)

    def test_plot_t1_output_filename_metadata(self, tmp_path):
        """Output filename gets shared metadata appended once per unique part."""
        p1 = self._make_t1_npz(tmp_path, "a.npz", defect="NV", cell_size="512")
        p2 = self._make_t1_npz(tmp_path, "b.npz", defect="NV", cell_size="512", calc_method="defect_band")
        out = tmp_path / "fig" / "overlay.png"

        plot_t1([p1, p2], output_path=out)
        saved = list((tmp_path / "fig").glob("*.png"))
        assert len(saved) == 1
        # unique metadata parts in order of first appearance
        assert saved[0].stem.startswith("overlay_NV_512")
        assert "defect_band" in saved[0].stem

    def test_plot_t1_log_scale(self, tmp_path):
        """The y axis is logarithmic."""
        p = self._make_t1_npz(tmp_path, "log.npz")

        fig, ax = plot_t1([p])
        assert ax.get_yscale() == "log"
        assert ax.get_xlabel() == "Temperature (K)"
        plt.close(fig)

    def test_zfs_plotter_with_spin_phonon_coupling_data(self, tmp_path, monkeypatch):
        """Tests that ZFSPlotter accepts SpinPhononCouplingData directly and converts units seamlessly."""
        from types import SimpleNamespace
        from spinlabs.models import SpinPhononCouplingData, ZFSTensor
        from spinlabs.plotter import ZFSPlotter
        from spinlabs.constants import CONSTANTS

        # Data in Joules
        freqs_j = np.array([20.0, 40.0, 60.0]) * CONSTANTS["meV2J"]
        v_00_j = np.array([1.5, 3.0, 4.5]) * CONSTANTS["MHz2J"]
        v_pm_j = np.array([0.5, 1.0, 1.5]) * CONSTANTS["MHz2J"]
        v_0pm_j = np.array([0.2, 0.4, 0.6]) * CONSTANTS["MHz2J"]

        data = SpinPhononCouplingData(
            order=1,
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            frequencies=freqs_j,
            frequency_unit="J",
            V_0_0=v_00_j,
            V_p_m=v_pm_j,
            V_0_pm=v_0pm_j,
            coupling_unit="J",
            ground_state_zfs=ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz"),
            symmetries=["A1", "Ex", "Ey"],
            iprs=np.array([0.1, 0.2, 0.3]),
        )

        plotter = ZFSPlotter()
        monkeypatch.chdir(path=tmp_path)

        (tmp_path / "figures").mkdir()

        args = SimpleNamespace(plot=False, ipr=False, output="test_output", format=".png")
        plotter.plot_data([data], args)

        out_file = tmp_path / "figures" / "test_output.png"
        assert out_file.exists()
