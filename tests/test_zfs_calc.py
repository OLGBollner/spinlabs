"""spinlabs calc zfs: the ZFS of a run folder from its OUTCAR(s), by method."""

import json

import numpy as np
import pytest

from spinlabs.cli import main
from spinlabs.models import decontaminate_tensors
from spinlabs.zfs_calc import compute_zfs, method_of


def _outcar(folder, diag, offdiag=(0.0, 0.0, 0.0), nupdown=None):
    """A run folder whose OUTCAR has a ZFS block (D_xx D_yy D_zz D_xy D_xz D_yz, MHz)."""
    folder.mkdir(parents=True, exist_ok=True)
    values = "  ".join(f"{v:.3f}" for v in (*diag, *offdiag))
    (folder / "OUTCAR").write_text(
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n ---\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n ---\n"
        f"  {values}\n ---\n")
    if nupdown is not None:
        (folder / "INCAR").write_text(f"ENCUT = 520\nNUPDOWN = {nupdown}\n")
    return folder


def test_ao_is_the_outcar_tensor_as_vasp_reports_it(tmp_path):
    folder = _outcar(tmp_path / "ZFS_AO", (-1000.0, -1000.0, 2000.0), nupdown=2)
    result = compute_zfs(folder)  # the method from the folder's name
    assert result.method == "AO" and result.spin == 1.0
    assert result.tensor.D == pytest.approx(3000.0) and result.tensor.E == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("nupdown, scale", [(2, 3 / 2), (3, 2 / 3)])
def test_uo_takes_the_defect_band_spin_correction_of_its_spin(tmp_path, nupdown, scale):
    folder = _outcar(tmp_path / "run", (-100.0, -100.0, 200.0), nupdown=nupdown)
    result = compute_zfs(folder, "unpaired")
    assert result.method == "UO" and result.spin == nupdown / 2
    assert result.tensor.D == pytest.approx(300.0 * scale)
    assert result.parts["OUTCAR"].D == pytest.approx(300.0)  # as reported
    assert any(f"x{scale:.4g}" in note for note in result.notes)


def test_full_decont_combines_its_sublevels(tmp_path):
    folder = tmp_path / "ZFS_full"
    high = np.array([0.0, 0.0, 0.0, 1050.0, 1050.0, 1050.0])
    flips = [np.array([-90.0, 170.0, -80.0, -760.0, -110.0, -800.0]), np.array([90.0, -170.0, 80.0, -360.0, -1000.0, -320.0])]
    _outcar(folder / "ms_1", high[:3], high[3:])
    for name, f in zip(("ms_0_127", "ms_0_128"), flips):
        _outcar(folder / name, f[:3], f[3:])

    def matrix(v):
        return np.array([[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]])

    result = compute_zfs(folder)  # ms_* folders: full_decont
    assert result.method == "full_decont" and result.spin == 1.0
    assert list(result.parts) == ["ms_1", "ms_0_127", "ms_0_128"]
    expected = decontaminate_tensors(matrix(high), [matrix(f) for f in flips], 1.0)
    assert np.allclose(result.tensor.matrix, expected)
    (folder / "ms_0_128" / "OUTCAR").unlink()
    with pytest.raises(ValueError, match="no OUTCAR in .*ms_0_128"):
        compute_zfs(folder)


def test_a_missing_spin_flip_or_zfs_block_or_method_is_an_error(tmp_path):
    folder = tmp_path / "set"
    _outcar(folder / "ms_1", (-1.0, -1.0, 2.0))
    _outcar(folder / "ms_0_4", (1.0, 1.0, -2.0))
    with pytest.raises(ValueError, match="2S = 2 spin-flip sublevels"):
        compute_zfs(folder)
    plain = tmp_path / "run"
    plain.mkdir()
    (plain / "OUTCAR").write_text("no ZFS here\n")
    with pytest.raises(ValueError, match="cannot tell the method"):
        compute_zfs(plain)
    with pytest.raises(ValueError, match="has no ZFS block"):
        compute_zfs(plain, "AO")
    assert method_of(tmp_path / "ZFS_occup") == "UO" and method_of(folder) == "full_decont"


def test_cli_prints_and_writes_json(tmp_path, capsys):
    folder = _outcar(tmp_path / "ZFS_UO", (-100.0, -100.0, 200.0), nupdown=2)
    out = tmp_path / "zfs.json"
    main(["calc", "zfs", str(folder), "--json", str(out)])
    text = capsys.readouterr().out
    assert "ZFS (UO): D = 450.000 MHz" in text and "defect-band spin correction x1.5" in text
    data = json.loads(out.read_text())
    assert data["method"] == "UO" and data["D_MHz"] == pytest.approx(450.0) and data["spin"] == 1.0
    with pytest.raises(SystemExit, match="give --method"):
        main(["calc", "zfs", str(_outcar(tmp_path / "other", (-1.0, -1.0, 2.0)))])
