"""Phonon pairing for the convergence GUI."""

from pathlib import Path

import numpy as np

from spinlabs.convergence_summary import mode_details
from spinlabs.gui.analysis import PackSession
from spinlabs.gui.phonons import resolve_phonon
from spinlabs.models import PhononSpectrum
from spinlabs.pack_convergence import ConvergenceScan, save_convergence
from spinlabs.parsers import save_phonon_npz

D0 = np.diag([-950.0, -1000.0, 1950.0])
STEPS = (-0.1, 0.1)


def _spectrum(aligned: bool) -> PhononSpectrum:
    n_modes = 5
    return PhononSpectrum(
        frequencies_mev=np.linspace(10.0, 50.0, n_modes),
        eigenvectors=np.zeros((n_modes, 2, 3)),
        atom_frac_coords=np.zeros((2, 3)),
        atom_symbols=["C", "C"],
        atomic_masses=np.array([12.0, 12.0]),
        lattice=np.eye(3) * 3.57,
        symmetries=["A1", "A2", "Ex", "Ey", "A1"],
        aligned=aligned,
    )


def _pack(root: Path, *, aligned: bool | None, phonon: Path | None) -> Path:
    path = root / "NV_512" / "20260101" / "NV_512_20260101_conv_test_AO.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = np.array([[D0 + q * np.eye(3) for q in STEPS] for _ in (1, 2)])
    scan = ConvergenceScan(
        path.parent, [1, 2], list(STEPS), tensors, np.zeros((2, 2)), 4, displacement="normalized",
        phonon_file=phonon, aligned=aligned,
    )
    gs = path.parent / "gs_OUTCAR"
    gs.write_text(
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n ---\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n ---\n"
        "   -950.0  -1000.0  1950.0  0.0  0.0  0.0\n ---\n"
    )
    return save_convergence(
        scan, path, ground_state=gs, defect="NV", cell="512", method="AO", method_key="test_AO",
        phonon_file=phonon, aligned=aligned,
    )


def test_resolve_prefers_the_aligned_full_spectrum(tmp_path):
    data = tmp_path / "data" / "NV_512"
    data.mkdir(parents=True)
    plain = data / "phonon_data.npz"
    aligned = data / "phonon_data_aligned.npz"
    save_phonon_npz(_spectrum(False), plain)
    save_phonon_npz(_spectrum(True), aligned)
    pack = _pack(tmp_path / "data", aligned=True, phonon=None)
    choice = resolve_phonon(pack, {"defect": "NV", "cell": 512, "aligned": True}, max_mode=2)
    assert choice.path == aligned
    assert choice.aligned is True


def test_mode_details_reads_the_chosen_phonon(tmp_path):
    data = tmp_path / "data" / "NV_512"
    data.mkdir(parents=True)
    path = data / "phonon_data_aligned.npz"
    save_phonon_npz(_spectrum(True), path)
    details = mode_details(tmp_path / "NV_512.npz", [1, 2], phonon_file=path)
    assert details[1][0] == "A1"
    assert details[2][0] == "A2"
    assert details[1][1] == 10.0


def test_pack_session_gets_symmetry_and_frequency_from_the_recorded_phonon(tmp_path):
    data = tmp_path / "data" / "NV_512"
    data.mkdir(parents=True)
    path = data / "phonon_data_aligned.npz"
    save_phonon_npz(_spectrum(True), path)
    pack = _pack(tmp_path / "data", aligned=True, phonon=path)
    entries = PackSession(pack, "AO", "normalized").modes()
    assert [(entry.mode, entry.symmetry, entry.frequency) for entry in entries] == [
        (1, "A1", 10.0), (2, "A2", 20.0)
    ]


def test_phonon_preference_can_override_the_pack_tag(tmp_path):
    data = tmp_path / "data" / "NV_512"
    data.mkdir(parents=True)
    plain = data / "phonon_data.npz"
    aligned = data / "phonon_data_aligned.npz"
    save_phonon_npz(_spectrum(False), plain)
    save_phonon_npz(_spectrum(True), aligned)
    pack = _pack(tmp_path / "data", aligned=None, phonon=None)
    session = PackSession(pack, "AO", "normalized", phonon_preference="aligned")
    assert session.modes()
    assert session.phonon_path == aligned
    session.set_phonon_preference("unaligned")
    assert session.modes()
    assert session.phonon_path == plain


def test_missing_phonon_is_reported(tmp_path):
    pack = _pack(tmp_path, aligned=None, phonon=None)
    choice = resolve_phonon(pack, {"defect": "NV", "cell": 512}, max_mode=2)
    assert choice.path is None
    assert choice.notes
