"""The method table: every name of AO / UO / full_decont, and folders found under new or legacy names."""

import pytest

from spinlabs.methods import (
    CLI_CHOICES,
    aliases,
    basis_dir,
    convergence_dir,
    ground_state_dir,
    method_from_path,
    normalize_calc_method,
    relax_dir,
    spin_corrected,
)


@pytest.mark.parametrize("name, canonical", [
    ("all", "AO"), ("AO", "AO"), ("ao", "AO"), ("all_bands", "AO"),
    ("unpaired", "UO"), ("UO", "UO"), ("defect_band_approx", "UO"), ("approx", "UO"),
    ("decont", "full_decont"), ("full_decont", "full_decont"),
    (None, None), ("", None), ("None", None), ("something_else", "something_else"),
])
def test_every_name_normalizes(name, canonical):
    assert normalize_calc_method(name) == canonical


def test_cli_offers_the_new_names_only():
    assert CLI_CHOICES == ("all", "AO", "unpaired", "UO", "decont")


def test_aliases_cover_the_stored_legacy_names():
    assert "all_bands" in aliases("AO") and "defect_band_approx" in aliases("unpaired")
    assert aliases("decont") == ("full_decont",)


def test_only_uo_is_spin_corrected():
    assert spin_corrected("UO") and spin_corrected("defect_band_approx")
    assert not spin_corrected("AO") and not spin_corrected("full_decont") and not spin_corrected(None)


def test_folders_prefer_the_new_name_and_fall_back_to_the_legacy_one(tmp_path):
    assert ground_state_dir(tmp_path, "AO") == tmp_path / "ZFS_AO"  # nothing there: the name to write
    (tmp_path / "ZFS_hyp").mkdir()
    (tmp_path / "relaxation_data").mkdir()
    assert ground_state_dir(tmp_path, "all") == tmp_path / "ZFS_hyp"
    assert relax_dir(tmp_path) == tmp_path / "relaxation_data"
    (tmp_path / "ZFS_AO").mkdir()
    assert ground_state_dir(tmp_path, "AO") == tmp_path / "ZFS_AO"
    (tmp_path / "pert_0.1" / "defect_band_approx").mkdir(parents=True)
    assert basis_dir(tmp_path / "pert_0.1", "UO") == tmp_path / "pert_0.1" / "defect_band_approx"
    assert basis_dir(tmp_path / "pert_0.1", "AO") == tmp_path / "pert_0.1" / "AO"
    (tmp_path / "convergence_test_occup").mkdir()
    assert convergence_dir(tmp_path, "UO") == tmp_path / "convergence_test_occup"
    assert convergence_dir(tmp_path, "full_decont") == tmp_path / "convergence_test_decont"


@pytest.mark.parametrize("path, method", [
    ("data/NV_64/pert_0.1/AO/20261120/NV_64_raw_zfs_data_AO_12d.npz", "AO"),
    ("data/NV_64/pert_0.1/all_bands/20261029/NV_64_raw_zfs_data_all_bands_12d.npz", "AO"),
    ("NV_64_raw_zfs_data_defect_band_approx_1d.npz", "UO"),
    ("NV_512/convergence_test_occup", "UO"),
    ("NV_512/ZFS_UO", "UO"),
    ("data/NV_64/pert_0.025/full_decont/20261118", "full_decont"),
    ("NV_512/first_order/pert_0.1", None),
])
def test_method_from_path(path, method):
    assert method_from_path(path) == method
