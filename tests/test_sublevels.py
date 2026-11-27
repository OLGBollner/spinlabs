"""Spin-sublevel INCARs derived at run time from the SCF INCAR of their run directory."""

import re

import pytest

from spinlabs.vasp.hse import hse_incar
from spinlabs.vasp.occupations import parse_occupations, spin_sublevels
from spinlabs.vasp.sublevels import main, sublevel_incar
from spinlabs.vasp.trees import decont_scf_incar, decont_sublevel_incar

# NV-like: 8 bands, bands 3 and 4 unpaired (occupied up, empty down)
ZFS = ("ENCUT = 520\nALGO = Normal\nISMEAR = 0\nNUPDOWN = 2\nNBANDS = 8\nLDMATRIX = .TRUE.\n"
       "DOCCUP = 4*1.0 4*0.0\nDOCCDO = 2*1.0 6*0.0\n")


def _tags(text):
    return {m[0].upper(): m[1].strip() for m in re.findall(r"^\s*(\w+)\s*=\s*(.*?)\s*(?:[!#].*)?$", text, re.M)}


def test_a_sublevel_incar_from_the_scf_is_the_one_setup_writes():
    scf = decont_scf_incar(ZFS)
    for level in spin_sublevels(*parse_occupations(ZFS)):
        assert _tags(sublevel_incar(scf, level.name)) == _tags(decont_sublevel_incar(ZFS, level)), level.name
    one = _tags(sublevel_incar(scf, "ms_0_3"))
    assert (one["ALGO"], one["ISTART"], one["ICHARG"], one["LDMATRIX"], one["NUPDOWN"]) == (
        "None", "1", "0", ".TRUE.", "0")


def test_a_setting_changed_in_the_scf_reaches_every_sublevel():
    scf = hse_incar(decont_scf_incar(ZFS)).replace("ENCUT = 520", "ENCUT = 600")
    for name in ("ms_1", "ms_0_3", "ms_0_4"):
        tags = _tags(sublevel_incar(scf, name))
        assert tags["ENCUT"] == "600" and tags["LHFCALC"] == ".TRUE." and tags["NCORE"] == "1", name
        assert tags["ALGO"] == "None" and tags["ISTART"] == "1", name  # still the sublevel's own run


def test_a_sublevel_the_scf_occupations_do_not_give_is_refused(tmp_path, capsys):
    scf = decont_scf_incar(ZFS)
    with pytest.raises(ValueError, match="ms_1, ms_0_3, ms_0_4, not ms_0_5"):
        sublevel_incar(scf, "ms_0_5")
    (tmp_path / "INCAR").write_text(scf)
    assert main([str(tmp_path / "INCAR"), "ms_0_4"]) == 0
    assert "NUPDOWN = 0" in capsys.readouterr().out
    assert main([str(tmp_path / "INCAR"), "ms_0_5"]) == 1
    assert "not ms_0_5" in capsys.readouterr().err
