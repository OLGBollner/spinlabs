"""relax_to_zfs: the ZFS INCAR made from a relaxation INCAR."""

import re

from spinlabs.vasp.occupations import patch_incar


def _tags(text, tag):
    return re.findall(rf"^\s*{tag}\s*=\s*(\S+)", text, re.M)


def test_the_zfs_incar_has_symmetry_off_whatever_the_relaxation_used():
    for relax in ("ENCUT = 600\nISYM = 2\n", "ENCUT = 600\nISYM = 3   # soft\n", "ENCUT = 600\n"):
        incar = patch_incar(relax, 128, 126, 256)
        assert _tags(incar, "ISYM") == ["0"]
        assert _tags(incar, "ENCUT") == ["600"]


def test_occupation_tags_are_left_as_they_were():
    relax = "ISMEAR = 0\nSIGMA = 0.01\nNSW = 50\nIBRION = 2\n"
    incar = patch_incar(relax, 128, 126, 256, {127, 128}, set())
    assert _tags(incar, "ISMEAR") == ["0"] and not _tags(incar, "FERWE") and not _tags(incar, "FERDO")
    assert _tags(incar, "NSW") == ["0"] and _tags(incar, "IBRION") == ["-1"] and _tags(incar, "NUPDOWN") == ["2"]
    assert "DOCCUP = 128*1.0 128*0.0" in incar and "# UNPAIRED_UP = 127 128" in incar
