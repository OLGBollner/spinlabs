"""NBANDS fitted to VASP's band groups: occupations.band_groups / nbands_for_ranks / nbands_problems, trees.script_ranks."""

import pytest

from spinlabs.vasp.occupations import (
    band_groups,
    describe_kpoints,
    kpoints_is_gamma_only,
    nbands_for_ranks,
    nbands_problems,
)
from spinlabs.vasp.trees import GROUND_STATE_SCRIPT, TEMPLATES, script_ranks


@pytest.mark.parametrize(
    "incar, ranks, groups",
    [
        ("NCORE = 32\n", 256, 8),
        ("NCORE = 32\nKPAR = 2\n", 256, 4),
        ("ENCUT = 520\n", 256, 256),  # NCORE and KPAR default to 1
        ("NCORE = 32\nNPAR = 4\n", 256, 4),  # an explicit NPAR wins
        ("NCORE = 4 # comment\nNSIM = 16\n", 128, 32),  # NSIM does not enter
    ],
)
def test_band_groups(incar, ranks, groups):
    assert band_groups(incar, ranks) == groups


def test_band_groups_needs_ranks_divisible_by_ncore_kpar():
    with pytest.raises(ValueError, match="not a multiple of NCORE x KPAR"):
        band_groups("NCORE = 48\n", 256)


@pytest.mark.parametrize("nbands, expected", [(540, 544), (544, 544), (1, 8), (1150, 1152)])
def test_nbands_rounds_up_to_the_band_groups(nbands, expected):
    assert nbands_for_ranks(nbands, "NCORE = 32\n", 256) == expected


def test_nbands_problems():
    good = "NBANDS = 544\nNCORE = 32\nDOCCUP = 500*1.0 44*0.0\nDOCCDO = 498*1.0 46*0.0\n"
    assert nbands_problems(good, 256) == []
    short = good.replace("44*0.0", "40*0.0")
    assert nbands_problems(short, 256) == ["DOCCUP lists 540 bands, NBANDS = 544"]
    off = "NBANDS = 540\nNCORE = 32\nDOCCUP = 500*1.0 40*0.0\nDOCCDO = 498*1.0 42*0.0\n"
    assert nbands_problems(off) == []  # without the job, only the lengths are checked
    (problem,) = nbands_problems(off, 256)
    assert "not a multiple of the 8 band groups" in problem and "raises it to 544" in problem
    assert nbands_problems("ENCUT = 520\n", 256) == []  # no NBANDS: VASP picks its own


@pytest.mark.parametrize(
    "lines, ranks",
    [
        ("#SBATCH -N 2\n#SBATCH --ntasks-per-node=128\n", 256),
        ("#SBATCH --nodes=4\n#SBATCH --ntasks-per-node 32\n", 128),
        ("#SBATCH -n 64\n#SBATCH --ntasks-per-node=128\n", 64),  # the total wins
        ("#SBATCH --ntasks=96\n", 96),
        ("#SBATCH -t 1:00:00\n", None),
    ],
)
def test_script_ranks(lines, ranks):
    assert script_ranks("#!/bin/bash\n" + lines) == ranks


def test_the_ground_state_template_runs_on_256_ranks():
    assert script_ranks((TEMPLATES / GROUND_STATE_SCRIPT).read_text()) == 256


# -- KPOINTS: the ZFS runs need the Gamma point alone ---------------------------------------------------------


@pytest.mark.parametrize(
    "text, gamma, described",
    [
        ("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", True, "Gamma 1 1 1"),
        ("auto\n0\nG\n1 1 1\n", True, "Gamma 1 1 1"),  # no shift line
        ("grid\n0\nGamma\n4 4 4\n0 0 0\n", False, "Gamma 4 4 4"),  # the finer grid the old check let through
        ("grid\n0\nMonkhorst-Pack\n2 2 2\n0 0 0\n", False, "Monkhorst-Pack 2 2 2"),
        ("grid\n0\nGamma\n1 1 1\n0.5 0.5 0.5\n", False, "Gamma 1 1 1 shifted 0.5 0.5 0.5"),
        ("list\n2\nReciprocal\n0 0 0 1\n0.5 0 0 1\n", False, "not an automatic mesh"),
        ("length\n0\nAuto\n20\n", False, "not an automatic mesh"),
    ],
)
def test_kpoints_gamma_check(text, gamma, described):
    assert kpoints_is_gamma_only(text) is gamma
    assert describe_kpoints(text).startswith(described)
