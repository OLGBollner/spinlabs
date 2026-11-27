"""load_poscar reads plain VASP 5 POSCARs itself (no pymatgen.io.vasp import) and leaves the rest to pymatgen."""

import numpy as np
import pytest
from pymatgen.io.vasp.inputs import Poscar

from spinlabs.vasp.structures import _read_plain_poscar, load_poscar

LATTICE = "   3.5  0.0  0.0\n   0.0  3.5  0.0\n   0.0  0.0  3.5\n"
DIRECT = "Direct\n  0.0 0.0 0.0\n  0.25 0.25 0.25\n  0.5 0.5 0.0\n"
CARTESIAN = "Cartesian\n  0.0 0.0 0.0\n  0.875 0.875 0.875\n  1.75 1.75 0.0\n"

CASES = {
    "direct": f"NV\n1.0\n{LATTICE}  C N\n  2 1\n{DIRECT}",
    "cartesian, scaled": f"NV\n2.0\n{LATTICE}  C N\n  2 1\n{CARTESIAN}",
    "vasp6 names": f"NV\n1.0\n{LATTICE}  C_GW/1a2b3c N/4d5e6f\n  2 1\n{DIRECT}",
}
TO_PYMATGEN = {
    "selective dynamics": f"NV\n1.0\n{LATTICE}  C N\n  2 1\nSelective dynamics\n"
                          "Direct\n  0.0 0.0 0.0 T T T\n  0.25 0.25 0.25 F F F\n  0.5 0.5 0.0 T T T\n",
    "vasp4, no species line": f"C N\n1.0\n{LATTICE}  2 1\n{DIRECT}",
    "volume scale": f"NV\n-42.875\n{LATTICE}  C N\n  2 1\n{DIRECT}",
}


def _same(a, b):
    return (np.allclose(a.lattice.matrix, b.lattice.matrix) and [s.symbol for s in a.species] == [s.symbol for s in b.species]
            and np.allclose(a.frac_coords, b.frac_coords) and np.allclose(a.cart_coords, b.cart_coords))


@pytest.mark.parametrize("case", list(CASES))
def test_a_plain_poscar_is_read_as_pymatgen_reads_it(tmp_path, case):
    path = tmp_path / "POSCAR"
    path.write_text(CASES[case])
    own = _read_plain_poscar(path)
    assert own is not None and _same(own, Poscar.from_file(path).structure)


@pytest.mark.parametrize("case", list(TO_PYMATGEN))
def test_other_poscars_are_left_to_pymatgen(tmp_path, case):
    path = tmp_path / "POSCAR"
    path.write_text(TO_PYMATGEN[case])
    assert _read_plain_poscar(path) is None
    assert _same(load_poscar(path), Poscar.from_file(path).structure)  # pymatgen reads it, as before
