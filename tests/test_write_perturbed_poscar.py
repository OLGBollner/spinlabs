"""write_perturbed_poscar keeps the template's header and writes positions in its coordinate mode."""

import numpy as np
import pytest

from spinlabs.vasp import load_poscar, write_perturbed_poscar

A = 3.567  # diamond lattice constant (Å)
FRAC = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.0]]


def _template(path, mode: str, scale: float) -> None:
    """A 3-atom POSCAR in Direct or Cartesian mode; line 2 holds ``scale`` (negative: volume)."""
    lattice = np.eye(3) * A
    written = lattice if scale < 0 else lattice / scale
    if mode == "Direct":
        coords = FRAC
    else:
        coords = (np.array(FRAC) @ lattice) / (1.0 if scale < 0 else scale)
    lines = ["template", f"{scale:g}"]
    lines += ["  " + "  ".join(f"{x:.10f}" for x in row) for row in written]
    lines += ["C N", "2 1", mode]
    lines += ["  " + "  ".join(f"{x:.10f}" for x in row) for row in coords]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("mode, scale", [("Direct", 1.0), ("Cartesian", 1.0), ("Cartesian", 2.0)])
def test_written_positions_read_back_as_the_perturbed_structure(tmp_path, mode, scale):
    template = tmp_path / "POSCAR"
    _template(template, mode, scale)
    structure = load_poscar(template)
    assert np.allclose(structure.lattice.matrix, np.eye(3) * A)

    structure.translate_sites([1], [0.05, -0.02, 0.01], frac_coords=False)  # displace one atom by Å
    out = tmp_path / "POSCAR_pert"
    write_perturbed_poscar(structure, out, template_poscar=template, comment="perturbed")

    lines = out.read_text().splitlines()
    assert lines[1:8] == template.read_text().splitlines()[1:8]  # header kept verbatim
    back = load_poscar(out)
    assert np.allclose(back.cart_coords, structure.cart_coords, atol=1e-8)


def test_cartesian_with_a_volume_scale_line_is_written_in_angstrom(tmp_path):
    # A negative line 2 is the cell volume; VASP then scales the lattice (and Cartesian
    # positions) by (V / V_written)^(1/3), here 1. pymatgen applies the raw number to the
    # positions instead, so the expected values are spelled out rather than read back.
    template = tmp_path / "POSCAR"
    _template(template, "Cartesian", -(A**3))
    structure = load_poscar(template)
    structure.replace(1, "C", coords=[0.9, 0.9, 0.9], coords_are_cartesian=True)
    out = tmp_path / "POSCAR_pert"
    write_perturbed_poscar(structure, out, template_poscar=template, comment="perturbed")
    written = np.array([[float(x) for x in line.split()] for line in out.read_text().splitlines()[8:11]])
    assert np.allclose(written[1], [0.9, 0.9, 0.9])
