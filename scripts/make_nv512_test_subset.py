"""Cut the small NV_512 test fixture out of the full 19 MB phonon file.

The full NV_512/phonon_data.npz lives in Git LFS, which CI does not pull. The
symmetrization tests only need a handful of modes, so this keeps whole
frequency clusters around the one genuinely mixed A1/Ey pair (313/315) plus
some clean clusters, and writes them with all atoms and the lattice. The
symmetrizer only mixes modes inside a cluster, so the subset symmetrizes
exactly like the full file. ``original_indices`` keeps the full-run mode
numbers, so tests address modes as 313, 315, ...

Usage (from the repo root):
    python scripts/make_nv512_test_subset.py [full.npz] [out.npz]
"""

import sys

import numpy as np

from spinlabs.symmetry import _build_clusters

FULL = "NV_512/phonon_data.npz"
OUT = "NV_512/phonon_data_subset.npz"
TOL_MEV = 0.01  # symmetrize_degenerate_groups default

KEEP = sorted(
    [
        # 313 Ey and 315 A1 are mixed (accidental degeneracy), 314 Ex is pure
        313, 314, 315,
        # clean neighbours, 309 is checked to stay unchanged
        306, 307, 308, 309, 310,
        # clean clusters with A1 and E at the same frequency
        42, 43, 44, 45, 46, 47, 48, 49,
        1223, 1224, 1225, 1226, 1227, 1228, 1229,
    ]
)  # fmt: skip


def main(full_path=FULL, out_path=OUT):
    d = np.load(full_path)
    freqs = d["freqs"]
    for cluster in _build_clusters(freqs, TOL_MEV):
        part = set(cluster) & set(KEEP)
        if part and part != set(cluster):
            raise SystemExit(f"subset splits frequency cluster {cluster}")
    np.savez_compressed(
        out_path,
        freqs=freqs[KEEP],
        eigs=d["eigs"][KEEP],
        n_atoms=d["n_atoms"],
        n_modes=len(KEEP),
        atom_symbols=d["atom_symbols"],
        atoms=d["atoms"],
        masses=d["masses"],
        lattice=d["lattice"],
        original_indices=np.array(KEEP),
    )
    print(f"wrote {len(KEEP)} modes to {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
