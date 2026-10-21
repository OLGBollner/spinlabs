# Cluster run scripts

ZFS perturbation runs on SLURM. Each tree folder (e.g. `first_order/pert_0.025/all_bands/`,
with `input/` next to it) holds a job script made from the templates in
`src/spinlabs/vasp/templates/` (`run_perturbation_first_order.sh` / `_second_order.sh`).
The job script holds only the tree's `#SBATCH` header and config (`DEFECT`, `BINARY`,
`PHONON_PATH`, `PERT`, `MASS_WEIGHTED`, `PAIR_MODE`, `SPINLABS_REPO`). Submit it with

```bash
sh run_perturbation_first_order.sh [sbatch options]
```

which snapshots `SPINLABS_REPO` at its current commit into `.spinlabs-code/<commit>/` and
submits itself; every task then runs `perturbation_runner.sh` and the helper scripts below
from that snapshot.

- `perturbation_runner.sh`: the loop. First order perturbs along single modes
  (`create_phonon_struct.py`, run dirs `runs/<mode>`), second order along mode pairs
  (`create_combined_phonon_struct.py`, `runs/<i>_<j>`). Run dirs whose OUTCAR already has a
  complete spin-spin ZFS block are skipped, so restarts resume where they stopped.
- `create_perturbation_tree.py`: build ready-to-go trees (`first_order`/`second_order` x
  `all_bands`/`defect_band_approx` x pert scales) from `<defect>/ZFS_hyp` and
  `<defect>/ZFS_occup`. The generated script gets `#SBATCH -J <defect>_<order>_<basis>` and
  a time limit / array size from the reference OUTCAR's wall time x tasks / `--array-jobs`;
  `--max-hours` caps the per-job limit. `--no-mass-weighted` makes a tree displace by Q*e.
- `update_tree_scripts.sh`: put the current template into existing trees, keeping each
  tree's `#SBATCH` lines and config.
- `get_n_modes.py`: the modes a tree runs (`spinlabs.vasp.modes`).

The scripts are tested in their own suite, `pytest scripts/tests` (plain `pytest` runs
only the spinlabs package tests).
