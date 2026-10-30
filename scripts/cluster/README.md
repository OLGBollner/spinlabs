# Cluster run scripts

ZFS perturbation runs on SLURM. Each tree folder (e.g. `first_order/pert_0.025/all_bands/`,
with `input/` next to it) holds a job script made from the templates in
`src/spinlabs/vasp/templates/` (`run_perturbation_first_order.sh` / `_second_order.sh`, and
`run_convergence.sh` for the step-size convergence folders). The job script holds only the
tree's `#SBATCH` header and config (`DEFECT`, `BINARY`, `PHONON_PATH`, `PERT`, `DISPLACEMENT`,
`PAIR_MODE`, `SPINLABS_REPO`; convergence: `MODES` and `PERTS`, step sizes each run at -p and +p, instead of `PERT`).
Submit it with

```bash
sh run_perturbation_first_order.sh [sbatch options]
```

which snapshots `SPINLABS_REPO` at its current commit into `.spinlabs-code/<commit>/` and
submits itself; every task then runs `perturbation_runner.sh` and the helper scripts below
from that snapshot.

`DISPLACEMENT` is how the step `PERT` moves the atoms along mode k (u = e/sqrt(m)):

- `normalized` (new trees): step * u/|u|, the same Cartesian length `PERT` (Angstrom) for every
  mode; the analysis uses each mode's normal coordinate step * n_k, n_k = 1/|u_k|.
- `mass_weighted`: Q*e/sqrt(m) with Q = `PERT` (Angstrom*sqrt(amu)); trees made before
  `DISPLACEMENT` (their scripts say `MASS_WEIGHTED=1` or nothing).
- `unweighted`: Q*e, the first NV_64 trees (`MASS_WEIGHTED=0`).

- `perturbation_runner.sh`: the loop. First order perturbs along single modes
  (`create_phonon_struct.py`, run dirs `runs/<mode>`), second order along mode pairs
  (`create_combined_phonon_struct.py`, `runs/<i>_<j>`), convergence each of `MODES` at -p and
  +p for each size p in `PERTS` (`runs/<mode>_pert_<step>`; a sign, a repeat or 0 in `PERTS`
  is dropped, so older signed lists run the same steps). A run dir is finished when it has
  `.done`, written by the task that finished it once its OUTCAR has the spin-spin ZFS block
  (that task alone reads the OUTCAR, and checks it with `check_run.py`); finished dirs are
  skipped, so restarts resume where they stopped. Run dirs from before `.done` (no `.done`, no
  `.attempted`) have their OUTCAR checked once and get `.done` if complete. Each array task
  first runs its own share (run index mod array size), then takes the other tasks' runs that
  are still free, so no task idles while others have work left; a run directory's `.lock`
  keeps two tasks out of it, and its `.attempted` (the array job id) stops the other tasks of
  the same submission from retrying a run that failed.
- `create_perturbation_tree.py`: build ready-to-go trees (`first_order`/`second_order` x
  `all_bands`/`defect_band_approx` x steps) from `<defect>/ZFS_hyp` and `<defect>/ZFS_occup`,
  plus the convergence folders `convergence_test_all/` and `convergence_test_occup/`.
  `--pert` (the step) is required; `--displacement` defaults to `normalized`;
  `--conv-modes` picks the convergence modes (or `--no-convergence`), `--conv-perts` their
  step sizes (default p/2, p, 2p and 4p for each step p, 4p being the 2Q helper of 2p; each run at
  both signs). The generated script gets
  `#SBATCH -J <defect>_<order>_<basis>` and a time limit / array size from the reference
  OUTCAR's wall time x tasks / `--array-jobs`; `--max-hours` caps the per-job limit.
- `add_convergence_tree.py`: add the convergence folders to an existing defect folder
  (`--modes`, and `--steps` or `--pert`); the displacement and VASP binary default to the
  ones its trees use.
- `update_tree_scripts.sh`: put the current templates into existing trees and convergence
  folders, keeping each script's `#SBATCH` lines, config and displacement (an older
  convergence script becomes `run_convergence.sh`), and copy in the current
  `verify_perturbation_setup.py`.
- `verify_perturbation_setup.py` (in each tree, from the templates): checks the INCAR and job
  script, then prints `x/N runs done` (a run is done when its folder has `.done`) and offers to
  set the script's `-t` for the runs left: ceil(left / array tasks) x the median wall time of
  the finished runs x 1.2, keeping the array size (smaller only when fewer runs are left than
  tasks). The logic is `spinlabs.vasp.verify.main`, so a tree's copy stays current.
- `get_n_modes.py`: the modes a tree runs (`spinlabs.vasp.modes`).

The scripts are tested in their own suite, `pytest scripts/tests` (plain `pytest` runs
only the spinlabs package tests).
