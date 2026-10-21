#!/bin/sh
#SBATCH -A naiss2025-3-15
#SBATCH -J defect_2ph
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -t 2:30:00
#SBATCH -p main
#SBATCH -a 0-9

# Second-order ZFS perturbation runs: perturb POSCAR along mode pairs.
#
# This file holds only what belongs to one tree: the SLURM header and the config below.
# The loop itself (scripts/cluster/perturbation_runner.sh), the structure scripts and the
# spinlabs package come from the repo in SPINLABS_REPO. Submit with
#     sh run_perturbation_second_order.sh [sbatch options]
# not plain sbatch: that snapshots SPINLABS_REPO at its current commit into
# .spinlabs-code/<commit>/ here and submits this file with it, so every task of the
# submission runs that one version even if the repo is pulled while the job runs.
# create_perturbation_tree.py places this file in a defect's second_order/<pert>/all_bands/ folder and fills
# the config (or set it by hand / export the variables before submitting).

# ---- config (override via environment, e.g. DEFECT=NV_512 sh run_perturbation_second_order.sh) ----
DEFECT=${DEFECT:-<defect>}
binary=${BINARY:-<vasp_binary>}
PHONON_PATH=${PHONON_PATH:-<path_to_phonon_data.npz>}
PERT=${PERT:-<pert_scale>}
MASS_WEIGHTED=${MASS_WEIGHTED:-1}  # 1: displace by Q*e/sqrt(m); 0: by Q*e (keep 0 for trees made that way, e.g. NV_64)
PAIR_MODE=${PAIR_MODE:-<pair_mode>}
SPINLABS_REPO=${SPINLABS_REPO:-<path_to_spinlabs_repo>}
# ------------------------------------------------------------------------------------------

ORDER=second_order
RUNNER=scripts/cluster/perturbation_runner.sh

if [ -z "${SLURM_JOB_ID:-}" ]; then
  # ---- submit (login node): snapshot the repo at its current commit, then sbatch this file ----
  case $SPINLABS_REPO in *'<'*) echo "FATAL: SPINLABS_REPO still has placeholder: '$SPINLABS_REPO'" >&2; exit 1 ;; esac
  command -v sbatch >/dev/null 2>&1 || { echo "FATAL: sbatch not found -- submit from a login node" >&2; exit 1; }
  commit=$(git -C "$SPINLABS_REPO" rev-parse --short=12 HEAD) \
    || { echo "FATAL: SPINLABS_REPO is not a git checkout: $SPINLABS_REPO" >&2; exit 1; }
  [ -z "$(git -C "$SPINLABS_REPO" status --porcelain --untracked-files=no -- src scripts/cluster)" ] \
    || echo "WARNING: $SPINLABS_REPO has uncommitted changes in src/ or scripts/cluster/; the job runs commit $commit without them" >&2
  code="$(pwd)/.spinlabs-code/$commit"
  if [ ! -f "$code/$RUNNER" ]; then
    mkdir -p "$code.tmp.$$"
    git -C "$SPINLABS_REPO" archive "$commit" src scripts/cluster | tar -x -C "$code.tmp.$$" \
      || { rm -rf "$code.tmp.$$"; echo "FATAL: could not snapshot $SPINLABS_REPO at $commit" >&2; exit 1; }
    mv "$code.tmp.$$" "$code" 2>/dev/null || rm -rf "$code.tmp.$$"  # another submission made it meanwhile
  fi
  [ -f "$code/$RUNNER" ] || { echo "FATAL: commit $commit of $SPINLABS_REPO has no $RUNNER -- update the checkout (git pull)" >&2; exit 1; }
  echo "Submitting $ORDER with spinlabs commit $commit (snapshot $code)"
  exec sbatch --export=ALL,SPINLABS_CODE="$code" "$@" "$0"
fi

# ---- job (started by SLURM): run the loop from the snapshot this submission made ----
if [ -z "${SPINLABS_CODE:-}" ] || [ ! -f "$SPINLABS_CODE/$RUNNER" ]; then
  echo "FATAL (task ${SLURM_ARRAY_TASK_ID:-?}): no code snapshot for this job -- submit with: sh $0 (not sbatch $0)" >&2
  exit 1
fi
ml PDC/26.03
ml miniconda3
source activate /cfs/klemming/projects/supr/adaq/obollner/conda-dirs/envs/sim-env || {
  echo "FATAL: failed to activate conda env -- aborting before any VASP run" >&2
  exit 1
}
PYTHONPATH="$SPINLABS_CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH DEFECT binary PHONON_PATH PERT MASS_WEIGHTED PAIR_MODE SPINLABS_CODE
exec sh "$SPINLABS_CODE/$RUNNER" "$ORDER"
