#!/bin/sh
# perturbation_runner.sh -- the loop behind run_perturbation_{first,second}_order.sh.
#
# The per-tree wrappers start it inside a SLURM job, from the tree's all_bands folder:
#     sh "$SPINLABS_CODE/scripts/cluster/perturbation_runner.sh" first_order|second_order
# with the tree's config in the environment: DEFECT, binary, PHONON_PATH, PERT, PAIR_MODE
# (second order) and SPINLABS_CODE, the snapshot of the spinlabs repo taken when the job
# was submitted. This file, the structure scripts and the spinlabs package all come from
# that snapshot, so every task of a submission runs the same code.

ORDER=${1:-}
case $ORDER in
  first_order|second_order) ;;
  *) echo "usage: $0 first_order|second_order" >&2; exit 2 ;;
esac

INPUT_DIR="input"
HEAD_DIR=$(pwd)
SCRIPTS="${SPINLABS_CODE:-}/scripts/cluster"
if [ "$ORDER" = "first_order" ]; then
  create_struct="$SCRIPTS/create_phonon_struct.py"
else
  create_struct="$SCRIPTS/create_combined_phonon_struct.py"
fi
get_n_modes="$SCRIPTS/get_n_modes.py"

# ---- preflight: verify every dependency BEFORE burning core hours ----
fail() { echo "FATAL (task ${SLURM_ARRAY_TASK_ID:-?}): $*" >&2; exit 1; }
[ -n "${SPINLABS_CODE:-}" ] || fail "SPINLABS_CODE not set -- submit with: sh run_perturbation_${ORDER}.sh"
[ -d "$INPUT_DIR" ] || fail "input dir '$INPUT_DIR' missing (run this from the all_bands folder)"
case ${DEFECT:-} in ''|*'<defect>'*) fail "DEFECT unset or still placeholder: '${DEFECT:-}'" ;; esac
case ${binary:-} in ''|*'<'*|*'/path/to/'*) fail "BINARY unset or still placeholder: '${binary:-}'" ;; esac
case ${PHONON_PATH:-} in ''|*'<path_to_phonon_data.npz>'*) fail "PHONON_PATH unset or still placeholder: '${PHONON_PATH:-}'" ;; esac
case ${PERT:-} in ''|*'<pert_scale>'*) fail "PERT unset or still placeholder: '${PERT:-}'" ;; esac
if [ "$ORDER" = "second_order" ]; then
  case ${PAIR_MODE:-} in diag|all) ;; *) fail "PAIR_MODE must be diag or all, got: '${PAIR_MODE:-}'" ;; esac
fi
[ -x "$binary" ] || fail "VASP binary not found/executable: $binary"
[ -f "$create_struct" ] || fail "create_struct script missing from the code snapshot: $create_struct"
[ -f "$get_n_modes" ] || fail "get_n_modes script missing from the code snapshot: $get_n_modes"
[ -f "$PHONON_PATH" ] || fail "phonon data file missing: $PHONON_PATH"
python -c "import numpy, pymatgen.core" >/dev/null 2>&1 \
  || fail "python env broken (numpy/pymatgen import failed) -- conda env did not activate?"
# spinlabs is a namespace package (no __file__): check a module the structure scripts use.
spinlabs_dir=$(python -c "import os, spinlabs.vasp.structures as m; print(os.path.dirname(m.__file__))" 2>/dev/null) \
  || fail "python cannot import spinlabs from the code snapshot"
case $spinlabs_dir in
  "$SPINLABS_CODE"/src/*) ;;
  *) fail "spinlabs is imported from $spinlabs_dir, not from the code snapshot $SPINLABS_CODE/src" ;;
esac
MODES=$(python "$get_n_modes" "$PHONON_PATH") || fail "get_n_modes exited nonzero on $PHONON_PATH"
[ -n "$MODES" ] || fail "get_n_modes returned no modes for $PHONON_PATH"
echo "Preflight OK: $ORDER, $(echo $MODES | wc -w) modes${PAIR_MODE:+, pair_mode=$PAIR_MODE}, binary=$binary"
echo "spinlabs code: $(basename "$SPINLABS_CODE") ($SPINLABS_CODE)"
# ------------------------------------------------------------------------

ID=$SLURM_ARRAY_TASK_ID
NJOBS=$SLURM_ARRAY_TASK_COUNT
ZFS_REGEX_PATTERN='Spin-spin contribution to zero-field splitting tensor \(MHz\)\s*-+\s*D_xx\s+D_yy\s+D_zz\s+D_xy\s+D_xz\s+D_yz\s*-+\s*([\s\d\.\-]+?)(?=\s*-{3,})'

echo $ID
echo $NJOBS

set -eu

# ---- run-directory lock ----
# Two arrays submitted on the same tree used to work in the same run directories at once:
# both copied input/ and perturbed POSCAR (2x-9x displacements in NV_64), and two VASP
# processes wrote one OUTCAR. A task now works only in a run directory it has locked
# (mkdir is atomic, also on Lustre). A lock whose job is no longer in squeue is stale and
# is taken over; if squeue cannot be asked, remove <run dir>/.lock by hand once no job
# runs there.
JOB_TAG=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}}
JOB_TAG=${JOB_TAG:-${SLURM_JOB_ID:-local-$$}}
HELD_LOCK=""

write_lock_owner() {
  printf 'job=%s\nhost=%s\nsince=%s\n' "$JOB_TAG" "$(hostname)" "$(date '+%Y-%m-%d %H:%M:%S')" > "$1/owner"
}

# acquire_lock DIR: 0 = this task now holds DIR's lock, 1 = another job holds it (skip DIR).
acquire_lock() {
  lock="$HEAD_DIR/$1/.lock"
  mkdir -p "$HEAD_DIR/$1"
  if mkdir "$lock" 2>/dev/null; then
    write_lock_owner "$lock"
    HELD_LOCK=$lock
    return 0
  fi
  owner_job=$(sed -n 's/^job=//p' "$lock/owner" 2>/dev/null || true)
  owner_info=$(tr '\n' ' ' < "$lock/owner" 2>/dev/null || true)
  if [ -z "$owner_job" ]; then
    echo "$ID: skipping $1 -- locked, owner not recorded yet (remove $lock if no job runs there)"
    return 1
  fi
  if ! jobs_now=$(squeue -h -u "$(id -un)" -o %i 2>/dev/null); then
    echo "$ID: skipping $1 -- locked by $owner_info and squeue is unavailable (remove $lock if no job runs there)"
    return 1
  fi
  if printf '%s\n' "$jobs_now" | grep -qx "$owner_job"; then
    echo "$ID: skipping $1 -- locked by running job: $owner_info"
    return 1
  fi
  # Stale: rename it first, so two tasks finding the same stale lock cannot both take it.
  stale="$lock.stale.$JOB_TAG.$$"
  if mv "$lock" "$stale" 2>/dev/null && mkdir "$lock" 2>/dev/null; then
    rm -rf "$stale"
    write_lock_owner "$lock"
    HELD_LOCK=$lock
    echo "$ID: took over stale lock on $1 (was: $owner_info)"
    return 0
  fi
  rm -rf "$stale" 2>/dev/null || true
  echo "$ID: skipping $1 -- another task took over its stale lock"
  return 1
}

release_lock() {
  if [ -n "$HELD_LOCK" ]; then
    rm -rf "$HELD_LOCK"
    HELD_LOCK=""
  fi
}

# A cancelled task (SLURM sends TERM before KILL) frees its lock; a KILL leaves it stale.
trap 'release_lock' EXIT
trap 'release_lock; exit 143' TERM
trap 'release_lock; exit 130' INT
# ---- end lock ----

# run_one LABEL DIR STRUCT_ARGS...: perturb input/POSCAR along the modes and run VASP in DIR.
run_one() {
  label=$1
  TARGET_DIR=$2
  shift 2
  if [ -f "$TARGET_DIR"/OUTCAR ] && grep -qzP "$ZFS_REGEX_PATTERN" "$TARGET_DIR"/OUTCAR; then
    echo "$ID: skipping $label"
    return 0
  fi
  acquire_lock "$TARGET_DIR" || return 0
  # Another job may have finished this directory before the lock was ours.
  if [ -f "$TARGET_DIR"/OUTCAR ] && grep -qzP "$ZFS_REGEX_PATTERN" "$TARGET_DIR"/OUTCAR; then
    echo "$ID: skipping $label (finished by another job)"
    release_lock
    return 0
  fi
  rm -f "$TARGET_DIR"/CHECK  # from an earlier attempt
  cp "$INPUT_DIR"/* "$TARGET_DIR"/
  cd "$TARGET_DIR"
  echo "$ID: starting job for $label"
  MSG=$(python "$create_struct" -o POSCAR POSCAR "$PHONON_PATH" "$@" "$PERT")
  if echo "$MSG" | grep -q "Success"; then
    echo "Successfully created perturbed structure"
  else
    echo "Failed to create structure."
    echo "Shutting down job $ID"
    echo "$MSG"
    exit 1
  fi
  # The displacement must be Q along exactly these modes before any VASP time is spent.
  if ! CHECK_MSG=$(python "$SCRIPTS/check_perturbation.py" POSCAR "$HEAD_DIR/$INPUT_DIR/POSCAR" "$PHONON_PATH" "$@" --pert "$PERT"); then
    echo "$ID: $CHECK_MSG"
    echo "$ID: $label: perturbed POSCAR is not the intended displacement -- VASP not started"
    printf 'perturbation check failed\n%s\n' "$CHECK_MSG" > CHECK
    cd "$HEAD_DIR"
    release_lock
    return 0
  fi
  echo "$ID: $CHECK_MSG"
  srun $binary
  echo "$ID: Done! $label"
  # Only drop the restart files once this directory really is complete -- the same test
  # the skip-guard above uses. VASP can exit 0 without writing a ZFS block; such a
  # directory is not skipped on relaunch, so it restarts from scratch and needs
  # CHGCAR/CHG to resume. (A cancelled job never reaches this line at all.)
  if [ -f OUTCAR ] && grep -qzP "$ZFS_REGEX_PATTERN" OUTCAR; then
    echo "$ID: removing excess VASP results (WAVECAR, CHGCAR, CHG)"
    rm -f WAVECAR CHGCAR CHG
    # Completion, VASP's symmetry handling and the ZFS change's shape; a suspicious run
    # keeps its result but gets a CHECK file for later filtering.
    python "$SCRIPTS/check_run.py" "$HEAD_DIR/$TARGET_DIR" --phonons "$PHONON_PATH" --write-check \
      | sed "s/^/$ID: /" || true
  else
    echo "$ID: $label did not finish -- keeping CHGCAR/CHG so it can resume"
  fi
  cd "$HEAD_DIR"
  release_lock
}

mkdir -p runs

PAIR_ID=0
for i in $MODES; do
  if [ "$ORDER" = "first_order" ]; then
    if [ $((PAIR_ID % NJOBS)) -eq $ID ]; then
      echo "$ID: processing mode $i"
      run_one "mode $i" "runs/$i" "$i"
    fi
    PAIR_ID=$((PAIR_ID + 1))
    continue
  fi
  for j in $MODES; do
    if [ "$PAIR_MODE" = "diag" ] && [ "$i" != "$j" ]; then
      continue
    fi
    if [ "$PAIR_MODE" = "all" ] && [ $i -gt $j ]; then
      continue
    fi
    if [ $((PAIR_ID % NJOBS)) -eq $ID ]; then
      echo "$ID: processing pair $PAIR_ID (modes $i, $j)"
      run_one "pair $i, $j" "runs/${i}_${j}" "$i" "$j"
    fi
    PAIR_ID=$((PAIR_ID + 1))
  done
done
