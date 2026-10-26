#!/bin/sh
# Replace the job scripts of existing trees with this repo's current wrappers, keeping each tree's
# own #SBATCH lines and config values.
#
#     sh scripts/cluster/update_tree_scripts.sh [--displacement CONV | --no-mass-weighted] [--dry-run] <dir> ...
#
# <dir>: a defect folder or any folder below it; every job script under it is updated
# (snapshots in .spinlabs-code/ are skipped): run_perturbation_{first,second}_order.sh of the
# perturbation trees, and run_convergence.sh of the convergence_test_* folders. An older
# convergence script (a run_perturbation_*_order.sh with MODES and PERTS in a convergence* folder)
# becomes run_convergence.sh. The original script is kept as <name>.bak (written once).
#
# The displacement convention stays what the tree's runs used: its DISPLACEMENT line, else its
# MASS_WEIGHTED (0: unweighted, else mass_weighted), else mass_weighted (scripts from before both).
# --displacement sets it for scripts that name none; --no-mass-weighted is --displacement unweighted
# (for trees displaced by Q*e, e.g. NV_64 and NV_512 pert 0.1). SPINLABS_REPO is set to this checkout.
set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
DISP=""
DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --no-mass-weighted) DISP=unweighted ;;
    --displacement) shift; DISP=${1:-} ;;
    --dry-run) DRY=1 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
  shift
done
case $DISP in ''|normalized|mass_weighted|unweighted) ;; *) echo "unknown displacement: $DISP" >&2; exit 2 ;; esac
[ $# -gt 0 ] || { echo "usage: $0 [--displacement CONV | --no-mass-weighted] [--dry-run] <dir> ..." >&2; exit 2; }

find "$@" -path '*/.spinlabs-code' -prune -o -type f \( -name 'run_perturbation_*_order.sh' -o -name 'run_convergence.sh' \) -print | while read -r old; do
  dir=$(dirname "$old")
  name=$(basename "$old")
  case $dir in
    */convergence*) name=run_convergence.sh ;;  # older convergence scripts were named run_perturbation_*
  esac
  new="$REPO/src/spinlabs/vasp/templates/$name"
  [ -f "$new" ] || { echo "skip $old: no $new" >&2; continue; }
  target="$dir/$name"
  tmp="$target.new"
  awk -v repo="$REPO" -v disp="$DISP" '
    function key(line) {
      if (line ~ /^#SBATCH /) { split(line, f, /[ =]+/); return "S" f[2] }
      if (match(line, /^(DEFECT|binary|PHONON_PATH|PERT|PERTS|MODES|PAIR_MODE|DISPLACEMENT|MASS_WEIGHTED|SPINLABS_REPO)=/))
        return substr(line, 1, RLENGTH)
      return ""
    }
    NR == FNR { k = key($0); if (k != "" && !(k in old)) old[k] = $0; next }
    FNR == 1 {
      # the convention the runs so far used
      if ("DISPLACEMENT=" in old) { conv = old["DISPLACEMENT="]; sub(/^DISPLACEMENT=/, "", conv); sub(/[ \t]*#.*/, "", conv)
                                   gsub(/["\047]/, "", conv); sub(/^\$\{DISPLACEMENT:-/, "", conv); sub(/\}$/, "", conv) }
      else if ("MASS_WEIGHTED=" in old) {  # 0 or its ${MASS_WEIGHTED:-0} default: unweighted
        mw = old["MASS_WEIGHTED="]; sub(/^MASS_WEIGHTED=/, "", mw); sub(/[ \t]*#.*/, "", mw); gsub(/["\047]/, "", mw)
        sub(/^\$\{MASS_WEIGHTED:?-/, "", mw); sub(/\}$/, "", mw)
        conv = (mw == "0") ? "unweighted" : "mass_weighted"
      }
      else conv = (disp != "") ? disp : "mass_weighted"
    }
    {
      k = key($0)
      if (k == "DISPLACEMENT=") { print "DISPLACEMENT=" conv; next }
      if (k == "SPINLABS_REPO=" && (!(k in old) || old[k] ~ /</)) { print "SPINLABS_REPO=${SPINLABS_REPO:-" repo "}"; next }
      if (k != "" && (k in old)) { print old[k]; next }
      print
    }' "$old" "$new" > "$tmp"
  if [ "$DRY" = 1 ]; then
    echo "== $old -> $target"; diff "$old" "$tmp" || true; rm -f "$tmp"; continue
  fi
  [ -e "$old.bak" ] || cp -p "$old" "$old.bak"  # keep the first original
  mv "$tmp" "$target"
  [ "$old" = "$target" ] || rm -f "$old"  # an older convergence script now lives on as run_convergence.sh (and .bak)
  echo "updated $target (DISPLACEMENT=$(sed -n 's/^DISPLACEMENT=\([^ ]*\).*/\1/p' "$target"))"
done
