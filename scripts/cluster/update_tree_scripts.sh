#!/bin/sh
# Replace the run_perturbation_{first,second}_order.sh of existing trees with this repo's
# current wrapper, keeping each tree's own #SBATCH lines and config values.
#
#     sh scripts/cluster/update_tree_scripts.sh [--no-mass-weighted] [--dry-run] <dir> ...
#
# <dir>: a defect folder or any folder below it; every job script under it is updated
# (snapshots in .spinlabs-code/ are skipped). The original script is kept as <name>.bak (written once).
# --no-mass-weighted sets MASS_WEIGHTED=0 in these trees, for trees whose runs were
# displaced by Q*e (NV_64, NV_512 pert 0.1). SPINLABS_REPO is set to this checkout.
set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
MW=1
DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --no-mass-weighted) MW=0 ;;
    --dry-run) DRY=1 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
  shift
done
[ $# -gt 0 ] || { echo "usage: $0 [--no-mass-weighted] [--dry-run] <dir> ..." >&2; exit 2; }

find "$@" -path '*/.spinlabs-code' -prune -o -type f -name 'run_perturbation_*_order.sh' -print | while read -r old; do
  new="$REPO/scripts/cluster/$(basename "$old")"
  [ -f "$new" ] || { echo "skip $old: no $new" >&2; continue; }
  tmp="$old.new"
  awk -v repo="$REPO" -v mw="$MW" '
    function key(line) {
      if (line ~ /^#SBATCH /) { split(line, f, /[ =]+/); return "S" f[2] }
      if (match(line, /^(DEFECT|binary|PHONON_PATH|PERT|PAIR_MODE|MASS_WEIGHTED|SPINLABS_REPO)=/))
        return substr(line, 1, RLENGTH)
      return ""
    }
    NR == FNR { k = key($0); if (k != "" && !(k in old)) old[k] = $0; next }
    {
      k = key($0)
      if (k == "MASS_WEIGHTED=" && mw == 0) { sub(/\$\{MASS_WEIGHTED:-1\}/, "0") ; print; next }
      if (k == "SPINLABS_REPO=" && (!(k in old) || old[k] ~ /</)) { print "SPINLABS_REPO=${SPINLABS_REPO:-" repo "}"; next }
      if (k != "" && (k in old)) { print old[k]; next }
      print
    }' "$old" "$new" > "$tmp"
  if [ "$DRY" = 1 ]; then
    echo "== $old"; diff "$old" "$tmp" || true; rm -f "$tmp"; continue
  fi
  [ -e "$old.bak" ] || cp -p "$old" "$old.bak"  # keep the first original
  mv "$tmp" "$old"
  echo "updated $old (MASS_WEIGHTED=$(sed -n 's/^MASS_WEIGHTED=\([^ ]*\).*/\1/p' "$old"))"
done
