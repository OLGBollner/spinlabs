#!/usr/bin/env bash
# Regenerate every paper figure and the manifest / numbers files (numbers.md, numbers_vsi.md), in the order M1-M5, S1-S6, F1-F3.
# Run from the repo root. Cached rates live in figures/out/cache/: delete it after a code or data change.
set -euo pipefail
cd "$(dirname "$0")/../.."
for s in m1_zfs_table m2_nv_t1 m3_rho m4_t1_vv_clv m5_vsi_t1 s1_coupling_scatter s2_s3_spectral s4_t1_vs_sigma s5_fill_ratio s6_debye f1_sic_rates f2_finite_size_nv f3_vsi_levels vsi_numbers build_reports; do
    echo "== $s"
    python "scripts/paper/$s.py"
done
