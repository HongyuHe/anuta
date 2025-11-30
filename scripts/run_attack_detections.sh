#!/usr/bin/env bash

set -euo pipefail

RULES_PATH="dt_cidds_all_e1_norm_wk12_portcls_tos.pl"
DATASET="cidds"

attack_files=(
  "data/attacks/cidds_intl_bruteForce.csv"
  "data/attacks/cidds_intl_dos.csv"
  "data/attacks/cidds_intl_portScan.csv"
  "data/attacks/cidds_intl_pingScan.csv"
  "/mnt/ann/hy/data/cidds_wk3_full.csv"
  "/mnt/ann/hy/data/cidds_wk4_full.csv"
)

for attack_file in "${attack_files[@]}"; do
  if [ ! -f "$attack_file" ]; then
    echo "Attack file '$attack_file' not found, skipping." >&2
    continue
  fi

#   label=$(basename "$attack_file")
#   label="${label%.*}"

  echo "Running detector for attack file: $attack_file"
  python anuta \
    -dataset="$DATASET" \
    -data="$attack_file" \
    -rules="$RULES_PATH" \
    -detect
done
