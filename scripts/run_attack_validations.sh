#!/usr/bin/env bash

set -euo pipefail

# RULES_PATH="denial_cidds_50000_p4_wk12norm_strictpreds_neg.pl"
# RULES_PATH="fpgrowth_cidds_5000_wk12norm.pl"
# DATASET="cidds"
# attack_files=(
#   "data/attacks/cidds_intl_bruteForce.csv"
#   "data/attacks/cidds_intl_dos.csv"
#   "data/attacks/cidds_intl_portScan.csv"
#   "data/attacks/cidds_intl_pingScan.csv"
#   "/mnt/ann/hy/data/cidds_wk3_full.csv"
#   "/mnt/ann/hy/data/cidds_wk4_full.csv"
# )

RULES_PATH="denial_yatesbury_1000000_p4_norm.pl"
DATASET="yatesbury"
attack_files=(
#   "data/Yatesbury/portscan_attack.csv"
#   "data/Yatesbury/synflood_attack.csv"
#   "data/Yatesbury/botnet_attack.csv"
  "data/Yatesbury/dbinjection_attack.csv"
  "data/Yatesbury/dnsamp_attack.csv"
#   "data/Yatesbury/ddosudp_attack.csv"
#   "data/Yatesbury/unauthaccess_attack.csv"
)


for attack_file in "${attack_files[@]}"; do
  if [ ! -f "$attack_file" ]; then
    echo "Attack file '$attack_file' not found, skipping." >&2
continue
  fi

#   label=$(basename "$attack_file")
#   label="${label%.*}"

  echo "Running validator for attack file: $attack_file"

#   limit_args=()
#   case "$attack_file" in
#     */cidds_wk3_full.csv|*/cidds_wk4_full.csv|*/cidds_intl_dos.csv|*/cidds_intl_portScan.csv)
#       limit_args=(-limit=100000)
#       ;;
#   esac

  python anuta \
    -dataset="$DATASET" \
    -data="$attack_file" \
    -rules="$RULES_PATH" \
    -limit=10000 \
    -validate
done
