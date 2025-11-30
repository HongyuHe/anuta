#!/usr/bin/env bash
set -euo pipefail

# Run dt classification across CIDDs train splits for selected attack labels.

labels=(bruteForce dos pingScan portScan)

PY_CMD="python anuta"
DATASET="-dataset=cidds"
TREE="-tree=dt"
CLASSIFY="-classify"
DATA_DIR="data/attacks"

for label in "${labels[@]}"; do
    data_file="${DATA_DIR}/cidds_intl_train_${label}.csv"

    if [[ ! -f "$data_file" ]]; then
        echo "Skipping ${label}: missing ${data_file}" >&2
        continue
    fi

    echo "Running classification for ${label}"
    $PY_CMD $DATASET -data="$data_file" $TREE $CLASSIFY -label="$label"
    echo "Done: ${label}"
    echo
done

echo "All classifications finished."
