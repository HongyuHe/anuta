#!/usr/bin/env bash
set -euo pipefail

# Labels to process
labels=(norm portScan dos pingScan bruteForce)

# Common command parts
PY_CMD="python anuta"
DATASET="-dataset=cidds"
LEARN="-learn"

LOGDIR="logs"
mkdir -p "$LOGDIR"

for lbl in "${labels[@]}"; do
    echo "===== Running label: $lbl ====="

    # Dynamically select the dataset file
    DATAFILE="data/attacks/cidds_intl_${lbl}.csv"

    if [[ ! -f "$DATAFILE" ]]; then
        echo "⚠️  Warning: File $DATAFILE not found, skipping."
        continue
    fi

    LOGFILE="$LOGDIR/run_${lbl}.log"

    # Execute the command
    $PY_CMD $DATASET -data="$DATAFILE" $LEARN -label="$lbl" 2>&1 | tee "$LOGFILE"

    echo "✅ Completed: $lbl (log: $LOGFILE)"
    echo
done

echo "All jobs finished."
