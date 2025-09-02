#!/usr/bin/env python3
"""
Sanity check that joblib.Parallel(loky) preserves order of results.

Usage:
  python sanity_joblib_order.py --n 20 --n-jobs 4
"""

import argparse
import random
import time
from joblib import Parallel, delayed


def _process_examples(i, chunk):
    # Add random jitter so tasks finish out of order.
    time.sleep(random.uniform(0.01, 0.2))
    # Return both the index and a deterministic transform of the chunk
    return i, chunk * 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12, help="Number of chunks")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Simple "chunks": 0..n-1
    chunks = list(range(args.n))

    # Run in parallel preserving submission order
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_process_examples)(i, chunk)
        for i, chunk in enumerate(chunks)
    )

    # --- Assertions: results must align with enumerate(chunks) ---
    assert len(results) == len(chunks), "Length mismatch"
    for k, (idx, value) in enumerate(results):
        assert idx == k, f"Result index out of order at position {k}: got {idx}"
        expected_value = chunks[k] * 10
        assert value == expected_value, f"Wrong value at {k}: {value} != {expected_value}"

    print("OK ✅  joblib.Parallel(loky) preserved order.")
    print("First 5 results:", results[:5])


if __name__ == "__main__":
    main()
