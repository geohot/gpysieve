#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"; lo=100000
for hi in 10000000 40000000 100000000 200000000 300000000; do
  d=runs/iris-semi-$hi; mkdir -p "$d"
  if [[ ! -f $d/split.txt ]]; then OMP_NUM_THREADS=12 ./split_scan "$lo" "$hi" | LC_ALL=C sort -k2,2n -k1,1n >"$d/split.tmp"; mv "$d/split.tmp" "$d/split.txt"; fi
  python3 semi.py "$d/split.txt" "$d/jobs.jsonl" --workers 12 >"$d/worker.log" 2>&1
  if grep -q '"event":"verified"' "$d/jobs.jsonl"; then exit 0; fi
  lo=$hi
done
