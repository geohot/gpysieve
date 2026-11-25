#!/usr/bin/env bash
# Continue after the initial worker exits; controller reclaims this dedicated VM at deadline.
set -eu
cd "$(dirname "$0")"
first=$1
while kill -0 "$first" 2>/dev/null; do sleep 2; done
if grep -q '"event":"verified"' runs/iris-native/jobs.jsonl; then exit 0; fi
# Require full receipts before advancing; never treat a crashed run as complete.
count=$(wc -l <runs/iris-native/jobs.jsonl); [[ $count == 72411 ]] || { echo "incomplete first wave: $count"; exit 1; }
lo=2000000
while :; do
  hi=$((lo+2000000)); dir=runs/iris-native-$lo; mkdir -p "$dir"
  python3 -u bigcpu.py "$dir/jobs.jsonl" --workers "$(nproc)" --lo "$lo" --hi "$hi" --width 47 --base 30000 >"$dir/worker.log" 2>&1
  if grep -q '"event":"verified"' "$dir/jobs.jsonl"; then exit 0; fi
  lo=$hi
done
