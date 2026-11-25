#!/usr/bin/env bash
# Run on the named dedicated VM. Local collector enforces allocation deadline.
set -eu
cd "$(dirname "$0")"
run(){ local d="runs/iris-mixed-$1"; shift; mkdir -p "$d"; python3 -u mixed.py "$d/jobs.jsonl" --workers 70 "$@" >"$d/worker.log" 2>&1; if grep -q '"event":"verified"' "$d/jobs.jsonl"; then exit 0; fi; }
run b --width 48 --base 35000 --hi 200000 --ds -15 17 -19 21
run c --width 50 --base 35000 --lo 150000 --hi 260000 --ds -7 -11 13
for lo in $(seq 260000 100000 2000000); do run "$lo" --width 50 --base 40000 --lo "$lo" --hi "$((lo+100000))" --ds -7 -11 13 -15 17 -19 21; done
