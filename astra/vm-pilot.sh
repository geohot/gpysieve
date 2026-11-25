#!/usr/bin/env bash
# One bounded, resumable D=5 pilot. No network access or provisioning.
set -euo pipefail
cd "$(dirname "$0")"
export ASTRA_GP="${ASTRA_GP:-$(command -v gp)}"
: "${ASTRA_GP:?install pari-gp or set ASTRA_GP}"
python -m unittest -q
out="${OUT:-runs/vm-anchor-30000-40000}"
mkdir -p "$out"
python campaign.py make "$out/jobs.tmp" --family 5 --width 41 --min-anchor 30000 --limit 40000
if [[ -f "$out/jobs.json" ]]; then cmp "$out/jobs.tmp" "$out/jobs.json"; rm "$out/jobs.tmp"; else mv "$out/jobs.tmp" "$out/jobs.json"; fi
sha256sum "$out/jobs.json"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
for i in {0..7}; do
  nice -n 10 python -u campaign.py run "$out/jobs.json" "$out/shard-$i.jsonl" --shard "$i" --shards 8 --gp "$ASTRA_GP" >"$out/worker-$i.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
trap - EXIT
python campaign_status.py "$out"/shard-*.jsonl
exit "$rc"
