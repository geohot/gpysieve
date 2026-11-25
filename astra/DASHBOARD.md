# Astra live monitor

Open **http://127.0.0.1:8765/** on the controller or **http://100.76.148.17:8765/** from its Tailscale network (subject to tailnet ACLs). Read-only JSON is at `/api/status`. There is no public listener, command endpoint, or file browser.

The server is in local tmux session `astra-monitor`. To restart if needed:

```sh
cd /home/light/fun/gpysieve/astra
python dashboard.py runs/bigcpu-501c65a3a23b73d3 --bind 127.0.0.1 --bind 100.76.148.17 --port 8765
```

`vm_metrics.py` samples `/proc` on the dedicated VM every five seconds and atomically updates `runs/heartbeat.json`. The `iris_collect.sh` supervisor mirrors VM receipts and telemetry approximately every 30 seconds (plus transfer time) into `runs/<vm>/remote/`. The web process does not SSH, provision VMs, or submit jobs. It tails completed JSONL records on the controller every three seconds; browsers poll every four seconds. A heartbeat older than two minutes is shown as stale. After release, the same URL serves the archive.

Rates measure receipt arrivals at the mirror; transfer batching can make the chart uneven. The ledger counts logical subset coverage, which may overlap across constructions. Jobs and GCD batches are not uniform amounts of work. Anchor and two-factor passes are preliminary congruence checks, not counterexamples. Only independent verification receipts increment the verified-hit counter. The integer solver's output is shown separately.

`python -m unittest test_dashboard` tests partial receipts, duplicate job IDs, integer precision, and separation of GCD, candidate, and verified counts.
