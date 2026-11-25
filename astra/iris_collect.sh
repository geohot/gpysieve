#!/usr/bin/env bash
# Local supervisor for ONE named VM. Mirror receipts; release only after a final successful copy.
set -u
cd "$(dirname "$0")"
vm=$1; end=$2
[[ $vm =~ ^bigcpu-[0-9a-f]{16}$ ]] || exit 2
out="runs/$vm"; host="runner@$vm"; mkdir -p "$out"
export RSYNC_RSH='ssh -oBatchMode=yes -oConnectTimeout=12 -oServerAliveInterval=15 -oServerAliveCountMax=2 -J iris'
sync_runs(){ rsync -a --timeout=90 "$host:astra/runs/" "$out/remote/"; }
while (( $(date +%s)<end )); do
  if sync_runs; then
    if grep -rq --include='*.jsonl' '"event":"verified"' "$out/remote/" || grep -q '^Verified composite:' "$out/remote/kit/solver.log" 2>/dev/null; then break; fi
  fi
  sleep 30
done
# Stop writers on the dedicated VM; save pending/completed records before deletion.
while ! ssh -oBatchMode=yes -oConnectTimeout=12 -J iris "$host" 'cd astra && python3 -' <<'PY'
from pathlib import Path
import os,signal
base=Path.cwd(); me=os.getpid(); parent=os.getppid()
for p in Path('/proc').glob('[0-9]*'):
    try:
        pid=int(p.name)
        if pid not in (me,parent) and (p/'cwd').resolve().is_relative_to(base): os.kill(pid,signal.SIGSTOP)
    except (FileNotFoundError,PermissionError,ProcessLookupError): pass
PY
do sleep 15; done
until sync_runs; do sleep 20; done
if grep -q '^Certified prime factors:' "$out/remote/kit/solver.log" 2>/dev/null; then
  # Replay the MILP kit's factor list, separately from its SymPy test.
  awk '/^Certified prime factors:/{printf "print(verify([";for(i=4;i<=NF;i++)printf "%s%s",(i>4?",":""),$i;print "]))"}' "$out/remote/kit/solver.log" | ssh -J iris "$host" 'cd astra; gp -fq verify.gp' >"$out/kit-gp.trace" || exit 1
fi
ssh -oBatchMode=yes iris delete "$vm" && date -u >"$out/released.txt"
