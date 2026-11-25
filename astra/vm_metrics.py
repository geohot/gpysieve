"""Atomic low-overhead VM heartbeat for the receipt mirror."""
import json
import os
from pathlib import Path
import sys
import time


def cpu():
    v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9])); return sum(v),v[3]+v[4]

if __name__=='__main__':
    out=Path(sys.argv[1]); old,idle=cpu()
    while True:
        time.sleep(5); new,idle2=cpu(); mem={a:int(b.split()[0])*1024 for a,b in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}; commands={}
        for f in Path('/proc').glob('[0-9]*/comm'):
            try: name=f.read_text().strip()
            except (OSError,UnicodeError): continue
            commands[name]=commands.get(name,0)+1
        record=dict(at=time.time(),busy=round(100*(1-(idle2-idle)/max(1,new-old)),1),cores=len(os.sched_getaffinity(0)),used=mem['MemTotal']-mem['MemAvailable'],total=mem['MemTotal'],joins=commands.get('join',0),gp=commands.get('gp',0),load=round(os.getloadavg()[0],2))
        tmp=out.with_suffix('.tmp'); tmp.write_text(json.dumps(record)); tmp.replace(out); old,idle=new,idle2
