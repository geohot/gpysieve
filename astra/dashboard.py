"""Read-only LAN/loopback dashboard. Reads a local mirror; never issues VM commands."""
import argparse
from collections import Counter, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time

HERE=Path(__file__).resolve().parent
FIELDS=('subsets','anchor_pass','two_pass','full_pass','prefixes','large_prime_divisors','prefix_pass')

def tail(p,n=2500):
    try:
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
    except OSError: return ''

class Monitor:
    def __init__(self,folder):
        self.folder=folder;self.remote=folder/'remote';self.lock=threading.Lock();self.snapshot={};self.reset()

    def reset(self):
        self.offset={};self.seen=set();self.counts=defaultdict(Counter);self.events=deque(maxlen=30);self.verdicts=Counter();self.errors=set();self.history=deque(maxlen=120);self.baseline=None;self.totals={};self.meta={};self.last=0

    def update(self):
        files=sorted(self.remote.glob('*/*.jsonl'))
        if any(p.stat().st_size<self.offset.get(p,0) for p in files):self.reset()
        for p in files:
            size=p.stat().st_size; pos=self.offset.get(p,0)
            if pos==size:continue
            with p.open('rb') as f:
                f.seek(pos)
                for line in f:
                    if not line.endswith(b'\n'):break
                    pos+=len(line)
                    try:
                        r=json.loads(line); self.record(p,r)
                    except (ValueError,KeyError,TypeError) as e:self.errors.add(f'{p.parent.name}/{p.name}: unreadable receipt ({type(e).__name__})')
                self.offset[p]=pos
            self.last=max(self.last,p.stat().st_mtime)
        waves=[]
        for name,c in sorted(self.counts.items(),key=lambda kv:kv[0],reverse=True):
            folder=self.remote/name;total=None
            if name not in self.totals:
                if (folder/'jobs.json').exists():self.totals[name]=len(json.loads((folder/'jobs.json').read_text()))
                elif name.startswith('iris-native'):
                    match=re.search(r'^(\d+) pending anchors;',tail(folder/'worker.log'),re.M)
                    # Pending count describes this invocation, so do not infer a denominator on resume.
                    if match and name=='iris-native':self.totals[name]=72411
            total=self.totals.get(name);waves.append(dict(name=name,jobs=c['done'],total=total,**{k:str(c[k]) for k in FIELDS}))
        heartbeat={}
        try:heartbeat=json.loads((self.remote/'heartbeat.json').read_text())
        except (OSError,ValueError):pass
        now=time.time(); counts=sum(self.counts.values(),Counter()); jobs=counts['done']
        # Rates are receipt arrivals at the mirror, not individual PRP calls per second.
        if self.baseline is None:self.baseline=(now,jobs)
        elif now-self.baseline[0]>=30:
            at,n=self.baseline;self.history.append(dict(at=now,rate=round((jobs-n)/(now-at),2)));self.baseline=(now,jobs)
        for f in self.remote.glob('*/*.log'):
            stat=f.stat(); stamp=(stat.st_size,stat.st_mtime)
            if self.meta.get(f)==stamp:continue
            self.meta[f]=stamp
            log=tail(f);label=f'{f.parent.name}/{f.name}'
            if 'Traceback (most recent call last)' in log or 'incomplete first wave:' in log:self.errors.add(label+': worker error')
        kit=tail(self.remote/'kit/solver.log');group=tail(self.remote/'kit/group.log');released=(self.folder/'released.txt').exists()
        snap=dict(vm=self.folder.name,at=now,heartbeat=heartbeat,last=self.last,released=released,deadline=tail(self.folder/'deadline').strip(),waves=waves,receipts=jobs,counts={k:str(counts[k]) for k in FIELDS},events=list(self.events),verdicts=dict(self.verdicts),history=list(self.history),errors=sorted(self.errors),solver=[s[:260] for s in (group+'\n'+kit).splitlines() if s.startswith(('Generated full','Solver returned','No candidate','Verified composite','2030 binary'))][-4:])
        with self.lock:self.snapshot=snap
        return snap

    def record(self,p,r):
        event=r['event']; name=p.parent.name
        key=(r['manifest'],r['id']) if 'manifest' in r else (tuple(r['params']),r.get('batch'))
        if event=='done':
            if key in self.seen:return
            self.seen.add(key); c=self.counts[name];c['done']+=1
            for k in FIELDS:c[k]+=int(r.get(k,0))
        elif event in ('candidate','disagreement','verified'):
            self.verdicts[event]+=1;self.events.append(dict(event=event,wave=name,id=str(r.get('id',r.get('batch',''))),factors=' × '.join(map(str,r.get('factors',[])))))

    def poll(self):
        while True:
            try:self.update()
            except (OSError,ValueError) as e:print('refresh:',e,flush=True)
            time.sleep(3)


def serve(monitor,hosts,port):
    page=(HERE/'dashboard.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=self.path.partition('?')[0]
            if path=='/':data=page; mime='text/html; charset=utf-8'
            elif path=='/api/status':
                with monitor.lock:data=json.dumps(monitor.snapshot).encode()
                mime='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    servers=[ThreadingHTTPServer((h,port),Handler) for h in hosts];monitor.update();threading.Thread(target=monitor.poll,daemon=True).start()
    for s in servers[:-1]:threading.Thread(target=s.serve_forever,daemon=True).start()
    servers[-1].serve_forever()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path);ap.add_argument('--bind',action='append');ap.add_argument('--port',type=int,default=8765);args=ap.parse_args();serve(Monitor(args.folder),args.bind or ['127.0.0.1'],args.port)
