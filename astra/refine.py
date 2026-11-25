"""Spend a bounded factorization budget on the smallest missing Lucas indices first."""
import argparse
from multiprocessing import Pool
import hashlib,json,subprocess
from pathlib import Path
from math import prod
from campaign import emit


def attempt(r):
    known=set(r['primes']); bad=[]
    for z in r['unresolved']:
        cmd=f'f=factor({z});for(i=1,matsize(f)[1], print([f[i,1],f[i,2],isprime(f[i,1])]))\n'
        try:
            got=subprocess.run(['gp','-s','128M','-fq'],input=cmd,text=True,capture_output=True,check=True,timeout=20)
            if got.stderr:raise RuntimeError(got.stderr)
            rows=[json.loads(l) for l in got.stdout.splitlines()];assert prod(q**e for q,e,_ in rows)==z
            for q,_,ok in rows:
                if ok:known.add(q)
                else:bad.append(q)
        except (RuntimeError,subprocess.SubprocessError):bad.append(z)
    return dict(id=r['id'],D=r['D'],m=r['m'],bits=r['bits'],primes=sorted(known),unresolved=bad,source_manifest=r['manifest'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('sources',nargs='+',type=Path);ap.add_argument('--T',type=int,required=True);ap.add_argument('--workers',type=int,default=12);args=ap.parse_args();rows={};digest=hashlib.sha256()
    for path in args.sources:
        data=path.read_bytes();digest.update(data)
        for line in data.splitlines():
            r=json.loads(line);rows[r['id']]=r
    tasks=sorted([r for r in rows.values() if r['unresolved'] and (r['m'] if r['m']%2 else r['m']//2)<=args.T],key=lambda r:r['m'] if r['m']%2 else r['m']//2);key=digest.hexdigest();done=set();print(len(tasks),'unresolved components',flush=True)
    with args.out.open('a+') as f,Pool(args.workers) as pool:
        f.seek(0)
        for l in f:
            v=json.loads(l);assert v['manifest']==key;done.add(v['id'])
        for row in pool.imap_unordered(attempt,(r for r in tasks if r['id'] not in done)):emit(f,dict(event='factored',manifest=key,**row))
