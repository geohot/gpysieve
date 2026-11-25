"""Sharded exhaustive last-prime completion for compatible two-prime prefixes."""
import argparse
from collections import Counter
import fcntl
import json
from math import gcd, lcm
from pathlib import Path
import subprocess
import time

from campaign import ROOT, compatible, emit, gp_check, rows_for
from paper_search import periods
from search import mr,is_strong_lucas_prp


def prefixes(bound,D):
    # 3 and 7 are inert for 5; neither qualifies for our -7 family.
    rows=([(p,*periods(p)) for p in (3,7)] if D==5 else [])+list(rows_for(D,bound//(3 if D==5 else 11)))
    for i,x in enumerate(rows):
        p,a,b=x
        if p*p>=bound: break
        for j in range(i+1,len(rows)):
            y=rows[j]; q,c,d=y
            if p*q>bound: break
            if (a&-a,b&-b)!=(c&-c,d&-d) or not compatible(x,y): continue
            yield p,q,lcm(a,c),lcm(b,d)


def compute(batch,D,gp):
    cmd=''.join(f'completion({p*q},{D},{max(q+1,((1<<64)+p*q-1)//(p*q))})\n' for p,q,_,_ in batch)
    r=subprocess.run([gp,'-s','64M','-fq',str(ROOT/'gcd_scan.gp')],input=cmd,capture_output=True,text=True,check=True,timeout=1800)
    if r.stderr: raise RuntimeError(r.stderr)
    result=[json.loads(l) for l in r.stdout.splitlines()]
    if len(result)!=len(batch): raise RuntimeError('incomplete batch')
    return result


def scan(bound,lower,D,shard,shards,out,gp):
    params=[bound,lower,D,shard,shards]; jobs=[v for i,v in enumerate(prefixes(bound,D)) if i%shards==shard and v[0]*v[1]>lower]
    with out.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); f.seek(0); done=set()
        for line in f:
            row=json.loads(line)
            if row['params']!=params: raise ValueError('wrong log for this shard')
            if row['event']=='verified': return
            if row['event']=='done': done.add(row['batch'])
        for i in range(0,len(jobs),16):
            if i in done: continue
            batch=jobs[i:i+16]; start=time.monotonic(); result=compute(batch,D,gp); counts=Counter(prefixes=len(batch)); biggest=0
            for (p,q,A,B),(g,rs) in zip(batch,result):
                biggest=max(biggest,g.bit_length()); counts['large_prime_divisors']+=len(rs)
                for r in rs:
                    n=p*q*r
                    if (n-1)%A or (n+1)%B: continue
                    counts['prefix_pass']+=1
                    if not mr(n,[2]) or not is_strong_lucas_prp(n): continue
                    fs=[p,q,r]; emit(f,dict(event='candidate',params=params,batch=i,factors=fs,n=n)); ok,trace=gp_check(fs,gp)
                    emit(f,dict(event='verified' if ok else 'disagreement',params=params,batch=i,factors=fs,n=n,trace=trace))
                    if not ok: raise RuntimeError('verifier disagreement')
                    return
            emit(f,dict(event='done',params=params,batch=i,max_gcd_bits=biggest,seconds=round(time.monotonic()-start,3),**counts))
        emit(f,dict(event='complete',params=params,jobs=len(jobs)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('log',type=Path)
    ap.add_argument('--bound',type=int,required=True); ap.add_argument('--lower',type=int,default=0); ap.add_argument('--D',type=int,choices=[5,-7],required=True)
    ap.add_argument('--shard',type=int,default=0); ap.add_argument('--shards',type=int,default=1); ap.add_argument('--gp',default='gp')
    args=ap.parse_args()
    if not (0<=args.lower<args.bound and 0<=args.shard<args.shards): ap.error('invalid range')
    scan(args.bound,args.lower,args.D,args.shard,args.shards,args.log,args.gp)
