"""Dynamic, fork-shared CPU queue; native mandatory-anchor join and exact replay."""
import argparse
from collections import defaultdict
from math import gcd,lcm,prod
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import subprocess
import time

from sympy import primerange
from campaign import ROOT, compatible, rows_for, periods, emit, gp_check
from search import mr,is_strong_lucas_prp


def native(job):
    p,a,b=job['anchor']; m=lcm(a,b); core=[r for r in job['core'] if gcd(r[0],m)==1]
    if m>=1<<62 or any(lcm(c,d)>=1<<62 for _,c,d in core): raise ValueError('oversize modulus')
    text=f'{p} {a} {b}\n'+''.join(f'{q} {c} {d} {pow(q,-1,m)}\n' for q,c,d in core)
    lines=subprocess.check_output([str(ROOT/'join')],input=text,text=True).splitlines()
    label,one,two,full=lines[-1].split(); assert label=='done' and int(full)==len(lines)-1
    factors=[]
    for line in lines[:-1]:
        label,mask=line.split(); assert label=='hit'; factors.append([q for i,(q,_,_) in enumerate(core) if int(mask)>>i&1]+[p])
    return dict(anchor_pass=int(one),two_pass=int(two),full_pass=int(full),subsets=(1<<(len(job['core'])-1))-1),factors


def task(p):
    start=time.monotonic(); a,b=periods(p); anchor=(p,a,b); A=a; B=b
    todo=[r for r in groups[a&-a,b&-b] if compatible(anchor,r)]
    core=[]
    while todo and len(core)<width:
        def cost(r):
            q,c,d=r; return c//gcd(c,A)*(d//gcd(d,B)),q
        x=min(todo,key=cost); core.append(x); A=lcm(A,x[1]);B=lcm(B,x[2]);todo=[y for y in todo if y!=x and compatible(x,y)]
    job=dict(anchor=anchor,core=sorted(core)); counts,fs=native(job) if len(core)>=2 else ({'subsets':0},[])
    return dict(id=p,family='5-native',core=[x[0] for x in job['core']],seconds=round(time.monotonic()-start,3),**counts),fs


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('out',type=Path);ap.add_argument('--workers',type=int,required=True);ap.add_argument('--lo',type=int,default=40000);ap.add_argument('--hi',type=int,default=2000000);ap.add_argument('--width',type=int,default=45);ap.add_argument('--base',type=int,default=12000)
    args=ap.parse_args(); width=args.width
    if not (8<=width<=48 and args.base<=args.lo<args.hi<1<<30): ap.error('invalid bounds')
    key=hashlib.sha256(repr((args.lo,args.hi,width,args.base)).encode()+b''.join((ROOT/f).read_bytes() for f in ('bigcpu.py','campaign.py','join.cpp','paper_search.py','search.py'))).hexdigest(); groups=defaultdict(list)
    for x in rows_for(5,args.base): groups[x[1]&-x[1],x[2]&-x[2]].append(x)
    with args.out.open('a+') as f:
        import fcntl
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); f.seek(0); done=set()
        for l in f:
            r=json.loads(l); assert r['manifest']==key
            if r['event']=='verified': quit()
            if r['event']=='done':done.add(r['id'])
        pending=[p for p in primerange(args.lo+1,args.hi+1) if p%5 in (2,3) and p not in done]
        print(len(pending),'pending anchors;',args.workers,'workers',flush=True)
        with Pool(args.workers) as pool:
            for row,fs in pool.imap_unordered(task,pending,chunksize=1):
                for factors in fs:
                    n=prod(factors); emit(f,dict(manifest=key,id=row['id'],event='candidate',factors=factors,n=n)); ok,trace=gp_check(factors,'gp')
                    if not(ok and mr(n,[2]) and is_strong_lucas_prp(n)): raise RuntimeError(trace)
                    emit(f,dict(manifest=key,id=row['id'],event='verified',factors=factors,n=n,trace=trace));quit()
                emit(f,dict(manifest=key,event='done',**row))
