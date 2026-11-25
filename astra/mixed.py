"""Discriminant-aware mixed split/inert, conditional-character campaign."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from math import gcd,lcm,prod
from multiprocessing import Pool
from pathlib import Path
import time

from sympy import factorint,primerange
from campaign import ROOT,compatible,emit,gp_check,lucas_u
from char_sieve import solve
from search import mr,is_strong_lucas_prp

DS=(5,-7,9,-11,13,-15,17,-19,21)

@lru_cache(maxsize=100000)
def symbol(D,p):
    x=pow(D,(p-1)//2,p)
    return -1 if x==p-1 else x

@lru_cache(maxsize=50000)
def order(p):
    a=p-1
    for f in factorint(a):
        while a%f==0 and pow(2,a//f,p)==1:a//=f
    return a


def getrow(D,p):
    prefix=DS[:DS.index(D)+1]; Q=(1-D)//4
    if Q%p==0 or any(d%p==0 for d in prefix):return None
    a=order(p); b=p-symbol(D,p)
    for f in factorint(b):
        while b%f==0 and lucas_u(b//f,Q,p)==0:b//=f
    return (p,a,b) if 2%gcd(a,b)==0 else None


def affine_count(n,eqs):
    basis={}
    for mask,rhs in eqs:
        while mask:
            i=mask.bit_length()-1
            if i not in basis: basis[i]=mask,rhs;break
            m,r=basis[i];mask^=m;rhs^=r
        else:
            if rhs:return 0
    return 1<<(n-len(basis))


def job_for(D,anchor,core):
    core=sorted(core);p=anchor[0];fixed=[]; n=len(core)
    for d in DS[:DS.index(D)+1]:
        mask=sum((symbol(d,q)<0)<<i for i,(q,_,_) in enumerate(core)); rhs=(d==D)^(symbol(d,p)<0)
        fixed.append((mask,int(rhs)))
    # For D=5 do not repeat the all-inert construction. For every D exclude
    # the empty core. The current discriminant already forces an odd inert count.
    anybits=(1<<n)-1
    if D==5 and symbol(D,p)<0:anybits=sum((symbol(D,q)>0)<<i for i,(q,_,_) in enumerate(core))
    space=affine_count(n,fixed)-affine_count(n,fixed+[(1<<i,0) for i in range(n) if anybits>>i&1])
    return dict(core=core,anchor=anchor,fixed=fixed,any=anybits,space=space)


def task(spec):
    D,anchor=spec;p,a,b=anchor;start=time.monotonic();A=a;B=b;key=(lcm(a,b),p)
    todo=[r for r in groups[D,a&-a,b&-b] if (lcm(r[1],r[2]),r[0])<key and compatible(anchor,r)]
    core=[]
    while todo and len(core)<width:
        def cost(r):return r[1]//gcd(r[1],A)*(r[2]//gcd(r[2],B)),r[0]
        options=todo
        if not core and D==5 and symbol(D,p)<0:
            options=[r for r in todo if symbol(D,r[0])>0]
            if not options:break
        x=min(options,key=cost);core.append(x);A=lcm(A,x[1]);B=lcm(B,x[2]);todo=[y for y in todo if y!=x and compatible(x,y)]
    j=job_for(D,anchor,core); counts=dict(subsets=j['space']);fs=[]
    if j['space'] and prod(r[0] for r in core)*p>=1<<64:counts,fs=solve(j)
    return dict(id=f'{D}:{p}',family=f'mixed-{D}',core=[q for q,_,_ in j['core']],seconds=round(time.monotonic()-start,4),**counts),fs


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--workers',type=int,required=True);ap.add_argument('--width',type=int,default=46);ap.add_argument('--base',type=int,default=20000);ap.add_argument('--lo',type=int,default=10);ap.add_argument('--hi',type=int,default=150000);ap.add_argument('--ds',nargs='+',type=int,choices=[v for v in DS if v!=9],default=[5,-7,-11,13]);args=ap.parse_args();width=args.width
    if not (2<=width<=55 and args.hi<=10000000 and args.base<=args.hi and 2<args.lo<args.hi):ap.error('invalid bounds')
    groups=defaultdict(list);tasks=[]
    for p in primerange(3,args.hi+1):
        for D in args.ds:
            r=getrow(D,p)
            if r:
                if p<=args.base:groups[D,r[1]&-r[1],r[2]&-r[2]].append(r)
                if p>args.lo:tasks.append((D,r))
    key=hashlib.sha256(repr((args.base,args.ds,args.lo,args.hi,width)).encode()+b''.join((ROOT/f).read_bytes() for f in ('mixed.py','chars.cpp','char_sieve.py','campaign.py','search.py'))).hexdigest()
    with args.out.open('a+') as f:
        import fcntl
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.seek(0);done=set()
        for line in f:
            r=json.loads(line);assert r['manifest']==key
            if r['event']=='verified':quit()
            if r['event']=='done':done.add(r['id'])
        pending=[v for v in tasks if f'{v[0]}:{v[1][0]}' not in done]; print(len(pending),'pending',flush=True)
        with Pool(args.workers) as pool:
            for row,fs in pool.imap_unordered(task,pending):
                for factors in fs:
                    n=prod(factors);emit(f,dict(event='candidate',id=row['id'],manifest=key,n=n,factors=factors));ok,trace=gp_check(factors,'gp');s=mr(n,[2]) and is_strong_lucas_prp(n)
                    if not (ok and s):emit(f,dict(event='disagreement',manifest=key,id=row['id'],trace=trace));raise RuntimeError('independent disagreement')
                    emit(f,dict(event='verified',manifest=key,id=row['id'],n=n,factors=factors,trace=trace));quit()
                emit(f,dict(manifest=key,event='done',**row))
