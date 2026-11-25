"""Deterministic anchored search campaign; see CAMPAIGN.md."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from math import gcd, lcm, prod
import os
from pathlib import Path
import subprocess
import time

from sympy import factorint, primerange
from sympy.ntheory.primetest import mr, is_strong_lucas_prp
from paper_search import anchored, periods
from pomerance import paper_pool

ROOT = Path(__file__).resolve().parent


def lucas_u(k, q, p):
    # x^k = U_k*x - Q*U_(k-1) in F_p[x]/(x^2-x+Q).
    a,b=0,1
    for bit in bin(k)[2:]:
        a,b=(a*(a+2*b))%p,(b*b-q*a*a)%p
        if bit=='1': a,b=(a+b)%p,-q*a%p
    return a


def rows_for(D,limit):
    for p in primerange(11,limit+1):
        if D==5:
            if p%5 not in (2,3): continue
            a,b=periods(p)
        elif D==-7:
            if p%5 not in (1,4) or pow(-7,(p-1)//2,p)!=p-1: continue
            a,b=p-1,p+1
            for f in factorint(a):
                while a%f==0 and pow(2,a//f,p)==1: a//=f
            for f in factorint(b):
                while b%f==0 and lucas_u(b//f,2,p)==0: b//=f
        else: raise ValueError(D)
        yield (p,a,b)


def compatible(x,y):
    p,a,b=x; q,c,d=y
    return (2%gcd(a,d)==2%gcd(c,b)==0 and gcd(p,c*d)==gcd(q,a*b)==1)


def make_jobs(width=39,limit=20000,min_anchor=0,family='all'):
    waves=[]
    if family in ('all','paper'):
        pool=paper_pool(2048,5,10**10)['primes']; core=[[p,*periods(p)] for p in pool[:width]]
        waves.append([dict(family='paper',anchor=[p,*periods(p)],core=core) for p in pool[width:] if p>min_anchor])
    for D in (5,-7):
        if family not in ('all',str(D)): continue
        groups=defaultdict(list); jobs=[]
        for row in rows_for(D,limit):
            p,a,b=row; group=groups[a&-a,b&-b]
            # Only retain mutually compatible core rows. Different anchors
            # in one family are disjoint because the anchor is the largest.
            if p<=min_anchor:
                group.append(row); continue
            todo=[y for y in group if compatible(row,y)]; group.append(row)
            if len(todo)<8: continue
            chosen=[]; A=a; B=b
            while todo and len(chosen)<width:
                def cost(y):
                    q,c,d=y
                    return c//gcd(c,A)*(d//gcd(d,B)),q
                best=min(todo,key=cost); chosen.append(best)
                A=lcm(A,best[1]); B=lcm(B,best[2]); todo=[y for y in todo if y!=best and compatible(best,y)]
            if len(chosen)>=8: jobs.append(dict(family=str(D),anchor=row,core=sorted(chosen)))
        waves.append(jobs)
    # Interleave families rather than serializing the whole paper sweep first.
    return [wave[i] for i in range(max(map(len,waves))) for wave in waves if i<len(wave)]


def emit(f,record):
    print(json.dumps(record,separators=(',',':')),file=f,flush=True); os.fsync(f.fileno())


def gp_check(fs,gp):
    # Only integers interpolated; do not treat GP's process exit alone as a verdict.
    if len(fs)<2 or not all(type(p) is int and p>1 for p in fs): raise ValueError(fs)
    cmd=f'print("VERDICT=",verify([{",".join(map(str,fs))}]))\n'
    r=subprocess.run([gp,'-fq',str(ROOT/'verify.gp')],input=cmd,text=True,capture_output=True,timeout=120,check=True)
    if r.stderr or not r.stdout.splitlines()[-1:] in (['VERDICT=0'],['VERDICT=1']): raise RuntimeError((r.stderr,r.stdout))
    return r.stdout.endswith('VERDICT=1\n'),r.stdout


def scan(job,on_candidate):
    p,a,b=job['anchor']; core=job['core']; sides={q:(c,d) for q,c,d in core}
    counts=dict(subsets=(1<<(len(core)-1))-1,anchor_pass=0,two_pass=0,full_pass=0)
    for fs in anchored(list(sides),p,lcm(a,b)):
        n=prod(fs)
        if n<1<<64: continue
        counts['anchor_pass']+=1
        # Choose a fixed second test (largest remaining prime).
        for i,q in enumerate(reversed(fs[:-1])):
            c,d=sides[q]
            if (n-1)%c or (n+1)%d: break
            if i==0: counts['two_pass']+=1
        else:
            counts['full_pass']+=1; on_candidate(fs)
    return counts


def run(manifest,out,shard,shards,gp):
    raw=manifest.read_bytes(); jobs=json.loads(raw); digest=hashlib.sha256(raw).hexdigest()
    with out.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); f.seek(0); done=set()
        for line in f:
            row=json.loads(line)
            if row['manifest']!=digest: raise ValueError('manifest changed')
            if row['event']=='done': done.add(row['id'])
            if row['event']=='verified': return
        for j in range(shard,len(jobs),shards):
            if j in done: continue
            job=jobs[j]; start=time.monotonic(); hit=False
            def log(event,**kw): emit(f,dict(manifest=digest,id=j,event=event,**kw))
            def verify(fs):
                nonlocal hit
                log('candidate',factors=fs,n=prod(fs)); verdict,trace=gp_check(fs,gp)
                if not (verdict and mr(prod(fs),[2]) and is_strong_lucas_prp(prod(fs))):
                    log('disagreement',trace=trace); raise RuntimeError('verifier disagreement')
                log('verified',factors=fs,n=prod(fs),trace=trace); hit=True
            counts=scan(job,verify)
            log('done',family=job['family'],anchor=job['anchor'][0],width=len(job['core']),seconds=round(time.monotonic()-start,3),**counts)
            if hit: return


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    m=sub.add_parser('make'); m.add_argument('manifest',type=Path); m.add_argument('--width',type=int,default=39); m.add_argument('--limit',type=int,default=20000)
    m.add_argument('--min-anchor',type=int,default=0,help='exclusive lower bound for the largest selected prime')
    m.add_argument('--family',choices=['all','paper','5','-7'],default='all')
    w=sub.add_parser('run'); w.add_argument('manifest',type=Path); w.add_argument('out',type=Path); w.add_argument('--shards',type=int,default=1); w.add_argument('--shard',type=int,default=0); w.add_argument('--gp',default='gp')
    args=ap.parse_args()
    if args.cmd=='make':
        if not 8<=args.width<=42: ap.error('width must be 8..42')
        jobs=make_jobs(args.width,args.limit,args.min_anchor,args.family); args.manifest.write_text(json.dumps(jobs,separators=(',',':'))+'\n'); print(len(jobs),'jobs')
    else:
        if not 0<=args.shard<args.shards: ap.error('invalid shard')
        run(args.manifest,args.out,args.shard,args.shards,args.gp)
