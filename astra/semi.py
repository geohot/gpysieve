"""Complete each split factor with its unbounded inert prime partner."""
import argparse
import json
import re
import hashlib
from math import prod
from multiprocessing import Pool
from pathlib import Path
import subprocess
import time

from sympy import factorint
from campaign import ROOT,emit,gp_check
from mixed import DS,symbol
from search import mr,is_strong_lucas_prp


def exponents(p,a,b):
    odd=factorint(p-1); A=(a&-a)*prod(q**e for q,e in odd.items() if q!=2 and b%q); B=(b&-b)*prod(q**e for q,e in odd.items() if q!=2 and a%q)
    assert 2*((p-1)//a)%B==0
    return A,B


def complete(row):
    D,p,a,b=row; start=time.monotonic(); A,B=exponents(p,a,b);result=subprocess.run(['gp','-s','192M','-D','parisizemax=2G','-fq',str(ROOT/'compact.gp')],input=f'compact({D},{p},{A},{B})\n',text=True,capture_output=True,check=True,timeout=600)
    # GP emits stack-growth telemetry on stderr even on a successful evaluation.
    unexpected=[s for s in result.stderr.splitlines() if s.strip() and not re.fullmatch(r'\s*\*\*\* [^:]+: Warning: increasing stack size to \d+\.',s)]
    if unexpected:raise RuntimeError('\n'.join(unexpected))
    bits,qs=json.loads(result.stdout);prefixpass=0;hits=[]
    for q in qs:
        n=p*q
        if (n-1)%a or (n+1)%b:continue
        if not all(symbol(d,p)*symbol(d,q)==(-1 if d==D else 1) for d in DS[:DS.index(D)+1]):continue
        prefixpass+=1
        if mr(n,[2]) and is_strong_lucas_prp(n):hits.append([p,q])
    return dict(id=f'{D}:{p}',family=f'semi-{D}',prefixes=1,large_prime_divisors=len(qs),prefix_pass=prefixpass,base_exp=A,lucas_exp=B,max_gcd_bits=bits,seconds=round(time.monotonic()-start,3)),hits


def wrap(row):
    try:return complete(row)
    except (OSError,RuntimeError,subprocess.SubprocessError,ValueError) as e:return dict(id=f'{row[0]}:{row[1]}',error=str(e)[:2000]),[]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--workers',type=int,default=12);args=ap.parse_args()
    data=args.input.read_bytes();key=hashlib.sha256(data+b''.join((ROOT/f).read_bytes() for f in ('semi.py','compact.gp','split_scan.cpp','split_segment.cpp'))).hexdigest();rows=sorted({tuple(map(int,line.split())) for line in data.splitlines()},key=lambda r:r[1])
    with args.out.open('a+') as f:
        import fcntl
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); f.seek(0); done=set()
        for line in f:
            r=json.loads(line);assert r['manifest']==key
            if r['event']=='verified':quit()
            if r['event']=='done':done.add(r['id'])
        with Pool(args.workers) as pool:
            for r,hits in pool.imap_unordered(wrap,(v for v in rows if f'{v[0]}:{v[1]}' not in done)):
                for fs in hits:
                    emit(f,dict(id=r['id'],manifest=key,event='candidate',factors=fs,n=prod(fs)));ok,trace=gp_check(fs,'gp');emit(f,dict(id=r['id'],manifest=key,event='verified' if ok else 'disagreement',factors=fs,n=prod(fs),trace=trace));
                    if not ok:raise RuntimeError(trace)
                    quit()
                emit(f,dict(event='incomplete' if 'error' in r else 'done',manifest=key,**r))
