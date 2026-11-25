"""Finite resultant sieve for arbitrary-sized triples with bounded additive diameter."""
import argparse,hashlib,json,subprocess
from collections import Counter
from pathlib import Path
from math import prod
from sympy import jacobi_symbol
from search import mr,is_strong_lucas_prp

ROOT=Path(__file__).resolve().parent
DS=(5,-7,-11,13,-15,17,-19,21)
def actual(n):
    d=5
    while True:
        j=jacobi_symbol(d,n)
        if j==-1:return d
        if j==0:return None
        d=-d-2 if d>0 else 2-d

def scan(limit,ds,gp):
    # If all residues mod 3 occur, the only admissible translate contains 3.
    # We keep even those patterns in the sieve; no exceptional translates lost.
    cmd=''.join(f'for(u=2,{limit}-2,if(u%2==0,forstep(v=2,{limit}-u,2,print([u,v,{d},shape(u,v,{d})]))))\n' for d in ds);stats=Counter();log=[];fermat=set();hits=[];missing=[];maxbits=0
    r=subprocess.run([gp,'-s','256M','-fq',str(ROOT/'shape.gp')],input=cmd,text=True,capture_output=True,check=True);assert not r.stderr,r.stderr[:500];lines=r.stdout.splitlines();assert len(lines)==len(ds)*(limit//2-1)*(limit//2)//2
    for line in lines:
        u,v,d,(s,bits,ct,pp)=json.loads(line);stats['shape_discriminants']+=1
        if s==-1:missing.append([u,v,d]);continue
        stats['center_anchor' if s==0 else 'endpoint_anchor']+=1;stats['prime_translates']+=ct;maxbits=max(maxbits,bits)
        for p in pp:
            fs=(p-u,p,p+v);n=prod(fs);fermat.add(fs)
            if actual(n)!=d or not mr(n,[2]):continue
            row=dict(factors=fs,D=d,n=n);log.append(row)
            if is_strong_lucas_prp(n):hits.append(row)
    return dict(diameter=limit,discriminants=ds,counts=dict(stats),distinct_fermat=len(fermat),largest_factored_gcd_union_bits=maxbits,trace_sha256=hashlib.sha256(r.stdout.encode()).hexdigest(),actual_D_and_MR=log,unresolved=missing,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--limit',type=int,default=120);ap.add_argument('--gp',default='gp');ap.add_argument('--ds',nargs='+',type=int,default=list(DS));args=ap.parse_args();r=scan(args.limit,args.ds,args.gp);args.out.parent.mkdir(exist_ok=True,parents=True);args.out.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ('actual_D_and_MR','hits','unresolved')},'MR',len(r['actual_D_and_MR']),'missing',len(r['unresolved']),'hits',len(r['hits']))
