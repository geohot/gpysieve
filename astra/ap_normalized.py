"""Unbounded-factor AP triples with small normalized gap h=d/gcd(d,p+1)."""
import argparse,json,os,subprocess
from collections import Counter
from math import gcd,prod
from pathlib import Path
from hashlib import sha256
from sympy import divisors
from search import mr,is_strong_lucas_prp


def audit(src,hs,gp):
    records={r['m']:r for s in src.read_text().splitlines() if (r:=json.loads(s))['D']==2};centers={};missing={}
    for h in hs:
        e=4*h*h;v=2**e-1;ps=set()
        for j in divisors(e):
            if j in records:ps.update(records[j]['primes'])
        for p in sorted(ps):
            assert v%p==0
            while v%p==0:v//=p
        if v==1:centers[h]=sorted(ps)
        else:missing[h]=v.bit_length()
    ps=sorted(set().union(*centers.values()));cmd='ck(p)={if(!isprime(p),error("center not prime"));my(f=factor(p+1));for(i=1,matsize(f)[1],if(!isprime(f[i,1]),error("uncertified divisor")));print(Vec(divisors(f)))};\n'+''.join(f'ck({p})\n' for p in ps);out=subprocess.run([gp,'-fq'],input=cmd,text=True,capture_output=True,check=True,timeout=180);assert not out.stderr,out.stderr;gs=dict(zip(ps,map(json.loads,out.stdout.splitlines()),strict=True));counts=Counter();basepass=[];hits=[];largest=0
    for h,pp in centers.items():
        for p in pp:
            for g in gs[p]:
                d=g*h
                if d%2 or p-d<3 or gcd(d,p+1)!=g:continue
                if pow(2,d*d%(p-1),p)!=1:continue
                counts['center_gap']+=1;n=p*(p-d)*(p+d);largest=max(largest,n.bit_length());counts['above64']+=int(n>2**64)
                if mr(n,[2]):
                    row=dict(p=p,d=d,h=h,n=n);basepass.append(row)
                    if is_strong_lucas_prp(n):hits.append(row)
    # Endpoint primality is deliberately not a filter: rejecting a superset is safe.
    return dict(source_sha256=sha256(src.read_bytes()).hexdigest(),complete_h=list(centers),incomplete_h_residual_bits=missing,distinct_certified_centers=len(ps),counts=dict(counts),largest_tested_bits=largest,base2_pass=basepass,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--gp',default='gp');ap.add_argument('--max-h',type=int,default=22);a=ap.parse_args();r=audit(a.src,range(1,a.max_h+1),a.gp);a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ('base2_pass','hits')},'MR pass',len(r['base2_pass']),'hits',len(r['hits']))
