"""Eliminate translation for every k-prime pattern in an additive-diameter box."""
import argparse,json,subprocess,hashlib
from collections import Counter
from itertools import combinations
from math import prod,comb
from pathlib import Path
from sympy import isprime,primerange
from shape_scan import DS,actual,ROOT
from search import mr,is_strong_lucas_prp


def tails(k,diam):
    for other in (2,4,6):
        for t in combinations([j for j in range(2,diam+1,2) if j%8 in (0,other)],k-1):
            if other==2 or any(s%8 for s in t):yield t

def scan(k,diam,gp,ds=DS):
    cmd=[];counts=Counter();hits=[];missing=[];survivors=[]
    counts['patterns']=comb(diam//2,k-1);counts['strong_signature_obstruction']=counts['patterns']
    for tail in tails(k,diam):
        H=(0,)+tail;counts['strong_signature_obstruction']-=1
        # Covering all residues mod ell forces a factor equal to ell. Exhaust
        # its finitely many translates, including small primes as exceptions.
        # Classes 3,5,7 mod 8 force distinct 2-parts 2,4,1 of ord(2).
        # Any translate meeting >=3 odd classes contains two incompatible ones.
        assert len({s%8 for s in H})<=2
        ell=next((q for q in primerange(3,k+1) if len({s%q for s in H})==q),None)
        if ell:
            counts['forced_small_prime']+=1
            for s in H:
                x=ell-s
                if x<3 or not all(isprime(x+j) for j in H):continue
                n=prod(x+j for j in H)
                if mr(n,[2]) and is_strong_lucas_prp(n):hits.append([x,H,n])
        else:
            for d in ds:cmd.append(f'print([{list(H)},{d},constellation({list(H)},{d})])')
    r=subprocess.run([gp,'-s','256M','-fq',str(ROOT/'shape.gp'),str(ROOT/'constellation.gp')],input='\n'.join(cmd)+'\n',text=True,capture_output=True,check=True);assert not r.stderr,r.stderr[:1000];assert len(r.stdout.splitlines())==len(cmd);counts['sieved_shape_D']=len(cmd)
    for l in r.stdout.splitlines():
        H,d,(i,b,ct,xs)=json.loads(l)
        if i<0:missing.append([H,d]);continue
        counts['prime_translates']+=ct
        for x in xs:
            n=prod(x+j for j in H)
            if mr(n,[2]) and actual(n)==d:
                survivors.append([x,H,d]);
                if is_strong_lucas_prp(n):hits.append([x,H,n])
    return dict(factors=k,diameter=diam,discriminants=ds,counts=counts,unresolved=missing,actual_D_and_MR=survivors,hits=hits,trace_sha256=hashlib.sha256(r.stdout.encode()).hexdigest())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--factors',type=int,default=5);ap.add_argument('--diameter',type=int,default=48);ap.add_argument('--gp',default='gp');ap.add_argument('--suite',action='store_true');a=ap.parse_args()
    if a.suite:
        r={k:scan(k,360 if k<=3 else 80 if k==4 else 48 if k==5 else 40,a.gp) for k in range(2,22)};print('unresolved:',sum(len(s['unresolved']) for s in r.values()),'hits:',sum(len(s['hits']) for s in r.values()))
    else:
        r=scan(a.factors,a.diameter,a.gp);print(r['counts'],'missing',len(r['unresolved']),'MR',len(r['actual_D_and_MR']),'hits',len(r['hits']))
    a.out.write_text(json.dumps(r,indent=2)+'\n')
