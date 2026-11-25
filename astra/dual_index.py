"""Semiprime reduction dual to the Lucas-index search. No D restriction or core-width limit.
Read GP-certified factorizations of Phi_m(2) from mersenne.py. Report complete indices u
and test EVERY distinct prime pair from the known divisors of 2^(2u)-1, with deduplication.
"""
import argparse
import hashlib
import json
from itertools import combinations
from math import prod
from pathlib import Path
from sympy import divisors
from campaign import gp_check
from search import mr, is_strong_lucas_prp


def audit(path, upper, gp='gp'):
    data=path.read_bytes(); records={r['m']:r for line in data.splitlines() if (r:=json.loads(line))['event']=='factored'}
    factors={1:set()}; complete={1:True}; phi={1:1}
    for m in range(2,2*upper+1):
        val,rem=divmod((1<<m)-1,prod(phi[d] for d in divisors(m) if d<m)); assert not rem
        phi[m]=val;r=records.get(m);factors[m]=set();complete[m]=False
        if r is None:continue
        assert r['D']==2; ps=set(r['primes']); cof=set(r['unresolved'])
        for f in sorted(ps|cof):
            assert f>1 and val%f==0
            while val%f==0:val//=f
        assert val==1;factors[m]=ps;complete[m]=not cof
    seen=set();large=base2=0;missing=[];hits=[]
    for u in range(1,upper+1):
        ds=divisors(2*u); ps=sorted(set.union(*(factors[d] for d in ds)))
        if not all(complete[d] for d in ds):missing.append(u)
        for p,q in combinations(ps,2):
            if (p,q) in seen:continue
            seen.add((p,q)); n=p*q
            if n<1<<64:continue
            large+=1
            if not mr(n,[2]):continue
            base2+=1
            if is_strong_lucas_prp(n):
                ok,trace=gp_check([p,q],gp);assert ok;hits.append(dict(n=n,factors=[p,q],trace=trace))
    return dict(source_sha256=hashlib.sha256(data).hexdigest(),upper=upper,complete_count=upper-len(missing),complete_through=min(missing,default=upper+1)-1,incomplete=missing,distinct_pairs=len(seen),above_bound=large,base2_pass=base2,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('components',type=Path);ap.add_argument('--upper',type=int,default=1024);ap.add_argument('--gp',default='gp');args=ap.parse_args();print(json.dumps(audit(args.components,args.upper,args.gp)))
