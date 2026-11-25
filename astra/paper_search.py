"""Search selected-factor (not fixed-pool) Pomerance conditions."""
import argparse
import json
from collections import Counter
from math import comb, gcd, lcm, log2, prod
from pathlib import Path
from sympy import factorint
from search import fib, products, mr, is_strong_lucas_prp


def periods(p):
    # All input primes are inert for D=5; base order divides p-1,
    # Fibonacci rank divides p+1. Strip factors only on a proven identity.
    def reduce(k, test):
        for q in factorint(k):
            while k%q==0 and test(k//q): k//=q
        return k
    return reduce(p-1,lambda k: pow(2,k,p)==1), reduce(p+1,lambda k: fib(k,p)==0)


def five(prev, p, m=None):
    """All increasing a,b,c,d with abcd = 1 mod m. No permutations/dedup set."""
    if m is None: m = lcm(p-1,p+1)
    ps = [q for q in prev if gcd(q,m)==1]
    inv = [pow(q,-1,m) for q in ps]
    pairs = {}
    for k in range(2,len(ps)-1):
        j = k-1
        for i in range(j):
            key = ps[i]*ps[j] % m
            pairs.setdefault(key, []).append((ps[i],ps[j]))
        for h in range(k+1,len(ps)):
            for ab in pairs.get(inv[k]*inv[h] % m, ()):
                yield (*ab, ps[k], ps[h], p)


def cluster(ps, seed, width, sides):
    todo = set(ps)-{seed}; chosen=[seed]; A,B=sides[seed]
    while len(chosen)<width:
        def cost(p):
            a,b=sides[p]
            return (a//gcd(A,a)*(b//gcd(B,b)),p)
        p=min(todo,key=cost); todo.remove(p); chosen.append(p)
        a,b=sides[p]; A=lcm(A,a); B=lcm(B,b)
    return sorted(chosen), log2(lcm(A,B))


def anchored(prev, p, m=None):
    """Odd subsets of size >=3 containing p as their largest prime."""
    if m is None: m = lcm(p-1,p+1)
    ps=[q for q in prev if gcd(q,m)==1]; k=len(ps)//2
    table={}
    for lo,v in products(ps[:k],m,0,1<<k):
        table.setdefault((v,lo.bit_count()%2),[]).append(lo)
    inv=[pow(q,-1,m) for q in ps[k:]]
    for hi,v in products(inv,m,0,1<<len(inv)):
        for lo in table.get((v,hi.bit_count()%2),()):
            mask=lo | hi<<k
            if mask:
                yield tuple(q for j,q in enumerate(ps) if mask>>j&1)+(p,)


def run(ps, mode, start, stop, width, exact=False):
    sides={p:periods(p) if exact else (p-1,p+1) for p in ps}
    counts=Counter(anchor_survivors=0, above_bound_survivors=0, hits=0); seen=set(); bits=[]
    for idx in range(start,stop):
        if mode=='five':
            batches=[(ps[:idx],ps[idx])]; counts['subsets']+=comb(idx,4)
        else:
            group,b=cluster(ps,ps[idx],width,sides)
            if tuple(group) in seen: continue
            seen.add(tuple(group)); bits.append(b)
            counts['subsets']+=(1<<(len(group)-1))-len(group)
            batches=[(group[:i],p) for i,p in enumerate(group) if i>=2]
        for prev,p in batches:
            for fs in (five if mode=='five' else anchored)(prev,p,lcm(*sides[p])):
                counts['anchor_survivors']+=1
                n=prod(fs)
                if n<2**64: continue
                counts['above_bound_survivors']+=1
                if not all((n-1)%sides[q][0]==0 and (n+1)%sides[q][1]==0 for q in fs): continue
                if not (mr(n,[2]) and is_strong_lucas_prp(n)):
                    raise RuntimeError(f'construction disagrees with verifier: {fs}')
                counts['hits']+=1; print(json.dumps(dict(n=n,factors=fs)),flush=True)
    if bits:
        counts.update(clusters=len(bits)); counts['min_mod_bits']=min(bits); counts['max_mod_bits']=max(bits)
    return dict(counts)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file'); ap.add_argument('mode',choices=['five','cluster'])
    ap.add_argument('--start',type=int,default=0); ap.add_argument('--stop',type=int)
    ap.add_argument('--width',type=int,default=28)
    ap.add_argument('--exact',action='store_true',help='use exact base orders and Fibonacci ranks instead of p-1,p+1')
    ap.add_argument('--first',type=int,help='restrict to the smallest N pool primes')
    args=ap.parse_args(); ps=json.loads(Path(args.file).read_text())[0]['primes']
    if args.first is not None:
        if not 1<=args.first<=len(ps): ap.error('invalid --first')
        ps=ps[:args.first]
    stop=len(ps) if args.stop is None else args.stop
    if ps!=sorted(set(ps)) or not 0<=args.start<=stop<=len(ps): ap.error('invalid pool or index range')
    if args.mode=='cluster' and not 3<=args.width<=min(40,len(ps)): ap.error('width must be between 3 and min(pool size,40)')
    print(json.dumps(dict(parameters=vars(args), counts=run(ps,args.mode,args.start,stop,args.width,args.exact)),sort_keys=True))
