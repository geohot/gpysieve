"""D=5 three-prime search with prefix-specific CRT progressions."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from math import gcd, lcm

from sympy import isprime
from search import catalog, target, mr, is_strong_lucas_prp


def progression(p, q, a, b, nmin):
    """Return first possible r, stride; a,b are prefix order/rank LCMs."""
    w = p*q
    m = lcm(a, b)
    if 2 % gcd(a, b) or gcd(w, m) != 1:
        return None
    t, _ = target(a, b)
    lo = max(q+1, (nmin+w-1)//w)
    return lo+(t*pow(w, -1, m)-lo) % m, m


def scan(limit, rmax, nmin, start, stop, shard, shards):
    groups = defaultdict(list)
    for row in catalog(limit):
        p,a,b = row
        groups[a & -a, b & -b].append(row)
    stats = Counter()
    for rows in groups.values():
        for (p,a,b),(q,c,d) in combinations(rows, 2):
            if p % shards != shard:
                continue
            stats['pairs'] += 1
            prog = progression(p,q,lcm(a,c),lcm(b,d),nmin)
            if prog is None: continue
            first, step = prog
            w = p*q
            for r in range(first+start*step, min(rmax+1,first+stop*step), step):
                stats['positions'] += 1
                if not r & 1 or r%5 not in (2,3): continue
                # For prime r, ord_r(2) divides r-1. Thus 2^(wr-1)=1
                # implies 2^(w-1)=1 mod r. Avoid primality tests until here.
                if pow(2,w-1,r) != 1 or not isprime(r): continue
                n = w*r
                if not mr(n,[2]): continue
                stats['mr_pass'] += 1
                if is_strong_lucas_prp(n):
                    stats['hits'] += 1
                    print(json.dumps(dict(n=n,factors=[p,q,r])), flush=True)
    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for key, default in [('limit',10000),('rmax',2**42),('nmin',2**64),('start',0),('stop',64),('shard',0),('shards',1)]:
        ap.add_argument('--'+key,type=int,default=default)
    args = ap.parse_args()
    if not (0 <= args.shard < args.shards and 0 <= args.start < args.stop and args.nmin >= 1 and 3 <= args.rmax < 2**64):
        ap.error('invalid bounds (rmax must fit uint64 for deterministic primality)')
    print(json.dumps(dict(parameters=vars(args), counts=scan(**vars(args)) ),sort_keys=True))
