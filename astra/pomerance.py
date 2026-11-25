"""Enumerate the paper's P_k(T) exactly by generating squarefree divisors of Q1."""
import argparse
import json
from math import prod, lcm
from pathlib import Path
from sympy import primerange, isprime


def paper_pool(T, k, pmax=None):
    qs = list(primerange(3, T))
    left = [q for q in qs if q % 4 == 1]
    right = [q for q in qs if q % 4 == 3]
    Q3 = prod(right)
    upper = T**k if pmax is None else min(pmax, T**k)
    bound = (upper-1)//2

    def divisors(i, d):
        yield d
        for j in range(i, len(left)):
            nd = d*left[j]
            if nd > bound: break
            yield from divisors(j+1, nd)

    ps = []
    for d in divisors(0, 1):
        p = 2*d+1
        if p < T or p % 5 not in (2,3): continue
        x = (p+1)//4
        if Q3 % x == 0 and isprime(p): ps.append(p)
    return dict(primes=sorted(ps), A=prod(left), B=Q3)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('T', type=int); ap.add_argument('k', type=int)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pmax', type=int, help='only enumerate the interval through pmax')
    a = ap.parse_args()
    upper = a.T**a.k if a.pmax is None else min(a.T**a.k, a.pmax)
    if a.T < 5 or a.k < 5 or not a.T <= upper < 2**64: ap.error('require T>=5, k>=5, T <= upper < 2^64')
    pool = paper_pool(a.T, a.k, upper)
    Path(a.out).write_text(json.dumps([pool])+'\n')
    print(f"{len(pool['primes'])} primes, {lcm(pool['A'], pool['B']).bit_length()} modulus bits")
