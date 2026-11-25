"""Bounded divisor-frontier search: add degrees of freedom at a FIXED strong target."""
import argparse,json
from functools import lru_cache
from math import gcd
from pathlib import Path
from sympy import factorint,isprime
from search import fib


def divisors(factors,limit,i=0,d=1):
    yield d
    for j in range(i,len(factors)):
        q,e=factors[j]; x=d*q
        if x>limit:break
        for _ in range(e):
            yield from divisors(factors,limit,j+1,x); x*=q
            if x>limit:break

@lru_cache(None)
def smallfib(n):
    a,b=0,1
    for _ in range(n):a,b=b,a+b
    return a

def generate(A,B,upper,hmax):
    assert A%4==2 and B%8==4 and gcd(A,B)==2 and upper<2**64
    hits=set();stats=[]
    for side,t in [('minus',A//2),('plus',B//4)]:
        fac=sorted(factorint(t).items());sign=1 if side=='minus' else -1;scale=2 if sign==1 else 4
        ones=all(q%4==1 for q,_ in fac);checked=0
        for h in range(1,hmax+1,2):
            # Avoid overlaps: h is the exact missing cofactor outside the allowed prime powers.
            if sign==1 and ((ones and h%4!=1) or h%5==0):continue
            lim=(upper-sign)//(scale*h)
            if lim<1:continue
            for d in divisors(fac,lim):
                x=d*h;p=scale*x+sign
                if p%8!=3 or p%5 not in (2,3) or gcd(t,x)!=d:continue
                checked+=1
                a=gcd(p-1,A);b=gcd(p+1,B)
                if a<p.bit_length():continue
                if b<=256:
                    if smallfib(b)%p:continue
                elif fib(b,p):continue
                if pow(2,a,p)==1 and isprime(p) and gcd(p,A*B)==1:hits.add(p)
        stats.append(dict(side=side,checked=checked,cumulative_hits=len(hits)))
    return sorted(hits),stats

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('audit',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--upper',type=int,default=10**9);ap.add_argument('--hmax',type=int,default=33);args=ap.parse_args();r=json.loads(args.audit.read_text())['historic'];hits,stats=generate(r['A'],r['B'],args.upper,args.hmax);args.out.write_text(''.join(f'{p}\n' for p in hits));print(stats)
