"""Recheck every enumerated split-factor row using Python integers, not the C++ or GP gcd."""
import argparse,json
from math import gcd,prod
from pathlib import Path
from sympy import isprime,factorint
from campaign import lucas_u
from semi import exponents
from mixed import symbol


def u_integer(k,q):
    # Fast doubling on consecutive terms U_k,U_(k+1), independent of GP's polynomial exponentiation.
    u,v=0,1
    for bit in bin(k)[2:]:
        x=u*(2*v-u);y=v*v-q*u*u
        u,v=(y,y-q*x) if bit=='1' else (x,y)
    return u


def audit(row):
    D,p,a,b=row;Q=(1-D)//4
    assert isprime(p) and symbol(D,p)==1 and (p-1)%a==(p-1)%b==0 and 2%gcd(a,b)==0
    assert pow(2,a,p)==1 and all(pow(2,a//ell,p)!=1 for ell in factorint(a))
    assert lucas_u(b,Q,p)==0 and all(lucas_u(b//ell,Q,p)!=0 for ell in factorint(b))
    A,B=exponents(p,a,b)
    if A<=B:
        m=(1<<A)-1;g=gcd(m,lucas_u(B,Q,m))
    else:
        m=abs(u_integer(B,Q));g=gcd(m,pow(2,A,m)-1)
    lower=((1<<64)+p-1)//p
    fs=factorint(g) if g>=lower else {};assert not fs or prod(q**e for q,e in fs.items())==g
    return dict(D=D,p=p,order=a,rank=b,A=A,B=B,g=g,lower=lower,possible=[int(q) for q in fs if q>=lower])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+',type=Path);args=ap.parse_args();rows={tuple(map(int,l.split())) for f in args.files for l in f.read_text().splitlines()}
    for row in sorted(rows):print(json.dumps(audit(row)))
