"""Binary subset banks in which every nontrivial product passes strong base 2 by construction."""
import argparse,json,hashlib
from collections import Counter,defaultdict
from functools import lru_cache
from itertools import combinations
from math import gcd,prod
from pathlib import Path
import numpy as np
from sympy import divisors,factorint,jacobi_symbol
from campaign import lucas_u,gp_check
from search import mr,is_strong_lucas_prp

DS=tuple((-1)**i*(5+2*i) for i in range(40))
@lru_cache(None)
def character(d,v):return int(jacobi_symbol(v,abs(d))) # D=1 mod 4; quadratic reciprocity
@lru_cache(None)
def word(p):
    chars=[character(d,p%abs(d)) for d in DS];return (sum((x<0)<<i for i,x in enumerate(chars)),sum((x==0)<<i for i,x in enumerate(chars)))

def subset_array(values,one,op):
    a=np.empty(1<<len(values),dtype=np.uint64);a[0]=one
    for j,v in enumerate(values):t=1<<j;a[t:2*t]=op(a[:t],np.uint64(v))
    return a

def positions(v):
    while v:b=v&-v;yield b.bit_length()-1;v-=b

@lru_cache(None)
def period(p,D):
    Q=(1-D)//4
    if gcd(Q,p)!=1:return 0
    b=p-character(D,p%abs(D));assert b>0
    # Scalar-power certificate; never assume primality for a local rejection.
    if lucas_u(b,Q,p):return None
    if p<2**32-1:
        for f in factorint(b):
            while b%f==0 and not lucas_u(b//f,Q,p):b//=f
    return b

def checked_orders(src):
    db={r['m']:r['primes'] for l in src.read_text().splitlines() if (r:=json.loads(l))['m']>1};valid={}
    for j,ps in db.items():
        valid[j]=[p for p in ps if p%j==1 and pow(2,j,p)==1 and (j%2 or pow(2,j//2,p)==p-1) and all(pow(2,j//ell,p)!=1 for ell in factorint(j))]
    return valid

def groups(src):
    valid=checked_orders(src)
    for m in sorted(valid):
        vs=defaultdict(list)
        for j in divisors(m):
            for p in valid.get(j,[]):
                if p%m==1:vs[j&-j].append(p)
        for ps in vs.values():
            if len(ps)>1:
                assert all(gcd(x,y)==1 for x,y in combinations(ps,2));yield m,sorted(ps)

def search_bank(m,ps,counts,hits,gp,forced=0,required=()):
    nps=len(ps);counts['banks']+=1
    assert all(gcd(x,y)==1 for x,y in combinations(ps,2))
    fixed=[ps[j] for j in positions(forced)];optional=[p for j,p in enumerate(ps) if not forced>>j&1];lookup={p:1<<j for j,p in enumerate(optional)};fn=fz=0
    for p in fixed:n,z=word(p);fn^=n;fz|=z
    a=subset_array([word(p)[0] for p in optional],fn,np.bitwise_xor);z=subset_array([word(p)[1] for p in optional],fz,np.bitwise_or);first=a|z;first&=-first;ids=np.arange(len(a),dtype=np.uint64);nontriv=((ids&(ids-1))!=0) if not fixed else (ids!=0) if len(fixed)==1 else np.ones(len(a),dtype=bool)
    for support in required:nontriv&=(ids&support)!=0
    counts['strong_base2_products']+=int(nontriv.sum())
    # If a Jacobi zero occurs first and gcd(D,n)=n (tiny products), leave the
    # convention to the direct implementation, rather than assume rejection.
    reject=nontriv & ((z&first)!=0);tiny=sum((1<<j) for j,p in enumerate(optional) if p<=abs(DS[-1]));
    def finish(mask,d):
        fs=sorted(fixed+[optional[j] for j in positions(mask)]);n=prod(fs)
        if d:
            Q=(1-d)//4
            for p in fs:
                if p<2**32-1 and period(p,d) is not None:continue # exact-rank vector sieve done
                b=period(p,d)
                if b==0:return
                if lucas_u((n+1)%b if b else n+1,Q,p):return
        counts['direct_Lucas_tests']+=1
        if is_strong_lucas_prp(n):
            assert mr(n,[2]);ok,trace=gp_check(fs,gp);assert ok,trace;hits.append(dict(m=m,factors=fs,n=n,trace=trace))
    counts['prior_gcd_rejections']+=int(reject.sum());
    for mask in ids[reject & ((ids&tiny)==ids)]:
        if prod(fixed)*prod(optional[j] for j in positions(int(mask)))<=abs(DS[-1]):finish(int(mask),None)
    for mask in ids[nontriv & (first==0)]:finish(int(mask),None)
    active=nontriv & ((first&a)!=0)
    for j in np.unique(first[active]):
        d=DS[int(j).bit_length()-1];live=ids[active & (first==j)];counts['eligible_character']+=len(live)
        for i,p in enumerate(ps):
            if p>=2**32-1 or not len(live):break
            bit=lookup.get(p,0);present=((live&bit)!=0) if bit else np.ones(len(live),dtype=bool)
            if not present.any():continue
            b=period(p,d)
            if b is None:continue
            if not b or 2%gcd(m,b):live=live[~present];continue
            if len(live)<8 and len(ids)>256:
                live=np.array([v for v in live if (bit and not(int(v)&bit)) or (prod(fixed)*prod(optional[k]%b for k in positions(int(v)))+1)%b==0],dtype=np.uint64);continue
            residues=subset_array([v%b for v in optional],prod(fixed)%b,lambda x,y:(x*y)%np.uint64(b));live=live[~present|(residues[live]==b-1)]
        counts['after_small_ranks']+=len(live)
        for mask in live:finish(int(mask),d)

def scan(src,gp):
    counts=Counter();hits=[]
    for m,ps in groups(src):search_bank(m,ps,counts,hits,gp)
    return dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),counts=dict(counts),hits=hits)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--gp',default='gp');x=ap.parse_args();r=scan(x.src,x.gp);x.out.write_text(json.dumps(r,indent=2)+'\n');print(r['counts'],'hits',len(r['hits']))
