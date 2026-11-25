"""Complementary search: construct strong base-2 pseudoprimes from cyclotomic factors of 2^k-1."""
import argparse,hashlib,json
from collections import defaultdict,Counter
from math import gcd,lcm,prod
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from sympy import divisors,factorint
from campaign import emit,gp_check,lucas_u
from index_scan import decompose
from search import mr,is_strong_lucas_prp


def character(a,n):
    # Jacobi (a/n), positive odd n.
    a%=n;s=1
    while a:
        t=(a&-a).bit_length()-1;a>>=t
        if t%2 and n%8 in (3,5):s=-s
        if a&n&2:s=-s
        a,n=n%a,a
    return s if n==1 else 0


def selfridge(n):
    d=5
    while True:
        j=character(n,abs(d)) # d = 1 mod 4, including negative d.
        if j<0:return d
        if j==0 and gcd(n,abs(d))!=n:return None
        d=-d-2 if d>0 else 2-d


@lru_cache(maxsize=200000)
def zbound(D,p):return p-character(D,p)


def halves(rows):
    vals=[(1,1,())]
    for p,a in rows:vals.extend([(x*p,lcm(o,a),fs+(p,)) for x,o,fs in vals])
    return vals


def sweep(spec):
    k,anchor,core=spec;p,a=anchor;middle=len(core)//2;table=defaultdict(list);c=Counter();hits=[]
    for x,o,fs in halves(core[:middle]):table[x%k].append((x,o,fs))
    for x,o,fs in halves(core[middle:]):
        if gcd(x*p,k)!=1:continue
        for y,u,gs in table.get(pow(x*p,-1,k),()):
            if not (gs or fs) or lcm(a,o,u)!=k:continue
            n=x*y*p
            if n<1<<64:continue
            c['constructed_mr']+=1;D=selfridge(n)
            if D is None:continue
            Q=(1-D)//4;factors=gs+fs+(p,)
            if any(Q%q==0 or lucas_u((n+1)%zbound(D,q),Q,q)!=0 for q in factors):continue
            c['ordinary_lucas_pass']+=1
            if is_strong_lucas_prp(n) and mr(n,[2]):hits.append(factors)
    return dict(id=f'{k}:{p}',common_order_multiple=k,anchor=p,width=len(core),**c),hits


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--limit',type=int,default=512);ap.add_argument('--workers',type=int,default=12);ap.add_argument('--width',type=int,default=22);args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True);key=hashlib.sha256(Path(__file__).read_bytes()+repr((args.width,args.limit)).encode()).hexdigest();vals={1:1};jobs=[];ranks={};fp=args.out/'components.jsonl';out=args.out/'jobs.jsonl';known={};done=set()
    if fp.exists():
        for l in fp.open():
            r=json.loads(l);assert r['manifest']==key;known[r['m']]=r['primes']
    if out.exists():
        for l in out.open():
            r=json.loads(l);assert r['manifest']==key
            if r['event']=='verified':quit()
            if r['event']=='done':done.add(r['id'])
    for m in range(2,args.limit+1):
        z,rem=divmod((1<<m)-1,prod(vals[d] for d in divisors(m) if d<m));assert rem==0;vals[m]=z
        if m not in known:jobs.append((2,m,z))
    with Pool(args.workers) as pool,fp.open('a') as f:
        for r in pool.imap_unordered(decompose,jobs):known[r['m']]=r['primes'];emit(f,dict(event='factored',manifest=key,**r))
    for m,ps in known.items():
        for p in ps:
            if p in ranks:continue
            a=m;assert pow(2,m,p)==1
            for q in factorint(m):
                while a%q==0 and pow(2,a//q,p)==1:a//=q
            ranks[p]=a
    jobs=[]
    for k in range(2,args.limit+1):
        # Same 2-adic order ensures that the MR branches line up.
        rows=sorted((p,a) for p,a in ranks.items() if k%a==0 and (a&-a)==(k&-k) and k%p)
        for i,(p,a) in enumerate(rows):
            core=rows[max(0,i-args.width):i]
            if core and f'{k}:{p}' not in done and lcm(a,*(u for _,u in core))==k:jobs.append((k,(p,a),core))
    print(len(jobs),'product jobs',flush=True)
    with Pool(args.workers) as pool,out.open('a') as f:
        for r,hits in pool.imap_unordered(sweep,jobs):
            for fs in hits:
                emit(f,dict(event='candidate',manifest=key,id=r['id'],n=prod(fs),factors=fs));ok,trace=gp_check(list(fs),'gp');emit(f,dict(event='verified' if ok else 'disagreement',manifest=key,id=r['id'],n=prod(fs),factors=fs,trace=trace));
                if not ok:raise RuntimeError(trace)
                quit()
            emit(f,dict(event='done',manifest=key,**r))
