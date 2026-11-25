"""Bound the index of 2, not the size of p. Factor Lucas components and search their products."""
import argparse
from collections import defaultdict,Counter
from itertools import combinations
from math import lcm,prod
from multiprocessing import Pool
from pathlib import Path
import hashlib,json,subprocess
from sympy import divisors,factorint
from campaign import ROOT,emit,gp_check,lucas_u
from mixed import DS,symbol
from search import mr,is_strong_lucas_prp


def components(D,T):
    # U_m = product_{d|m,d>1} Phi_d(alpha,beta); preserve exact multiplicities.
    u,v=0,1; q=(1-D)//4; parts={1:1}
    for m in range(1,2*T+1):
        u,v=v,v-q*u
        if m==1:continue
        den=prod(parts[d] for d in divisors(m) if d<m);part,rem=divmod(abs(u),den)
        assert rem==0 and part>0;parts[m]=part
        if m%2==0 or m<=T:yield m,part


def decompose(job):
    D,m,g=job; nbits=g.bit_length()
    def factor(full):
        cmd=f'f=factor({g}{"" if full else ",100000"}); for(i=1,matsize(f)[1], print([f[i,1],f[i,2],isprime(f[i,1])]))\n'
        r=subprocess.run(['gp','-s','96M','-fq'],input=cmd,text=True,capture_output=True,timeout=6,check=True)
        if r.stderr:raise RuntimeError(r.stderr)
        rows=[json.loads(l) for l in r.stdout.splitlines()];assert prod(p**e for p,e,_ in rows)==g
        return [p for p,e,prime in rows if prime],[p for p,e,prime in rows if not prime]
    try:pr,res=factor(nbits<=220) if g>1 else ([],[])
    except (subprocess.TimeoutExpired,RuntimeError):
        # Time/stack limit is not a failed search. Retain (or explicitly mark)
        # the unfactored cofactor rather than aborting the other workers.
        try:pr,res=factor(False)
        except (subprocess.TimeoutExpired,RuntimeError):pr,res=[],[g]
    return dict(id=f'{D}:{m}',D=D,m=m,bits=nbits,primes=pr,unresolved=res)


def check_prods(spec):
    D,k,entries,arity=spec; Q=(1-D)//4; c=Counter();hits=[]
    for v in combinations(entries,arity):
        # Use the least even multiple of all ranks, rather than repeating the
        # same product at every index. Strong Lucas requires matching 2-parts.
        if lcm(2,*(z for p,z in v))!=k:continue
        c['products']+=1;n=prod(p for p,z in v)
        if n<1<<64 or (n+1)%(k//(2 if all(z%2 for _,z in v) else 1)):continue
        if not all(prod(symbol(d,p) for p,z in v)==(-1 if d==D else 1) for d in DS[:DS.index(D)+1]):continue
        c['lucas_congruence_pass']+=1
        # Filter by necessary base-2 Fermat at the smallest prime before MR.
        p=v[0][0]
        if pow(2,(n-1)%(p-1),p)!=1:continue
        c['fermat_one_pass']+=1
        if mr(n,[2]) and is_strong_lucas_prp(n):hits.append([p for p,z in v])
    return dict(id=f'{D}:{k}:{arity}:{entries[0][1]&-entries[0][1]}',D=D,index=k,arity=arity,**c),hits


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--T',type=int,default=128);ap.add_argument('--workers',type=int,default=12);ap.add_argument('--arity',type=int,nargs='+',default=[2,3,5]);ap.add_argument('--reuse',nargs='*',type=Path,default=[]);args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    key=hashlib.sha256(repr((args.T,args.arity)).encode()+Path(__file__).read_bytes()).hexdigest();data=args.out/'components.jsonl';reports=args.out/'products.jsonl';old={};done=set()
    if data.exists():
        for line in data.open():
            row=json.loads(line);assert row['manifest']==key;old[row['id']]=row
    if reports.exists():
        for line in reports.open():
            r=json.loads(line);assert r['manifest']==key
            if r['event']=='verified':quit()
            if r['event']=='done':done.add(r['id'])
    Ds=[D for D in DS if D!=9];tasks=[(D,m,g) for D in Ds for m,g in components(D,args.T) if f'{D}:{m}' not in old]
    saved={}
    for path in args.reuse:
        for line in path.open():
            r=json.loads(line)
            if r.get('event')=='factored' and not r['unresolved']:saved[r['id']]=r
    with Pool(args.workers) as pool,data.open('a') as f:
        todo=[]
        for D,m,g in tasks:
            k=f'{D}:{m}'
            if k not in saved:todo.append((D,m,g));continue
            r=saved[k].copy();rem=g
            for p in r['primes']:
                assert p>1 and rem%p==0
                while rem%p==0:rem//=p
            assert rem==1;r.update(source_manifest=r['manifest'],manifest=key);emit(f,r);old[k]=r
        for row in pool.imap_unordered(decompose,todo,chunksize=1):
            row.update(event='factored',manifest=key);emit(f,row);old[row['id']]=row
    jobs=[];coverage={}
    for D in Ds:
        ranks={};Q=(1-D)//4;unresolved={r['m'] for r in old.values() if r['D']==D and r['unresolved']};coverage[D]=[t for t in range(1,args.T+1) if any(2*t%d==0 for d in unresolved)]
        for r in old.values():
            if r['D']!=D:continue
            for p in r['primes']:
                if p<3 or Q%p==0 or any(d%p==0 for d in DS[:DS.index(D)+1]) or p in ranks:continue
                z=r['m'];assert lucas_u(z,Q,p)==0
                for q in factorint(z):
                    while z%q==0 and lucas_u(z//q,Q,p)==0:z//=q
                ranks[p]=z
        for k in range(2,2*args.T+1,2):
            groups=defaultdict(list)
            for p,z in sorted(ranks.items()):
                if k%z==0:groups[z&-z].append((p,z))
            for s,entries in groups.items():
                for a in args.arity:
                    if len(entries)>=a and f'{D}:{k}:{a}:{s}' not in done:jobs.append((D,k,entries,a))
    (args.out/'incomplete-indices.json').write_text(json.dumps(coverage)+'\n');print('product batches',len(jobs),flush=True)
    with Pool(args.workers) as pool,reports.open('a') as f:
        for row,hits in pool.imap_unordered(check_prods,jobs):
            for fs in hits:
                emit(f,dict(event='candidate',manifest=key,id=row['id'],factors=fs,n=prod(fs)));ok,trace=gp_check(fs,'gp');emit(f,dict(event='verified' if ok else 'disagreement',manifest=key,id=row['id'],factors=fs,n=prod(fs),trace=trace));
                if not ok:raise RuntimeError(trace)
                quit()
            emit(f,dict(event='done',manifest=key,**row))
