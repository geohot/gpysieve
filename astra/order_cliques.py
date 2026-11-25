"""All pairwise-compatible strong-MR subsets, not just templates m listed as a table index."""
import argparse,json,hashlib
from collections import Counter,defaultdict
from math import gcd,lcm,prod,comb
from pathlib import Path
from order_bank import search_bank,positions,checked_orders

def coprime_product(ps):
    if len(ps)<2:return ps[0] if ps else 1
    h=len(ps)//2;a=coprime_product(ps[:h]);b=coprime_product(ps[h:]);assert gcd(a,b)==1;return a*b

def small_count(ps,forced,limit=1<<64):
    mandatory=prod(ps[j] for j in positions(forced));opts=[p for i,p in enumerate(ps) if not forced>>i&1]
    def visit(n,i,t):
        if n>=limit:return 0
        total=int(t>=2)
        for j in range(i,len(opts)):
            if n*opts[j]>=limit:break
            total+=visit(n*opts[j],j+1,t+1)
        return total
    return visit(mandatory,0,forced.bit_count())

def boxes(src):
    buckets=defaultdict(list);valid=checked_orders(src);coprime_product(sorted(p for ps in valid.values() for p in ps))
    for m,ps in valid.items():buckets[m&-m].extend((p,m) for p in ps)
    for verts in buckets.values():
        verts.sort();assert len({p for p,m in verts})==len(verts);by=defaultdict(int)
        for i,(p,m) in enumerate(verts):by[m]|=1<<i
        # For each exact order j, bitset of vertices whose entries equal 1 mod j.
        ok={m:sum(1<<i for i,(p,j) in enumerate(verts) if p%m==1) for m in by};neigh=[]
        for i,(p,m) in enumerate(verts):
            b=1<<i;nb=0
            for j,js in by.items():
                if ok[j]&b:nb|=js
            neigh.append(nb&ok[m]);assert neigh[i]&b
        def rec(fixed,optional):
            if not optional and fixed.bit_count()<2:return
            worst=max(positions(optional),key=lambda i:(optional&~neigh[i]).bit_count(),default=None)
            if worst is None or not optional&~neigh[worst]:
                ids=list(positions(fixed|optional));members=[verts[j] for j in ids];m=lcm(*(a for p,a in members));ps=[p for p,a in members];assert all(p%m==1 for p in ps);yield m,ps,sum(1<<k for k,j in enumerate(ids) if fixed>>j&1);return
            b=1<<worst;yield from rec(fixed,optional^b);yield from rec(fixed|b,(optional^b)&neigh[worst])
        for i,adj in enumerate(neigh):yield from rec(1<<i,((1<<i)-1)&adj)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--gp',default='gp');ap.add_argument('--count-only',action='store_true');a=ap.parse_args();cnt=Counter();hits=[];mx=0;maxbits=0;small=0;weights=Counter()
    for m,ps,f in boxes(a.src):
        mx=max(mx,len(ps));maxbits=max(maxbits,prod(ps).bit_length());small+=small_count(ps,f);k=f.bit_count()
        for j in range(max(0,2-k),len(ps)-k+1):weights[j+k]+=comb(len(ps)-k,j)
        if a.count_only:cnt['distinct_selections']+=(1<<(len(ps)-f.bit_count()))-int(f.bit_count()==1);cnt['boxes']+=1
        else:search_bank(m,ps,cnt,hits,a.gp,f)
    r=dict(source_sha256=hashlib.sha256(a.src.read_bytes()).hexdigest(),max_box=mx,largest_bits=maxbits,below_2_64=small,above_2_64=sum(weights.values())-small,by_weight=dict(weights),counts=dict(cnt),hits=hits,count_only=a.count_only);a.out.write_text(json.dumps(r,indent=2)+'\n');print(cnt,'max',mx,'hits',len(hits))
