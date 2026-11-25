"""Discharge later Selfridge discriminants by a binary kernel on the earlier characters."""
import argparse,hashlib,json
from collections import Counter,defaultdict
from functools import lru_cache
from math import gcd,lcm,prod
from pathlib import Path
from lucas_first import templates,metadata,join
from order_bank import checked_orders,character,positions,period
from order_cliques import coprime_product
from campaign import lucas_u,gp_check
from search import mr,is_strong_lucas_prp

def kernel(columns):
    basis={};ker=[]
    for i,x in enumerate(columns):
        a=1<<i
        while x:
            j=x.bit_length()
            if j not in basis:basis[j]=(x,a);break
            y,b=basis[j];x^=y;a^=b
        if not x:ker.append(a)
    return ker

def table(values,m):
    v=[1]
    for p in values:v+=list(x*p%m for x in v)
    return v

def selfridge(fs):
    i=0
    while True:
        d=(-1)**i*(5+2*i);ch=prod(character(d,p%abs(d)) for p in fs)
        if ch<0:return d
        if not ch:return None
        i+=1

def finish(m,group,ker,counts,hits,gp):
    # Gray traversal of the character nullspace; small tables give n mod m
    # without constructing thousand-bit integers for every subset.
    counts['tail_patterns']+=len(group)>0;counts['tail_masks']+=(1<<len(ker))-1
    tables=[table([p for p,j in group[i:i+8]],m) for i in range(0,len(group),8)];out=sum((p%m!=1)<<i for i,(p,j) in enumerate(group));mask=0
    for step in range(1,1<<len(ker)):
        mask^=ker[(step&-step).bit_length()-1]
        if (mask&out).bit_count()<3 or prod(t[(mask>>(8*j))&255] for j,t in enumerate(tables))%m!=1:continue
        rows=[group[j] for j in positions(mask)]
        if lcm(*(a for p,a in rows))!=m:continue
        fs=sorted(p for p,a in rows);n=prod(fs)
        if n<2**64:continue
        counts['tail_actual_products']+=1;d=selfridge(fs)
        if d is None:continue
        Q=(1-d)//4
        if gcd(Q,n)!=1:continue
        for p in fs:
            b=period(p,d)
            if lucas_u((n+1)%b if b else n+1,Q,p):break
        else:
            counts['tail_direct']+=1
            if is_strong_lucas_prp(n):
                assert mr(n,[2]);ok,trace=gp_check(fs,gp);assert ok,trace;hits.append(dict(m=m,D=d,factors=fs,n=n,trace=trace))

def plan(m,group,direct,work,ct,dim):
    # At stage i all previous symbols must be +1. Zero removes that
    # catalogue element (proper gcd for products >2^64 in this run).
    g=list(group);cols=[0]*len(g)
    for i in range(256):
        ker=kernel(cols)
        if len(ker)<=dim:direct.append((m,g,ker));return
        d=(-1)**i*(5+2*i);cc=[character(d,p%abs(d)) for p,j in g];keep=[j for j,x in enumerate(cc) if x];g=[g[j] for j in keep];cols=[cols[j] for j in keep];neg=sum((cc[j]<0)<<k for k,j in enumerate(keep));ker=kernel(cols)
        if sum(p%m!=1 for p,_ in g)<3 or lcm(*(j for p,j in g),1)!=m:return
        if any((v&neg).bit_count()%2 for v in ker):work[d].append((m,g))
        for j in positions(neg):cols[j]|=1<<i
    raise ValueError('tail not resolved')

def scan(src,bin,gp,dim):
    db=checked_orders(src);coprime_product(sorted(p for ps in db.values() for p in ps));work=defaultdict(list);direct=[];counts=Counter();hits=[];report={}
    for m,g in templates(db):plan(m,g,direct,work,counts,dim);counts['templates']+=1;counts['raw_subset_visits']+=2**len(g)
    for m,g,ker in direct:finish(m,g,ker,counts,hits,gp)
    for d,groups in work.items():
        data=metadata({p for m,g in groups for p,j in g},d,bin);cnt=Counter()
        for m,g in groups:join(m,d,g,data,cnt,hits,gp,19)
        report[d]=dict(groups=len(groups),**cnt)
    return dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),dimension=dim,counts=counts,early=report,hits=hits)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--gp',default='gp');ap.add_argument('--bin',type=Path,default=Path('/tmp/astra-rank'));ap.add_argument('--dim',type=int,default=9);a=ap.parse_args();r=scan(a.src,a.bin,a.gp,a.dim);a.out.write_text(json.dumps(r,indent=2)+'\n');print(r['counts'],'early D:',len(r['early']),'hits:',len(r['hits']))
