"""Compress unfactored exact-order branches to affine progressions of the common scale."""
import argparse,json
from functools import lru_cache
from itertools import product
from math import gcd
from pathlib import Path
from scale_refine import catalogue,classes
from scale_shapes import indices


def merge(a,m,b,n):
    g=gcd(m,n)
    if (b-a)%g:return None
    return a+m*(((b-a)//g)*pow(m//g,-1,n//g)%(n//g)),m*(n//g)


def branches(cs,opts):
    es=indices(cs);rows=[[(*c,m) for m in opts(e)[1] for c in classes(cs,i,m)] if e else [(0,2,0,0)] for i,e in enumerate(es)];out=set()
    for (a,m,s,i),(b,n,t,j) in product(rows[0],rows[1]):
        if s and t and s!=t:continue
        ab=merge(a,m,b,n)
        if ab is None:continue
        for c,l,u,k in rows[2]:
            if u and (s or t) and u!=(s or t):continue
            abc=merge(*ab,c,l)
            if abc is not None:out.add((*abc,i,j,k))
    return sorted(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('refined',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();opts,_=catalogue(a.src);r=json.loads(a.refined.read_text());n=0
    with a.out.open('w') as f:
        for cs in map(tuple,r['remaining']):
            br=branches(cs,opts);assert br;n+=len(br)
            for row in br:print(*cs,*row,file=f)
    print(len(r['remaining']),'shapes;',n,'order-progressions')
