"""Dual search: fix the exact Lucas rank LCM (not necessarily a listed index), screen base 2 before joining."""
import argparse,json,hashlib
from collections import Counter,defaultdict
from math import gcd,lcm
from pathlib import Path
from sympy import factorint
from campaign import lucas_u
from order_cliques import coprime_product
from lucas_first import metadata,join


def ranks(files,d):
    Q=(1-d)//4;by={}
    for f in files:
        for line in f.open():
            r=json.loads(line)
            if r['D']!=d:continue
            j=r['m']
            for p in r['primes']:
                if gcd(p,2*d*Q)!=1 or p in by:continue
                assert not lucas_u(j,Q,p);k=j
                for ell in factorint(j):
                    while k%ell==0 and not lucas_u(k//ell,Q,p):k//=ell
                by[p]=k
    coprime_product(sorted(by));return by

def scan(files,d,B,bin,gp):
    cat=ranks(files,d);meta=metadata(cat,d,bin,True);by=defaultdict(list)
    for p,j in cat.items():by[j].append(p)
    # Sieve rank templates once; classes with LCM < candidate m cannot yield a canonical subset.
    indices=defaultdict(list)
    for j in by:
        for m in range(j,B+1,j):indices[m,j&-j].append(j)
    cnt=Counter();hits=[]
    for (m,s),js in indices.items():
        if lcm(*js)!=m:continue
        group=[(p,j) for j in js for p in by[j] if gcd(m,p)==1]
        if len(group)<2:continue
        cnt['rank_templates']+=1;cnt['raw_patterns']+=2**len(group)
        join(m,d,group,meta,cnt,hits,gp,19,True)
    return dict(D=d,rank_lcm_bound=B,catalogue_size=len(cat),source_sha256={f.parent.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},exact_binary_orders=sum(v[0]==v[1] for v in meta.values()),counts=dict(cnt),hits=hits)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('files',type=Path,nargs='+');ap.add_argument('--D',type=int,required=True);ap.add_argument('--bound',type=int,default=50000);ap.add_argument('--bin',type=Path,default=Path('/tmp/astra-rank'));ap.add_argument('--gp',default='gp');a=ap.parse_args();r=scan(a.files,a.D,a.bound,a.bin,a.gp);a.out.write_text(json.dumps(r,indent=2)+'\n');print(a.D,r['catalogue_size'],r['counts'],len(r['hits']))
