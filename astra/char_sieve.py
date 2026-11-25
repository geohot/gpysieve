"""Compile conditional necessary QR equations for one mutually compatible inert core."""
from functools import lru_cache
from math import lcm
from subprocess import check_output
from sympy import factorint
from campaign import ROOT
from search import target

@lru_cache(maxsize=100000)
def parts(k):
    return tuple([int(v) for v in factorint(k) if v!=2]+([4] if k%4==0 else [])+([8] if k%8==0 else []))

@lru_cache(maxsize=500000)
def chi(q,h):
    if q%h==0:raise ValueError('nonunit in quadratic equation')
    if h==4:return int(q%4==3)
    if h==8:return int(q%8 in (3,5))
    return int(pow(q,(h-1)//2,h)==h-1)


def system(job):
    rows=job['core']; p,a,b=job['anchor']; cache={}; allbits=(1<<len(rows))-1
    fixed=job.get('fixed',[(allbits,0)])
    text=[f"{len(rows)} {p} {job.get('any',allbits)} {len(fixed)}"]+[f'{mask} {rhs}' for mask,rhs in fixed]
    for q,c,d in [*rows,(p,a,b)]:
        eq=set()
        for k,sign in [(c,1),(d,-1)]:
            for h in parts(k):
                if h not in cache:cache[h]=sum(chi(r[0],h)<<j for j,r in enumerate(rows))
                rhs=chi(p,h)^chi(sign,h);mask=cache[h]
                if mask or rhs:eq.add((mask,rhs))
        tgt,m=target(c,d); rem=tgt*pow(q,-1,m)%m
        text.append(f'{q} {m} {rem} {len(eq)}');text.extend(f'{mask} {rhs}' for mask,rhs in sorted(eq))
    return '\n'.join(text)+'\n'


def solve(job):
    lines=check_output([str(ROOT/'chars')],input=system(job),text=True).splitlines()
    tag,nodes,leaves,cp,exact=lines[-1].split();assert tag=='done';fs=[]
    for line in lines[:-1]:
        tag,v=line.split();assert tag=='hit';fs.append([r[0] for j,r in enumerate(job['core']) if int(v)>>j&1]+[job['anchor'][0]])
    return dict(char_nodes=int(nodes),leaf_masks=int(leaves),char_pass=int(cp),exact_checks=int(exact),full_pass=len(fs),subsets=job.get('space',(1<<(len(job['core'])-1))-1)),fs
