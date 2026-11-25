"""Exactly two nonidentity residues at the actual order LCM, cancelling jointly."""
import argparse,json,hashlib
from pathlib import Path
from itertools import combinations
from collections import Counter,defaultdict
from math import gcd,lcm
from sympy import divisors,factorint
from order_bank import checked_orders,search_bank
from order_cliques import coprime_product


def pairs(src):
    db=checked_orders(src);order={p:j for j,ps in db.items() for p in ps};coprime_product(sorted(order))
    for m in sorted(db):
        levels=[q**e for q,e in factorint(m).items()];by=defaultdict(list)
        for j in divisors(m):
            for p in db.get(j,[]):
                if gcd(p,m)==1:by[j&-j].append(p)
        for ps in by.values():
            core=sorted(p for p in ps if p%m==1);outside=sorted(set(ps)-set(core));sup={q:sum((1<<j) for j,p in enumerate(core) if order[p]%q==0) for q in levels}
            for p,q in combinations(outside,2):
                if p*q%m!=1:continue
                l=lcm(order[p],order[q]);need=[sup[z] for z in levels if l%z];
                if 0 in need:continue
                vs=sorted(core+[p,q]);f=(1<<vs.index(p))|(1<<vs.index(q));yield m,vs,f,need

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--gp',default='gp');a=ap.parse_args();cnt=Counter();hits=[]
    for m,ps,f,need in pairs(a.src):search_bank(m,ps,cnt,hits,a.gp,f,need)
    a.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(a.src.read_bytes()).hexdigest(),counts=cnt,hits=hits),indent=2)+'\n');print(cnt,len(hits))
