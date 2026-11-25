"""Treat Cunningham Table 2 as hints; reconstruct primitive components by exact division."""
import argparse,json,re
from collections import defaultdict,Counter
from hashlib import sha256
from math import gcd,prod
from pathlib import Path
from sympy import isprime
from scale_refine import cyclo


def hints(path):
    groups=defaultdict(set);mode=None;n=None
    for ln in re.sub(r'\\\s*\n\s*','',path.read_text(encoding='latin1')).splitlines():
        if ln.startswith('    Table '):
            mode=ln.split()[1]
            if mode.startswith('3'):break
            n=None;continue
        if not mode:continue
        start=re.match(r'^ {0,4}(\d+)(L?)\s+(.*)$',ln)
        if start:n=int(start[1]);ln=start[3]
        if n is None:continue
        m=n if mode=='2-' else 2*n
        ln=re.sub(r'\([^()]*\)',' ',ln);groups[m].update(int(v) for v in re.findall(r'(?<![\w])\d+(?![\w])',ln) if int(v)>1)
    return groups


def build(src,table,out,bits):
    old={r['m']:set(r['primes']) for l in src.read_text().splitlines() if (r:=json.loads(l))['D']==2};hh=hints(table);counts=Counter();new=set()
    with out.open('w') as f:
        for m in sorted((set(hh)|set(old))-{1}):
            z=cyclo(m);qs=old.get(m,set()).copy();assert all(z%p==0 for p in qs)
            for p in hh.get(m,()):
                if p.bit_length()<=bits and z%p==0 and isprime(p):qs.add(p)
            parts=[z]
            if m%8==4:
                t=m//4;L=2**t+1-2**((t+1)//2);g=gcd(z,L);parts=[g,z//g]
            rem=[]
            for v in parts:
                for p in qs:
                    while v%p==0:v//=p
                if v>1:
                    if v.bit_length()<=bits and isprime(v):qs.add(v)
                    else:rem.append(v)
            # 'primes' is provisional pending GP certification by downstream
            # sieve; unresolved entries are not assumed prime or semiprime.
            new.update(qs-old.get(m,set()));print(json.dumps(dict(event='factored',m=m,D=2,primes=sorted(qs),unresolved=rem),separators=(',',':')),file=f);counts['components']+=1;counts['complete']+=not rem
    return dict(**counts,new_probable_primes=len(new),bits_cap=bits,source_sha256=sha256(src.read_bytes()).hexdigest(),table_sha256=sha256(table.read_bytes()).hexdigest(),output_sha256=sha256(out.read_bytes()).hexdigest())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('table',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--bits',type=int,default=420);a=ap.parse_args();print(json.dumps(build(a.src,a.table,a.out,a.bits)))
