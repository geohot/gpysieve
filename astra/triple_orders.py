"""Test every strong-MR triple in the supplied order catalogue against actual Selfridge."""
import argparse,json,hashlib,subprocess
from collections import Counter
from math import comb,gcd,lcm,prod
from pathlib import Path
from order_bank import checked_orders,word,DS,period
from order_cliques import coprime_product
from search import mr,is_strong_lucas_prp
from campaign import lucas_u,gp_check


def scan(src,binary,gp):
    db=checked_orders(src);rows=sorted((j,p) for j,ps in db.items() for p in ps);coprime_product([p for j,p in rows]);cmd=''.join(f'{j} {p}\n' for j,p in rows);r=subprocess.run([str(binary.resolve())],input=cmd,text=True,capture_output=True,check=True);counts=Counter(json.loads(r.stderr));counts['same_signature_triples']=sum(comb(k,3) for k in Counter(j&-j for j,p in rows).values());hits=[];dm=Counter();above=0
    for l in r.stdout.splitlines():
        ids=list(map(int,l.split()));v=[rows[i] for i in ids];fs=sorted(p for j,p in v);m=lcm(*(j for j,p in v));n=prod(fs);above+=n>2**64
        assert n%m==1;outside=sum(p%m!=1 for p in fs);counts[f'nonidentity_{outside}']+=1
        if m not in db:counts['unlisted_lcm']+=1
        neg=zero=0
        for p in fs:a,b=word(p);neg^=a;zero|=b
        first=(neg|zero)&-(neg|zero);D=DS[first.bit_length()-1] if first&neg else None
        if first&zero and n>abs(DS[first.bit_length()-1]):counts['prior_gcd']+=1;continue
        if D is not None:
            dm[D]+=1;Q=(1-D)//4
            for p in fs:
                t=period(p,D)
                if t==0 or lucas_u((n+1)%t if t else n+1,Q,p):break
            else:p=None
            if p is not None:continue
        counts['direct_tests']+=1
        if is_strong_lucas_prp(n):
            assert mr(n,[2]);ok,trace=gp_check(fs,gp);assert ok,trace;hits.append(dict(n=n,factors=fs,D=D,trace=trace))
    assert counts['all_three']==len(r.stdout.splitlines());return dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),triple_trace_sha256=hashlib.sha256(r.stdout.encode()).hexdigest(),counts=dict(counts),above_2_64=above,discriminants=dm,hits=hits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--bin',type=Path,default=Path('/tmp/astra-triples'));ap.add_argument('--gp',default='gp');a=ap.parse_args();r=scan(a.src,a.bin,a.gp);a.out.write_text(json.dumps(r,indent=2)+'\n');print(r)
