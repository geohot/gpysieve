"""Exact support probabilities, not probabilities of passing BPSW."""
import json,argparse
from fractions import Fraction
from math import comb,lcm,log2
from collections import Counter
from pathlib import Path
from sympy import factorint

def audit(rows,weights):
    n=len(rows);support=Counter()
    for p,a,b in rows:
        for q,e in factorint(lcm(a,b)).items():
            if q!=2:support.update((int(q),j) for j in range(1,e+1))
    res={}
    for k in weights:
        prob={v:(Fraction(1,2**v) if k is None else Fraction(comb(n-v,k),comb(n,k))) for v in set(support.values())};union=sum(prob[v] for v in support.values());saved=sum(float(prob[v])*log2(q if j>1 else q-1) for (q,j),v in support.items());res[str(k)]=dict(union_bound=float(min(1,union)),expected_saved_odd_ambient_bits=saved)
    return dict(primes=n,odd_prime_power_levels=len(support),smallest_support=min(support.values()),by_weight=res)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--base',type=Path,default=Path('runs/lift12'));a=ap.parse_args();rows=json.loads((a.base/'rows--7.json').read_text());reports={}
    for tag in ('16','four-lists'):
        ps=set(map(int,(a.base/'trim'/f'trim-{tag}.txt').read_text().split()));reports[tag]=audit([r for r in rows if r[0] in ps],[None,3,5,31,101,301]);assert reports[tag]['primes']==len(ps)
    a.out.write_text(json.dumps(reports,indent=2)+'\n');print({k:v['by_weight']['None'] for k,v in reports.items()})
