"""Summarize completed work; skip a partial final line from a live writer."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def report(files):
    totals=defaultdict(Counter); seen=set()
    for file in files:
        for line in Path(file).read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'): continue
            r=json.loads(line)
            if r['event'] in ('candidate','disagreement','verified'):
                print(file,json.dumps(r)); continue
            if r['event']!='done': continue
            if 'manifest' in r:
                key=r['manifest'],r['id']; label='anchor '+r['family']
            else:
                key=tuple(r['params']),r['batch']; label='gcd D='+str(r['params'][2])
            if key in seen: continue
            seen.add(key); c=totals[label]; c['receipts']+=1
            for k in ('subsets','anchor_pass','two_pass','full_pass','prefixes','large_prime_divisors','prefix_pass'):
                if k in r: c[k]+=r[k]
    for k,v in sorted(totals.items()): print(k,json.dumps(v,sort_keys=True))

if __name__=='__main__': report(sys.argv[1:])
