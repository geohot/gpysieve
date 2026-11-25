"""Probe generator span (not binary coverage) in four disjoint supports on the common target."""
import argparse,json,random
from math import gcd,log2
from pathlib import Path
from pool_audit import coords,joint
from search import target

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--seed',type=int,default=1);args=ap.parse_args();d=args.folder;ps=list(map(int,(d/'trim-four-lists.txt').read_text().split()));g=json.loads((d/'trim.json').read_text())['four-lists'];A,B=g['A'],g['B'];assert A%5==B%7==0 and all(p%8==3 and p%7 in (3,5,6) and gcd(p,A*B)==1 for p in ps)
    random.Random(args.seed).shuffle(ps);t,m=target(A,B);report=[]
    for i in range(4):
        part=ps[i::4];rs=list(coords(m,[*part,t]))+[(2,[1]*(len(part)+1)),(2,[int(p%5 in (2,3)) for p in part]+[0])];r=joint(rs);report.append(dict(size=len(part),group_bits=log2(r['image']),index=g['image']//r['image'],target_reachable=r['reachable']));assert g['image']%r['image']==0
    (d/'blocks.json').write_text(json.dumps(dict(seed=args.seed,blocks=report),indent=2)+'\n');print(report)
