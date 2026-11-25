"""Shared-cost prime pool selection via maximum closure / exact integer min-cut."""
import json
from math import lcm,log2
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from sympy import factorint
from pool_audit import analyze, periods

@lru_cache(None)
def levels(a,b):
    return tuple((q,j) for q,e in factorint(lcm(a,b)).items() if q!=2 for j in range(e))

class Closure:
    def __init__(self,rows):
        self.rows=rows;self.used=[levels(a,b) for p,a,b in rows]
        self.coords=sorted(set().union(*self.used)); self.u={k:i for i,k in enumerate(self.coords)}
        self.cost=np.array([log2(q if j else q-1) for q,j in self.coords]);self.n=n=len(rows);self.s=n+len(self.coords);self.t=self.s+1
        self.i=np.array([i for i,ls in enumerate(self.used) for _ in ls]);self.j=np.array([n+self.u[k] for ls in self.used for k in ls]);self.scale=10000
    def cut(self,price,ban=()):
        n=self.n;s=self.s;t=self.t;cap=n*self.scale+1
        cost=np.rint(self.scale*price*self.cost).astype(np.int64)
        for k in ban:cost[self.u[k]]=cap
        v=csr_matrix((np.r_[np.full(n,self.scale),np.full(len(self.i),cap),cost],(np.r_[np.full(n,s),self.i,np.arange(n,s)],np.r_[np.arange(n),self.j,np.full(s-n,t)])),shape=(t+1,t+1))
        flow=maximum_flow(v,s,t);res=v-flow.flow;res.eliminate_zeros();side=breadth_first_order(res,s,return_predecessors=False);return tuple(sorted(int(i) for i in side if i<n))
    def bits(self,ids):
        keys={k for i in ids for k in self.used[i]};return 2+sum(self.cost[self.u[k]] for k in keys) # upper bound incl constant mod-4 and parity

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('pool',type=Path);ap.add_argument('out',type=Path);args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    ps=list(map(int,args.pool.read_text().split()));o=Closure([(p,*periods(p)) for p in ps]);seen={}
    for x in [0.1,0.2,0.4,0.6,0.8,1,1.05,1.1,1.13,1.15,1.155,1.16,1.17,1.2,1.3,1.5]:
        ids=o.cut(x)
        if ids and ids not in seen:
            rows=[o.rows[i] for i in ids];r=analyze(rows);seen[ids]=r;r['price']=x;(args.out/f'cut-{len(ids)}.txt').write_text(''.join(f'{p}\n' for p,_,_ in rows));print(x,len(ids),r['group_bits'],r['surplus_bits'],r['reachable'],flush=True)
    (args.out/'cuts.json').write_text(json.dumps(list(seen.values()),indent=2)+'\n')
