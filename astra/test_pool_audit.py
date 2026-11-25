import itertools
import random
import unittest
from math import gcd
from sympy import primerange
from pool_audit import joint,periods
from extend_pool import generate


class PoolTests(unittest.TestCase):
    def test_joint(self):
        # Both proper subgroups with full projections and mixed prime powers.
        cases=[([2,2],[[1,1]],[1,0]),([4,8],[[2,4],[0,2]],[2,0])]
        rng=random.Random(42)
        for _ in range(80):
            ks=[rng.choice([2,3,4,6,8,9,12]) for j in range(3)]
            cases.append((ks,[[rng.randrange(k) for k in ks] for _ in range(3)],[rng.randrange(k) for k in ks]))
        for ks,cols,t in cases:
            rows=[(k,[c[j] for c in cols]+[t[j]]) for j,k in enumerate(ks)];res=joint(rows)
            group={tuple(0 for k in ks)};queue=list(group)
            for x in queue:
                for c in cols:
                    y=tuple((u+v)%k for u,v,k in zip(x,c,ks))
                    if y not in group:group.add(y);queue.append(y)
            self.assertEqual(res['image'],len(group));self.assertEqual(res['reachable'],tuple(t) in group)

    def test_frontier(self):
        A=2*641*809;B=4*3*107*11*331;upper=16000;hmax=9
        found,_=generate(A,B,upper,hmax);want=[]
        for p in primerange(3,upper+1):
            if p%8!=3 or p%5 not in (2,3) or gcd(p,A*B)!=1:continue
            if min((p-1)//gcd(A,p-1),(p+1)//gcd(B,p+1))>hmax:continue
            a,b=periods(p)
            if A%a==B%b==0:want.append(p)
        self.assertIn(1283,want);self.assertEqual(found,want)
