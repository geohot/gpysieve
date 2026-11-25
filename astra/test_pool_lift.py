from itertools import combinations
from math import gcd,prod
from pathlib import Path
import unittest
from sympy import primerange
from closure_pool import Closure
from pool_lift import discover
from mixed import getrow
from paper_search import periods


class LiftTests(unittest.TestCase):
    def test_cut_against_every_subset(self):
        rows=[(p,2*a,4*b) for p,a,b in [(11,13,3),(19,13,9),(43,13,3),(59,17,7),(83,13,7),(107,17,9)]];o=Closure(rows)
        for x in [0.04,0.17,0.3,0.61,1.0]:
            def score(ids):
                ks={k for i in ids for k in o.used[i]};return len(ids)*o.scale-sum(round(x*o.scale*o.cost[o.u[k]]) for k in ks)
            best=max(score(ids) for t in range(len(rows)+1) for ids in combinations(range(len(rows)),t));self.assertEqual(score(o.cut(x)),best)

    @unittest.skipUnless(Path(__file__).resolve().with_name('divisor_lift').exists(),'compile divisor_lift.cpp')
    def test_native(self):
        upper=45000;H=21;C=64
        ds,A,B=discover(2*13*17*641,4*3*7*11*107,upper,H,C,20)
        for D in (5,-7):
            want=set()
            for p in primerange(3,upper+1):
                if p%8!=3 or (D==5 and p%5 not in (2,3)) or (D==-7 and p%7 not in (3,5,6)):continue
                u=(p-1)//gcd(A,p-1);v=(p+1)//gcd(B,p+1)
                if min(u,v)>H or max(u,v)>C:continue
                if D==5:a,b=periods(p)
                else:
                    r=getrow(D,p)
                    if not r:continue
                    _,a,b=r
                if A%a==B%b==0:want.add(p)
            self.assertTrue(want);self.assertEqual(ds[D],want)
