import itertools,json,random,tempfile,unittest
from collections import Counter
from math import gcd,lcm,prod
from pathlib import Path
from sympy import jacobi_symbol
from scale_lucas import DS,SEQ,config,inert_possible
from scale_progressions import merge
from order_cliques import boxes
from order_bank import checked_orders,search_bank,positions,period
from search import mr,is_strong_lucas_prp

class NextTests(unittest.TestCase):
    def test_crt(self):
        for a,m,b,n in [(8,12,2,6),(4,5,3,14),(1,8,0,2)]:
            vals=[k for k in range(lcm(m,n)) if k%m==a and k%n==b];ab=merge(a,m,b,n);self.assertEqual(vals,[ab[0]] if ab else [])

    def test_characters(self):
        rng=random.Random(120);dset=(5,-7,-11,13,-15,17,-19,21)
        for d in dset:
            prev,Q,z,odds=config(d);M=8*prod(odds)
            for _ in range(120):
                cs=tuple(sorted(rng.sample(range(1,65),3)));k=2*rng.randrange(40,300000);fs=[c*k-1 for c in cs];n=prod(fs);mn=min(c&-c for c in cs);neg={i for i,p in enumerate(fs) if jacobi_symbol(Q,p)<0};want=(all(jacobi_symbol(dd,n)==1 for dd in prev) and all(jacobi_symbol(d,p)==-1 and gcd(p,Q)==1 for p in fs) and (not neg or neg=={i for i,c in enumerate(cs) if c&-c==mn}));self.assertEqual(want,inert_possible(cs,d,k%M,M))

    def test_cancelling_pair(self):
        # 29*113=1 mod 84; neither factor is 1 mod 84. One further
        # order component is required to make 84 the exact LCM.
        ps=[29,113,1429,14449];counts=Counter();hits=[];search_bank(84,ps,counts,hits,'gp',forced=3,required=[3]);self.assertEqual(counts['strong_base2_products'],3);self.assertFalse(hits)
        for q in [1429,14449,1429*14449]:self.assertTrue(mr(29*113*q,[2]));self.assertFalse(is_strong_lucas_prp(29*113*q))

    def test_boxes(self):
        # Deliberately omit common order templates; a pairwise clique must
        # still be found, and no subset may be visited twice.
        from sympy import primerange,n_order
        ps=list(primerange(3,48))+[73,89];orders={p:int(n_order(2,p)) for p in ps};records=[dict(m=m,primes=[p for p in ps if orders[p]==m]) for m in set(orders.values())];got=set();ct=Counter();hits=[]
        with tempfile.TemporaryDirectory() as dd:
            path=Path(dd)/'x';path.write_text(''.join(json.dumps(r)+'\n' for r in records));
            for m,sub,forced in boxes(path):
                search_bank(m,sub,ct,hits,'gp',forced)
                for mask in range(1<<len(sub)):
                    if mask.bit_count()>=2 and mask&forced==forced:
                        key=tuple(sub[j] for j in positions(mask));self.assertNotIn(key,got);got.add(key)
        expected=set()
        for k in range(2,len(ps)+1):
            for group in itertools.combinations(ps,k):
                ms=[orders[p] for p in group]
                if len({a&-a for a in ms})==1 and all(p%lcm(*ms)==1 for p in group):expected.add(group)
        self.assertEqual(got,expected);self.assertIn((23,89),got);self.assertEqual(ct['strong_base2_products'],len(expected));self.assertFalse(hits)
        for group in got:self.assertTrue(mr(prod(group),[2]));self.assertFalse(is_strong_lucas_prp(prod(group)))
