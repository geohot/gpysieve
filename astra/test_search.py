from itertools import combinations, product
from math import prod
import unittest

from search import catalog, matches, target, profile, mr, is_strong_lucas_prp


class TestSearch(unittest.TestCase):
    def test_components(self):
        for n, expected in [(101,(1,1)), (2047,(1,0)), (5459,(0,1)), (49,(0,0))]:
            self.assertEqual((mr(n,[2]), is_strong_lucas_prp(n)), expected)

    def test_orders(self):
        for p, a, b in catalog(500):
            self.assertEqual(a, next(k for k in range(1,p) if pow(2,k,p)==1))
            u,v = 1,1
            for k in range(1,p+2):
                if not u: break
                u,v = v,(u+v)%p
            self.assertEqual((k,u),(b,0))

    def test_sharded_matches_with_collisions(self):
        ps = [3,5,7,11,13,17,19]
        for t in range(8):
            expected = {c for k in (3,5,7) for c in combinations(ps,k) if prod(c)%8==t}
            found = [tuple(c) for a,b in [(0,3),(3,11),(11,16)] for c in matches(ps,t,8,3,a,b)]
            self.assertEqual(set(found),expected)
            self.assertEqual(len(found),len(expected))

    def test_group(self):
        for a,b in [(12,5),(8,9),(4,3),(1,11),(1,2)]:
            t,m = target(a,b); ps=[7,13]
            group={(1,0)}
            while True:
                new=group|{(x*p%m,1-y) for (x,y),p in product(group,ps)}
                if new==group: break
                group=new
            self.assertEqual(profile(dict(A=a,B=b,primes=ps)),(len(group),(t,1) in group))


if __name__ == '__main__':
    unittest.main()
