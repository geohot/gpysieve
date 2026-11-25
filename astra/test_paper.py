import unittest
from itertools import combinations
from math import prod
from sympy import primerange, factorint
from paper_search import anchored, five
from pomerance import paper_pool


class PaperSearchTests(unittest.TestCase):
    def test_pair_join_and_general_join(self):
        ps=[3,5,7,11,13,17,19,23,29,31,37,41,43]
        for m in (8,15,60,252):
            want=[c+(47,) for k in range(2,len(ps)+1,2) for c in combinations(ps,k) if prod(c)%m==1]
            want5=[c for c in want if len(c)==5]
            self.assertTrue(want5)
            self.assertCountEqual(five(ps,47,m),want5)
            self.assertCountEqual(anchored(ps,47,m),want)

    def test_bounded_generator(self):
        T=2048; upper=25000
        def valid(x,res):
            return all(q<T and q%4==res and e==1 for q,e in factorint(x).items())
        brute=[p for p in primerange(T,upper+1) if p%8==3 and p%5 in (2,3) and valid((p-1)//2,1) and valid((p+1)//4,3)]
        self.assertTrue(brute)
        self.assertEqual(paper_pool(T,5,upper)['primes'],brute)
