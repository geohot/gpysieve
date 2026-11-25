from math import prod
from itertools import product
import unittest
from campaign import ROOT,lucas_u
from char_sieve import solve,chi,parts
from mixed import affine_count,getrow,job_for,DS,symbol


class CharacterTests(unittest.TestCase):
    def test_affine_counts(self):
        self.assertEqual(affine_count(4,[(3,1),(6,0)]),4)
        self.assertEqual(affine_count(4,[(3,1),(6,0),(5,0)]),0)

    def test_periods_split_and_inert(self):
        for D,p in product((5,-7,-11,13),(11,13,17,19,23,29,31,37,41,43)):
            r=getrow(D,p)
            if r:
                _,a,b=r;self.assertEqual(next(k for k in range(1,p) if pow(2,k,p)==1),a)
                self.assertEqual(next(k for k in range(1,p+2) if lucas_u(k,(1-D)//4,p)==0),b)

    @unittest.skipUnless((ROOT/'chars').is_file(),'compile chars.cpp first')
    def test_planted_mixed_residues(self):
        # Controlled periods give real positive and negative CRT matches above 2^64.
        # Include primes that are +1 as well as -1 mod b: cofactor target is not always 1.
        ps=[101,103,107,109,113,127,131,137,139,149,151,157,163,167,173]; core=[(q,2,3 if i%3 else 4) for i,q in enumerate(ps)];anchor=(179,2,3)
        for D in (5,-7,-11):
            j=job_for(D,anchor,core);count,fs=solve(j);expect=[];space=0;cp=0
            for mask in range(1,1<<len(core)):
                selected=[r for i,r in enumerate(core) if mask>>i&1]+[anchor]; n=prod(r[0] for r in selected)
                if not all(prod(symbol(d,r[0]) for r in selected)==(-1 if d==D else 1) for d in DS[:DS.index(D)+1]):continue
                if D==5 and not any(symbol(5,r[0])>0 for r in selected):continue
                space+=1
                if not all(chi(n,h)==chi(sign,h) for _,a,b in selected for k,sign in [(a,1),(b,-1)] for h in parts(k)):continue
                cp+=1
                if n>=1<<64 and all((n-1)%a==(n+1)%b==0 for _,a,b in selected):expect.append([r[0] for r in selected])
            self.assertEqual(count['subsets'],space);self.assertEqual(count['char_pass'],cp);self.assertTrue(expect);self.assertCountEqual(fs,expect)
