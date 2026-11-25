import os
from math import gcd,lcm,prod
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from campaign import lucas_u,compatible,scan,run,rows_for,gp_check
from gcd_scan import prefixes,compute


class CampaignTests(unittest.TestCase):
    def test_rank(self):
        for p,a,b in rows_for(-7,2000):
            u,v=0,1
            for k in range(1,b+1):
                u,v=v,(v-2*u)%p
                self.assertEqual(u,lucas_u(k,2,p))
                self.assertEqual(u==0,k==b)

    def test_exact_anchor_counts(self):
        ps=[487,1237,2687,4637,8167,16567,33037,68147,123457]
        # Synthetic periods plant many positives; test scan(), not primality.
        core=[[p,2,4] for p in ps[:-1]]; anchor=[ps[-1],2,2]
        hits=[]; result=scan(dict(core=core,anchor=anchor),hits.append)
        expect=[tuple(sub)+(ps[-1],) for k in (2,4,6,8) for sub in combinations(ps[:-1],k) if prod(sub)*ps[-1]>=1<<64 and (prod(sub)*ps[-1]+1)%4==0]
        self.assertTrue(expect); self.assertCountEqual(hits,expect);self.assertEqual(result['full_pass'],len(expect))

    @unittest.skipUnless(os.environ.get('ASTRA_GP'),'set ASTRA_GP to exercise PARI/GP')
    def test_gp_completion(self):
        gp=os.environ['ASTRA_GP']
        for D in [5,-7]:
            batch=list(prefixes(10000,D))[:6]; self.assertTrue(batch)
            for (p,q,_,_),(g,rs) in zip(batch,compute(batch,D,gp)):
                k=p*q-1; u,v=0,1
                for _ in range(k): u,v=v,v-((1-D)//4)*u
                self.assertEqual(g,gcd((1<<k)-1,u))
                self.assertTrue(all(g%r==0 for r in rs))
        self.assertFalse(gp_check([53,103],gp)[0])
