import json,os,subprocess,unittest
from fractions import Fraction
from itertools import combinations
from math import gcd,lcm,prod
from pathlib import Path
from shutil import which
from sympy import isprime,n_order,primerange
from activation_audit import audit
from scale_shapes import indices
from scale_refine import classes
from search import mr
from shape_scan import ROOT

class Shapes(unittest.TestCase):
    def test_resultants(self):
        # Compare to direct local Fermat exponent, including degenerate shapes.
        for cs in combinations(range(1,12),3):
            es=indices(cs)
            for k in range(2,52,2):
                ps=[c*k-1 for c in cs];n=prod(ps)
                for i,p in enumerate(ps):
                    if p<3 or not isprime(p):continue
                    m=int(n_order(2,p))
                    if (n-1)%m:continue
                    self.assertEqual(es[i]%m,0);self.assertTrue(any(k%M==x for x,M,_ in classes(cs,i,m)))

    def test_small_sparse(self):
        rows=[(11,10,12),(19,18,4),(43,14,44),(59,58,60)];r=audit(rows,[None]);missing=sum(lcm(*(lcm(a,b) for p,a,b in c))!=lcm(*(lcm(a,b) for p,a,b in rows)) for k in range(5) for c in combinations(rows,k));self.assertGreaterEqual(r['by_weight']['None']['union_bound'],missing/16)

    def test_signature_patterns(self):
        from constellations import tails
        for k in range(2,10):
            raw=list(tails(k,28));want={t for t in combinations(range(2,29,2),k-1) if len({s%8 for s in (0,)+t})<=2};self.assertEqual(len(raw),len(want));self.assertEqual(set(raw),want)

    def test_reports_controls(self):
        d=ROOT/'runs/shape';ap=json.loads((d/'ap.json').read_text());self.assertEqual([r['n'] for r in ap['base2_pass']],[29341]);allD=json.loads((d/'anyD.json').read_text());ps=list(primerange(3,251));base={tuple(v) for v in allD['MR']};self.assertIn((13,37,61),base)
        for a,b,c in combinations(ps,3):
            if c-a<allD['first_unresolved_diameter'] and mr(a*b*c,[2]):self.assertIn((a,b,c),base)

    @unittest.skipUnless(which(os.environ.get('GP','gp')),'GP not installed')
    def test_gp_sieve(self):
        # Independent exact integer Lucas recurrence and polynomial evaluations.
        cmd=[];answers=[]
        for D in (5,-7,-11,13,21):
            Q=(1-D)//4;U=[0,1]
            for i in range(1,1200):U.append(U[-1]-Q*U[-2])
            for u,v in [(2,4),(6,6),(8,14)]:
                for s in (-u,0,v):
                    i,j=[x-s for x in (-u,0,v) if x!=s];r=(1+i)*(1+j)-1;w=(i-1)*(j-1)-1;answers.append(lcm(gcd(2**abs(r)-1,U[abs(w)]),gcd(2**abs(r)-1,U[abs(r+2)])) if r else -1);cmd.append(f'print(shapeg({u},{v},{s},{D}))')
        r=subprocess.run([os.environ.get('GP','gp'),'-fq',str(ROOT/'shape.gp')],input='\n'.join(cmd)+'\n',capture_output=True,text=True,check=True);self.assertEqual(r.stderr,'');self.assertEqual(list(map(int,r.stdout.splitlines())),answers)
