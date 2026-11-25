import itertools,json,os,shutil,subprocess,tempfile,unittest
from collections import Counter
from math import gcd,lcm,prod
from pathlib import Path
from order_bank import period,character,positions
from lucas_first import join,metadata,crt,prefix
from lucas_tail import kernel,plan,finish,selfridge
from campaign import lucas_u
from search import mr,is_strong_lucas_prp
from sympy import primerange,n_order

ROOT=Path(__file__).resolve().parent
class LucasFirstTests(unittest.TestCase):
    def test_bidirectional_controls(self):
        # Deliberately weaken the lower bound to 1: known one-sided passes
        # must reach the opposite test, not silently disappear at the join.
        for reverse,m,rows in [(False,153,[(7,3),(73,9),(103,51)]),(True,27,[(53,27),(109,27)])]:
            counts=Counter();hits=[];fs=[p for p,j in rows];self.assertEqual(selfridge(fs),5);self.assertEqual(mr(prod(fs),[2]),not reverse);self.assertEqual(is_strong_lucas_prp(prod(fs)),reverse)
            data={p:(1,p-1 if reverse else period(p,5),int(character(5,p%5)<0)) for p in fs}
            for maxhalf in (0,19):
                counts.clear();join(m,5,rows,data,counts,hits,None,maxhalf,reverse);self.assertEqual(counts['canonical_matches'],1);self.assertEqual(counts['all_divisors_pass'],1);self.assertEqual(hits,[])

    def test_kernel(self):
        cols=[7,19,0,9,3,14,5,2,7];k=kernel(cols);actual=set()
        for choices in itertools.product([0,1],repeat=len(k)):
            v=0
            for b,w in zip(choices,k):
                if b:v^=w
            actual.add(v)
        want=set()
        for m in range(1<<len(cols)):
            x=0
            for i in positions(m):x^=cols[i]
            if x==0:want.add(m)
        self.assertEqual(want,actual)

    @unittest.skipUnless(shutil.which('g++'),'requires native GMP development headers')
    def test_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rank=root/'rank';tri=root/'tri'
            for f,b in [('rank_divisors.cpp',rank),('triple_orders.cpp',tri)]:subprocess.run(['g++','-O2','-std=c++17',str(ROOT/f),'-lgmpxx','-lgmp','-o',str(b)],check=True)
            ps=list(primerange(3,330));rows=sorted((int(n_order(2,p)),p) for p in ps);cmd=''.join(f'{j} {p}\n' for j,p in rows);proc=subprocess.run([str(tri)],input=cmd,capture_output=True,text=True,check=True);got={tuple(map(int,l.split())) for l in proc.stdout.splitlines()};want={ix for ix in itertools.combinations(range(len(rows)),3) if mr(prod(rows[i][1] for i in ix),[2])};self.assertEqual(want,got);self.assertTrue(want)
            for d in (5,-7):
                for reverse in (False,True):
                    data=metadata(ps,d,rank,reverse)
                    for p,(lo,hi,_) in data.items():
                        self.assertEqual(lo,hi);self.assertEqual(lo,int(n_order(2,p)) if reverse else period(p,d))

    def test_tail_partition(self):
        # Every subset with actual D and required LCM is either in exactly
        # one eligible early group at that D or is in the tail nullspace.
        from collections import defaultdict
        m=153;g=[(p,int(n_order(2,p))) for p in primerange(3,1500) if gcd(p,m)==1 and (p-1)%2==0 and pow(2,m,p)==1];direct=[];early=defaultdict(list);plan(m,g,direct,early,Counter(),1);covered=0
        for k in range(3,len(g)+1):
            for s in itertools.combinations(g,k):
                ps=[p for p,j in s];n=prod(ps)
                if lcm(*(j for p,j in s))!=m or n%m!=1 or sum(p%m!=1 for p in ps)<3:continue
                d=selfridge(ps)
                if d is None:continue
                w=sum(set(s)<=set(h) for mm,h in early.get(d,[]));
                for mm,h,ker in direct:
                    if not set(s)<=set(h):continue
                    bits=sum(1<<h.index(v) for v in s);bas={}
                    for v in ker:
                        while v and v.bit_length() in bas:v^=bas[v.bit_length()]
                        if v:bas[v.bit_length()]=v
                    while bits and bits.bit_length() in bas:bits^=bas[bits.bit_length()]
                    w+=not bits
                covered+=1;self.assertEqual(w,1)
        self.assertTrue(covered)
