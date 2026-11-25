import itertools,json,math,unittest
from pathlib import Path
from quotient_cover import Projection,target,insert
from quartic_lift import certify

D=Path(__file__).resolve().parent/'runs/lift12/trim'
class LiftCertificates(unittest.TestCase):
    def test_saved_moves(self):
        c=json.loads((D/'cover-5103.json').read_text());r=json.loads((D/'moves24.json').read_text());P=Projection(c['A'],c['B'],c['parts']);t,m=target(c['A'],c['B']);pairs=[tuple(math.prod(b['primes'][j] for j in side)%m for side in ab) for b in c['blocks'] for ab in b['switches']];start=int(r['start'],16);rat=[];base=math.prod(c['blocks'][0]['primes'][i] for i in r['reserve'])%m
        for j,(a,b) in enumerate(pairs):
            if start>>j&1:a,b=b,a
            base=base*a%m;rat.append(b*pow(a,-1,m)%m)
        self.assertEqual((P.sig(base),P.odd(base)),(P.sig(t),P.odd(t)));qs=r['qs'];ds=[int(x,16) for x in r['directions']];span={}
        def delta(v):
            z=1
            while v:
                bit=v&-v;z=z*rat[bit.bit_length()-1]%m;v-=bit
            return z
        for v in ds:
            z=delta(v);self.assertEqual((P.sig(z),P.odd(z)),(0,0));lift=[pow(z,(q-1)//4,q) for q in qs];self.assertTrue(all(a in (1,q-1) for q,a in zip(qs,lift)));self.assertTrue(insert(span,sum(int(x!=1)<<j for j,x in enumerate(lift))))
        # Pair overlap appears twice in the ordinary product and zero times in XOR.
        for v,w in itertools.combinations(ds,2):
            z=delta(v&w);self.assertTrue(all(pow(z,(q-1)//2,q)==1 for q in qs))
        sel=r['target_selection'];self.assertEqual(len(sel),len(set(sel)));z=math.prod(sel)%m;self.assertEqual((P.sig(z),P.odd(z)),(P.sig(t),P.odd(t)));self.assertTrue(all(pow(z,(q-1)//4,q)==pow(t,(q-1)//4,q) for q in qs));self.assertNotEqual(z,t)

    def test_bounds(self):
        for filename,tag,n4,n8 in [('fourier.json','four-lists',101,24),('fourier-small.json','16',74,0)]:
            old=json.loads((D/filename).read_text());r=certify(D,n8,n4,tag);self.assertEqual(r['quotient_size'],old['quotient_size']);self.assertEqual(r['fourier_sum_upper_thousandths'],old['fourier_sum_upper_thousandths']);self.assertLess(r['fourier_sum_upper_thousandths'],2000)
