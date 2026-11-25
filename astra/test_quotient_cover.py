import json,random,tempfile,unittest
from pathlib import Path
from quotient_cover import Projection,complete,decode,target
from verify_cover import check

D=Path(__file__).resolve().parent/'runs/lift12/trim';C=D/'cover-5103.json'
class CoverTests(unittest.TestCase):
    def test_cert(self):
        self.assertEqual(check(C,D)['quotient_size'],2**192*5103)

    def test_corruption(self):
        r=json.loads(C.read_text());r['blocks'][0]['reserve'].pop()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.json';path.write_text(json.dumps(r))
            with self.assertRaises(ValueError):check(path,D)

    def test_completion(self):
        r=json.loads(C.read_text());P=Projection(r['A'],r['B'],r['parts']);t,_=target(r['A'],r['B']);rng=random.Random(71)
        for j,c in enumerate(r['blocks']):
            free={c['primes'][i] for i in c['unused']}|{p for b in r['blocks'] if b!=c for p in b['primes']};partial=set(rng.sample(sorted(free),len(free)//2));w=complete(r,P,partial,j);self.assertEqual(set(w)&free,partial);v=1
            for p in w:v=v*p%P.m
            self.assertEqual((P.sig(v),P.odd(v)),(P.sig(t),P.odd(t)))
            ps=c['primes'];s=P.sig(ps[11])^P.sig(ps[50]);z=rng.randrange(P.h);ii=decode(c,P,s,z);v=1
            for i in ii:v=v*ps[i]%P.m
            self.assertEqual((P.sig(v),P.odd(v)),(s,z))
