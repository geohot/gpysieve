import unittest
from math import gcd,lcm,prod
from itertools import product
from sympy import jacobi_symbol,divisors
from semi import exponents
from mersenne import sweep,selfridge,character
from index_scan import components
from search import mr,is_strong_lucas_prp


class Special(unittest.TestCase):
    def test_partner_exponents(self):
        for D,p,a,b in [(5,61681,40,1542),(-15,87211,54,95),(-11,15467,814,19)]:
            A,B=exponents(p,a,b);self.assertEqual((2*((p-1)//a))%B,0)
            for z in divisors(p-1):
                if gcd(a,z)<=2 and z&-z==b&-b:self.assertEqual(B%z,0)
                if gcd(b,z)<=2 and z&-z==a&-a:self.assertEqual(A%z,0)

    def test_cyclotomic_reassembly(self):
        for D in (5,-7,13):
            phi=dict(components(D,64));u,v=0,1
            for m in range(1,129):
                u,v=v,v-((1-D)//4)*u
                if m%2==0:self.assertEqual(prod(phi[d] for d in divisors(m) if d>1),abs(u))

    def test_mersenne_control(self):
        p,q=193707721,761838257287;n=p*q
        self.assertEqual(n,2**67-1);self.assertTrue(mr(n,[2]));self.assertFalse(is_strong_lucas_prp(n));r,h=sweep((67,(q,67),[(p,67)]));self.assertEqual(r['constructed_mr'],1);self.assertFalse(h)
        for n,d in product(range(501,701,2),[5,-7,9,-11,13]):self.assertEqual(character(n,abs(d)),int(jacobi_symbol(d,n)))
        self.assertEqual(selfridge(5459),-7)
