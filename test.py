import gmpy
from fractions import gcd
import math

def qfac(n):
  s = 0
  while n&1 == 0 and n > 0:
    n /= 2
    s += 1
  return n,s

# Fermat's (Little) Theorem check
# i think these psuedoprimes are a strict subset of b2spp ones
def flt(n):
  return pow(2, n-1, n) == 1

# strong base 2 psuedoprimes
def b2spp(n):
  a = 2
  d,s = qfac(n-1)
  is_prime = False

  if pow(a, d, n) == 1:
    is_prime = True

  # is this part what makes it strong?
  for r in range(s):
    if is_prime:
      break
    if pow(a, d*(1<<r), n) == n-1:
      is_prime = True

  return is_prime


def jacobi(a, n):
  # a / n
  # n must be odd?

  # for all p that are prime factors of n
  #   0 if a === 0 (mod p)
  #   1 if a !== 0 (mod p) and there exists x s.t. a === x^2 (mod p)
  #  -1 if a !== 0 (mod p) and there does not exist x

  # 1. Reduce the "numerator" modulo the "denominator" using rule 2.
  a = a%n

  # 2. Extract any factors of 2 from the "numerator" using rule 4 and evaluate them using rule 8.
  ret = 1
  while a&1 == 0 and a != 0:
    n8 = n%8
    if n8 == 3 or n8 == 5:
      ret *= -1
    a /= 2
    #print a

  #print a, n, ret
  if a == 1:
    return ret
  elif gcd(a, n) > 1:
    return 0
  else:
    # we know they are odd positive coprime integers
    # flip the symbol using rule 6?
    if n%4 == 3 and a%4 == 3:
      ret *= -1
    return ret * jacobi(n, a)

# http://codegolf.stackexchange.com/questions/10701/fastest-code-to-find-the-next-prime
def legendre(a, m):
  return pow(a, (m-1) >> 1, m)

def ltest(n):
  D = 5
  #while legendre(D, n) != n-1:

  # ahh, i think this fails iff n is a perfect square
  # clearly, no perfect squares are prime
  if int(math.sqrt(n))**2 == n:
    return False

  #print n
  while gmpy.jacobi(D,n) != -1:
    #print "    %4d %5d %2d %2d" % (D, n, jacobi(D, n), gmpy.jacobi(D,n))
    if jacobi(D, n) != gmpy.jacobi(D, n):
      print "FAILED AT",D,n
      exit(0)
    if D > 0:
      D += 2
    else:
      D -= 2
    D *= -1
    if D > 100:
      print "FAILURE"
      exit(0)
  P = 1
  Q = (1-D)/4

  # test n for Lucas psuedoprime using D, P, Q
  #print n, D, P, Q

  # delta n is n - (D/n), so n+1
  d,s = qfac(n+1)

  # a strong Lucas psuedoprime is when
  #    U_d     === 0 mod n
  # or V_d*2^r === 0 mod n for some r < s

  # U_0(P,Q) = 0
  # U_1(P,Q) = 1
  # V_0(P,Q) = 2
  # V_1(P,Q) = P

  U = [0]
  V = [2]

  # compute the Lucas sequence for D,P,Q in a stupid way
  for i in range(1, n+1):
    U.append((P*U[i-1] + V[i-1])/2)
    V.append((D*U[i-1] + P*V[i-1])/2)

  is_prime = False
  if U[d]%n == 0:
    is_prime = True
  for r in range(s):
    if is_prime:
      break
    if V[d*(1<<r)]%n == 0:
      is_prime = True
  return is_prime

def many():
  for n in range(3, 10000, 2):
    ft = flt(n)
    bt = b2spp(n)
    lt = ltest(n)
    if ft != bool(gmpy.is_prime(n)):
      print "FT", ft, n
    if bt != bool(gmpy.is_prime(n)):
      print "BT", bt, n
    if lt != bool(gmpy.is_prime(n)):
      print "LT", lt, n

#for i in range(1, 100, 2):
#for i in range(9, 30, 2):
  #print i, gmpy.jacobi(5,i), jacobi(5,i) 
  
#print gmpy.jacobi(13, 179), jacobi(13, 179)
#print gmpy.jacobi(-7, 29)
#print jacobi(-7, 29)
many()


