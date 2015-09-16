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
  # n must be odd
  assert(n&1)

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


def getd(n):
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

  return D

def ltest(n):
  D = getd(n)
  P = 1
  Q = (1-D)/4
  print D, P, Q

  # D = P^2 - 4Q

  # test n for Lucas psuedoprime using D, P, Q
  # Lucas sequence is x_n = P*x_n-1 - Q*x_n-2
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

  # U in the (1, -1) case is fibonacci numbers
  U = [0]
  V = [2]

  # compute the Lucas sequence for D,P,Q in a stupid way
  for i in range(1, n+1):
    U.append((P*U[i-1] + V[i-1])/2)
    V.append((D*U[i-1] + P*V[i-1])/2)

  print U, V

  # check the fib number
  is_prime = False
  if U[d]%n == 0:
    is_prime = True

  # check the lucas numbers
  for r in range(s):
    if is_prime:
      break
    if V[d*(1<<r)]%n == 0:
      is_prime = True
  return is_prime

def many():
  for n in range(3, 10000, 2):
    ft = flt(n)
    if ft != bool(gmpy.is_prime(n)):
      print "FT", ft, n

    bt = b2spp(n)
    if bt != bool(gmpy.is_prime(n)):
      print "BT", bt, n

    """
    lt = ltest(n)
    if lt != bool(gmpy.is_prime(n)):
      print "LT", lt, n
    """

#ltest(123)
#exit(0)


#for i in range(1, 100, 2):
#for i in range(9, 30, 2):
  #print i, gmpy.jacobi(5,i), jacobi(5,i) 
  
#print gmpy.jacobi(13, 179), jacobi(13, 179)
#print gmpy.jacobi(-7, 29)
#print jacobi(-7, 29)
#many()

# there are no strong Carmichael numbers

# search only where jacobi(5,i) == -1?

from sieve import vfactorize
import numpy as np
import itertools
import math

T = 8*20 + 3
k = 5

#print "frac", ((T**2) * (1-4./k)),  math.log(pow(T, k))

P = filter(gmpy.is_prime, range(2, T))
print P

p1 = filter(lambda x: x%4 == 1, P)
p2 = filter(lambda x: x%4 == 3, P)

print p1, p2

# (p-1)/2 is made up of primes in p1
# (p+1)/4 is made up of primes in p2
def mprod(x):
  ret = []
  for i in range(1<<len(x)):
    app = 1
    for j in range(0, len(x)):
      if (i>>j)&1 == 1:
        app *= x[j]
    ret.append(app)
  return ret

print "doing mprods"
P1 = set(mprod(p1))
P2 = set(mprod(p2))
print "mprods done"

Q1 = reduce(lambda x,y: x*y, p1)
Q2 = reduce(lambda x,y: x*y, p2)
print Q1, Q2

"""
227
29867
73883
5696123
"""

test = 227*5696123

print test
print test%Q1
print test%Q2
print flt(227*5696123)


# Carmichael numbers
#   that are also strong base 2 psuedoprimes
#   jacobi(5, n) == -1


# condition b
for pp in P1:
  p = pp*2+1
  # half of condition a
  if p%8 != 3:
    continue
  # condition c
  if (p+1)/4 not in P2:
    continue
  # other half of condition a
  if gmpy.jacobi(5, p) != -1:
    continue
  # prime
  if not gmpy.is_prime(p):
    continue
  print p

"""
# half of condition a
for p in xrange(T, pow(T, k), 8):
  # prime
  if not gmpy.is_prime(p):
    continue
  # other half of condition a
  if gmpy.jacobi(5, p) != -1:
    continue
  # condition b and c
  if (p-1)/2 in P1 and (p+1)/4 in P2:
    print p
"""



"""
p = 83
tp1 = vfactorize((p-1)/2, P)
tp2 = vfactorize((p+1)/4, P)

print tp1, tp2
"""

"""
# now we multiply this shit
for p in xrange(T, pow(T, k), 8):
  if p%8 == 3 and gmpy.is_prime(p) and gmpy.jacobi(5, p) == -1:
    p1 = vfactorize((p-1)/2, P)
    p2 = vfactorize((p+1)/4, P)

    # composed solely of primes q < T
    if p1 == None or p2 == None:
      continue

    # square free
    if np.any(p1 > 1) or np.any(p2 > 1):
      continue

    print p
"""

