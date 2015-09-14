import gmpy
from fractions import gcd
import math

def qfac(n):
  s = 0
  while n&1 == 0 and n > 0:
    n /= 2
    s += 1
  return n,s

def b2spp(n):
  a = 2
  d,s = qfac(n-1)
  is_prime = False
  if pow(a, d, n) == 1:
    is_prime = True
  for r in range(s):
    if is_prime:
      break
    if pow(a, d*(1<<r), n) == n-1:
      is_prime = True
  return is_prime

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

  while gmpy.jacobi(D,n) != -1:
    #print "    ",D, n, legendre(D, n), gmpy.jacobi(D,n)
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

for n in range(3, 10000, 2):
  bt = b2spp(n)
  lt = ltest(n)
  if bt != bool(gmpy.is_prime(n)):
    print "BT", bt, n
  if lt != bool(gmpy.is_prime(n)):
    print "LT", lt, n


  

