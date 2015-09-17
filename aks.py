import gmpy
from math import factorial

def fact_mod(x, n):
  ret = 1
  for i in range(2, x+1):
    ret *= i
    ret %= n
  return ret


def choose(n, k):
  top = factorial(n)
  bottom = factorial(k) * factorial(n-k)

  print n, k, top, bottom, top // bottom, (top // bottom)%n

  return top // bottom

#  i believe if it's true for a=1 it's true for all
#    (x - a) ^ n === (x^n - a) mod n

def comp(x, n, a=1):
  lhs = pow(x-a, n, n)
  rhs = (pow(x,n,n) - a) % n
  return lhs==rhs

# expand out (x - a) ^ n
def isprime_alt(n):
  for k in range(1, n):
    if choose(n,k)%n != 0:
      return False
  return True

def isprime(n):
  for x in range(1, n):
    if not comp(x,n):
      return False
  return True


"""
if __name__ == "__main__":
  # hmm, finds carmichael number 561
  for n in range(2, 1000):
    aks = isprime_alt(n)
    if aks != bool(gmpy.is_prime(n)):
      print "OMGOMG", n, aks, gmpy.is_prime(n)
"""

print isprime_alt(23)
#print isprime_alt(561)

