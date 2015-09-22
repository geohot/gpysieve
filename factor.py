import sys
import random
from fractions import gcd

def factor(n):
  ret = []
  nn = n

  phi = n

  d = 2
  while nn > 1:
    done = False
    while nn%d == 0:
      if not done:
        phi = phi - (phi/d)
        done = True
      print d,
      ret.append(d)
      sys.stdout.flush()
      nn /= d
    d += 1

  """
  print ""
  print "phi: ", phi
    
  for i in range(30):
    rr = random.randint(1, n)
    rr = rr/gcd(rr, n)
    print pow(rr, phi, n),
  print ""
  """

  return ret, phi

if __name__ == "__main__":
  n = int(sys.argv[1])
  factor(n)

