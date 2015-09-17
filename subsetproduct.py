
from factor import factor
from math import log
import gmpy

primes = filter(gmpy.is_prime, range(1000))

P = primes[0:20]
print P
print "2^%d" % len(P)

N = 123673

ret, phi = factor(N)

print "2^%f" % (log(phi)/log(2))

# brute force

for i in xrange(1 << len(P)):
  tr = 1
  for j in range(len(P)):
    if (i>>j)&1 == 1:
      tr *= P[j]
  #print i, tr
  if (tr%N) == 1:
    print tr





