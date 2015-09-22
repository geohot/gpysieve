import numpy as np
from factor import factor
from math import log
import gmpy

primes = filter(gmpy.is_prime, range(1000))

P = primes[20:28]
print P
print "2^%d" % len(P)

N = reduce(lambda x,y: x*y, primes[0:4])
print "finding", N

ret, phi = factor(N)

print "2^%f" % (log(phi)/log(2))

# brute force
for i in xrange(1 << len(P)):
  # check, but don't add it to choice
  tr = 1
  for j in range(len(P)):
    if (i>>j)&1 == 1:
      tr *= P[j]

  # see if this one is good
  if (tr%N) == 1:
    # reconstruct the ones we multiplied together
    choice = []
    for j in range(len(P)):
      if (i>>j)&1 == 1:
        choice.append(P[j])

    # x*N + 1 == tr
    x = (tr-1)/N
    print tr, x, bin(i)[2:].rjust(len(P), '0')[::-1], choice

    # x*N === -1 mod p
    for p in choice:
      print p, x*N % p

# we know the target number is === 1 mod <all factors>
#                          and === 0 mod <all things used in subset>

# ok so we talking search
# intersection of the subsets that work for each factor

# so theres a matrix here

rret = ret[1:]
re = [x-1 for x in rret]  # mods for isomorphism
print rret
print P

mat = []
for x in rret:
  mat.append([y%x for y in P])
mat = np.asarray(mat).astype(np.int).T
print mat

# mserrano: there's an isomorphism between (Z/pZ)^* and (Z/(p-1)Z)^+ for odd primes p
def isomorphism(x):
  quack = {1: 0}

  # get generator
  if x == 7:
    g = 3
  if x == 5:
    g = 2
  if x == 3:
    g = 2

  for i in range(1, x-1):
    quack[(g**i) % x] = i

  return quack

pmat = []
for x in rret:
  quack = isomorphism(x)
  pmat.append([quack[y%x] for y in P])

pmat = np.asarray(pmat).astype(np.int).T
print pmat

#test = [1,0,0,0,1,1,1,1]

test = [1,1,0,1,1,0,1,1]
print test
print np.dot(test, pmat), re, np.dot(test, pmat) % np.asarray(re)

#exit(0)

# for 7 -- 1 -> 0, 3 -> 1, 2 -> 2, 6 -> 3

# for example, x*y === 1 mod 3, how do we know?

from z3 import *

s = Solver()

bb = []
for i in range(pmat.shape[0]):
  b = Int('b_'+str(i))
  s.add(b >= 0)
  s.add(b <= 1)
  if i == 0:
    bsum = b
  else:
    bsum += b
  bb.append(b)
s.add(bsum != 0)

for j in range(pmat.shape[1]):
  for i in range(pmat.shape[0]):
    if i == 0:
      ss = bb[i] * pmat[i,j]
    else:
      ss += bb[i] * pmat[i,j]
  s.add(ss % re[j] == 0)
  
print s.check()
print s.model()


"""
test = 4336848158096904451
print test

for p in ret:
  print test%p, p

for p in P:
  print test%p, p
"""


