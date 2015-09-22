import numpy as np
from factor import factor
from math import log
import gmpy

primes = filter(gmpy.is_prime, range(1000))

# fast demo
P = primes[20:30]
N = reduce(lambda x,y: x*y, primes[0:4])

#P = primes[20:35]
#N = reduce(lambda x,y: x*y, primes[0:5])

print P
print "2^%d" % len(P)
print "finding", N

ret, phi = factor(N)

odds = log(phi)/log(2)
print "2^%f" % odds

print "will find",len(P)-odds

if len(P)-odds < 1.0:
  exit(-1)

"""
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
"""

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

def generator(x):
  fact, phi = factor(x-1)
  #g = sorted(fact)[::-1][0]
  return fact

# mserrano: there's an isomorphism between (Z/pZ)^* and (Z/(p-1)Z)^+ for odd primes p
def isomorphism(x):

  # get potential generators
  #G, phi = factor(x-1)

  #for g in G:
  for g in range(2, x):
    quack = {1: 0}
    for i in range(1, x-1):
      quack[(g**i) % x] = i
    if len(quack) == x-1:
      print x,"has generator",g
      break

  if len(quack) != x-1:
    print "FAILED FAILED FAILED", x, G
    exit(-2)

  return quack

pmat = []
for x in rret:
  quack = isomorphism(x)
  pmat.append([quack[y%x] for y in P])

pmat = np.asarray(pmat).astype(np.int).T
print pmat

#test = [1,0,0,0,1,1,1,1]

"""
test = [1,1,0,1,1,0,1,1]
print test
print np.dot(test, pmat), re, np.dot(test, pmat) % np.asarray(re)
"""

#exit(0)

# for 7 -- 1 -> 0, 3 -> 1, 2 -> 2, 6 -> 3

# for example, x*y === 1 mod 3, how do we know?

print "WE ARE IMPORTING Z3"
from z3 import *
from time import time

start = time()
s = Solver()

ms = 8

bb = []
for i in range(pmat.shape[0]):

  #b = Bool('b_'+str(i))

  #b = BitVec('b_'+str(i), 1)
  #bi = to_int(b)
  #bi = BV2Int(b)
  #b = BV2Int(BitVec('b_'+str(i), 1))

  #b = Int('b_'+str(i))
  #s.add(b >= 0), s.add(b <= 1)

  b = BitVec('b_'+str(i), ms)
  s.add(b & 0xFE == 0)

  if i == 0:
    bor = b
  else:
    #bor = Or(b, bor)
    #bor += b
    bor |= b
  bb.append(b)

# at least one must be True
#s.add(bor == True)
s.add(bor > 0)

for j in range(pmat.shape[1]):
  for i in range(pmat.shape[0]):
    #iv = IntVal(pmat[i,j])
    iv = BitVecVal(pmat[i,j], ms)
    if i == 0:
      ss = iv * bb[i]
      print ss
      #ss = If(bb[i], pmat[i,j], 0)
    else:
      ss += iv * bb[i]
      #ss += If(bb[i], pmat[i,j], 0)
  s.add(ss % re[j] == 0)
  
print s.check()
print s.model()
print time()-start, "seconds elapsed"

ret = [0]*pmat.shape[0]
m = s.model()
for w in m:
  if m[w].as_long() == 1:
    ret[int(str(w).split("_")[1])] = 1

print np.dot(ret, pmat), re, np.dot(ret, pmat) % np.asarray(re)


