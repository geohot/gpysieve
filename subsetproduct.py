import numpy as np
from factor import factor
from math import log
import gmpy

primes = filter(gmpy.is_prime, range(10000))

# P and N
# (1248, 70) takes ???? seconds
#from findd import *
# max sum is 18 bits
# after reduction it's (1248, 169)

# (60, 14) takes ??? seconds
# reduce to (60, 27)
#P = primes[20:200]
#N = reduce(lambda x,y: x*y, primes[0:12])

# (40, 9) takes 11 seconds, 2 seconds with hacks
# reduce to (40, 15)
#P = primes[20:60]
#N = reduce(lambda x,y: x*y, primes[0:10])

# swag
P = primes[20:60]
N = reduce(lambda x,y: x*y, primes[0:9])

# fast demo
# (15, 4) takes 0.02 seconds
#P = primes[20:35]
#N = reduce(lambda x,y: x*y, primes[0:5])

# really fast demo, (10,4)
#P = primes[20:40]
#N = reduce(lambda x,y: x*y, primes[0:5])

# this is a bullshit parameter
# max size any of the additions can be
# could be a bit wrong in the mods
#ms = 20
#ms = 16
#ms = 6
#ms = 6

#print P
print "2^%d" % len(P)
print "finding", N

ret, phi = factor(N)
ret = sorted(list(set(ret)))
print ret
#exit(0)

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
#print rret
#print P

"""
mat = []
for x in rret:
  mat.append([y%x for y in P])
mat = np.asarray(mat).astype(np.int).T
print mat
"""

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
      #print x,"has generator",g
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

print "mod targets"
print re

# crappy optimization is crappy
import itertools
fpmat = []
ree = []

for i in range(len(re)):
  x = re[i]
  ff, _ = factor(x)
  # wrong
  # not wrong anymore
  #uff = sorted(list(set(ff)))
  uff = [list(g) for k, g in itertools.groupby(ff)]

  for f in uff:
    f = reduce(lambda x,y: x*y, f)
    fpmat.append(pmat[:, i] % f)
    ree.append(f)
  print uff
fpmat = np.asarray(fpmat).T


order = sorted(zip(ree, range(len(ree))))
print order

pmat = fpmat[:, map(lambda x: x[1], order)]
re = map(lambda x: x[0], order)

print pmat.shape
print pmat
print re

#exit(0)

# crappy optimization done

print "max possible sums"
maxsum = pmat.sum(axis=0)
print maxsum

ms = int(log(max(maxsum))/log(2) + 1)
print "bits needed is",ms

#exit(0)

"""
exit(0)



print "FUCK PRODUCT IS BIG NUMBER"
#print reduce(lambda x,y: x*y, map(int, pmat.sum(axis=0)))
print reduce(lambda x,y: x*y, map(int, re))


# note how all the mod targets must be === 0 mod 2, so the solution must also solve the subset xor problem

#from sieve import gje
#print gje(pmat%2)

test=[0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,1,0,0]
# [  4  12  36  70  84  96  90 154 196]

exit(0)
"""


# for 7 -- 1 -> 0, 3 -> 1, 2 -> 2, 6 -> 3

# for example, x*y === 1 mod 3, how do we know?
#exit(0)

print "WE ARE IMPORTING Z3", pmat.shape
from z3 import *
from time import time

start = time()
#s = Solver()
s = Goal()
#ONE = (1<<ms)-1
ONE = 1

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

  # i think this is as close to free as you can get
  #s.add(b == 0 or b == ONE)
  #for j in range(ms):
  #  s.add((b&1) == (b>>j))

  if ONE == 1:
    s.add(b & ((1<<ms)-2) == 0)
  else:
    s.add(Or(b == 0, b == ONE))


  if i == 0:
    bor = b
  else:
    #bor = Or(b, bor)
    #bor += b
    bor |= b
  bb.append(b)

# at least one must be True
#s.add(bor == True)
s.add(bor == ONE)

for j in range(pmat.shape[1]):
  mod = re[j]

  #if (mod%2) == 0:
  if mod == 2:
    # special case for mod 2 is xor
    # we might be able to do even better using linear algebra stuff
    ss = None
    for i in range(pmat.shape[0]):
      if (pmat[i,j]%2) == 1:
        if ss == None:
          ss = bb[i]
        else:
          ss ^= bb[i]
    s.add(ss == 0)

  if mod != 2:
    for i in range(pmat.shape[0]):
      #iv = IntVal(pmat[i,j])
      #iv = BitVecVal(pmat[i,j], ms)
      iv = pmat[i,j]
      if i == 0:
        #ss = (iv * bb[i]) % mod
        ss = iv * bb[i]
        #print ss
        #ss = If(bb[i], pmat[i,j], 0)
      else:
        #ss = (ss + iv * bb[i]) % mod
        ss += iv * bb[i]
        #ss += If(bb[i], pmat[i,j], 0)
    s.add(ss % mod == 0)
    #s.add(ss == 0)


from z3_out import z3_to_cnf, write_cnf
ncnf = z3_to_cnf(s)
write_cnf(ncnf, "tmp")

"""
print s.check()
#print s.model()
print time()-start, "seconds elapsed"

#exit(0)

ret = [0]*pmat.shape[0]
m = s.model()
for w in m:
  if m[w].as_long() == 1:
    ret[int(str(w).split("_")[1])] = 1

print ret
print np.dot(ret, pmat), re, np.dot(ret, pmat) % np.asarray(re)

# print solution
out = []
prod = 1
for x,y in zip(P, ret):
  if y:
    out.append(x)
    prod *= x
print prod, prod%N, out
"""


"""
print s.to_smt2()
"""


