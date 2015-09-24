import numpy as np
from factor import factor
from math import log
import gmpy
import os
import subprocess
from sieve import gjee

def print_solution(ret):
  # print solution
  out = []
  prod = 1
  for x,y in zip(P, ret):
    if y:
      out.append(x)
      prod *= x
  print N, prod, prod%N, out

primes = filter(gmpy.is_prime, range(10000))

# P and N
# (1248, 70) takes ???? seconds
#from findd import *
# max sum is 18 bits
# after reduction it's (1248, 169)
# each has max 256 possible

# (60, 14) takes ??? seconds
# reduce to (60, 27)
P = primes[20:120]
N = reduce(lambda x,y: x*y, primes[0:20])

# (40, 9) takes 11 seconds, 2 seconds with hacks
# reduce to (40, 15)
#P = primes[20:35]
#N = reduce(lambda x,y: x*y, primes[0:6])

# swag
#P = primes[20:30]
#N = reduce(lambda x,y: x*y, primes[0:4])

# fast demo
# (15, 4) takes 0.02 seconds
P = primes[20:45]
N = reduce(lambda x,y: x*y, primes[0:8])

# really fast demo, (10,4)
#P = primes[20:300]
#N = reduce(lambda x,y: x*y, primes[0:9])

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
print phi
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

#exit(0)

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


def cmpr(x, y):
  a = x[0]
  b = y[0]
  if a%2 == 0:
    a -= 100000
  if b%2 == 0:
    b -= 100000
  return cmp(a,b)

order = sorted(zip(ree, range(len(ree))), cmpr)
print order

pmat = fpmat[:, map(lambda x: x[1], order)]
re = map(lambda x: x[0], order)

print pmat.shape
print pmat
print re

# ITS LLL O'CLOCK


# build LLL matrix
lllme = np.hstack((np.identity(pmat.shape[0], dtype=np.int), pmat))
modme = np.identity(pmat.shape[1], dtype=np.int)
for i in range(len(re)):
  modme[i][i] = re[i]
botme = np.hstack((np.zeros(pmat.shape).T.astype(np.int), modme))
lllme = np.vstack((lllme, botme))

# HACKS TO BE BIG
NNN = 19
for i in range(pmat.shape[0], lllme.shape[0]):
  lllme[:, i] *= NNN

def run_lll(lllme, fn='LLL.sage'):
  print "calling sage", lllme.shape
  f = open("/tmp/lll_me_please", "w")
  f.write(str(lllme.tolist()))
  f.close()
  p = subprocess.Popen(['sage', fn], stdout=subprocess.PIPE)
  lllmats, _ = p.communicate()

  lllmat = []
  for ln in lllmats.split("\n")[:-1]:
    lllmat.append(map(int, ln.strip('[]').split()))
  lllmat = np.asarray(lllmat)
  print "sage done"

  return lllmat

lllmat = run_lll(lllme)

#print lllmat.shape

# print good
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=np.inf)
# hacks
#print lllme
#print "LLL"
#print lllmat


# find a all 1 vector from these
good_rows = filter(lambda i: all(lllmat[i, pmat.shape[0]:] == 0), range(lllmat.shape[0]))
lm = lllmat[good_rows, 0:pmat.shape[0]]

print "hamming weights", abs(lm).sum(axis=1)
print lm

AB = np.hstack((lm.T, np.identity(lm.shape[0], dtype=np.int)*1))
print AB
print AB.shape

def runlp(AB):
  f = open("ext/tmp.lp", "w")
  f.write("maximize\n")
  f.write("x1\n")
  f.write("subject to\n")

  for row in range(AB.shape[0]):
    s = []
    for i in range(AB.shape[1]):
      if AB[row,i] != 0:
        if AB[row,i] == 1:
          s.append("x" + str(i+1))
        elif AB[row,i] == -1:
          s.append("-x" + str(i+1))
        else:
          s.append(str(AB[row,i]) + "x" + str(i+1))
    f.write(' + '.join(s).replace("+ -", "- ")+" = 1\n")

  f.write("bounds\n")
  for i in range(AB.shape[1]/2, AB.shape[1]):
    f.write("x"+str(i+1)+">0\n")
    f.write("x"+str(i+1)+"<1\n")

  f.write("integer\n")
  for i in range(AB.shape[1]):
    f.write("x"+str(i+1)+"\n")

  f.write("end\n")
  f.close()

  os.system("glpsol --cpxlp ext/tmp.lp --output ext/tmp.out")

  dat = open("ext/tmp.out").read().split("Column name")[1].split("\n")[2:2+AB.shape[1]]

  sol = [0]*AB.shape[1]

  for c in dat:
    tmp = c.split()
    sol[int(tmp[1][1:])-1] = int(tmp[3])

  return sol

sol = runlp(AB)
print sol
print np.dot(AB, sol)

hf = AB.shape[1]/2
sol = np.dot(AB, sol[0:hf] + [0]*hf)
print_solution(sol)


exit(0)

"""
# Z3
print "It's Z3 time"
from z3 import *

s = Solver()
xx = []
ms = 8
for i in range(AB.shape[1]):
  x = Int('x_'+str(i))
  #x = BitVec('x_'+str(i), ms)
  xx.append(x)
  if i >= AB.shape[1]/2:
    s.add(x >= 0)
    s.add(x <= 1)
    #s.add(x & ((1<<ms)-2) == 0)
    pass

for j in range(AB.shape[0]):
  b = None
  for i in range(AB.shape[1]):
    if AB[j,i] != 0:
      if b == None:
        b = xx[i] * AB[j,i]
      else:
        b += xx[i] * AB[j,i]
  s.add(b == 1)

print s.check()
#print s.model()

ret = [0]*AB.shape[1]
m = s.model()
for w in m:
  ret[int(str(w).split("_")[1])] = m[w].as_long()

print ret
print np.dot(AB, ret)

# best quality
hf = AB.shape[1]/2
sol = np.dot(AB, ret[0:hf] + [0]*hf)

print_solution(sol)
exit(0)
"""


# part 2
lll = run_lll(AB, fn='matmat.sage')
print lll[(lll.shape[0]/2):, (lll.shape[0]/2):]
exit(0)


# LOL
import requests
data = {}
AB = np.hstack((AB, np.asarray([1]*AB.shape[0]).reshape(-1, 1)))
data['rows'] = str(AB.shape[0])
data['cols'] = str(AB.shape[1]-1)
data['matrix'] = " ".join(str(AB).replace("[", "").replace("]", "").replace("\n", " ").split(" "))
data['SUBMIT'] = 'GO'
print "wtf this shit uses import requests"
r = requests.post("http://www.numbertheory.org/php/axb.php", data=data)
open("ext/out.html", "w").write(r.text)

soll = map(int, r.text.split("Y = (")[1].split(")")[0].split(','))
print soll

# right half must be all 0 or 1
mul = soll[0:AB.shape[0]]
print mul

# final solution
rsol = np.dot(lm.T, mul)
print rsol

print_solution(rsol)


exit(0)


"""
lm[lm <= -1] *= 113
print lm
print "NEW"
lm2 = run_lll(lm)
print lm2
"""

#print gjee(lm)
#print np.asarray(sorted(lm.as_list()))

#exit(0)


# solve better than this
for row in lllmat:
  # has to be a solution
  if not all(row[pmat.shape[0]:] == 0):
    continue

  # row of zeros is useless
  sol = row[0:pmat.shape[0]]

  print sol
  if all(sol == 0):
    continue

  # fuck 2's
  if any(abs(sol) > 1):
    continue

  # mixed 1 and -1 is useless
  if any(sol == 1) and any(sol == -1):
    continue
    #pass

  print sol, np.dot(sol, pmat), np.dot(sol, pmat)%re

  #print np.dot(row, lllmat)
  print_solution(sol)

exit(0)


"""
# run LLL on matrix
from liblll import *
print lllme
lllmat = create_matrix(lllme)
lllred = lll_reduction(lllmat)
print islll(lllred)
print_mat(lllred)
"""


"""
# USELESS IDEAS START HERE

rep = map(lambda x: factor(x)[0][0], re)
repp = reduce(lambda x,y: x*y, list(set(rep)))
print rep, repp

pmat_id = np.hstack((pmat % rep, np.identity(pmat.shape[0], dtype=np.int)))
sol = gjee(pmat_id, rep + [repp]*pmat.shape[0], len(rep))
#print sol

nul = sol[pmat.shape[1]:, pmat.shape[1]:]
for r in nul:
  ss = np.dot(r, pmat) 
  #print r, ss, ss%re
  #print ss%re
  print r, ss%re

exit(0)

end = map(lambda x: x%2, re).index(1)

is_mod_2 = pmat[:,0:end]%2
is_mod_2_id = np.concatenate((is_mod_2, np.identity(is_mod_2.shape[0], dtype=np.int)), axis=1)
#print is_mod_2_id

aa = gjee(is_mod_2_id)
nul = aa[is_mod_2.shape[1]:, is_mod_2.shape[1]:].T
print nul
print nul.shape

print "searching 2 ^",(nul.shape[1] - (len(P)-odds))
"""

"""
# solve Ax = b
tes = np.asarray([0,1,0,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,1,1,1])
swag = gjee(np.hstack((nul, tes.reshape(-1, 1))))
swag = swag[:nul.shape[1], -1]
print swag
print np.dot(nul, swag)%2
print tes
"""


"""
print nul
print nul.shape

print tes
"""

"""
aa, bb = gje(is_mod_2)
nul = bb[list(aa.sum(axis=1)).index(0):].T
#print aa
print nul
tes = np.asarray(tes).reshape(len(tes), 1)
aug = np.concatenate((nul, tes), axis=1)
print aug
exit(0)

#print tes
#print nul.shape

#slv = gje(
#print slv

print np.dot(slv, nul)
print tes[:, 0]
"""

#exit(0)



# for each column we can find tiny disjoint sets that satisfy the condition, doesn't have to be maximal
# hmm, might not work
# for example this is a matrix A for the column
# then we find an x s.t. Ax = Bx = ... = Zx

# what i really want here is a way to factorize each condition into multiple mod 2 things


exit(0)

# crappy optimization done

"""
# this is wrong???
prod = 1
# deoptimize
ri = []
for x in re:
  ri.append(prod)
  prod *= x

# is subset sum mod bullshit
theset = list(np.dot(pmat, ri))
print theset, prod

exit(0)
"""


print "max possible sums"
maxsum = pmat.sum(axis=0)
print maxsum

ms = int(log(max(maxsum))/log(2) + 1)
print "bits needed is",ms



"""
exit(0)



print "FUCK PRODUCT IS BIG NUMBER"
#print reduce(lambda x,y: x*y, map(int, pmat.sum(axis=0)))
print reduce(lambda x,y: x*y, map(int, re))


# note how all the mod targets must be === 0 mod 2, so the solution must also solve the subset xor problem

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
s = Solver()
#s = Goal()
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


"""
from z3_out import z3_to_cnf, write_cnf
print "converting to cnf..."
ncnf = z3_to_cnf(s)
print "writing output"
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

print_solution(ret)

