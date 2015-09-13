import random
import gmpy
import itertools
import numpy as np
from fractions import gcd
import sys

# hella bootleg
def gje(a):
  ova = 0

  #track = map(lambda x: [x], range(a.shape[0]))
  track = np.identity(a.shape[0], dtype=np.int)

  # loop through the columns
  for pivot in range(a.shape[1]):
    #sys.stdout.write(".")
    #sys.stdout.flush()
    #print pivot, ova
    found = None

    # find a row i with the pivot p after and including ova
    for i in range(ova, a.shape[0]):
      # found it
      if a[i, pivot] == 1:
        if found is None:
          # found it, switch it with the top row
          if i != ova:
            #print a
            #print "  switch", ova, i
            a[[ova, i],:] = a[[i, ova],:]
            track[[ova, i],:] = track[[i, ova],:]
            #print a
          found = ova
          ova = ova + 1
        else:
          # otherwise xor it with the found row
          a[i] ^= a[found]
          track[i] += track[found]

  return a, track%2

def factorize(zn, P):
  ret = []
  while zn != 1:
    good = False
    for p in P:
      if zn%p == 0:
        zn /= p
        ret.append(p)
        good = True
        break
    if not good:
      break

  #print z, zn
  if zn == 1:
    return ret
  else:
    return None


def ton(z, P):
  ret = [0]*len(P)
  for zz in z:
    ret[P.index(zz)] += 1
  return ret

def rsieve(n):
  # ONLY POSITIVE THINING
  # smaller B means less needed
  # bigger B means faster to find

  #B = 347
  B = 700
  #B = 40
  #B = 80
  #B = 100
  #B = 10000
  #B = 1500
  #B = 1000
  #B = 100000
  #B = 500
  P = filter(gmpy.is_prime, range(2, B+1))
  print B, "has", len(P)

  # find valid z
  # not sure if there is a faster way
  pfactor_count = 1
  zs = []

  # it's generally sufficient that the number of relations be a few more than the size of P

  # len(P)*2+1 is guaranteed to be enough

  target = len(P)*2+1
  #target = len(P)+9


  print "finding relations started..."
  while len(zs) < target:
    #print pfactor_count, len(zs)
    for f in itertools.combinations_with_replacement(P, pfactor_count):
      z = reduce(lambda x,y: x*y, f)
      # z is the candidate
      #print f, z, z+n
      ff = factorize(z+n, P)

      if ff != None:
        zs.append((ton(f, P), ton(ff, P)))
        sys.stdout.write("finding relations %6d/%6d\r" % (len(zs), target))
        sys.stdout.flush()

      # duplicate
      if len(zs) >= target:
        break

    pfactor_count += 1
  print "found"


  # got len(P)+3 z's
  #print zs

  # we need even exponents
  # each pair is a valid congruence
  arr = np.asarray(zs)
  arr = arr.reshape(arr.shape[0], -1)
  #print arr

  a = (arr%2)
  #a = np.hstack((a, np.zeros((a.shape[0], 1), dtype=np.int)))

  # find a vector in the nullity of arr
  # we want to figure out what xors together to make 0
  print a.shape, len(P)
  print "running gauss jordan"
  a,track = gje(a)
  print "printing factors hopefully"
  #print a
  #print track


  print a
  filt = filter(lambda (x,y): x == 0, zip(a.sum(axis=1), track))
  #print filt


  def pp(x):
    ret = 1
    for i in range(len(x)):
      ret *= pow(int(P[i]), int(x[i]/2))
    return ret

  # but with luck we will get a nontrivial pair of factors of n, and the algorithm will terminate

  # WTF THIS ALGORITHM NEEDS LUCK
  # the y form a basis we could use better TODO: use better luck here
  for x,y in filt:
    s1 = np.dot(y, arr)[0:len(P)]
    s2 = np.dot(y, arr)[len(P):]

    a,b = pp(s1), pp(s2)

    print n, gcd(b-a, n), gcd(b+a, n)

  

BITS = 24

if __name__ == "__main__":
  p = gmpy.next_prime(random.randint(0, 1 << BITS))
  q = gmpy.next_prime(random.randint(0, 1 << BITS))
  print p,q
  n = p*q
  print "factoring",n

  rsieve(n)

