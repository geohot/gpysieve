import random
import gmpy
import itertools
import numpy as np
from fractions import gcd

# hella bootleg
def gje(a):
  ova = 0

  track = map(lambda x: [x], range(a.shape[0]))

  # loop through the columns
  for pivot in range(a.shape[1]):
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

            t = track[ova]
            track[ova] = track[i]
            track[i] = t
            #print a
          found = ova
          ova = ova + 1
        else:
          # otherwise xor it with the found row
          a[i] ^= a[found]
          track[i] += track[found]

  # cancel pairs in track

  ret = []
  for x in track:
    rret = np.zeros(a.shape[0], dtype=np.int)
    #print x
    for xx in x:
      rret[xx] += 1
    ret.append(rret%2)

  return a, ret

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
  B = 7
  P = filter(gmpy.is_prime, range(2, B+1))
  #print P

  # find valid z
  # not sure if there is a faster way
  pfactor_count = 1
  zs = []

  # it's generally sufficient that the number of relations be a few more than the size of P

  # len(P)*2+1 is guaranteed to be enough
  while len(zs) < len(P)+3:
    print pfactor_count, len(zs)
    for f in itertools.combinations_with_replacement(P, pfactor_count):
      z = reduce(lambda x,y: x*y, f)
      # z is the candidate
      #print f, z, z+n
      ff = factorize(z+n, P)

      if ff != None:
        zs.append((ton(f, P), ton(ff, P)))
    pfactor_count += 1


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
  #print a
  a,track = gje(a)
  #print a
  #print track

  s1 = np.dot(track[-1], arr)[0:len(P)]
  s2 = np.dot(track[-1], arr)[len(P):]

  def pp(x):
    ret = 1
    for i in range(len(x)):
      ret *= pow(P[i], x[i]/2)
    return ret


  a,b = pp(s1), pp(s2)

  print n, gcd(b-a, n), gcd(b+a, n)

  

BITS = 6

if __name__ == "__main__":
  p = gmpy.next_prime(random.randint(0, 1 << BITS))
  q = gmpy.next_prime(random.randint(0, 1 << BITS))
  print p,q
  n = p*q
  print "factoring",n

  rsieve(n)
  """
  rsieve(187)
  rsieve(273)    # 13*21
  rsieve(19*21)    # 17*21
  """


