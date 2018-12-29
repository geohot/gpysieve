#!/usr/bin/env python3

import gmpy

# find base-2 psuedoprimes

for n in range(3, 1<<24, 2):

  # 3
  if n%5 not in [2,3]:
    continue

  # 1
  if pow(2,n,n) != 2:
    continue

  # 4
  if gmpy.is_prime(n):
    continue

  # 2
  fres = gmpy.fib(n+1)%n
  if fres != 0:
    continue

  print("OMG OMG OMG", n)
  #exit(0)

