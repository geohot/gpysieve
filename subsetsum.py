from z3 import *
from math import log

def solve(arr, prod):
  ms = int(log(sum(arr))/log(2) + 1)

  s = Solver()
  bb = []
  for i in range(len(arr)):
    b = BitVec('b_'+str(i), ms)
    s.add(b & ((1<<ms)-2) == 0)
    if i == 0:
      bor = b * arr[i]
    else:
      bor += b * arr[i]
    bb.append(b)
  s.add(bor > 0)
  s.add(bor % prod == 0)
  print s.check()
  print s.model()

if __name__ == "__main__":
  #arr = [42, 32, 39, 35, 14, 7, 46, 21]
  #prod = 48
  #print solve(arr, prod)
  print solve([1,2,3,4], 7)

