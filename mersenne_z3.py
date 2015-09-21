from z3 import *

s = Solver()

p = Int('p')
s.add(p > 5)

m = 2**p
for a in [2,3]:
  s.add(((a**m) % m) == a)

print s.check()
m = s.model()




