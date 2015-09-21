from z3 import *

s = Solver()

p = Int('p')

s.add(p > 5)

m = 2**p

s.add(2**m % m == 2)
s.add(3**m % m == 3)

print s.check()
m = s.model()




