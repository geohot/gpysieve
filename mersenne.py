p = 4253
#p = 110503
#p = 1398269

print "testing",p
s = 4
m = pow(2, p)-1
print "computed m"

print "prob tests"
print pow(2, m, m) == 2
print pow(3, m, m) == 3
print pow(4, m, m) == 4

print "lucas"

for i in xrange(p-2):
  s = (s*s - 2) % m

if s == 0:
  print p,"is prime"
else:
  print p,"is composite"

