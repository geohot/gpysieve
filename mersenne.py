p = 4253
#p = 110503
#p = 1398269

print "testing",p
s = 4
m = pow(2, p)-1
print "computed m"

for i in xrange(p-2):
  s = (s*s - 2) % m

if s == 0:
  print p,"is prime"
else:
  print p,"is composite"

