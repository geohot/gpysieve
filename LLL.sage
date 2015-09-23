import sys

M = Matrix(ZZ, eval(sys.argv[1]))
result = M.LLL()
print result.str()

