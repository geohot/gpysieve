import sys

M = Matrix(ZZ, eval(open("/tmp/lll_me_please").read()))
result = M.LLL()
print result.str()

