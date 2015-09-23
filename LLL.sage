import sys

M = Matrix(ZZ, eval(open("/tmp/lll_me_please").read()))
sys.stderr.write("LLL running\n")
result = M.LLL()
sys.stderr.write("LLL done\n")

print result.str()

