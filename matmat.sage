import sys

A = Matrix(ZZ, eval(open("/tmp/lll_me_please").read()))
sys.stderr.write("smith starting\n")
B,U,V = A.smith_form()
sys.stderr.write("smith done\n")
#sys.stderr.write(B.str() + "\n")
#sys.stderr.write(U.str() + "\n")

#print V[V.nrows()/2:, V.nrows()/2:].T.LLL().T.str()
print V[V.nrows()/2:, V.nrows()/2:].str()

