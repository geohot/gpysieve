from z3 import *
def z3_to_cnf(g):
  t = Then('simplify', 'bit-blast', 'tseitin-cnf')

  subgoal = t(g)
  assert len(subgoal) == 1
  sg = subgoal[0]
  print "got subgoal"

  def proc(aa):
      aa = str(aa)
      ret = []
      for a in aa.split(","):
          #print a
          vn = int(a.split("!")[1].strip(")")) + 1
          if "Not" in a:
              vn *= -1
          ret.append(vn)
      return ret

  cnf = map(proc, sg)
  print "parsed cnf"

  abs_flat_cnf = map(abs, [item for sublist in cnf for item in sublist])

  ll = list(set(abs_flat_cnf))
  ll_0 = zip(ll, range(1, len(ll)+1))
  ll_1 = zip(map(lambda x: x*-1, ll), map(lambda x: x*-1, range(1, len(ll)+1)))
  repl = dict(ll_0 + ll_1)
  varcount = len(ll)
  print "vars", varcount

  return [[repl[k] for k in c] for c in cnf]

import itertools
def write_cnf(ncnf, filename):
  varcount = max(map(abs, list(itertools.chain.from_iterable(ncnf))))
  f = open(filename, "w")
  f.write("p cnf %d %d\n" % (varcount, len(ncnf)))
  for c in ncnf:
      f.write(' '.join(map(str, c)) + " 0\n")
  f.close()

