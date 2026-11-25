# Choosing the discriminant with the CRT

Target n=1 mod A and -1 mod B, where gcd(A,B)=2, 5|A and 7|B. For odd n the symbol (5/n)=(n/5)=+1 and (-7/n)=(n/7)=-1, so the *actual* Selfridge choice is D=-7, not a Lucas test of our own choosing. P=1,Q=2. Restrict prime factors to p=3 mod 8 with p mod 7 in {3,5,6}, excluding divisors of AB. Now v2(ord_p(2))=1 by Euler's criterion. Also alpha^p=beta for the roots of x^2-x+2, alpha*beta=2; (alpha/beta)^((p+1)/2)=2^(-(p-1)/2)=-1 mod p. Thus v2(rank_p(U))=2 because v2(p+1)=2. A product of distinct primes meeting both CRT constraints passes the two strong tests. Nontrivial composite selection and independent direct verification are still required. This design allows 5 to divide p-1, unlike the previous all-inert D=5 construction. We recompute the exact LCMs of orders and ranks per output and check joint membership including odd cardinality and the prior D=5 symbol. Membership allows integer exponents, not just 0 and 1.

The bounded generator relaxes cheap small-prime powers of the historic template, lending the minus side 5^6. It visits divisors d of one odd template to generate p=2dh+1 or 4dh-1 with missing cofactor h at most 9, and rejects pairs with missing cofactor above 256 on either side. The new-candidate bound was 10^12; seeds from the existing certified catalogue may exceed it. The output is not claimed exhaustive outside these restrictions. Every retained prime and period is revalidated. On the full D=-7 catalogue PARI/GP independently certified all 2767 primes, base orders, Lucas ranks (companion matrix powering, including divisor-stripping checks), and strong signatures.

## Reducing the group

A coordinate for the first power of an odd prime q costs log2(q-1) ambient group bits, and each further power costs log2(q). These costs are shared: pay once if *any* selected candidate needs that level. Fix a price per bit; give each prime node profit 1 and add an implication arc to every level it requires. A source/sink min-cut chooses an exact maximum-weight closed set for this surrogate (edge weights rounded to 1/10000 unit); the 2-part is fixed. This is not an optimality certificate for the true subgroup, arbitrary templates, or the zero-one witness problem. Audit each resulting joint group exactly. Separately use greedy rare-level peeling for smaller groups at a prescribed margin. A `four-lists` margin leaves enough subset choices for four disjoint lists of the size called for by a uniform generalized-birthday cost model. No such astronomically large join was run. Full group generation of each support is not proof of short-subset equidistribution.

```sh
export OPENBLAS_NUM_THREADS=1
g++ -std=c++20 -O3 -march=native divisor_lift.cpp -lgmpxx -lgmp -o divisor_lift
python pool_lift.py runs/lift12 --upper 1000000000000 --hmax 9 --defect 256 --extra 71
python trim_pool.py runs/lift12/D-7-price0.0.txt runs/lift12/trim --D=-7
python block_profile.py runs/lift12/trim
python -m unittest test_pool_lift test_pool_audit
```

See `runs/lift12/report.json`, `runs/lift12/trim/trim.json`, and `runs/lift12/trim/blocks.json`. The files contain integer moduli, exact group sizes, or split metadata. `check_lift.gp` supplies the independent verifier for the period catalogue; input rows are in `runs/lift12/rows--7.json`.
