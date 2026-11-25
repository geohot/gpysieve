# Character and semiprime campaigns

Target: base-2 strong MR and the strong Lucas–Selfridge test, n >= 2^64. The global counterexample problem is open. Logs for the dedicated Iris run are mirrored under `runs/bigcpu-501c65a3a23b73d3/remote/`. A partial factorization, timed-out solver, or interrupted shard is not an exclusion proof.

## Conditional character solver

Write a candidate as an anchor times a subset of distinct core primes, with Boolean selectors x. For prime q let a be the order of 2 and b the Lucas rank for P=1,Q=(1-D)/4. For each selected q we require n=1 mod a and n=-1 mod b. Equal 2-parts of all a's and of all b's synchronize the strong-test branches. Within a core all factors are units modulo each other's periods. Every odd prime divisor h of a or b gives a quadratic character equation: the xor of selected Legendre-sign bits (including the anchor sign) must equal the sign of +1 or -1. Two analogous real characters are available modulo 4 and 8. Crucially each equation is activated **only if q is selected**. Apply unconditional Jacobi sign equations for the first Selfridge discriminant. Branch on selection and maintain an exact affine GF(2) basis. Enumerate its remaining solutions at low dimension, then check each exact modular condition. `chars.cpp` handles both split and inert primes by testing n/q against the CRT target multiplied by q^-1, not always 1. A full candidate is saved and verified in SymPy and `verify.gp`.

Example comparison on the same previously searched 45-factor core, anchor 49477: native residue join receipt 88.047 s, character replay ~0.88 s under contemporaneous VM load; 859 full modular checks. This is a selected dense example, not an average speedup. `test_chars.py` includes positive planted mixed-residue controls checked by brute force.

## Semiprime reduction

For squarefree n=pq passing the tests, (D/n)=-1 implies exactly one split prime p and one inert prime q. Exclude primes dividing 2QD as in the actual test. Put a=ord_p(2), b=rank_p(U), k=p-1, t=k/a. Both a,b divide k, and gcd(a,b) divides gcd(n-1,n+1)=2. Hence b divides 2t. Also a divides q-1 (reduce n-1 modulo a). Rank_q divides q+1 by inertness, and divides n+1 by the Lucas test, hence divides p-1. Its gcd with a divides gcd(q-1,q+1)=2. Thus **rank_q divides 2t as well**. In particular pq divides U_(2t). This is a sufficient search space, not a claim that arbitrary factors of that Lucas number pass.

1. `split_scan.cpp` and `split_segment.cpp` sieve the split primes p with gcd(a,b)|2. After finding a, first test U_gcd(k,2t) modulo p rather than computing full Lucas ranks for every prime. The segmented version uses quadratic reciprocity and reciprocal modular multiplication (uint32 input range). Its output was differentially checked against the first sieve over (21,10^7]. The Python exact-period implementation also matched the first sieve through 10^5.
2. For a surviving split p, the inert partner has ord_q(2) and rank_q dividing k. Remove from its allowed order exponent A every odd prime power of k with prime dividing b; similarly remove from the allowed rank exponent B those with prime dividing a. Use 2-adic parts matching a and b respectively. Necessarily q divides gcd(2^A-1,U_B). `compact.gp` computes this gcd by reducing at the smaller operand. For p=249595441,D=13 the exponents shrink from k=249595440 to A=37010,B=1686. The possible-partner cutoff is q >= ceil(2^64/p); there is no upper cutoff. `audit_semi.py` independently rechecks orders, ranks, and gcd results in Python integers. `runs/u32-audit.jsonl` contains all 38 (D,p) records, with no admissible large prime divisor. The original full-exponent GP pass produced a stack-growth warning on that p; it was logged as incomplete and successfully replayed, not counted as an exclusion until then.
3. For bounded t **without a size bound on p or q**, factor U_(2t) via its cyclotomic components. `index_scan.py` certifies prime divisors in GP, computes actual ranks, groups by 2-part, and tests squarefree products at the least even multiple of their ranks. All pairs must be included for a semiprime exclusion. Higher arities tested by this program do not constitute an exhaustive BPSW search. `incomplete-indices.json` lists every index with unresolved cofactors; require completion of all product batches as well. `refine.py` targets missing factorizations. Multiple runs can overlap.

All eight discriminants D=5,-7,-11,13,-15,17,-19,21 were tested for the split-prime bound p<2^32. D=9 cannot have Jacobi symbol -1. The results say nothing about other Selfridge discriminants or arbitrary composites with more factors.

## Complementary construction

`mersenne.py` factors cyclotomic components of 2^k-1 and joins subsets all having the same 2-part of their orders. Require n=1 mod k and actual order LCM=k. These constructed composites pass base-2 MR by construction; check actual Selfridge selection, Lucas residues and any hit directly. `test_special.py` checks a positive MR control (2^67-1). Search windows, incomplete factors and results are retained in `runs/iris-base2-*` on the VM. Only the dual-verifier path reports a hit.

## Reproduction

From `astra/` with Python/SymPy, C++20 and PARI/GP:

```sh
g++ -std=c++20 -O3 -march=native chars.cpp -o chars
g++ -std=c++20 -O3 -march=native join.cpp -o join
g++ -O3 -march=native -fopenmp split_segment.cpp -o split_segment
ASTRA_GP=$(command -v gp) python3 -m unittest -v
python3 mixed.py mixed.jsonl --workers 4 --width 46 --base 20000 --hi 150000
OMP_NUM_THREADS=4 ./split_segment 300000000 4294967295 >split-u32.txt
python3 semi.py split-u32.txt semi-u32.jsonl --workers 4
python3 index_scan.py index-128 --T 128 --workers 4
```

Use `semi_waves.sh` for the lower split-prime intervals. Preserve sources and parameters with receipts; do not merge duplicate confirmations of the same split factor into a larger coverage count. Jobs reported above ran on the Iris VM, not on the controller.
