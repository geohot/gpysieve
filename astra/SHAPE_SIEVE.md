# Eliminating a translation or scale (strong BPSW)

## Why switch away from dense conditional branching?

For each odd prime-power level L of the pool's order/rank LCM, let s be the number of factors requiring it. At uniform binary selection its absence probability is exactly 2^-s, or binom(n-s,k)/binom(n,k) at fixed weight k. A union bound controls the fraction missing *any* level, without an independence assumption. Weight each absence probability by the log of the unit-group cost for that level to get the exact expectation of saved odd ambient bits (not generated image bits). In the 1,495-prime core the union bound is 0.000218703 and expected savings are 0.000665 bits or less. At weight 3 the expected savings instead exceed 1297 ambient bits. These are statements about selections, not the conditional distribution of passing selections. `activation_audit.py` and `runs/shape/activation.json` provide both cores and several fixed weights.

## Translation-free sparse test

Let n be the product of distinct odd primes x+h, h in H, min H=0. At an anchor q=x+s define W_s(X)=product over h!=s of (X+h-s). If n passes strong MR, a=ord_q(2) divides both q-1 and n-1, hence divides R=W_s(1)-1. If it passes strong Lucas-Selfridge at actual D, b=rank_q(U) divides q-epsilon and n+1, epsilon=(D/q). Here q does not divide 2DQ. Thus b divides S=W_s(-1)-1 for epsilon=-1, or T=W_s(1)+1 for epsilon=1. These exponents depend on offsets, **not on x**. Therefore q divides one of gcd(2^|R|-1,U_|S|), gcd(2^|R|-1,U_|T|). Negative Lucas indices are harmless since the roots are units mod q. Use the LCM to include both cases. Skip uninformative R=0. Choose an affordable anchor, certify all prime divisors with PARI, translate back to x and check the actual pattern. No heuristic primality or distribution assumption; a size limit on exponent or factorization means *unresolved*, never excluded. For every row of the final compact suite at least one anchor succeeded, or a cheaper obstruction applied.

Cheaper obstructions: ord_q(2) has 2-part respectively 2,4,1 for q=3,5,7 mod 8 (Euler's criterion). A strong MR product must have a common 2-part, so any offset set occupying three or more residue classes mod 8 is impossible. If H covers all classes mod a small prime ell, one of the factors must equal ell: check those few actual shifts instead of silently discarding the exceptions. `constellations.py --suite` gives zero unresolved and zero hits for 2/3 factors up to diameter 360, 4 up to 80, 5 up to 48 and every 6..21 up to 40. A diameter-40 interval has at most 21 odd slots. Hence this excludes **all odd squarefree composites of prime-factor diameter <=40** for actual D in {5,-7,-11,13,-15,17,-19,21}, without an upper bound on n. The narrower D-independent triple result below is separate. A different actual D, larger diameter, or repeated factor is not excluded by this statement. `compact.json` supersedes the earlier smaller pilots (`tight.json` still records their exponent cutoff).

## Scale-free sparse test, unrestricted Selfridge D

Write p_i=c_i*K-1 for three increasing primitive positive coefficients, K=gcd(p_i+1) even. Set K=2s. At anchor c with other coefficients a,b, ord_{p_i}(2) divides 2*gcd(cs-1,2ab*s-a-b), because gcd(s,cs-1)=1. It therefore divides **twice the part of |2ab-c(a+b)| coprime to c**. Nonzero bounds force the anchor to divide a *known Mersenne integer*, with no cutoff on K. Any fully factored anchor suffices. Even incomplete anchors help: test all K obtained from known prime factors, then ask whether three unknown exact orders can coexist. Order m implies c_i K=2 mod m, n=1 mod m, and compatible residue class mod 8 by Euler's criterion. All three orders must have equal 2-parts. `scale_refine.py` checks congruences jointly; for order m the still-unknown prime must divide Phi_m(2), and primes dividing m are not primitive. Known prime divisors are PARI-certified. A compatible unknown branch is recorded, not counted as an exclusion. In the box max c<=64: 25,704/35,203 primitive shapes excluded, including *all* max c<=18. No factor-size bound and no restriction on actual D. Coarse and refinement reports keep survivors separate; an endpoint need not be certified prime for a rejection on a superset to be valid.

Other unlimited-size corollaries:
- AP triples p-d,p,p+d: a=ord_p(2) divides d^2. With g=gcd(d,p+1), h=d/g, gcd(a,g^2) divides gcd(p-1,(p+1)^2), which divides 4. Hence a|4h^2. `ap_normalized.py` verifies factorizations by division, certifies centers and p+1 factorizations, and checks every g|p+1. For every h<=12 only 91 distinct prime centers in total; 722 possible (center,gap) pairs after necessary tests. Only 29341 passes MR; it fails strong Lucas. h=13..22 are **not** asserted complete.
- `shape_binary.py` applies the anchor-order/CRT sieve to ordinary gap triples without preselecting D. No solution at **any actual D for all triple diameters <=50**; 1,544/1,770 individual gap shapes through diameter 120 covered. It tests some composite-endpoint products too, a safe superset. PARI's `verify.gp` independently rejected all six distinct strong-MR survivors of the recorded binary/scale/AP screens (`MR-control-gp.txt`).

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/tmp/astra-pari/usr/lib
export GP=/tmp/astra-pari/usr/bin/gp
d=runs/shape
src=runs/bigcpu-501c65a3a23b73d3/remote/iris-base2-2048/components.jsonl
python activation_audit.py "$d/activation.json"
python ap_normalized.py "$src" "$d/ap.json" --gp "$GP"
python scale_shapes.py "$src" "$d/scales.json" --box 64 --gp "$GP"
python scale_refine.py "$src" "$d/scales.json" "$d/refined.json" --gp "$GP"
python shape_binary.py "$src" "$d/anyD.json" --gp "$GP"
python constellations.py "$d/compact.json" --suite --gp "$GP"
python -m unittest test_shapes
```
