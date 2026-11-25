# Start here: strong Baillie–PSW search

**Status: no counterexample.** Workdir `/home/light/fun/gpysieve/astra`. The target is one actual composite passing BOTH strong base-2 Miller–Rabin and strong Lucas–Selfridge at the *first eligible* discriminant. Ordinary Fermat/Fibonacci pseudoprimes, subgroup reachability, and a subset satisfying only a quotient do not count. Preserve the factors and verify any hit with the independent PARI `verify.gp`. The user prefers mathematical reductions to blind scaling. All latest work was local; no VM is running.

## Latest completed work (`runs/lucas-first/`)

- **Forward, listed orders, all Selfridge D:** the base-2 catalogue is `runs/scale-next/hints-big.jsonl` (SHA256 `c83aa8ceb173447103132ea4b9f8a662dcbd2be708520ff5f3620520491a81cd`). Previously `SCALE_NEXT.md` closed identity-only and exactly two cancelling nonidentity residues. The new `lucas_tail.py` run covers products above 2^64 with **three or more** nonidentity residues, at **any arity**, when the exact common base-2 order LCM m is a **listed** index. Disjoint factors, one use per catalogue entry. 4,181 groups; 7,456,413,972,495,536 raw subset placements before filters (not distinct tests or unique composites). The adaptive search used 1,330,804 half-table entries for early discriminants and 897,202 nonempty character-kernel masks in the tails. 23,141 tail products reached actual character testing; no hit. See `all-D.json`. The seven `D*.json` are earlier fixed-D pilots, not additional disjoint scope.
- **No LCM bound for triples:** `triple_orders.cpp` + `triple_orders.py` exhausted triples of the 12,172 base-2 catalogue entries. For anchor p with order a, the other two factors must multiply to 1 mod a: hash inverse residues, then check the orders at both other factors. Of 31,937,609,854 same-signature triples, exactly 210,644 passed strong MR, including 202,034 above 2^64 and 82,910 at unlisted LCMs. All failed the actual BPSW test (`triples.json`). This is not a result about triples outside the input catalogue.
- **Forward, unlisted orders, D=5:** `forward5-50k.json` covers three-or-more cancelling residues at exact order LCM <=50,000. It needed one further branch rule for oversized joins: an absent prime-power level of m requires at least one user; partition by its first selected user. If no level is absent, include/exclude an optional factor. The only canonical product surviving the lower-bound sieve was a 328-bit triple with orders 169, 1001, 143 (LCM 13013). This was a weak rank-divisor match, **not** a near counterexample: GP confirmed MR pass, actual D=5, strong Lucas fail, and nonzero U_(n+1) at all three factors. Full factors in JSON; independent residues in `forward5-gp.txt`. This overlaps the triple run.
- **Reverse search:** `rank_banks.py` unions the recorded Lucas components separately per D, computes each input's exact Lucas rank by divisor stripping, and fixes the rank LCM rather than the binary order LCM. Covered any number of selected factors at rank LCM <=1,000,000 for actual D in {5,-7,-11,13,-15,17,-19}, over their respective catalogues. 1,168,393 rank templates and 10,142,004 half-table entries; no hit (`reverse*-1M.json`). The earlier `reverse5.json` pilot is contained in this result.

## What makes the reduction work

If n=1 mod m and n=-1 mod rank, their moduli have gcd dividing 2. `rank_divisors.cpp` certifies a scalar power of the Lucas companion matrix locally, then strips certified prime divisors to get a proven divisor and a proven multiple of the rank; the 2-part is exact. In reverse mode it uses modular powers of 2. An uncertified large residual is **not** assumed to be part of the rank. `lucas_first.py` removes incompatible entries, groups by exact 2-part, forces an anchor, and joins half-products on BOTH the CRT residue and earlier Selfridge character bits. Direct local Lucas powers settle ambiguous upper/lower bounds. Matched products are checked with the actual strong tests. `lucas_tail.py` uses a separate observation: reaching a late D requires all earlier characters to be +1. Remove proper-gcd entries, compute the GF(2) nullspace of sign columns, and enumerate only this nullspace once small. The tail is not cut off at an assumed maximum D. See `test_lucas_first.py` for controls: 52633 passes MR but fails Lucas; 5777 passes strong Lucas but fails MR. Deliberately weakened bounds send these to the opposite test to check the joins do not just reject everything.

## Resume plan

1. **Don't rerun the listed-order search or same-catalogue triples as new campaigns.** The clearest open forward region is 4+ factors with **unlisted** order LCM, especially actual D other than 5. `bounded_templates` generates only LCMs supported by available entries. Start with an unlisted-only pilot at e.g. D=-7 and m<=50,000, record width/half entries and actual hits before growing it; extending `lucas_tail` to these templates could remove a D cutoff. For oversized groups use recursive prime-power support branching, not a giant flat table. These are proposed steps, not completed runs.
2. For a materially different search, **change the factor supply** rather than only increasing the LCM ceiling: merge new verified Lucas sequence factors, or exploit the dense D=-7 pools from `DISCRIMINANT_LIFT.md` and `KERNEL_LIFTS.md`. The current order/rank catalogues are finite, not all primes with those orders. In the dense pools the kernel remains very large; coverage certificates are not full-CRT solutions.
3. Record exact candidate domain, discriminants, and whether counts are distinct or overlapping. For a hit: save complete factorization immediately, check product and nonsquareness, run actual strong tests and `gp_check`/`verify.gp`; report disagreement rather than claiming discovery. There is no justified probability estimate of success.

## Reproduce / sanity check

```sh
cd /home/light/fun/gpysieve/astra
export OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/tmp/astra-pari/usr/lib GP=/tmp/astra-pari/usr/bin/gp
g++ -O3 -std=c++17 rank_divisors.cpp -lgmpxx -lgmp -o /tmp/astra-rank
g++ -O3 -std=c++17 triple_orders.cpp -lgmpxx -lgmp -o /tmp/astra-triples
src=runs/scale-next/hints-big.jsonl; out=runs/lucas-first
python lucas_tail.py "$src" "$out/all-D.json" --gp "$GP"
python triple_orders.py "$src" "$out/triples.json" --gp "$GP"
python lucas_first.py "$src" "$out/forward5-50k.json" --D 5 --bound 50000 --maxhalf 16 --gp "$GP"
for d in 5 -7 -11 13 -15 17 -19; do
  python rank_banks.py "$out/reverse$d-1M.json" runs/bigcpu-*/remote/iris-{index,refine}-*/components.jsonl --D="$d" --bound 1000000 --gp "$GP"
done
python -m unittest -q
```

Release check: **40 tests passed with none skipped** from a fresh export of the Git index using `make -j2 test GP=/tmp/astra-pari/usr/bin/gp` and the local PARI library path. A plain Python-only run may skip optional integration tests. The adaptive listed-order and reverse JSON logs predate the final early-LCM pruning / oversized-join refactor; incidental work counters can differ on rerun. Regression tests cover forced branching and both join orientations. Earlier campaign context: `SEARCH_NOTES.md`, `SHAPE_SIEVE.md`, `SCALE_NEXT.md`, `QUOTIENT_COVER.md`, `KERNEL_LIFTS.md`.
