# Scale and exact-order follow-up

No BPSW counterexample. Two separate results: unlimited-scale exclusions for specified coefficient shapes, and a direct candidate search using a finite factor catalogue. Nothing here gives a probability of finding a witness.

## Unresolved scale branches

Use p_i=c_i K-1, sorted primitive c_i<=64 (all scope restrictions from `SHAPE_SIEVE.md` apply). `scale_progressions.py` intersects the exact-order congruences on K before testing factors. The 9,499 previously remaining shapes gave 23,963 order-labelled progressions. `scale_frontier.cpp` visited 1,256,940,869 progression entries through K=2^44, including duplicates between order branches. Only three passed the first unresolved order after trial division; none passed all three. Combined with the original known-factor audit this settles the **entire coefficient box for K<=2^44**, regardless of D. The original MR survivors at K=2 are recorded; a zero in `bounded44.json` refers only to the unresolved-order frontier.

Cunningham `pmain1126.txt` was downloaded from https://homes.cerias.purdue.edu/~ssw/cun/ . All table numbers are *hints*. `cunningham_merge.py` checks exact divisibility in Phi_m(2), separates the two algebraic factors for m=4 mod 8, and recognizes large residual probable primes. Downstream scale exclusions certify the anchor divisors they use with GP isprime. `merged.jsonl` used a 420-bit cap on additions. `hints-big.jsonl` removes that cap up to 4096 bits; it is not in itself a prime certificate. Updated, any-D, arbitrary-K completion: 32,659 / 35,203 coefficient shapes including every max c<=30; 2,544 shapes remain.

For a split anchor p=cK-1, the Lucas rank b divides p-1 and n+1. Multiply the latter congruence by c^2 and substitute cK=2 mod b and p=1 mod b: b divides 2*(2ab-c*(a+b)+c^2), with a,b here the *other coefficients*. Intersect the corresponding Lucas term with the Mersenne order bound by gcd, and enumerate only certified prime factors. Every split anchor in the remaining shapes was resolved for each of the tested D's. If all anchors are inert then (alpha/beta)^((p+1)/2)=(Q/p). Thus a negative symbol forces v2(rank)=v2(cK); a positive symbol forces strictly less. For equal 2-parts of the three ranks the negatives are either absent or occur at exactly the coefficients of minimal v2(c). `scale_lucas.py` checks this simultaneously with all earlier Selfridge symbols and the exact order progression using CRT and binary characters. The local exclusions p equal to a small character prime only apply beyond the already covered K range. D=21 is vacuous: if 5,-7,-15 were all skipped with symbol +1, their product gives (21/n)=+1 as well. Don't count it as evidence.

At D=17 no shapes remain; at D=5 there were 65 order branches in 26 shapes after the 420-bit-cap pass. `scale_last.py` targeted those and checked further anchor factors from the uncapped hints / cached https://factordb.com/api responses. Exact multiplication and GP isprime were required; 202 distinct anchor primes certified, up to 1471 bits. Only five order branches remain in shapes (3,53,61) and (11,53,61); these two and the later-D cases are **not** excluded at arbitrary scale.

## Searching actual strong MR candidates

For each catalogue element x check its exact order j by modular powering, and for even j check 2^(j/2)=-1 mod x. Verify the catalogue elements are globally pairwise coprime by divide-and-conquer product gcds. No unproven claim that all the inputs are prime is necessary for this search. For a subset with a common v2(j), if its product n is 1 mod the LCM m of the orders, n passes strong MR by construction. Nontrivial subsets have distinct composite products. Two disjoint searches:

* `order_cliques.py`: search **all** subsets of size >=2 such that every selected element equals 1 mod every selected order. The pairwise-compatibility graph is decomposed into disjoint boxes with mandatory and optional factors. Each box has an all-optional clique. No restriction that m be a listed table index. 5,119,163 products; 5,108,378 exceed 2^64.
* `order_pairs.py`: at an index m in the catalogue, require exactly two nonidentity residues that cancel, with optional identity residues. Require coverage of all prime powers in m so the actual order LCM is exactly m. Unique m and unique exceptional pair prevent double counting; none of these products is in the first search. 38,125,368 products. Not exhaustive over three or more cancelling residues.

For the 43,244,531 products these procedures distinguish the actual Selfridge character (including earlier zero-symbol cases), rather than fitting a convenient D after selection. At invertible Q the companion matrix's scalar powers form a subgroup. A direct U_t=0 check therefore certifies a rank multiple t even for composite moduli; factor stripping proves exact ranks at small inputs. Reject if U_(n+1) != 0 at any selected factor, since that excludes strong Lucas. If no certified short period is available, use the actual exponent. Any unresolved character or fully surviving candidate goes to a direct strong Lucas test. No hits. `verify.gp` independently confirmed [MR pass, strong Lucas fail] for 41 controls from each search. Count is distinct **within these two runs combined**, not deduplicated against all earlier campaigns.

```sh
export OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/tmp/astra-pari/usr/lib GP=/tmp/astra-pari/usr/bin/gp
d=runs/scale-next; src=runs/bigcpu-501c65a3a23b73d3/remote/iris-base2-2048/components.jsonl
python scale_progressions.py "$src" runs/shape/refined.json "$d/orders.txt"
g++ -std=c++17 -O3 scale_frontier.cpp -o /tmp/astra-scale
/tmp/astra-scale $((1<<44)) < "$d/orders.txt" > "$d/pass44.txt" 2> "$d/bounded44.json"
python cunningham_merge.py "$src" "$d/cunningham/pmain1126.txt" "$d/merged.jsonl" > "$d/merge-report.json"
python cunningham_merge.py "$d/merged.jsonl" "$d/cunningham/pmain1126.txt" "$d/hints-big.jsonl" --bits 4096 > "$d/merge-big-report.json"
python scale_shapes.py "$d/merged.jsonl" "$d/coarse.json" --box 64 --gp "$GP"
python scale_refine.py "$d/merged.jsonl" "$d/coarse.json" "$d/refined.json" --gp "$GP"
python scale_progressions.py "$d/merged.jsonl" "$d/refined.json" "$d/merged-orders.txt"
python scale_lucas.py "$d/merged-orders.txt" "$d/merged-lucas.json" --gp "$GP"
python scale_last.py "$d" --gp "$GP"  # cached data; no network required
python order_cliques.py "$d/hints-big.jsonl" "$d/cliques.json" --gp "$GP"
python order_pairs.py "$d/hints-big.jsonl" "$d/pairs.json" --gp "$GP"
python check_order_sample.py "$d/hints-big.jsonl" "$d/sample.json" --gp "$GP"
python check_order_sample.py "$d/hints-big.jsonl" "$d/sample-pairs.json" --pairs --gp "$GP"
python -m unittest test_scale_next
```
