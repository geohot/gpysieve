# Constructive quotient coverage, D=-7 four-block pool

Use the same 2,327 primes and four disjoint supports as `block_profile.py` (shuffle seed 1). Let m=lcm(A,B), H the previously audited joint image. The extra character tags are redundant on this pool: odd cardinality is visible mod 4 and the earlier Selfridge symbol is visible mod 5. For units mod m form a signature from the sign mod 4 and the Legendre symbols at every odd q|m. This is a binary linear image V; the primes span dimension 192 in 193 ambient sign coordinates. Append two odd coordinates: the order-729 quotient of units mod 2187 and the order-7 quotient of units mod 2401. Concretely at modulus k of totient f and desired order d take x -> x^(f/d) mod k. The odd target group has order h=5103. Q=V × C729 × C7 has log2 size 204.3171299264. Includes full units mod 2187 (order 2·729), but not full units mod every prime power. Q is the same quotient for all blocks.

## Proof and recovery

In each block reserve 16 primes. Their binary subset products cover the entire odd coordinate: accumulate S starting at {1}, replacing it by S union pS for each reserved prime; the final set has 5103 elements. The remaining construction uses 192 independent switches. A switch consists of two alternatives (each a small subset, possibly empty). The alternatives have equal odd-coordinate product. Their binary signatures differ by a vector; the 192 differences are independent. Switch supports and reserve supports are pairwise disjoint. Consequently:

1. Fix the unused inputs to any desired values; start with the first alternative of every switch.
2. Use backtracking in the reserve recurrence to hit the desired odd coordinate after the fixed choices and default switches. This may change the signs.
3. Solve a binary linear system for the switches needed to repair *all* signs. Flipping them leaves the odd coordinate unchanged.

This realizes every class of Q with 0/1 coefficients, not just the projected BPSW target. Switch construction uses collisions of 1-, 2-, and 3-factor residues or equal pair products, not an enumeration of composite candidates. Exhaustive checks are confined to the small odd quotient. Sign generation uses no 2^192 table. `verify_cover.py` independently checks disjointness, global sign rank, odd power residues, and the saved target witnesses without discrete logs. The checker binds the certificate to the four-list pool by SHA256 and checks its partition. For the fourth block the choices outside it (1746) and unused within it (70) are completely free. Hence *every* target fiber in the whole pool has at least 2^1816 distinct binary subsets, without an assumption of uniformity. This is a quotient-fiber lower bound, **not** a bound on the number of full CRT solutions. The full residual group has log2 size 1515.11897852; our correction switches are not guaranteed to preserve any chosen kernel coordinates. Neither independent generation nor large fibers imply short-subset coverage of the kernel.

```sh
export OPENBLAS_NUM_THREADS=1
python quotient_cover.py runs/lift12/trim/cover-5103.json
python verify_cover.py runs/lift12/trim/cover-5103.json
python -m unittest test_quotient_cover
```

Use `quotient_cover.complete(report, projection, partial, block=3)` to correct a partial selection, preserving the exact on/off pattern on all its free factors. `decode` accepts any target sign vector in V and odd residue (in the builder's cyclic log coordinates). The default run's target witnesses have respectively 273, 259, 267 and 251 factors; none satisfies the two full CRT conditions. Exploratory attempts at odd quotients of order 11907 and 35721 found no certificate with seeds 0–3 in block 0; this is a limitation of the greedy construction, not a coverage obstruction.
