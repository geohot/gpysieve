# Pomerance pool experiment

Run from `astra/` with SymPy installed:

```sh
python -m unittest -v
python pomerance.py 2048 5 --pmax 10000000000 --out pool.json
python paper_search.py pool.json five --exact
python paper_search.py pool.json cluster --width 30 --exact
python paper_search.py pool.json cluster --first 40 --width 40 --stop 1 --exact
```

The first command after the tests enumerates exactly the paper's eligible primes in the specified subinterval, not all the way to T^5. For every candidate product n with largest selected prime p, write n=pw. Since p is 1 mod its base-2 order and -1 mod its Fibonacci rank, a passing n requires w=1 mod their LCM. The default uses p-1 and p+1 for the stronger paper conditions; `--exact` uses the actual orders/ranks. On these primes the 2-adic valuations are fixed, so the congruences imply synchronized strong-test passes. The full product is tested independently on any hit; use `gp -fq verify.gp` then `verify([factor1,factor2,...])` to cross-check.

`five` joins ordered pairs to find every admissible four-factor cofactor w. `cluster` grows a group around each seed by least added CRT modulus, then joins subsets of two halves for each possible largest prime. In both modes, filters on the largest factor are necessary conditions only; survivors are checked against every selected factor. Reported subset counts measure mathematical coverage, not full BPSW calls. Cluster counts include overlaps. Identical clusters are deduplicated within a run, not across shards. Hits below 2^64 are outside the search target.

Use disjoint `--start I --stop J` ranges indexing anchors (five) or seeds (cluster). For five the late anchors cost more; worst-case memory is quadratic in the anchor index. For cluster, fix `--width` per run; memory grows exponentially with half-width. Pool files should come from `pomerance.py`; the search assumes their family restrictions. Do not feed it arbitrary prime lists and interpret a negative result as exhaustive for those lists.
