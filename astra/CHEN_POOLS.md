# Chen–Greene pools, strong-signature audit, and a fixed-modulus extension

`runs/chen-greene/Baillie-PSW.html` is the saved [Greene data page](https://www.d.umn.edu/~jgreene/baillie/Baillie-PSW.html), SHA-256 `7dfc6796f0064149efb6ee316ab3b214f9ede4bbc92a0cb94049140bcd38cd7c`. The *larger displayed brace-delimited list* has 4,816 distinct entries despite the page's label of 4,838. All are certified primes and inert for D=5. Of these, 19 Fibonacci ranks divide twice the published N but not N (an extra factor 2); those 19 are audited separately. Combining 4,797 valid entries, 35 more from the smaller published list, and 9 additional divisor-frontier primes gives a 4,841-prime reconstruction **for the published moduli**, not a reconstruction of the precise unpublished/missing entries.

Generator idea: if a=ord_p(2) and b=rank_p(F), use a|M,b|N rather than requiring p-1|M,p+1|N. Find near-consecutive smooth integers by generating p=2dh+1 or p=4dh-1 where d divides the allowed odd modulus on that side, and h is a small missing cofactor. Test actual divisibility via pow(2,gcd(p-1,M),p) and F_gcd(p+1,N) mod p. Our bounded generator focuses on p=3 mod 8, p=±2 mod 5: then v2(a)=1 and v2(b)=2. The latter follows from Frobenius exchanging the Fibonacci roots alpha,beta and (alpha/beta)^((p+1)/2)=-1. An odd subset ensures D=5. These primes need no further subdivision into strong-test signatures. The frontier is not asserted complete outside its stated bounds.

Audit: for each signature independently use A=lcm(a), B=lcm(b). Combine n=1 mod A, n=-1 mod B into a CRT target. Embed each input in U(lcm(A,B)) × C2, with the C2 column bit 1 for every input (all are inert); target bit 1. Decompose prime-power unit groups into cyclic coordinates, with two coordinates for powers of two >=8. Per-primary elimination computes exact generated image sizes and joint target membership. The elimination concerns arbitrary integer exponents. It does not prove a nontrivial zero-one solution. The heuristic score `|P|-log2(|H|)` gives expected log2 hits **only under a uniform subset-product model**. An obstruction excludes only this fixed target; a different subpool can have different LCMs.

Reproduce (Python 3, NumPy, SymPy; run in astra):

```sh
export OPENBLAS_NUM_THREADS=1
d=runs/chen-greene
python pool_audit.py "$d/Baillie-PSW.html" "$d" --historic ../psw/lists/primes620.txt
python extend_pool.py "$d/audit.json" "$d/frontier.txt" --upper 10000000000 --hmax 65
python pool_audit.py "$d/Baillie-PSW.html" "$d" --historic ../psw/lists/primes620.txt --extra "$d/frontier.txt"
python regen_chen.py "$d"
python trim_pool.py "$d/extended.txt" "$d"
python -m unittest test_pool_audit
```

`chen-extension.json` and `audit.json` give per-class results and exact target group sizes. `extended.txt` has 2,081 primes in the Alford–Grantham signature at its original order LCMs. `trim-atleast2030.txt` has 2,034 primes, the original A, and B divided by 67·863. Its reachable joint group is **28,877 times smaller** than the historic 2,030-prime pool's. `trim-16.txt` trades more choices for a smaller group. The filenames' numerical margins are conservative ranking parameters, not exact final surplus values. None of these results is a composite witness or a bound on the difficulty of every search algorithm.
