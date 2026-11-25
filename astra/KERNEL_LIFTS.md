# Moving inside the quadratic/odd quotient fiber

These are results for the specific D=-7 pools in `runs/lift12/trim`. H is their generated full CRT image including the Selfridge tags; neither a subgroup calculation nor coverage of a proper quotient is a BPSW witness.

## Constructive moves (four-list pool)

Use the previously certified disjoint switches, i.e. 768 binary choices between two disjoint alternatives. Each switch preserves the order-5103 odd coordinate. Apply the reserve correction once, and solve the linear sign system once to get a starting point. Toggle any XOR vector in the nullspace of the sign matrix; every such change preserves the old quotient. At a given start, let a_i be the **oriented** order-four log change for toggling switch i. The change for vector v is sum a_i v_i mod 4; on the sign kernel it is even. BUT the change for v XOR w is the sum of their changes minus 2 sum a_i v_i w_i. Thus naive addition of kernel moves is incorrect.

For each already chosen direction w and each controlled quartic coordinate, constrain the next v to be orthogonal mod 2 to (a_i w_i)_i. These are linear constraints, not subset constraints. Among the resulting null vectors keep a move whose quartic changes give a new independent vector after division by two. Pairwise orthogonality suffices for **every** XOR combination (higher intersection terms are multiples of 4). The saved 24 directions give a full rank-24 toggle system on one old-target fiber. Its 2^24 configurations hit the 2^24 lifts bijectively, with no enumeration. Saved 987-prime selection realizes the old target and its 24 prescribed next bits; it misses the full CRT. The direct modular checks in `test_kernel_lift.py` use switch product ratios, verify *all* pairs and test the saved selection. Blocks need not individually cover the lifted quotient.

## Larger, nonconstructive simultaneous coverage

Work in V, the actual binary sign image (includes mod 4 and the Legendre symbols), augmented by C5103. For the selected primes q=1 mod 4 refine each sign coordinate to an order-four character; for selected q=1 mod 8 further refine it to order eight. Think of the codomain as all refinements of V consistent with reduction to signs, times C5103. Do not assume the subset map covers it. Uniform fair binary selection gives a probability measure. At each character its absolute Fourier coefficient is the product over input primes of |cos(theta_i)|, where theta_i is **half** the character phase. If the sum S of these absolute coefficients, including the trivial character, is less than 2, Fourier inversion gives every target probability >= (2-S)/|quotient|. This tests all target fibers simultaneously; it does not require any independence model for the projected residues.

Here is the tractable bound used instead of enumerating characters. At fixed higher-digit and odd characters, summing over the dual of V shifts theta_i by a multiple of pi/2. Greedily pack b disjoint full binary column bases of the sign matrix. Drop other factors (initially <=1). Apply Holder with exponent b to the sum over dual sign characters of the product over bases. Inside each base the shifts are independent uniform binary labels when enumerating characters, so the bound is the product over retained i of

    F1(theta_i)=(|cos(theta_i)|^b + |sin(theta_i)|^b)^(1/b).

This is pi/2-periodic. Now sum over the additional order-four character digits. At a given coordinate i, only the parity of the resulting log matters: the shift is either 0 or pi/4 modulo the current period. Its binary evaluation column consists of the Legendre bits restricted to the lifted primes q. Pack k disjoint bases of these columns *inside the retained inputs*. Drop unselected F1 factors (<=1 for b>=2) and apply Holder again. The bound becomes a product of F2=(F1(theta)^k+F1(theta+pi/4)^k)^(1/k). It is pi/4-periodic. For order-eight digits repeat with pi/8 and l disjoint bases. **F2 can slightly exceed one: discarding F2 factors without compensation is invalid.** The implementation must charge a uniform upper bound for every discarded F2 factor. Finally sum the product bounds over the 5103 fixed odd characters. The binary-digit character enumeration uses one representative per refinement digit, with all binary sign characters at the bottom: products differing by lower characters have already been summed. Coordinate powers correspond to true multiplicative characters, not independent projections of the target.

`quartic_lift.py` uses outward-rounded mpmath interval trig and powers. Per-residue log2 costs are rounded UP on a 2^-20 grid; exact integer dot sums are binned UP to 1/256 bit. The final sum is rounded UP to thousandths. The proof controls the full pool, not each of the four blocks. A kernel size is an exact remaining group order, **not** an automatic improvement of the same number of bits in runtime. Efficient decoding of the largest covered quotient remains open.

```sh
export OPENBLAS_NUM_THREADS=1
d=runs/lift12/trim
python kernel_moves.py "$d/cover-5103.json" "$d/moves24.json" --bits 24
python quartic_lift.py --out "$d/fourier.json"
python quartic_lift.py --tag 16 --quartics 74 --octics 0 --out "$d/fourier-small.json"
python -m unittest test_kernel_lift
```
