"""Construct squarefree D=5 strong BPSW candidates. See README.md."""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from sympy import primerange
from sympy.ntheory.primetest import mr, is_strong_lucas_prp


def fib(k, p):
    a, b = 0, 1
    for bit in bin(k)[2:]:
        c, d = a * (2*b-a) % p, (a*a+b*b) % p
        a, b = (d, (c+d) % p) if bit == '1' else (c, d)
    return a


def strip(k, spf, identity):
    x = k
    while x > 1:
        q = spf[x]
        while x % q == 0:
            x //= q
        while k % q == 0 and identity(k//q):
            k //= q
    return k


def catalog(limit):
    spf = list(range(limit+2))
    for i in range(2, math.isqrt(limit+1)+1):
        if spf[i] == i:
            for j in range(i*i, limit+2, i):
                if spf[j] == j:
                    spf[j] = i
    for p in range(3, limit+1, 2):
        if spf[p] == p and p % 5 in (2, 3):
            a = strip(p-1, spf, lambda k: pow(2, k, p) == 1)
            b = strip(p+1, spf, lambda k: fib(k, p) == 0)
            yield p, a, b


def discover(limit, trials, seed, smooth):
    rows = list(catalog(limit))
    rng = random.Random(seed)
    # Assign each odd prime power to at most one side of the CRT.
    powers = []
    for q in primerange(3, smooth+1):
        x = int(q)
        while x*q <= limit+1:
            x *= q
        powers.append(x)
    pools = {}
    for _ in range(trials):
        A, B = 2, 2
        if rng.randrange(2):
            A <<= limit.bit_length()
        else:
            B <<= limit.bit_length()
        for x in powers:
            if rng.randrange(2):
                A *= x
            else:
                B *= x
        groups = defaultdict(list)
        for p, a, b in rows:
            if A % a == B % b == 0:
                groups[a & -a, b & -b].append((p, a, b))
        for group in groups.values():
            # Shrink to actual LCMs, discard nonunits, repeat via next trial
            # if a different allocation would recover discarded factors.
            a = math.lcm(*(r[1] for r in group))
            b = math.lcm(*(r[2] for r in group))
            group = [r for r in group if math.gcd(r[0], a*b) == 1]
            if len(group) < 3:
                continue
            ps, aa, bb = zip(*group)
            a, b = math.lcm(*aa), math.lcm(*bb)
            pools[ps] = dict(primes=ps, A=a, B=b)
    def score(pool):
        # Proxy only: odd subsets per residue, NOT a success probability.
        return len(pool['primes'])-1-math.log2(math.lcm(pool['A'], pool['B']))
    return len(rows), sorted(pools.values(), key=score, reverse=True)


def target(a, b):
    g = math.gcd(a, b)
    if 2 % g:
        raise ValueError('incompatible congruences')
    m = math.lcm(a, b)
    return (1+a*((-2//g)*pow(a//g, -1, b//g) % (b//g))) % m, m


def products(values, m, start, end):
    # Gray order permits random access to a shard with one multiply per step.
    inv = [pow(x, -1, m) for x in values]
    mask = start ^ (start >> 1)
    value = math.prod(x for j, x in enumerate(values) if mask >> j & 1) % m
    for i in range(start, end):
        yield mask, value
        if i+1 < end:
            bit = ((i+1) & -(i+1)).bit_length()-1
            mask ^= 1 << bit
            value = value * (values[bit] if mask >> bit & 1 else inv[bit]) % m


def matches(ps, t, m, left, start, end):
    table = defaultdict(list)
    # Store required complements; preserve collisions, including singletons.
    for mask, v in products([pow(p, -1, m) for p in ps[:left]], m, 0, 1 << left):
        table[t*v % m, 1-(mask.bit_count() % 2)].append(mask)
    for mask, v in products(ps[left:], m, start, end):
        for lo in table.get((v, mask.bit_count() % 2), ()):
            full = lo | mask << left
            if full.bit_count() >= 3:
                yield [p for j, p in enumerate(ps) if full >> j & 1]


def profile(pool):
    """Exact membership in the generated group (not necessarily a subset)."""
    from sympy import Matrix, diag, factorint, primitive_root, discrete_log
    from sympy.matrices.normalforms import hermite_normal_form
    t, m = target(pool['A'], pool['B'])
    values = [*pool['primes'], t]
    # Include subset-size parity as a C2 coordinate.
    orders, rows = [2], [[1]*len(values)]
    for q, k in factorint(m).items():
        h = q**k
        if q == 2:
            if k == 1:
                continue
            signs = [int(v % 4 == 3) for v in values]
            orders.append(2); rows.append(signs)
            if k == 2:
                continue
            vals = [(-v if s else v) % h for v,s in zip(values, signs)]
            base, order = 5, h//4
        else:
            vals = [v % h for v in values]
            base, order = primitive_root(h), h-h//q
        orders.append(order)
        rows.append([int(discrete_log(h, v, base)) if v != 1 else 0 for v in vals])
    coords = Matrix(rows)
    lattice = diag(*orders)
    for j in range(len(values)-1):
        lattice = hermite_normal_form(lattice.row_join(coords[:, j]), D=math.prod(orders))
    index = int(lattice.det())
    reachable = index == hermite_normal_form(lattice.row_join(coords[:, -1]), D=index).det()
    return math.prod(orders)//index, bool(reachable)


def main():
    arg = argparse.ArgumentParser()
    subs = arg.add_subparsers(dest='cmd', required=True)
    d = subs.add_parser('discover')
    for key, default in [('limit', 1000000), ('trials', 100), ('seed', 1), ('smooth', 47)]:
        d.add_argument('--'+key, type=int, default=default)
    d.add_argument('--out', default='pools.json')
    pr = subs.add_parser('profile'); pr.add_argument('file'); pr.add_argument('--pool', type=int, default=0)
    s = subs.add_parser('search')
    s.add_argument('file'); s.add_argument('--pool', type=int, default=0)
    s.add_argument('--left', type=int); s.add_argument('--start', type=int, default=0)
    s.add_argument('--end', type=int)
    args = arg.parse_args()
    if args.cmd == 'discover':
        count, pools = discover(args.limit, args.trials, args.seed, args.smooth)
        Path(args.out).write_text(json.dumps(pools)+'\n')
        print(f'{count} eligible primes, {len(pools)} distinct pools')
        for i, pool in enumerate(pools[:5]):
            print(i, len(pool['primes']), 'factors;', math.lcm(pool['A'],pool['B']).bit_length(), 'modulus bits')
    elif args.cmd == 'profile':
        pool = json.loads(Path(args.file).read_text())[args.pool]
        size, reachable = profile(pool)
        print(f'group size (including parity)={size}, target in group={reachable}')
    else:
        pool = json.loads(Path(args.file).read_text())[args.pool]
        ps = pool['primes']; t, m = target(pool['A'], pool['B'])
        left = len(ps)//2 if args.left is None else args.left
        if not 0 <= left <= min(22, len(ps)):
            arg.error('choose --left between 0 and 22 (RAM guard)')
        size = 1 << (len(ps)-left)
        end = size if args.end is None else args.end
        if not 0 <= args.start <= end <= size or (args.end is None and size > 2**24):
            arg.error('choose a bounded right interval with --start/--end')
        hits = 0
        for factors in matches(ps, t, m, left, args.start, end):
            n = math.prod(factors)
            if not (all(p > 1 for p in factors) and mr(n,[2]) and is_strong_lucas_prp(n)):
                raise RuntimeError(f'candidate failed independent verification: {factors}')
            print(json.dumps(dict(n=n, factors=factors)), flush=True); hits += 1
        print(f'pool {args.pool}: left={1<<left}, right=[{args.start},{end})/{size}, verified hits={hits}')


if __name__ == '__main__':
    main()
