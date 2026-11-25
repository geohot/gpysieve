"""Exhaust every small pool from a discovery run; skip larger pools explicitly."""
import argparse
import json
import math
from pathlib import Path

from search import matches, target, mr, is_strong_lucas_prp

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file')
    ap.add_argument('--max-factors', type=int, default=24)
    args = ap.parse_args()
    done = skipped = candidates = hits = 0
    for i, pool in enumerate(json.loads(Path(args.file).read_text())):
        ps = pool['primes']; k = len(ps)
        if k > args.max_factors:
            skipped += 1
            continue
        done += 1
        candidates += (1 << (k-1))-k
        for factors in matches(ps, *target(pool['A'], pool['B']), k//2, 0, 1 << (k-k//2)):
            n = math.prod(factors)
            if not (mr(n, [2]) and is_strong_lucas_prp(n)):
                raise RuntimeError(f'verifier disagreement: {factors}')
            print(json.dumps(dict(pool=i, n=n, factors=factors)), flush=True)
            hits += 1
    print(f'{done=} {skipped=} {candidates=} (with overlap) {hits=}')
