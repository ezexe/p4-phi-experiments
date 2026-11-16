"""
Measurements of the Zeckendorf register against the binary register.

1. Single-bit flips over every register of a width in normal form: the share
   detected and the share uniquely correctable.
2. Rewrite steps per sum, for every pair of values below 21, 55 and 144.
3. Normalization rounds, by normalize_rounds, for every pair below 21 .. 610.
"""

import statistics as st

from phi_prune.register import (
    BINARY, KINDS, PHI, add, correction_candidates, flip_detected, from_int, normalize_rounds, place_values, pour,
)


def flips():
    print("1. single-bit flips")
    for n in (4, 6, 8, 10, 13):
        for kind in KINDS:
            top = (1 << n) if kind == BINARY else place_values(PHI, n + 1)[-1]
            words = [from_int(v, kind, n) for v in range(top)]
            det = cor = tot = 0
            for w in words:
                for i in range(n):
                    f = list(w)
                    f[i] ^= 1
                    tot += 1
                    if flip_detected(f, kind):
                        det += 1
                        c = correction_candidates(f)
                        assert w in c
                        cor += len(c) == 1
            print(f"  {n:>2} digits {kind:<6} {len(words):>5} values  detected {det / tot:.3f}  "
                  f"correctable {cor / tot:.3f}")


def equal_values():
    print("\n2. rewrite steps per sum, same values in both registers, lowest position first")
    for n in (21, 55, 144):
        for kind in KINDS:
            steps = [add(from_int(a, kind), from_int(b, kind), kind)[1] for a in range(n) for b in range(n)]
            print(f"  values < {n:<4} {kind:<6} mean {st.mean(steps):.2f} max {max(steps)}")


def all_rounds(n, kind):
    return [normalize_rounds(pour(from_int(a, kind), from_int(b, kind)), kind)[1] for a in range(n) for b in range(n)]


def rounds():
    print("\n3. normalization rounds per sum")
    for n in (21, 55, 144, 610):
        for kind in KINDS:
            r = all_rounds(n, kind)
            print(f"  values < {n:<4} {kind:<6} width {len(from_int(n - 1, kind)):>2}  mean {st.mean(r):.2f} worst {max(r)}")


if __name__ == "__main__":
    flips()
    equal_values()
    rounds()
