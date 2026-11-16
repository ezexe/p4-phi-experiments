"""
The logic a stream of node addresses needs, per code: what the receiver must decide at each item, what one
flipped bit does, and what happens when the tree outgrows what was agreed in advance. For
ezexe/phi's .docs/p4/nested-store.md, section 5.

Streams: random sequences of Zeckendorf store addresses (p4 = 1, .claude = 0; start and end with 1, no 11) of
up to MAXLEN digits, the addresses run Z grows. Codes:
  fibonacci   the address digits least significant first, then 1: the comma is the pair the tree never makes
  gamma       Elias-gamma of the address's integer value (universal: no width agreed either)
  fixed       the value in W bits, W agreed in advance from the largest address then known
  lenfield    a fixed-width length field, then the address digits

Measured:
  decisions   branch decisions per decoded item, counted by instrumented decoders (one per comparison the decoder
              makes to know where it is: a digit test, a mode switch, a counter test)
  flips       every single-bit flip of the stream: whether the decoder can tell (a 11 inside a codeword, a value out
              of range, a stream that does not end cleanly) and how many items decode wrong before it is back in step
  growth      a stream whose addresses grow one digit past what fixed and lenfield agreed on: which codes still parse

Run (2026-10-27):  python scripts/stream_logic.py --items 2000 --maxlen 12
"""

import argparse
import itertools
import json
import math
import random
import sys


def addresses(maxlen):
    return [a for m in range(1, maxlen + 1) for t in itertools.product("01", repeat=m)
            for a in ["".join(t)] if a[0] == "1" and a[-1] == "1" and "11" not in a]


def zval(a):
    pl = [1, 2]
    while len(pl) < len(a):
        pl.append(pl[-1] + pl[-2])
    return sum(int(c) * p for c, p in zip(reversed(a), pl))


class Count:
    n = 0

    def __call__(self, cond):
        self.n += 1
        return cond


# ─── encoders and counting decoders: each returns (values, ok), ok False when the stream is visibly wrong ────

def enc_fib(addrs):
    return "".join(a[::-1] + "1" for a in addrs)


def dec_fib(bits, c):
    out, cur, ok, prev = [], "", True, "0"
    for b in bits:
        if c(b == "1" and prev == "1"):          # the comma, and the only rule: 11 ends a codeword
            out.append(cur[::-1])
            cur, prev = "", "0"
            continue
        cur += b
        prev = b
    if c(cur != ""):
        ok = False                               # the stream ended inside a codeword
    for a in out:                                # an address must start and end with a store
        if c(not a or a[0] != "1" or a[-1] != "1"):
            ok = False
    return out, ok


def enc_gamma(addrs):
    out = []
    for a in addrs:
        b = bin(zval(a))[2:]
        out.append("0" * (len(b) - 1) + b)
    return "".join(out)


def dec_gamma(bits, c, valid=None):
    out, i, ok = [], 0, True
    while c(i < len(bits)):
        z = 0
        while c(i < len(bits) and bits[i] == "0"):
            z += 1
            i += 1
        if c(i + z + 1 > len(bits)):
            return out, False
        out.append(int(bits[i:i + z + 1], 2))
        i += z + 1
    if valid is not None:
        for v in out:
            if c(v not in valid):
                ok = False                       # a value no address has: needs the table of every address
    return out, ok


def enc_fixed(addrs, w):
    return "".join(format(zval(a), f"0{w}b") for a in addrs)


def dec_fixed(bits, c, w, valid=None):
    out, ok = [], True
    if c(len(bits) % w):
        ok = False
    for i in range(0, len(bits) - len(bits) % w, w):
        out.append(int(bits[i:i + w], 2))
    if valid is not None:
        for v in out:
            if c(v not in valid):
                ok = False
    return out, ok


def enc_len(addrs, lw):
    return "".join(format(len(a), f"0{lw}b") + a for a in addrs)


def dec_len(bits, c, lw):
    out, i, ok = [], 0, True
    while c(i < len(bits)):
        if c(i + lw > len(bits)):
            return out, False
        n = int(bits[i:i + lw], 2)
        i += lw
        if c(n == 0 or i + n > len(bits)):
            return out, False
        out.append(bits[i:i + n])
        i += n
    for a in out:
        if c("11" in a or a[0] != "1" or a[-1] != "1"):
            ok = False
    return out, ok


def wrong_items(got, want):
    """Items lost or decoded wrong, the sequences aligned (difflib): how far one flip carries before the decoder
    is back in step. A shift that merges or splits one item costs that item, not every item after it."""
    import difflib
    sm = difflib.SequenceMatcher(None, got, want, autojunk=False)
    return len(want) - sum(b.size for b in sm.get_matching_blocks())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=2000)
    ap.add_argument("--maxlen", type=int, default=12)
    ap.add_argument("--flips", type=int, default=3000, help="single-bit flips sampled per code")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    pool = addresses(args.maxlen)
    addrs = [rng.choice(pool) for _ in range(args.items)]
    vals = [zval(a) for a in addrs]
    valid = {zval(a) for a in pool}
    w = max(vals).bit_length()
    lw = max(1, math.ceil(math.log2(args.maxlen + 1)))
    # each code: its stream, a decoder that uses only the code's own structure, and one that also checks every
    # value against the table of all addresses (a table agreed in advance)
    codes = {
        "fibonacci": (enc_fib(addrs), lambda b, c: dec_fib(b, c), None, addrs),
        "gamma": (enc_gamma(addrs), lambda b, c: dec_gamma(b, c), lambda b, c: dec_gamma(b, c, valid), vals),
        f"fixed {w}-bit": (enc_fixed(addrs, w), lambda b, c: dec_fixed(b, c, w),
                           lambda b, c: dec_fixed(b, c, w, valid), vals),
        f"lenfield {lw}-bit": (enc_len(addrs, lw), lambda b, c: dec_len(b, c, lw), None, addrs),
    }
    report = {"items": args.items, "maxlen": args.maxlen, "addresses": len(pool), "codes": {}}
    for name, (bits, dec, dec_table, want) in codes.items():
        c = Count()
        got, ok = dec(bits, c)
        assert ok and got == want, name
        flips = rng.sample(range(len(bits)), min(args.flips, len(bits)))
        caught = caught_t = spread = 0
        for i in flips:
            fb = bits[:i] + ("1" if bits[i] == "0" else "0") + bits[i + 1:]
            got2, ok2 = dec(fb, Count())
            caught += (not ok2)
            caught_t += (not (dec_table or dec)(fb, Count())[1])
            spread += wrong_items(got2, want)
        report["codes"][name] = {
            "bits_per_item": round(len(bits) / args.items, 2),
            "decisions_per_item": round(c.n / args.items, 2),
            "flips_detected_by_structure": round(caught / len(flips), 3),
            "flips_detected_with_table": round(caught_t / len(flips), 3) if dec_table else None,
            "items_lost_per_flip": round(spread / len(flips), 2),
        }
    # growth: the tree grows one digit past what was agreed when the stream format was fixed
    grow_to = max(args.maxlen + 2, (1 << lw) + 1)                 # past the fixed width and the length field
    longer = [a for a in addresses(grow_to) if len(a) > max(args.maxlen, (1 << lw) - 1)]
    grown = addrs[:50] + [rng.choice(longer) for _ in range(50)]
    g = {}
    fb = enc_fib(grown)
    g["fibonacci"] = dec_fib(fb, Count()) == (grown, True)
    gb = enc_gamma(grown)
    g["gamma"] = dec_gamma(gb, Count())[0] == [zval(a) for a in grown]
    g[f"fixed {w}-bit"] = all(zval(a) < (1 << w) for a in grown)
    g[f"lenfield {lw}-bit"] = all(len(a) < (1 << lw) for a in grown)
    report["growth_past_agreement"] = {"grown_to_digits": grow_to, **g}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
