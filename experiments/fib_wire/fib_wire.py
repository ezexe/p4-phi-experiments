#!/usr/bin/env python3
"""
fib_wire.py: a fibinary wire between the programs around a p4 store, and what it does to the store's real labels.

p4-recall's hooks, its trainer and the phi models pass each other labels as rows of text (`labels.tsv`: task id,
entry id, label, source, weight, time, note). This writes the same statements as a stream of Fibonacci codes
(phi_prune.gaps: a number's Zeckendorf digits, least significant first, closed by a 1, so the first `11` ends each
code and the stream needs no separators or lengths), and measures the trade on a store.

A record is four codes:
    act      what was said about the entry for the task (ACTS: edited, cited, ..., and three a session could state
             itself: applied, set-aside, lacked)
    task     the task's number, as the gap from the previous record's (records are sorted by task, then entry)
    entry    the entry's number, as the gap from the previous entry of the same task
    check    ((3 act + 5 task + 7 entry) mod 89) + 1, so a flipped bit that still parses is caught
The names (task ids, entry ids) are not on the wire: a table beside it lists them once, and the stream's first
code is the store's node number in phi_rank.tree's addresses (1, 101, 1001, ...).

What is measured (`run --store <project>/.claude/p4`), free and offline:
    lossless     the records read back are the records written, so training on them is the same training
    size         the stream against the same statements as text rows and as fixed-width numbers, with and
                 without the name table
    append       two streams end to end read as one
    integrity    of random single-bit flips: caught by the stream's own structure, caught by the check code, and
                 the ones that get through
    speed        records written and read per second

It does not measure whether a model learns more from the stream: the stream carries exactly what the rows carry,
by construction. What it changes is how the statements travel, not what is known.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

ACTS = ("edited", "invoked", "cited", "dropped", "unopened", "used", "unused", "positive", "negative",
        "applied", "set-aside", "lacked")          # the last three: what a session could state itself
CHECK_MOD = 89                                      # F(11): the check fits ten digits
FIELDS = 4


def gaps_mod():
    from phi_prune import gaps
    return gaps


def act_of(row):
    """The act a labels.tsv row states: its note's first word when that is an act, else positive / negative."""
    word = (row.get("note") or "").split(" ")[0].strip("();")
    if word in ACTS:
        return word
    return "positive" if str(row.get("label")) == "1" else "negative"


def read_labels(store):
    """[(task id, entry id, act)] of a store's models/labels.tsv, each statement once."""
    path = Path(store) / "models" / "labels.tsv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return sorted({(r["task"], r["entry"], act_of(r)) for r in rows if r.get("task") and r.get("entry")})


def tables(statements):
    """(task ids, entry ids): each name once, in the order the stream numbers them."""
    return sorted({s[0] for s in statements}), sorted({s[1] for s in statements})


def check(a, t, e):
    return (3 * a + 5 * t + 7 * e) % CHECK_MOD + 1


def encode(statements, node=1, with_check=True, names=None):
    """(bitstream, task table, entry table): the statements as records, sorted by task then entry then act.
    `names` is (task table, entry table) fixed in advance, for writers that share one."""
    g = gaps_mod()
    tasks, entries = names or tables(statements)
    t_at, e_at = {t: i for i, t in enumerate(tasks)}, {e: i for i, e in enumerate(entries)}
    nums = sorted((t_at[t], e_at[e], ACTS.index(a) + 1) for t, e, a in statements)
    out, prev_t, prev_e = [g.fib_code(node)], 0, 0
    for t, e, a in nums:
        if t != prev_t:
            prev_e = 0
        dt, de = t - prev_t + 1, e - prev_e + 1      # a gap of 1 is "the same again": codes start at 1
        out += [g.fib_code(a), g.fib_code(dt), g.fib_code(de)] + ([g.fib_code(check(a, dt, de))] if with_check else [])
        prev_t, prev_e = t, e
    return "".join(out), tasks, entries


class Corrupt(ValueError):
    pass


def decode(bits, tasks, entries, with_check=True):
    """(node, [(task id, entry id, act)]) back from a stream; Corrupt when it does not read as one."""
    g = gaps_mod()
    try:
        vals, _ = g.read_codes(bits)
    except ValueError as e:
        raise Corrupt(f"framing: {e}") from None
    n = FIELDS if with_check else FIELDS - 1
    if not vals or (len(vals) - 1) % n:
        raise Corrupt(f"structure: {len(vals)} codes do not make a node and whole records of {n}")
    node, out, t, e = vals[0], [], 0, 0
    for k in range(1, len(vals), n):
        a, dt, de = vals[k:k + 3]
        if with_check and vals[k + 3] != check(a, dt, de):
            raise Corrupt(f"check: record {(k - 1) // n}")
        if not 1 <= a <= len(ACTS):
            raise Corrupt(f"structure: act {a}")
        if dt > 1:
            t, e = t + dt - 1, 0
        e += de - 1
        if t >= len(tasks) or e >= len(entries):
            raise Corrupt(f"structure: record {(k - 1) // n} names task {t} or entry {e}, past the table")
        out.append((tasks[t], entries[e], ACTS[a - 1]))
    return node, out


def pack(bits):
    """The stream as bytes: its length in bits first, then the bits."""
    return len(bits).to_bytes(4, "big") + int(bits or "0", 2).to_bytes((len(bits) + 7) // 8, "big") if bits else b"\0\0\0\0"


def unpack(data):
    n = int.from_bytes(data[:4], "big")
    return bin(int.from_bytes(data[4:], "big"))[2:].zfill(n) if n else ""


def entropy_bits(values):
    c = Counter(values)
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def flip_test(bits, tasks, entries, statements, n=2000, seed=0, with_check=True):
    """Of n random single-bit flips: how many the structure caught, the check caught, and how many read as a valid
    stream that says something else (silent)."""
    rng = random.Random(seed)
    want = sorted(statements)
    got = Counter()
    for _ in range(n):
        i = rng.randrange(len(bits))
        bad = bits[:i] + ("1" if bits[i] == "0" else "0") + bits[i + 1:]
        try:
            _node, back = decode(bad, tasks, entries, with_check=with_check)
        except Corrupt as e:
            got[str(e).split(":")[0]] += 1
            continue
        got["silent" if sorted(back) != want else "harmless"] += 1
    return {k: got.get(k, 0) for k in ("framing", "structure", "check", "silent", "harmless")}


def run(store, flips=2000):
    statements = read_labels(store)
    t0 = time.time()
    bits, tasks, entries = encode(statements)
    t_enc = time.time() - t0
    t0 = time.time()
    node, back = decode(bits, tasks, entries)
    t_dec = time.time() - t0
    plain, _, _ = encode(statements, with_check=False)
    text_rows = sum(len(f"{t}\t{e}\t{a}\n".encode("utf-8")) for t, e, a in statements)
    table = sum(len(x.encode("utf-8")) + 1 for x in tasks + entries)
    fixed = len(statements) * (math.ceil(math.log2(len(ACTS))) + math.ceil(math.log2(max(2, len(tasks))))
                               + math.ceil(math.log2(max(2, len(entries)))))
    ent = len(statements) * (entropy_bits([s[2] for s in statements]) + entropy_bits([s[0] for s in statements])
                             + entropy_bits([s[1] for s in statements]))
    half = len(statements) // 2
    a_bits = encode(statements[:half], names=(tasks, entries))[0]
    b_bits = encode(statements[half:], names=(tasks, entries))[0]
    joined = decode_many([a_bits, b_bits], tasks, entries)
    res = {
        "statements": len(statements), "tasks": len(tasks), "entries": len(entries), "acts": dict(Counter(s[2] for s in statements)),
        "lossless": sorted(back) == sorted(statements), "node": node,
        "bits": len(bits), "bits_per_record": round(len(bits) / max(1, len(statements)), 2),
        "bits_without_check": len(plain), "wire_bytes": len(pack(bits)), "text_bytes": text_rows, "table_bytes": table,
        "fixed_width_bits": fixed, "entropy_bits": round(ent),
        "append_reads_as_one": sorted(joined) == sorted(statements),
        "flips_with_check": flip_test(bits, tasks, entries, statements, n=flips),
        "flips_without_check": flip_test(plain, tasks, entries, statements, n=flips, with_check=False),
        "encode_per_s": round(len(statements) / max(t_enc, 1e-9)), "decode_per_s": round(len(statements) / max(t_dec, 1e-9)),
    }
    return res


def decode_many(streams, tasks, entries):
    """Streams written one after another against the same tables, read in turn: each begins with its node code."""
    out = []
    for s in streams:
        out += decode(s, tasks, entries)[1]
    return out


def report(res, path):
    f = res["flips_with_check"]
    p = res["flips_without_check"]
    n = sum(f.values())
    L = ["# fib_wire: a store's labels as a fibinary stream", "",
         f"{res['statements']:,} statements over {res['tasks']:,} tasks and {res['entries']:,} entries.", "",
         f"- Read back exactly: {'yes' if res['lossless'] else 'NO'}; two streams end to end read as one: "
         f"{'yes' if res['append_reads_as_one'] else 'NO'}.",
         f"- Size: {res['wire_bytes']:,} bytes on the wire ({res['bits_per_record']} bits a record) against "
         f"{res['text_bytes']:,} bytes as text rows; the name table both need once is {res['table_bytes']:,} bytes.",
         f"- Without the check code the wire is {res['bits_without_check'] // 8:,} bytes; the check costs "
         f"{(res['bits'] - res['bits_without_check']) // 8:,}. The same numbers at fixed width: "
         f"{res['fixed_width_bits'] // 8:,} bytes. Coded one at a time with no order, their entropy is "
         f"{res['entropy_bits'] // 8:,} bytes: the gaps between sorted records are what gets the wire under it.",
         f"- With the name table, which the stream cannot do without: {(res['wire_bytes'] + res['table_bytes']):,} bytes "
         f"against the text's {res['text_bytes']:,}.",
         f"- Of {n:,} single-bit flips with the check: {f['framing'] + f['structure']:,} caught by the stream's own "
         f"structure, {f['check']:,} by the check, {f['silent']:,} read as a valid stream saying something else.",
         f"- Without the check: {p['framing'] + p['structure']:,} caught, {p['silent']:,} silent.",
         f"- Speed: {res['encode_per_s']:,} records a second written, {res['decode_per_s']:,} read.", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--store", required=True, help="a store directory (<project>/.claude/p4)")
    ap.add_argument("--flips", type=int, default=2000)
    ap.add_argument("--work", default=str(HERE / "out"))
    a = ap.parse_args(argv)
    res = run(a.store, a.flips)
    out = Path(a.work) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(report(res, out / "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
