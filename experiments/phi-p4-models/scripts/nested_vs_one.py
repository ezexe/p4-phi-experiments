"""
The shard and pair helpers that nested_suite.py, pathways_suite.py and directions_suite.py import: first_added,
shards, pairs_of and zmean.
"""

import math
import os
import subprocess
import sys

import phi_rank as pr  # noqa: E402


def first_added(root):
    """{path: unix time git first added it}: one git log, oldest commit last."""
    out = subprocess.run(["git", "-C", root, "log", "--format=@%at", "--name-only", "--diff-filter=A"],
                         capture_output=True, text=True).stdout
    when, t = {}, 0
    for line in out.split("\n"):
        if line.startswith("@"):
            t = int(line[1:])
        elif line.strip():
            when[line.strip()] = t            # later lines are older commits: keep overwriting, oldest wins
    return when


def shards(root, k):
    files = pr.text_files(root)
    when = first_added(root)
    rel = lambda p: os.path.relpath(p, root).replace(os.sep, "/")  # noqa: E731
    files.sort(key=lambda p: (when.get(rel(p), 1 << 62), rel(p)))
    sizes = [min(os.path.getsize(p), pr.MAX_FILE_BYTES) for p in files]
    total, out, cur, acc = sum(sizes), [[] for _ in range(k)], 0, 0
    for p, s in zip(files, sizes):
        if acc >= total * (cur + 1) / k and cur < k - 1:
            cur += 1
        out[cur].append(p)
        acc += s
    return out


def pairs_of(root, files):
    pairs, total = [], 0
    for p in files:
        ps = pr.passages(p)
        total += sum(len(x) for x in ps)
        r = os.path.relpath(p, root)
        pairs += [(f"{r}\n{a}", b) for a, b in zip(ps, ps[1:])]
        if len(ps) == 1:
            pairs.append((r, ps[0]))
    return pairs, total


def zmean(score_lists):
    out = None
    for sc in score_lists:
        mu = sum(sc) / len(sc)
        sd = math.sqrt(sum((x - mu) ** 2 for x in sc) / len(sc)) or 1.0
        z = [(x - mu) / sd for x in sc]
        out = z if out is None else [a + b for a, b in zip(out, z)]
    return out
