#!/usr/bin/env python3
"""
context_growth.py: a model of how a session's context grows, trained on the sizes p4-recall's view.py records.

Input is numbers only: for each turn of each session, the context when the turn began and when it ended
(view.sizes, from the usage every API call reports). The model predicts how many tokens the next turn will add
from what is known when it begins: the context so far, what the session began at, how far in it is, and how its
earlier turns grew. A least-squares fit on logarithms; nothing to download.

    python experiments/context_growth/context_growth.py run --sessions <folder of session records> [--limit 1000000]

Measured on the newest fifth of the sessions, held out, against two guesses that need no model: the median growth
of all training turns, and the median of the session's own turns so far. Error is the median factor a prediction
is off by, and the share within a factor of two. Then what the prediction is for: turns left before the limit.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "p4" / "lbiza" / "plugins" / "p4-recall" / "scripts"))
HOLDOUT = 0.2
LIMIT = 1_000_000


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def rows_of(folder):
    """[[session, {features}, growth]] in session order: one row a turn that grew the context."""
    import view
    out = []
    names = sorted((os.path.getmtime(os.path.join(folder, n)), n) for n in os.listdir(folder) if n.endswith(".jsonl"))
    for _m, n in names:
        try:
            s = view.sizes(os.path.join(folder, n))
        except (OSError, ValueError):
            continue
        past = []
        for k, (_turn, start, end) in enumerate(s["turns"]):
            grew = end - start
            if grew > 0 and start > 0:
                out.append([n, {"start": start, "baseline": s["baseline"], "turn": k, "prev": past[-1] if past else 0,
                                "own": median(past)}, grew])
                past.append(grew)
    return out


def vec(f):
    lg = lambda x: math.log1p(max(x, 0))
    return [1.0, lg(f["start"]), lg(f["baseline"]), lg(f["turn"]), lg(f["prev"]), lg(f["own"]), 1.0 if f["prev"] else 0.0]


def fit(rows):
    import numpy as np
    x = np.array([vec(f) for _s, f, _g in rows])
    y = np.array([math.log1p(g) for _s, _f, g in rows])
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    return [float(v) for v in w]


def predict(w, f):
    return max(0.0, math.expm1(sum(a * b for a, b in zip(w, vec(f)))))


def score(pairs):
    """(the median factor a prediction is off by, the share within a factor of two)."""
    off = sorted(max((p + 1) / (g + 1), (g + 1) / (p + 1)) for p, g in pairs)
    return round(off[len(off) // 2], 2) if off else 0, round(sum(o <= 2 for o in off) / max(1, len(off)), 3)


def run(folder, limit=LIMIT):
    rows = rows_of(folder)
    sessions = list(dict.fromkeys(r[0] for r in rows))
    cut = set(sessions[int(len(sessions) * (1 - HOLDOUT)):])
    train, held = [r for r in rows if r[0] not in cut], [r for r in rows if r[0] in cut]
    w = fit(train)
    overall = median([g for _s, _f, g in train])
    res = {"sessions": len(sessions), "turns": len(rows), "train": len(train), "held_out": len(held),
           "weights": dict(zip(("bias", "context so far", "baseline", "turn", "previous turn", "own median", "has previous"),
                               [round(v, 3) for v in w])),
           "model": score([(predict(w, f), g) for _s, f, g in held]),
           "all-turns median": score([(overall, g) for _s, _f, g in held]),
           "own median so far": score([(f["own"] or overall, g) for _s, f, g in held])}
    # what it is for: at each held-out turn, turns left before the limit by the model against what really happened
    left = []
    by = {}
    for s, f, g in held:
        by.setdefault(s, []).append((f, g))
    for s, seq in by.items():
        for i, (f, _g) in enumerate(seq):
            room = limit - f["start"]
            actual, total = 0, 0
            for _f2, g2 in seq[i:]:
                if total + g2 > room:
                    break
                total, actual = total + g2, actual + 1
            else:
                continue                       # the session ended before the limit: nothing to compare with
            left.append((room / max(1.0, predict(w, f)), actual))
    res["turns_left"] = {"cases": len(left), "score": score(left)}
    return res


def report(res, path):
    L = ["# context_growth: how many tokens the next turn adds", "",
         f"{res['turns']:,} growing turns of {res['sessions']} sessions; the newest sessions' {res['held_out']:,} turns held out.", "",
         "| predicts a turn's growth by | off by (median factor) | within a factor of two |", "|---|---|---|"]
    for k in ("model", "own median so far", "all-turns median"):
        L.append(f"| {k} | {res[k][0]} | {res[k][1]:.0%} |")
    t = res["turns_left"]
    L += ["", f"Turns left before the limit, where a held-out session reached it ({t['cases']} cases): off by a median factor "
          f"of {t['score'][0]}, within a factor of two in {t['score'][1]:.0%}.", "",
          "Weights (on logarithms): " + ", ".join(f"{k} {v}" for k, v in res["weights"].items())]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument("--work", default=str(HERE / "out"))
    a = ap.parse_args(argv)
    res = run(a.sessions, a.limit)
    out = Path(a.work) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(report(res, out / "report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
