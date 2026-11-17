#!/usr/bin/env python3
"""measure.py: the free, offline numbers an upgrade is judged by, from one copy of the plugin's scripts.

  python3 -I measure.py --scripts DIR --p4 STORE --suite TASKS.jsonl [--mac STORE] [--now T] [--no-loo]

Prints one JSON line: {"silent_heldout": share of the suite's held-out (test-split) general prompts the trained
ranker prints nothing for, "silent_all": the same over every general prompt (the train split is fitted when the
scripts train on it), "r_precision", "precision_lines" (hits over printed lines, leave-one-task-out),
"precision_task" (train.py's own mean), "hits", "wrong", "threshold", "regressions": the two ap-p4_mac queries}.
The model is what train.run_round would write: train.build_sets, train.fit_sets, ranker.best_threshold.
Standard library only; nothing is written outside the store copies given."""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--p4", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--mac", default="")
    ap.add_argument("--now", default="2026-11-01 18:00")
    ap.add_argument("--no-loo", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.scripts))
    import ranker
    import store as st
    import train
    import p4 as cli

    idx, _ = st.ensure_index(a.p4)
    sets = train.build_sets(a.p4, idx, now=a.now) if "now" in train.build_sets.__code__.co_varnames \
        else train.build_sets(a.p4, idx)
    live = [s for s in sets if not s["missed"]]
    labelled = [s for s in live if s.get("source") != "silent"]
    w = train.fit_sets(live)
    t = ranker.best_threshold([(s["files"], s["ys"], train.scored(w, s)) for s in labelled], 2, train.MIN_PRECISION)
    model = {"kind": ranker.KIND, "w": w, "threshold": t, "per_file": 2, "features": ["base"]}

    rows = [json.loads(l) for l in open(a.suite, encoding="utf-8") if l.strip()]
    loud = {"test": [0, 0], "all": [0, 0]}
    for r in rows:
        lines = cli.query(a.p4, r["prompt"], None, a.now, idx=idx, model=model)["lines"]
        for k in (("test",) if r.get("split") == "test" else ()) + ("all",):
            loud[k][0] += bool(lines)
            loud[k][1] += 1
    out = {"silent_heldout": 1 - loud["test"][0] / max(1, loud["test"][1]),
           "silent_all": 1 - loud["all"][0] / max(1, loud["all"][1]), "threshold": t,
           "labelled_tasks": len(labelled), "silence_tasks": len(live) - len(labelled)}
    if not a.no_loo:
        ev = train.evaluate(sets)
        hits = sum(m["precision"] * m["picked"] for m in ev["rows"].values())
        picked = sum(m["picked"] for m in ev["rows"].values())
        out.update({"r_precision": ev["r-precision"], "precision_task": ev["precision"],
                    "precision_lines": hits / picked if picked else 0.0, "hits": hits, "wrong": picked - hits,
                    "silent_tasks_loo": sum(1 for m in ev["rows"].values() if not m["picked"]), "tasks": ev["n"]})
    if a.mac:
        midx, _ = st.ensure_index(a.mac)
        reg = {}
        for q, needles in (("which branch do mac commits go on", ("on this Mac the work is committed on _mac",
                                                                  "the Mac's branch is 0.0.1-apple&win&mac_mac")),
                           ("where does the p4 repo live", ("the p4 repo is nested in the project",))):
            lines = cli.query(a.mac, q, None, a.now, idx=midx, model=None)["lines"]
            reg[q] = any(n in l for l in lines for n in needles)
        out["regressions"] = reg
    print(json.dumps(out))


if __name__ == "__main__":
    main()
