"""recall_levers: what moves p4-recall's JSON ranker on a store, measured with the plugin's own train.evaluate.

The JSON ranker is what recall serves; served_back measured it at 0.01-0.07 held-out r-precision on its store against
0.36-0.47 for a .pt ranker on the same labels. This finds where that gap comes from and which lever closes it, each
variant evaluated exactly as `train.py eval` does (grouped 5-fold by time, missed tasks counted as zero):

  candidates   how many positives the BM25 candidate stage draws at all (the ranker can't rank what it never sees)
  features     the base feature set with groups removed: the per-file one-hot (`file=`), kind/state one-hots, age
  weights      the hand-set label weights (repo cited 0.7, its-last-commit 0.35, dropped 0.3 as a stated negative,
               usage 0.3; none measured): stated negatives off, commit labels off, usage off and up, all uniform
  curve        held-out numbers against the number of labelled tasks (3 draws per size): is it short of tasks?

Works on a copy of the store (the index is rebuilt in place). Results: out/levers-<name>.json.

    python -I experiments/recall_levers/recall_levers.py --store <a copy of a .claude/p4 store>
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent.parent / "p4" / "lbiza" / "plugins" / "p4-recall" / "scripts"
sys.path.insert(0, str(PLUGIN))
import labels as lb  # noqa: E402
import ranker  # noqa: E402
import store as st  # noqa: E402
import train  # noqa: E402

KEYS = ("n", "missed", "r-precision", "recall@20", "precision", "f1", "baseline", "top5")


def short(ev):
    if not ev:
        return None
    out = {k: (round(ev[k], 4) if isinstance(ev[k], float) else ev[k]) for k in KEYS if k in ev}
    rows = ev.get("rows", {}).values()
    # the same numbers over the drawn tasks only: what the ranker does with what it was handed
    if rows:
        out["drawn_r-precision"] = round(sum(m["r-precision"] for m in rows) / len(rows), 4)
        out["drawn_baseline"] = round(sum(m["baseline"] for m in rows) / len(rows), 4)
    return out


def strip_features(groups):
    """Wrap ranker.candidate_features so the named one-hot / numeric groups are left out of every candidate."""
    orig = ranker.candidate_features

    def wrapped(task_terms, cands, idf, avgdl, now, scores=None, weights=None, extra=None):
        xs, sc, sh = orig(task_terms, cands, idf, avgdl, now, scores, weights, extra)
        for x, c in zip(xs, cands):
            drop = []
            if "file=" in groups:
                drop.append(ranker.h("file=" + c["file"]))
            if "kind=" in groups:
                drop.append(ranker.h("kind=" + (c.get("fkind") or c["kind"])))
            if "state=" in groups:
                drop.append(ranker.h("state=" + c.get("state", "LIVE")))
            if "age" in groups:
                drop += [ranker.h("age"), ranker.h("recent"), ranker.h("no-time")]
            if "bm25" in groups:
                drop += [ranker.h(k) for k in ("bm25", "bm25-rank", "head-bm25", "top10", "overlap", "rare")]
            for k in drop:
                x.pop(k, None)
        return xs, sc, sh
    return orig, wrapped


def reweight(rows, rule):
    out = []
    for r in rows:
        r = dict(r)
        note, src = r.get("note", ""), r["source"]
        w = rule(r, note, src)
        if w is None:
            continue
        r["weight"] = w
        out.append(r)
    return out


WEIGHTS = {
    "as shipped": lambda r, n, s: r["weight"],
    "no stated negatives": lambda r, n, s: None if not r["label"] else r["weight"],
    "no commit labels": lambda r, n, s: None if "last commit" in n else r["weight"],
    "no usage labels": lambda r, n, s: None if s == "usage" else r["weight"],
    "usage x3 (0.9)": lambda r, n, s: r["weight"] * 3 if s == "usage" else r["weight"],
    "uniform 1.0": lambda r, n, s: 1.0,
    "cited only positives": lambda r, n, s: r["weight"] if (not r["label"] or n.startswith("cited") or n.startswith("edited")) else None,
    "edited only positives": lambda r, n, s: r["weight"] if (not r["label"] or n.startswith("edited")) else None,
}


def run(p4, out_path, draws):
    report = {"store": p4}
    t0 = time.time()
    idx, _ = st.ensure_index(p4)
    rows = lb.training_labels(p4, idx)
    report["labels"] = len(rows)
    report["by_note"] = {}
    for r in rows:
        k = f"{r['source']}:{r['label']}:{r.get('note', '').split(' (')[0]}"
        report["by_note"][k] = report["by_note"].get(k, 0) + 1

    def ev(label, **kw):
        t = time.time()
        sets = train.build_sets(p4, idx, rows=kw.get("rows", rows))
        e = train.evaluate(sets)
        res = short(e)
        report.setdefault("variants", {})[label] = res
        print(f"{label:<34} {res} ({time.time() - t:.0f} s)", flush=True)
        Path(out_path).write_text(json.dumps(report, indent=1))
        return sets, e

    # baseline, and how many positives the candidate stage draws at each width
    sets, base = ev("base (as shipped)")
    for c in (200, 1000, 5000):
        train.CANDIDATES = c
        ev(f"candidates {c}")
    train.CANDIDATES = 200

    # features
    for groups in (("file=",), ("kind=", "state="), ("age",), ("file=", "kind=", "state=", "age"), ("bm25",)):
        orig, wrapped = strip_features(set(groups))
        ranker.candidate_features = wrapped
        try:
            ev("without " + "+".join(groups))
        finally:
            ranker.candidate_features = orig

    # weights
    for name, rule in WEIGHTS.items():
        if name != "as shipped":
            ev("weights: " + name, rows=reweight(rows, rule))

    # learning curve over the live tasks
    live = [s for s in sets if not s["missed"] and s.get("source") != "silent"]
    rest = [s for s in sets if s["missed"] or s.get("source") == "silent"]
    curve = {}
    for n in (10, 20, 40, 60, len(live)):
        if n > len(live):
            continue
        got = []
        for d in range(draws if n < len(live) else 1):
            pick = random.Random(d).sample(live, n)
            missed_share = [s for s in rest if s["missed"]][: round(len(pick) * sum(s["missed"] for s in rest) / max(1, len(live)))]
            e = train.evaluate(pick + [s for s in rest if s.get("source") == "silent"] + missed_share)
            got.append(short(e))
        curve[n] = {k: round(sum(g[k] for g in got) / len(got), 4) for k in ("r-precision", "precision", "drawn_r-precision", "drawn_baseline")}
        print(f"curve {n}: {curve[n]}", flush=True)
        report["curve"] = curve
        Path(out_path).write_text(json.dumps(report, indent=1))
    report["seconds"] = round(time.time() - t0)
    Path(out_path).write_text(json.dumps(report, indent=1))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="a COPY of a .claude/p4 store")
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args(argv)
    p4 = str(Path(args.store).resolve())
    name = Path(p4).parent.parent.name if Path(p4).name == "p4" else Path(p4).name
    (HERE / "out").mkdir(exist_ok=True)
    run(p4, str(HERE / "out" / f"levers-{name}.json"), args.draws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
