#!/usr/bin/env python3
"""node_eval.py: the retrieval lines with and without the session's place in the session tree, replayed on a store's
labelled tasks.

It measures p4-phi's `recall.node.on` against the scripts of the p4 checkout beside this repository, at
`p4/lbiza/plugins/p4-phi/scripts`, or at the folder P4_SCRIPTS names.
A replay takes minutes on a store of a few hundred labelled tasks, so it runs by hand and never in the plugin's tests:

    .venv/Scripts/python -X utf8 experiments/p4-suite/node_eval.py --p4 <store> [--json out.json] [--train]

Every task of models/tasks.tsv whose session the app's records place under another session is replayed at its own
time, with the entries written after it hidden, four ways:

  today   the message alone, as the hook runs now
  words   the message with the words of the session's own task and of the tasks above it, at NODE_WEIGHT
  used    the message, with the entries the tasks above it were labelled positive with as candidates of a one-word gate
  both    words and used together

For each way it counts the tasks, the share of tasks with no line, the lines printed, the share of those lines a label
marks positive for the task, the tasks with at least one positive line, and the characters.
It counts twice: every line, and the lines the store's inject mode lets through, which is every line when p4.json
`recall.inject` is true and otherwise the rule and fact lines that `rules-facts` shows.
A label written after the task is never read for its place in the tree, and the task's own labels are the answer key.
Nothing is written to the store but the index cache a hook would refresh anyway.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.environ.get("P4_SCRIPTS") or os.path.join(HERE, "..", "..", "p4", "lbiza", "plugins", "p4-phi", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))
import node as nd  # noqa: E402
import p4 as cli  # noqa: E402
import p4conf  # noqa: E402
import store as st  # noqa: E402

RULES_FACTS = re.compile(r"^- \S+ (rule|fact): ")
ARMS = ("today", "words", "used", "both")


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run(p4, records_dirs=None, limit=None):
    models = p4conf.sub(p4, "models")
    tasks = read_tsv(os.path.join(models, "tasks.tsv"))
    labels = read_tsv(os.path.join(models, "labels.tsv"))
    tree = nd.record_tree(os.path.join(models, "tasks.tsv"), records_dirs)
    phi = nd.phi_tree(p4)                             # the live notes and tasks p4-phi keeps
    for sid, s in phi.sessions.items():
        tree.add(sid, s["parent"], s["cli"], s["created"], s["tasks"], s["notes"])
    tree.finish()
    pos, by_session = {}, {}
    task_session = {t["id"]: tree.session_of(t["session"]) for t in tasks}
    for r in labels:
        if r.get("label") != "1":
            continue
        pos.setdefault(r["task"], set()).add(r["entry"])
        sid = task_session.get(r["task"])
        if sid:
            by_session.setdefault(sid, []).append(((r.get("time") or "")[:16], r["entry"]))
    idx, _ = st.ensure_index(p4)
    docs = idx["docs"]
    times = [(d.get("time") or "")[:16] for d in docs]
    mode = "all" if p4conf.section(p4, "recall", write=False).get("inject") is True else "rules-facts"
    rows = []
    picked = [t for t in tasks if (n := tree.at(t["session"], (t.get("time") or "")[:16])) is not None and n.up]
    for t in picked[:limit]:
        when = (t.get("time") or "")[:16]
        node = tree.at(t["session"], when)
        hide = {docs[i]["id"] for i, w in enumerate(times) if w and when and w > when}
        used = {e for sid, _ in node.up for w, e in by_session.get(sid, []) if not w or w <= when}
        answer = pos.get(t["id"], set())
        row = {"task": t["id"], "positives": len(answer), "up": len(node.up), "used": len(used)}
        for arm in ARMS:
            r = cli.query(p4, t["text"], (t.get("session") or "")[:8], when, idx=idx, model={},
                          node=node if arm in ("words", "both") else None,
                          node_ids=used if arm in ("used", "both") else None, hide=hide)
            lines = r["lines"]
            pairs = [(l, i) for l, i in zip(r["lines"], r["ids"])]
            ids = [i for _, i in pairs]
            shown = [(l, i) for l, i in pairs if mode == "all" or RULES_FACTS.match(l)]
            row[arm] = {"lines": len(lines), "ids": ids, "hits": sum(1 for i in ids if i in answer),
                        "chars": sum(len(l) + 1 for l in lines),
                        "shown": len(shown), "shown_hits": sum(1 for _, i in shown if i in answer),
                        "shown_chars": sum(len(l) + 1 for l, _ in shown)}
        rows.append(row)
    return rows, mode


def summary(rows, labelled_only=False):
    rs = [r for r in rows if r["positives"]] if labelled_only else rows
    out = {"tasks": len(rs)}
    for arm in ARMS:
        for pre in ("", "shown_"):
            k = "lines" if not pre else "shown"
            lines = sum(r[arm][k] for r in rs)
            hits = sum(r[arm][pre + "hits"] for r in rs)
            out[f"{arm}/{pre or 'all_'}silent"] = round(sum(1 for r in rs if not r[arm][k]) / max(1, len(rs)), 3)
            out[f"{arm}/{pre or 'all_'}lines"] = lines
            out[f"{arm}/{pre or 'all_'}precision"] = round(hits / lines, 3) if lines else None
            out[f"{arm}/{pre or 'all_'}hit_tasks"] = sum(1 for r in rs if r[arm][pre + "hits"])
            out[f"{arm}/{pre or 'all_'}chars_per_task"] = round(sum(r[arm][pre + "chars"] if pre else r[arm]["chars"]
                                                                  for r in rs) / max(1, len(rs)), 1)
    return out


def table(s):
    head = f"{'arm':6} {'lines':>9} {'silent':>8} {'precision':>10} {'hit tasks':>10} {'chars/task':>11}"
    out = [f"tasks: {s['tasks']}", "every line:", head]
    for pre, title in (("all_", None), ("shown_", "lines the inject mode lets through:")):
        if title:
            out += [title, head]
        for arm in ARMS:
            p = s[f"{arm}/{pre}precision"]
            out.append(f"{arm:6} {s[f'{arm}/{pre}lines']:>9} {s[f'{arm}/{pre}silent']:>8} "
                       f"{'-' if p is None else p:>10} {s[f'{arm}/{pre}hit_tasks']:>10} {s[f'{arm}/{pre}chars_per_task']:>11}")
    return "\n".join(out)


def trained(p4, records_dirs=None):
    """The ranker's held-out numbers (train.evaluate, nothing written) with and without the node feature, over every
    labelled task and over the tasks of sessions that have a parent."""
    import labels as lb
    import train
    train.configure(p4)
    idx, _ = st.ensure_index(p4)
    tasks = lb.read_tasks(p4)
    tree = nd.record_tree(None, records_dirs)
    child = {tid for tid, t in tasks.items()
             if (sid := tree.session_of(t.get("session") or "")) and tree.sessions[sid]["parent"]}
    out = {}
    for name, on in (("today", False), ("node", True)):
        sets = train.build_sets(p4, idx, node=on)
        ev = train.evaluate(sets)
        if ev is None:
            out[name] = None
            continue
        rows = ev.pop("rows")
        sub = [m for t, m in rows.items() if t in child]
        out[name] = {"all": {k: ev[k] for k in ("n", "missed", "r-precision", "recall@20", "precision", "top5",
                                                 "baseline", "threshold")},
                     "child": train.summarize(sub, 0) if sub else None,
                     "node_hits": sum(1 for s in sets if not s["missed"] and s.get("source") != "silent"
                                      and any(s["ys"][k] for k in range(len(s["ys"]))))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p4", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train", action="store_true", help="the trained ranker's held-out numbers with and without the node")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.train:
        got = trained(a.p4)
        print(json.dumps(got, indent=1))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(got, f, indent=1)
        print(f"{time.time() - t0:.0f}s")
        return 0
    rows, mode = run(a.p4, limit=a.limit)
    every, labelled = summary(rows), summary(rows, labelled_only=True)
    print(f"store {a.p4}; inject {mode}; {time.time() - t0:.0f}s")
    print("== every child-session task")
    print(table(every))
    print("== the child-session tasks with a positive label")
    print(table(labelled))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "every": every, "labelled": labelled, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
