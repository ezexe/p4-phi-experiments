#!/usr/bin/env python3
"""tune_fallback.py: choose the gate of p4-phi's retrieval road without a ranker, p4.py's FALLBACK constants, on a
held-out split of a store's labelled tasks.

It measures against the scripts of the p4 checkout beside this repository, at `p4/lbiza/plugins/p4-phi/scripts`, or
at the folder P4_SCRIPTS names, and it runs by hand, since it builds every store's index and replays every task:

    .venv/Scripts/python -X utf8 experiments/p4-suite/tune_fallback.py <store> [<store> ...] [--suite F] [--history N]

What should print: each store's pool tasks, the `pool` rows of labels-raw.tsv, and its history tasks, the completed
message rows whose reply shares rare words with a document, which labels.from_history labels and this script keeps in
memory. Each is a message with the documents that bore on it. What must stay silent: the general prompts of the
overrides suite's tasks, every category but `lessons`, which are messages a session would send that the store has
nothing for. Tasks are split by a hash of their id: the even half chooses the gate and the odd half reports it.

The gate, as p4.fallback_order holds it: a document prints when it shares at least MIN_RARE of the message's words,
covering `cover` of them and `icover` of their idf, where a word the index lacks counts at the highest idf, at `dense`
shared words per square root of its length; a file document also prints on its own gate of FILE_SHARED words,
FILE_COVER and FILE_DENSE. The query's dictionary and the session tree add documents this script leaves out.
The choice puts silence first, since a wrong line costs more than a missing one: among the gates that pass the
regression cases and leave every store's tuning-half prompts silent, those within a fifth of the best tuning F1, and
of those the one printing the fewest lines on the tuning half.

The stores are read and their index caches refreshed as a hook would; the history labels stay in memory.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.environ.get("P4_SCRIPTS") or os.path.join(HERE, "..", "..", "p4", "lbiza", "plugins", "p4-phi", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))
import labels as lb  # noqa: E402
import p4 as cli  # noqa: E402
import ranker  # noqa: E402
import store as st  # noqa: E402

GRID_COVER = (0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
GRID_ICOVER = (0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
GRID_DENSE = (0.0, 0.2, 0.35, 0.4, 0.45)
F1_TOLERANCE = 0.2            # a fifth of the best tuning F1 may be traded for fewer lines
# regression cases of the store ap-p4: a gate is eligible only when each message finds, among its top 5, a document
# whose text starts with one of the prefixes, and the ordinary messages print nothing. They constrain the choice and
# are in neither the tuning numbers nor the held-out ones.
MUST = (("which branch do mac commits go on", ("on this Mac the work is committed on _mac",
                                               "the Mac's branch is 0.0.1-apple&win&mac_mac")),
        ("where does the p4 repo live", ("the p4 repo is nested in the project",)))
MUST_SILENT = ("hi", "keep going", "thanks", "ok")


def half(tid):
    return int(hashlib.sha1(tid.encode()).hexdigest()[:4], 16) % 2


def candidates(idx, text):
    """[(doc index, shared, cover, icover, density)] in p4.fallback_order's candidate order, before its gate: the same
    BM25 over the message's words and their other forms, and the same counts the gate compares."""
    terms = list(dict.fromkeys(ranker.tokens(text)))
    if not terms or not cli.content_terms(idx, terms):
        return []
    docs, n = idx["docs"], idx["n"]
    q = cli.content_terms(idx, terms)
    words = cli.message_words(terms)
    forms = {t: [t] + cli.variants(idx, t) for t in q}
    w = {}
    for t in q:
        for v in forms[t][1:]:
            if v not in w and v not in terms:
                w[v] = cli.VARIANT_WEIGHT
    qterms = list(dict.fromkeys(list(terms) + list(w)))
    scores = cli.bm25_scores(idx, qterms, w)
    cand = sorted(scores, key=lambda i: (-scores[i], i))[:cli.CANDIDATES]
    idf = {v: ranker.idf_of(len(st.postings(idx, v) or []) // 2, n) for t in q for v in forms[t] if st.postings(idx, v)}
    unseen = ranker.idf_of(0, n)
    widf = {t: max(idf.get(v, 0.0) for v in forms[t]) if t in forms else unseen for t in words}
    total = sum(widf.values()) or 1.0
    out = []
    for i in cand:
        toks = set(docs[i]["toks"].split())
        got = [t for t in q if any(v in toks for v in forms[t])]
        out.append((i, len(got), len(got) / max(1, len(words)), sum(widf.get(t, 0.0) for t in got) / total,
                    len(got) / max(1.0, docs[i]["dl"]) ** 0.5))
    return out


def lines(cands, g, idx):
    """The document indices p4.fallback_order would print under the gate g = (cover, icover, dense): MAX_LINES at most
    and PER_FILE a file, a file document passing on its own gate too."""
    cover, icover, dense = g
    docs = idx["docs"]
    keep = [c for c in cands if c[1] >= cli.MIN_RARE and c[2] >= cover and c[3] >= icover and c[4] >= dense
            or docs[c[0]]["kind"] == "file" and c[1] >= cli.FILE_SHARED and c[2] >= cli.FILE_COVER
            and c[4] >= cli.FILE_DENSE]
    keep.sort(key=lambda c: -c[1])           # stable: BM25 within a tie
    out, per = [], {}
    for i in cli.standing_first(idx, [c[0] for c in keep]):
        path = idx["docs"][i]["path"]
        if per.get(path, 0) >= cli.PER_FILE:
            continue
        per[path] = per.get(path, 0) + 1
        out.append(i)
        if len(out) >= cli.MAX_LINES:
            break
    return out


def score(tasks, silent, g):
    hit = printed = want = 0
    found = 0
    for idx, cands, pos in tasks:
        got = lines(cands, g, idx)
        ids = {idx["docs"][i]["id"] for i in got}
        h = len(ids & pos)
        hit += h
        printed += len(got)
        want += min(5, len(pos))
        found += h > 0
    noisy = 0
    for idx, cands in silent:
        n = len(lines(cands, g, idx))
        printed += n
        noisy += n > 0
    p = hit / printed if printed else 0.0
    r = hit / want if want else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "found": found / max(1, len(tasks)), "silent_broken": noisy / max(1, len(silent)), "printed": printed}


def silent_by_store(silent, g):
    """{store index id: share of its prompts that printed nothing}."""
    out = {}
    for idx, cands in silent:
        a = out.setdefault(id(idx), [0, 0])
        a[0] += not lines(cands, g, idx)
        a[1] += 1
    return {k: a / b for k, (a, b) in out.items()}


def regressions_ok(idx_cands, g):
    for (idx, cands, prefixes) in idx_cands["must"]:
        shown = [idx["docs"][i]["show"] for i in lines(cands, g, idx)]
        if not any(x.startswith(p) for x in shown for p in prefixes):
            return False
    return all(not lines(c, g, i) for i, c in idx_cands["silent"])


def regression_data(p4_dirs):
    out = {"must": [], "silent": []}
    for p4 in p4_dirs:
        idx, _ = st.ensure_index(p4)
        if not any(d["show"].startswith(MUST[0][1][0]) for d in idx["docs"]):
            continue
        for text, prefixes in MUST:
            out["must"].append((idx, candidates(idx, text), prefixes))
        for text in MUST_SILENT:
            out["silent"].append((idx, candidates(idx, text)))
    return out


def load(p4_dirs, suite, n_history):
    by_half = {0: ([], []), 1: ([], [])}
    prompts = []
    if suite and os.path.exists(suite):
        with open(suite, encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                if d.get("category") != "lessons":
                    prompts.append((d["id"], d["prompt"]))
    for p4 in p4_dirs:
        idx, _ = st.ensure_index(p4)
        files = st.load_store(p4)
        tasks = lb.read_tasks(p4)
        pos = {}
        for r in lb.read_tsv(lb.mpath(p4, lb.RAW)):
            if r.get("source") == "pool" and r.get("label") == "1":
                pos.setdefault(r["task"], set()).add(r["entry"])
        texts = {t: tasks[t]["text"] for t in pos if t in tasks}
        hist = history_in_memory(p4, idx, files, n_history)
        for t, (text, ents) in hist.items():
            texts[t], pos[t] = text, ents
        ids = {d["id"] for d in idx["docs"]}
        for t, text in texts.items():
            ps = pos[t] & ids
            if ps:
                by_half[half(t)][0].append((idx, candidates(idx, text), ps))
        for pid, text in prompts:
            by_half[half(pid)][1].append((idx, candidates(idx, text)))
    return by_half


def history_in_memory(p4, idx, files, limit):
    """{row id: (fingerprint, {document ids})}: the labels labels.from_history would write, kept in memory."""
    tmp = {}
    real_write = lb.write_tsv
    real_add = lb.add_tasks
    got = {}

    def cap_write(path, head, rows):
        for r in rows:
            if r.get("source") == "store":
                got.setdefault(r["task"], set()).add(r["entry"])

    def cap_add(p4_, rows):
        for r in rows:
            tmp[r["id"]] = r["text"]
        return 0
    lb.write_tsv, lb.add_tasks = cap_write, cap_add
    try:
        lb.from_history(p4, idx=idx, files=files)
    finally:
        lb.write_tsv, lb.add_tasks = real_write, real_add
    keys = sorted(got)[-limit:] if limit else sorted(got)
    return {k: (tmp.get(k, ""), got[k]) for k in keys if tmp.get(k)}


def choose(tune, reg):
    """The gate the module docstring's rule chooses, or None when no gate passes the regression cases."""
    ok = []
    for g in itertools.product(GRID_COVER, GRID_ICOVER, GRID_DENSE):
        if reg["must"] and not regressions_ok(reg, g):
            continue
        if min(silent_by_store(tune[1], g).values() or [1.0]) < 1.0:
            continue
        ok.append((score(tune[0], tune[1], g), g))
    if not ok:
        return None
    best = max(m["f1"] for m, _ in ok)
    near = [(m, g) for m, g in ok if m["f1"] >= best * (1 - F1_TOLERANCE) - 1e-12]
    return min(near, key=lambda x: (x[0]["printed"], -x[0]["f1"], x[1]))[1]


def report(name, data, reg, g):
    both = data[0][1] + data[1][1]
    held = score(data[1][0], data[1][1], g)
    print(f"  {name}: held " + json.dumps({k: round(v, 3) for k, v in held.items()}))
    print(f"    every suite prompt silent, by store: "
          + ", ".join(f"{v:.0%}" for v in silent_by_store(both, g).values())
          + (f"; regression cases pass: {regressions_ok(reg, g)}" if reg["must"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("p4", nargs="+")
    ap.add_argument("--suite", default=os.path.join(HERE, "..", "..", "p4", "scripts", "session",
                                                    "overrides_tasks.jsonl"))
    ap.add_argument("--history", type=int, default=300)
    a = ap.parse_args(argv)
    data = load(a.p4, a.suite, a.history)
    tune, held = data[0], data[1]
    reg = regression_data(a.p4)
    g = choose(tune, reg)
    print(f"regression cases: {len(reg['must'])} must recall, {len(reg['silent'])} must stay silent")
    print(f"tuning half: {len(tune[0])} tasks, {len(tune[1])} silent prompts; held-out half: {len(held[0])} tasks, "
          f"{len(held[1])} silent prompts")
    if g is None:
        print("no gate passes the regression cases with the tuning half silent")
        return 1
    print(f"chosen gate: cover {g[0]}, icover {g[1]}, dense {g[2]}; in p4.py: cover {cli.FALLBACK_COVER}, icover "
          f"{cli.FALLBACK_ICOVER}, dense {cli.FALLBACK_DENSE}")
    print("  tuning  " + json.dumps({k: round(v, 3) for k, v in score(tune[0], tune[1], g).items()}))
    report("chosen", data, reg, g)
    report("in p4.py", data, reg, (cli.FALLBACK_COVER, cli.FALLBACK_ICOVER, cli.FALLBACK_DENSE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
