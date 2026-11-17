#!/usr/bin/env python3
"""recall_contexts.py: the context files of the fresh validation's recall pathways (fresh-marketplace.md 8.2-8.4).

For each task of the given task files, `p4.py query` runs against the migrated store the task belongs to, and the
lines it prints become that task's context, written as `<pathway>.<task id>.md` for overrides_suite.py's
--context-dir. A task with no line gets no file, so the suite sends it exactly what the built-in gets and answers it
from the built-in's cache: silence costs nothing and changes nothing.

  recall           the recall lines alone (the hook's additionalContext)
  recall+pinned    the store's form: and pinned: rulings, newest first, at most 1,500 characters, under "Standing
                   rules from the owner:", then the recall lines
  emission-plain   the 10-line plain rewrite of the emission gate's testable rules (fresh/emission-plain.md), the
                   same for every task

Which store a task's query runs on:
  - a task with `from_store` (lesson, short-turn and newer tasks): that store;
  - a general task: each store in turn (`--general-repo all`, the default: one pathway per store, named
    `recall@<store>`, the store-bound tasks answering the same in each, so from the cache), or the one store named.
A short-turn task (category `short`) is queried on a held-out copy of its store: the held sessions' message rows and
every store entry dated at or after the held cut are left out, and only the rows of its own session before it are
added back, so the turn's own reading never reaches its context. A newer task's store copy carries the facts of
fresh/newer_facts.md in its facts.md (8.4: "the facts are entered in each fixture's facts.md").

Stores are given as NAME=PATH: an old layout (sessions/) is migrated, free, into --work; a store already on the new
layout (store/) is copied there. The sources are never written. `--train` runs one training round on each copy
first (free; the round is promoted only past the floors of 5.6), as the Stop hook would after migration.

  python3 scripts/session/recall_contexts.py --tasks scripts/session/overrides_tasks.jsonl,<lessons.jsonl> \\
      --out-dir <dir> --work <dir> [--stores NAME=PATH,...] [--general-repo all|NAME] [--train]
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLUGIN = REPO / "lbiza" / "plugins" / "p4-recall" / "scripts"
FRESH = HERE / "fresh"
FIXTURES = os.environ.get("P4_FIXTURES", "/Users/ez/gritz/p4/.claude/p4/scratchpad/.out/p4-colab/overrides-1-work/"
                                         "stores")
STORE_NAMES = ("AudioPlugin-p4", "ap-p4_mac", "p4-p4", "phi-p4")
PINNED_CHARS = 1500
PINNED_HEAD = "Standing rules from the owner:"
COPY_SUBS = ("store", "memory", "src", "models")
SKIP = ("index.json", "index-cache.tmp", "served", "*.tmp")

if str(PLUGIN) not in sys.path:
    sys.path.insert(0, str(PLUGIN))


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def default_stores():
    return {n: os.path.join(FIXTURES, n) for n in STORE_NAMES if os.path.isdir(os.path.join(FIXTURES, n))}


def parse_stores(text):
    if not text:
        return default_stores()
    out = {}
    for part in [p for p in text.split(",") if p]:
        name, _, path = part.partition("=")
        out[name] = path
    return out


def prepare(name, src, work, now, fresh=False):
    """The new-layout copy of store `name` under `work`, migrated (old layout) or copied (new layout) once."""
    import migrate
    dest = Path(work) / name
    if dest.joinpath("store").is_dir() and not fresh:
        return str(dest)
    if dest.exists():
        shutil.rmtree(dest)
    if os.path.isdir(os.path.join(src, "sessions")):
        dest.parent.mkdir(parents=True, exist_ok=True)
        migrate.run(src, str(dest), apply=True, now=now, report_path=str(Path(work) / f"migration-{name}.md"))
    elif os.path.isdir(os.path.join(src, "store")):
        for sub in COPY_SUBS:
            if os.path.isdir(os.path.join(src, sub)):
                shutil.copytree(os.path.join(src, sub), dest / sub, ignore=shutil.ignore_patterns(*SKIP))
    else:
        raise SystemExit(f"{name}: {src} is neither an old layout (sessions/) nor a new one (store/)")
    return str(dest)


def copy_of(src, dest):
    """A throwaway copy of a prepared store (its index left behind, so it is built for the copy)."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    for sub in COPY_SUBS:
        if os.path.isdir(os.path.join(src, sub)):
            shutil.copytree(os.path.join(src, sub), os.path.join(dest, sub), ignore=shutil.ignore_patterns(*SKIP))
    return dest


def newer_facts():
    """[(store name, task id, entry pairs)] from fresh/newer_facts.md."""
    import store as st
    text = (FRESH / "newer_facts.md").read_text(encoding="utf-8")
    out = []
    for e in st.parse(text, "facts.md"):
        pairs = [(k, v) for k, v in e.pairs if k not in ("store", "task")]
        out.append((e.fields.get("store", ""), e.fields.get("task", ""), pairs))
    return out


def with_newer_facts(src, dest, name):
    """A copy of the prepared store with its newer facts appended to store/facts.md."""
    import store as st
    copy_of(src, dest)
    mine = [p for s, _t, p in newer_facts() if s == name]
    if mine:
        path = os.path.join(dest, "store", "facts.md")
        have = [list(e.pairs) for e in st.load_file(dest, "facts.md")]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(st.render("facts.md", have + mine))
    return dest


def held_copy(src, dest, held_sessions, cut):
    """A copy of the prepared store as it stood before the held-out turns: the held sessions' message files gone and
    every entry and row dated at or after `cut` left out (8.3: the newest sessions' short rows, held out)."""
    import store as st
    copy_of(src, dest)
    for s8 in held_sessions:
        p = os.path.join(dest, "store", st.MESSAGES, f"{s8}.md")
        if os.path.exists(p):
            os.remove(p)
    for fname in list(st.STORE_FILES) + st.message_files(dest):
        ents = st.load_file(dest, fname)
        keep = [list(e.pairs) for e in ents if not e.time or e.time[:16] < cut[:16]]
        if len(keep) != len(ents):
            with open(os.path.join(dest, "store", fname), "w", encoding="utf-8", newline="\n") as f:
                f.write(st.render(fname, keep))
    return dest


def pinned_block(p4):
    """The store's form: and pinned: rulings, LIVE, newest first, within PINNED_CHARS (8.2's =recall+pinned)."""
    import store as st
    rules = [e for e in st.load_store(p4).get("rules.md", []) if e.state == "LIVE" and
             (e.fields.get("form", "").strip() or e.fields.get("pinned", "").strip().lower() in ("yes", "true", "1"))]
    rules.sort(key=lambda e: e.time or "", reverse=True)
    out = PINNED_HEAD
    for e in rules:
        line = "\n- " + st.collapse(e.fields.get("plain") or e.value)
        if len(out) + len(line) > PINNED_CHARS:
            break
        out += line
    return out if "\n" in out else ""


def query_lines(p4, idx, text, session=None, now=None, extra_rows=()):
    """p4.query's lines for one message; `extra_rows` ([time, fingerprint, addressed]) are this session's earlier
    rows, added to the index's short-turn and session tables for this query alone."""
    import p4 as cli
    import store as st
    added = []
    if extra_rows and session:
        rows = idx["sessions"].setdefault(session, [])
        base = len(rows)
        for t, fp, addressed in extra_rows:
            tok = st.norm_token(fp)
            rows.append([t, tok[:300], 1 if addressed else 0, addressed[:160]])
            if tok and len(tok.split()) <= st.SHORT_WORDS:
                idx["short"].setdefault(tok, []).append([t, session, 1 if addressed else 0, addressed[:300], "", 0])
                added.append(tok)
    try:
        r = cli.query(p4, text, session, now, idx=idx)
    finally:
        if extra_rows and session:
            del idx["sessions"][session][base:]
            if not idx["sessions"][session]:
                del idx["sessions"][session]
            for tok in added:
                idx["short"][tok].pop()
                if not idx["short"][tok]:
                    del idx["short"][tok]
    return r


def build(tasks, stores, out_dir, work, general="all", now=None, train=False, fresh=False):
    """Write the context files; returns the manifest {pathways, tasks, stores}."""
    import store as st
    now = now or time.strftime("%Y-%m-%d %H:%M")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.md"):
        f.unlink()
    work = Path(work)
    prepared = {n: prepare(n, p, work / "stores", now, fresh) for n, p in stores.items()}
    if train:
        import train as tr
        for n, p in prepared.items():
            row = tr.run_round(p, now=now, quiet=True, dictionary=False)
            say(f"{n}: training round {row['promoted']} ({row['why']})")
    general_stores = list(prepared) if general == "all" else [general]
    for g in general_stores:
        if g not in prepared:
            raise SystemExit(f"--general-repo {g}: no such store (have {', '.join(prepared)})")
    suffix = len(general_stores) > 1

    def names(store_for_general):
        tag = f"@{store_for_general}" if suffix else ""
        return f"recall{tag}", f"recall+pinned{tag}"

    # the store each task is queried on, and the copy (held-out, newer facts, or the store itself)
    copies, idxs, pins = {}, {}, {}

    def handle(key, make):
        if key not in copies:
            copies[key] = make()
            idxs[key] = st.build_index(copies[key])
            pins[key] = pinned_block(copies[key])
        return copies[key], idxs[key], pins[key]

    manifest = {"now": now, "stores": prepared, "general": general_stores, "pathways": {}, "tasks": {}}
    shutil.copyfile(FRESH / "emission-plain.md", out / "emission-plain.md")
    manifest["pathways"]["=emission-plain"] = {"base": True}
    for t in tasks:
        cat = t["category"]
        # a store-bound task is queried once, on its store, and its lines go to every pathway; a general task is
        # queried on each general store, its lines going to that store's pathways
        runs = [(t["from_store"], general_stores)] if t.get("from_store") else [(g, [g]) for g in general_stores]
        for store, pathway_stores in runs:
            if store not in prepared:
                manifest["tasks"].setdefault(t["id"], {})[store] = {"missing_store": True}
                continue
            if cat == "short":
                cut = t["held_cut"]
                p4, idx, pin = handle(("held", store, cut), lambda: held_copy(
                    prepared[store], str(work / "held" / store), t["held_sessions"], cut))
                r = query_lines(p4, idx, t.get("message") or t["prompt"], t.get("session"), t.get("time") or now,
                                t.get("session_rows", ()))
            elif cat == "newer":
                p4, idx, pin = handle(("newer", store), lambda: with_newer_facts(
                    prepared[store], str(work / "newer" / store), store))
                r = query_lines(p4, idx, t["prompt"], None, now)
            else:
                p4, idx, pin = handle(("store", store), lambda: prepared[store])
                r = query_lines(p4, idx, t.get("message") or t["prompt"], None, now)
            lines = r["lines"]
            rec = "\n".join(lines)
            pinned = "\n\n".join(x for x in (pin, rec) if x)
            for g in pathway_stores:
                rec_name, pin_name = names(g)
                for n, text in ((rec_name, rec), (pin_name, pinned)):
                    if text:
                        (out / f"{n}.{t['id']}.md").write_text(text + "\n", encoding="utf-8")
                    m = manifest["pathways"].setdefault("=" + n, {"tasks": 0, "silent": 0, "lines": 0,
                                                                 "by_category": {}})
                    m["tasks"] += 1
                    m["silent"] += not text
                    m["lines"] += len(lines) if n == rec_name else 0
                    c = m["by_category"].setdefault(cat, [0, 0])
                    c[0] += 1
                    c[1] += not text
            manifest["tasks"].setdefault(t["id"], {})[store] = {"lines": len(lines), "road": r.get("road", ""),
                                                               "ids": r.get("ids", [])}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load(paths):
    out = []
    for p in [x for x in paths.split(",") if x]:
        if os.path.exists(p):
            out += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", required=True, help="task files, comma-separated (a missing file is skipped)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work", required=True, help="where the migrated store copies live")
    ap.add_argument("--stores", default="", help="NAME=PATH,...; default the four clones under P4_FIXTURES")
    ap.add_argument("--general-repo", default="all")
    ap.add_argument("--now", default="")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="migrate or copy the stores again")
    a = ap.parse_args(argv)
    tasks = load(a.tasks)
    m = build(tasks, parse_stores(a.stores), a.out_dir, a.work, a.general_repo, a.now or None, a.train, a.fresh)
    for n, p in sorted(m["pathways"].items()):
        if "tasks" in p:
            say(f"{n}: {p['tasks']} tasks, {p['silent']} silent, {p['lines']} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
