#!/usr/bin/env python3
"""
says_vs_does.py: what the p4-recall plugin and the phi models say about themselves, against what they do.

Every claim is a sentence a reader of the two codebases would take as fact: a module docstring, a store file's
`Fields:` header, a finding marked "shipped", a README line, a constant the design names. Each gets one mechanical
check against the code as it stands and, where the claim is about a store, against a real store checked out beside
this file (any branch of a p4-recall store; by default the worktree experiments/served_back left under its out/).
Two checks are live measurements rather than readings: recall's query latency against the 300 ms budget run-hook.sh
cites, and its silence on the plugin's own ordinary prompts.

Verdicts: HOLDS (the code or the data does what the sentence says), DRIFTED (it does not), PARTLY (true with a
qualification the sentence does not carry), UNTESTABLE (the check could not run here; the reason is given). The
report is results/claims.md: one row per claim with where it is said, what it says, what was found, and the verdict,
so a reader can see each sentence's standing without trusting either the docs or this file. Nothing here calls an
API or costs anything.

    python experiments/says_vs_does/says_vs_does.py [--store DIR]
    python experiments/says_vs_does/test_says_vs_does.py
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P4 = ROOT / "p4"
PHI = ROOT / "phi"
RECALL = P4 / "lbiza" / "plugins" / "p4-recall"
SCRIPTS = RECALL / "scripts"
DEFAULT_STORE = ROOT / "experiments" / "served_back" / "out" / "branches" / "win"
WORK = HERE / "out"
LATENCY_TASKS = 20                              # owner messages timed through recall's query
LATENCY_BUDGET_MS = 300                         # run-hook.sh: "the recall hook has a 300 ms budget"

HOLDS, DRIFTED, PARTLY, UNTESTABLE = "HOLDS", "DRIFTED", "PARTLY", "UNTESTABLE"
CLAIMS = []


def claim(id_, where, says, bar="says", fix=""):
    """Register a check: `where` is the file (and section) the sentence is in, `says` the sentence as a reader
    takes it. The function returns (verdict, what was found). `bar` is "says" for a sentence the docs make, or
    "better" for a target set above them: what the code should do so that it does better than it says, judged
    MET or NOT YET. `fix` names where a drifted "says" claim is resolved: "code" (make the code keep the
    promise) or "doc" (the promise was never worth keeping; say what is)."""
    def deco(fn):
        CLAIMS.append(SimpleNamespace(id=id_, where=where, says=says, check=fn, bar=bar, fix=fix))
        return fn
    return deco


MET, NOT_YET = "MET", "NOT YET"


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def mods():
    s = str(SCRIPTS)
    if s not in sys.path:
        sys.path.insert(0, s)
    import hook, labels, p4 as cli, p4conf, ranker, rows, store, train  # noqa: E401
    return SimpleNamespace(hook=hook, lb=labels, cli=cli, conf=p4conf, rk=ranker, rows=rows, st=store, train=train)


def phi():
    os.environ.setdefault("PHI_DEVICE", "cpu")
    import phi_model, phi_rank, phi_serve, phi_store  # noqa: E401
    return SimpleNamespace(model=phi_model, rank=phi_rank, serve=phi_serve, store=phi_store)


class Ctx:
    """What the checks read: the two checkouts and, when one is given, a store worktree."""

    def __init__(self, store=None):
        self.store = Path(store) if store else None
        self._files = None

    def has_store(self):
        return bool(self.store and (self.store / "store").is_dir())

    def files(self):
        if self._files is None:
            self._files = mods().st.load_store(str(self.store)) if self.has_store() else {}
        return self._files

    def need_store(self):
        if not self.has_store():
            raise LookupError("no store worktree given (--store), so a claim about the data cannot be checked")


# ---- p4-recall: the hooks ----
@claim("hooks-registered", "p4-recall/scripts/hook.py docstring; hooks/hooks.json",
       "The hooks are layout (SessionStart), recall (UserPromptSubmit), agent (PreToolUse Agent), close and train "
       "(Stop), subagent-stop (SubagentStop); subagent-start is registered only if the PreToolUse road fails.")
def _(ctx):
    hooks = json.loads(read(RECALL / "hooks" / "hooks.json"))["hooks"]
    got = {}
    for event, entries in hooks.items():
        for e in entries:
            for h in e["hooks"]:
                got[h["command"].split()[-1]] = (event, e.get("matcher", ""))
    want = {"layout": "SessionStart", "recall": "UserPromptSubmit", "agent": "PreToolUse", "close": "Stop",
            "train": "Stop", "subagent-stop": "SubagentStop"}
    wrong = {k: v for k, v in want.items() if got.get(k, ("",))[0] != v}
    extra = set(got) - set(want)
    found = f"registered: {', '.join(f'{k}@{v[0]}' for k, v in sorted(got.items()))}"
    if wrong or extra:
        return DRIFTED, found + f"; off: {wrong or ''} {('extra ' + str(extra)) if extra else ''}"
    return HOLDS, found + "; subagent-start absent, as said"


@claim("recall-max-lines", "hook.py docstring: 'recall ... additionalContext: 0-5 lines'; p4.py MAX_LINES; p4.json recall.query.max_lines",
       "Recall prints at most five lines for a message.")
def _(ctx):
    m = mods()
    conf = m.conf.DEFAULTS["recall"]["query"]["max_lines"]
    got = f"MAX_LINES={m.cli.MAX_LINES}, p4.json default max_lines={conf}"
    return (HOLDS if m.cli.MAX_LINES == 5 == conf else DRIFTED), got


@claim("pinned-chars", "hook.py docstring: 'Standing rules from the owner: (<= 1,500 characters)'",
       "The standing-rules block at session start is at most 1,500 characters.")
def _(ctx):
    m = mods()
    conf = m.conf.DEFAULTS["recall"]["pinned_chars"]
    return (HOLDS if m.hook.PINNED_CHARS == 1500 == conf else DRIFTED), f"PINNED_CHARS={m.hook.PINNED_CHARS}, p4.json default {conf}"


@claim("recall-latency", "hooks/run-hook.sh: 'the recall hook has a 300 ms budget (fresh-marketplace.md 4.5)'",
       "A recall query answers within 300 ms.")
def _(ctx):
    ctx.need_store()
    m = mods()
    p4 = str(ctx.store)
    tasks = [t for t in m.lb.read_tasks(p4).values() if t.get("kind") == "new" and not t["text"].lstrip().startswith("<")]
    tasks = sorted(tasks, key=lambda t: t["time"], reverse=True)[:LATENCY_TASKS]
    if not tasks:
        return UNTESTABLE, "no owner tasks in the store"
    idx, _ = m.st.ensure_index(p4)                 # the index as the hook finds it: built once, then read
    times = []
    for t in tasks:
        t0 = time.perf_counter()
        m.cli.query(p4, t["text"], t["session"], now=t["time"], idx=idx)
        times.append((time.perf_counter() - t0) * 1000)
    med, mx = statistics.median(times), max(times)
    found = f"{len(times)} queries, in-process with the index loaded: median {med:.0f} ms, max {mx:.0f} ms (the hook adds a python start and the index load)"
    return (HOLDS if mx <= LATENCY_BUDGET_MS else PARTLY if med <= LATENCY_BUDGET_MS else DRIFTED), found


@claim("recall-silence", "p4/.docs/p4/findings.md 30: 'training on silence tasks takes it to 92% silent'; precision-upgrades.md: 'held-out silent share >= 0.9'",
       "Recall stays silent on at least nine of ten ordinary prompts.")
def _(ctx):
    ctx.need_store()
    m = mods()
    p4 = str(ctx.store)
    prompts = [json.loads(l) for l in read(SCRIPTS / "silent-prompts.jsonl").splitlines() if l.strip()]
    idx, _ = m.st.ensure_index(p4)
    model = m.rk.load(os.path.join(p4, "models", "ranker.json"))
    loud = [p["id"] for p in prompts if m.cli.query(p4, p["prompt"], None, idx=idx, model=model)["lines"]]
    share = 1 - len(loud) / len(prompts)
    found = f"{len(prompts)} prompts, road {'ranker' if model else 'no ranker (fallback)'}: silent on {share:.0%}" + (f"; loud: {loud}" if loud else "")
    return (HOLDS if share >= 0.9 else DRIFTED), found


@claim("p4py-subcommands", "p4.py docstring (the subcommand list)",
       "p4.py's docstring lists exactly the subcommands its parser accepts.")
def _(ctx):
    src = read(SCRIPTS / "p4.py")
    doc = src.split('"""')[1]
    said = set(re.findall(r"^\s{2}([a-z-]+)\b", doc, re.M))
    parsed = set(re.findall(r'add_parser\("([a-z-]+)"', src))
    loop = re.search(r"for name in \(([^)]*)\):\s*\n\s*p = sub\.add_parser\(name\)", src)
    if loop:
        parsed |= set(re.findall(r'"([a-z-]+)"', loop.group(1)))
    src_names = set(re.findall(r'sub\.add_parser\(s', src))
    stubs = set(re.findall(r'for s in \(([^)]*)\):\s*\n\s*sub\.add_parser\(s', src))
    found = f"said {sorted(said)}; parsed {sorted(parsed)}"
    missing = said - parsed
    if missing and src_names:
        return PARTLY, found + f"; {sorted(missing)} parsed from a list the regex could not read ({stubs})"
    return (HOLDS if said <= parsed else DRIFTED), found + (f"; said but not parsed: {sorted(missing)}" if missing else "")


@claim("labels-columns", "labels.py docstring: 'labels.tsv ... columns: task, entry, label, source, weight, time, teacher, edited, note'",
       "labels.tsv has those nine columns, in the code and in the store.")
def _(ctx):
    m = mods()
    said = ["task", "entry", "label", "source", "weight", "time", "teacher", "edited", "note"]
    found = f"TRAIN_HEAD={m.lb.TRAIN_HEAD}"
    if ctx.has_store() and (ctx.store / "models" / "labels.tsv").exists():
        head = read(ctx.store / "models" / "labels.tsv").split("\n", 1)[0].split("\t")
        found += f"; store header {head}"
        return (HOLDS if head == said == m.lb.TRAIN_HEAD else DRIFTED), found
    return (HOLDS if m.lb.TRAIN_HEAD == said else DRIFTED), found


@claim("task-kinds", "labels.py docstring: 'tasks.tsv holds the tasks: id, session, time, text, kind (first | new | short | agent)'",
       "Tasks are of kind first, new, short or agent, and the code writes all four.", fix="code")
def _(ctx):
    src = read(SCRIPTS / "hook.py") + read(SCRIPTS / "labels.py") + read(SCRIPTS / "rows.py")
    written = {k for k in ("first", "new", "short", "agent") if re.search(rf'"{k}"', src)}   # each kind as a literal the scripts write
    found = f"kinds the code writes: {sorted(written)}"
    if ctx.has_store():
        kinds = Counter(t["kind"] for t in mods().lb.read_tasks(str(ctx.store)).values())
        found += f"; in the store: {dict(kinds)}"
    said = {"first", "new", "short", "agent"}
    never = said - written
    return (HOLDS if not never else PARTLY), found + (f"; never written by hook.py or labels.py: {sorted(never)}" if never else "")


@claim("usage-weight", "labels.py: 'USAGE_WEIGHT = 0.3 (until 8.1's agreement test raises it)'; train.py: 'usage 0.3'",
       "Usage labels weigh 0.3 pending an agreement test.")
def _(ctx):
    m = mods()
    src = read(SCRIPTS / "train.py")
    doc_says = "usage 0.3" in src
    found = f"USAGE_WEIGHT={m.lb.USAGE_WEIGHT}; train.py docstring names usage 0.3: {doc_says}; served_back measured the label's agreement with a judge at 0.45 over 20 rows"
    return (HOLDS if m.lb.USAGE_WEIGHT == 0.3 and doc_says else DRIFTED), found


@claim("row-chars", "rows.py docstring: 'fingerprint: (the message, quoted, at most 300 characters)' and 'addressed: (the reply's first paragraph, at most 300 characters)'",
       "A message row keeps at most 300 characters of the message and of the reply.")
def _(ctx):
    m = mods()
    found = f"FP_CHARS={m.rows.FP_CHARS}, ADDRESSED_CHARS={m.rows.ADDRESSED_CHARS}"
    if ctx.has_store():
        longest = 0
        for rel, es in ctx.files().items():
            if rel.startswith("messages/"):
                for e in es:
                    longest = max(longest, len(m.st.unquote(e.fields.get("fingerprint", ""))), len(e.fields.get("addressed", "")))
        found += f"; longest value in the store {longest}"
        if longest > 302:
            return PARTLY, found + " (migrated rows run longer than the hook writes)"
    return (HOLDS if m.rows.FP_CHARS == m.rows.ADDRESSED_CHARS == 300 else DRIFTED), found


def echo_writers():
    """Scripts (migrate.py aside) with a line that writes a state other than FRESH."""
    writers = []
    for p in sorted(SCRIPTS.glob("*.py")):
        if p.name.startswith("test_") or p.name == "migrate.py":
            continue
        for n, line in enumerate(read(p).splitlines(), 1):
            if re.search(r"(state:\s*(ECHO|SUPERSEDED)|\"(ECHO|SUPERSEDED)\")", line) and not line.strip().startswith(("#", '"""')) \
                    and re.search(r"write|append|lines\[|= *f?\"|return\s+\"(ECHO|SUPERSEDED)\"", line):
                writers.append(f"{p.name}:{n}")      # a write of the state, or a judge returning it for the writer
    return writers


@claim("echo-offline", "rows.py docstring (as it stood): 'ECHO and SUPERSEDED are judged offline, never here.'",
       "Some step of the plugin judges a row ECHO or SUPERSEDED.", fix="code")
def _(ctx):
    writers = echo_writers()
    found = "no script writes a state other than FRESH (migrate.py carries the old store's ECHO/SUPERSEDED rows over)" if not writers else f"writers: {writers}"
    if ctx.has_store():
        states = Counter((e.fields.get("state", "").split(" ") or [""])[0] for rel, es in ctx.files().items() if rel.startswith("messages/") for e in es)
        found += f"; states in the store: {dict(states)}"
    return (PARTLY if not writers else HOLDS), found + ("; 'judged offline' names no step that exists" if not writers else "")


@claim("store-fields", "store.py HEADERS: each store file's 'Fields:' line",
       "Each store file's entries carry only the fields its header names.")
def _(ctx):
    ctx.need_store()
    m = mods()
    out, drift = [], False
    for fname, header in m.st.HEADERS.items():
        said = set(re.findall(r"[a-z][a-z-]*", re.sub(r"\([^)]*\)", "", header[2].split(":", 1)[1])))   # a parenthetical qualifies, never names
        entries = ctx.files().get(fname, []) if fname != m.st.MESSAGES else [e for rel, es in ctx.files().items() if rel.startswith("messages/") for e in es]
        if not entries:
            continue
        got = Counter(k for e in entries for k, _v in e.pairs)
        extra = {k: n for k, n in got.items() if k not in said}
        if extra:
            drift = True
        out.append(f"{fname}: {len(entries)} entries" + (f", fields not in the header: {extra}" if extra else ", all fields named"))
    return (DRIFTED if drift else HOLDS), "; ".join(out)


@claim("entry-ids", "store.py docstring: 'An entry's id is <file>:<time>:<sha8 of its first field's text> ... a second entry with the same id in one file gets ~2'",
       "Entry ids are file, time and the head's sha8, with ~n on a collision.")
def _(ctx):
    ctx.need_store()
    m = mods()
    n, bad, tilde = 0, 0, 0
    for rel, es in ctx.files().items():
        for e in es:
            n += 1
            tilde += "~" in e.id
            base = f"{rel}:{e.time}:{m.st.sha8(e.value)}"
            if not e.id.startswith(base):
                bad += 1
    return (HOLDS if not bad else DRIFTED), f"{n} entries, {bad} ids off the rule, {tilde} with a ~n suffix"


@claim("seeded-gitignore", "hook.py GITIGNORE: 'seeded once by p4-recall; your edits win' (models/served/, models/index.json, models/repo.json, models/*.tmp ...)",
       "A store's .gitignore holds the lines p4-recall seeds.", fix="code")
def _(ctx):
    ctx.need_store()
    m = mods()
    seeded = [l for l in m.hook.GITIGNORE.splitlines() if l and not l.startswith("#")]
    have = set(l.strip() for l in read(ctx.store / ".gitignore").splitlines()) if (ctx.store / ".gitignore").exists() else set()
    missing = [l for l in seeded if l not in have]
    return (HOLDS if not missing else PARTLY), f"{len(seeded)} seeded lines, {len(missing)} missing from the store's .gitignore{': ' + str(missing) if missing else ''}"


@claim("close-completes", "hook.py docstring: 'close (Stop): completes the row, writes the turn's usage and repo labels'",
       "Owner rows get completed and labels get written at turn close.")
def _(ctx):
    ctx.need_store()
    m = mods()
    rows = [e for rel, es in ctx.files().items() if rel.startswith("messages/") for e in es if m.st.owner_message(e)]
    done = sum(m.st.completed(e) for e in rows)
    labels = Counter(x.get("source") for x in m.lb.read_tsv(m.lb.mpath(str(ctx.store), m.lb.TRAIN)))
    found = f"{done} of {len(rows)} owner rows completed; labels by source {dict(labels)}"
    return (HOLDS if done and labels.get("usage") and labels.get("repo") else PARTLY), found


@claim("labels-raw-sources", "labels.py docstring: 'labels-raw.tsv: the store's own labels ... pool, store, short, miss'",
       "labels-raw.tsv exists with sources pool, store, short or miss.", fix="doc")
def _(ctx):
    ctx.need_store()
    m = mods()
    path = ctx.store / "models" / m.lb.RAW
    if not path.exists():
        return PARTLY, "no labels-raw.tsv in this store: `p4.py labels --from-history` was never run on it"
    srcs = Counter(x.get("source") for x in m.lb.read_tsv(str(path)))
    return (HOLDS if set(srcs) <= {"pool", "store", "short", "miss"} else DRIFTED), f"sources {dict(srcs)}"


@claim("train-if-owed", "train.py docstring: '--if-owed returns at once unless >= 10 labels landed since the last round and the last round is older than an hour'",
       "The owed thresholds are 10 labels and one hour.")
def _(ctx):
    t = mods().conf.DEFAULTS["recall"]["train"]
    return (HOLDS if t.get("owed_labels") == 10 and t.get("owed_age_s") == 3600 else DRIFTED), f"p4.json defaults owed_labels={t.get('owed_labels')}, owed_age_s={t.get('owed_age_s')}"


@claim("promotion-floor", "train.py: 'promote past the floor'; p4.json recall.train.min_precision",
       "A round is promoted only when its held-out precision at threshold reaches 0.6.")
def _(ctx):
    m = mods()
    floor = m.conf.DEFAULTS["recall"]["train"]["min_precision"]
    found = f"min_precision={floor}"
    if ctx.has_store() and (ctx.store / "models" / "eval.tsv").exists():
        evals = m.lb.read_tsv(str(ctx.store / "models" / "eval.tsv"))
        prom = Counter(e.get("promoted") for e in evals)
        found += f"; {len(evals)} rounds in the store, promoted: {dict(prom)}; last: precision {evals[-1].get('precision')}, {evals[-1].get('why')}"
    return (HOLDS if floor == 0.6 else DRIFTED), found


@claim("ranker-json-shared", "p4-recall/scripts/ranker.py: 'The model file is models/ranker.json {kind: p4 recall ranker ...}'; phi_rank: 'models/ranker.json names the anchor and the chain'",
       "In a p4-recall store, models/ranker.json has one owner.", fix="code")
def _(ctx):
    m, ph = mods(), phi()
    name = ph.rank.sidecar_name("/nonexistent") if hasattr(ph.rank, "sidecar_name") else ph.rank.SIDECAR
    in_store = ph.rank.sidecar_name(str(ctx.store)) if ctx.has_store() and hasattr(ph.rank, "sidecar_name") else name
    same = in_store == "ranker.json"
    found = f"p4-recall writes models/ranker.json (kind {m.rk.KIND!r}); phi_rank's sidecar in a p4-recall store is {in_store!r}: {'the same path, two formats' if same else 'a different file'}"
    return (DRIFTED if same else HOLDS), found


@claim("teacher-models", "teacher.py docstring: second opinion by 'Claude Opus 5.5'; p4.json recall.teacher",
       "The teacher is Fable 5.1 and the second opinion Opus 5.5.")
def _(ctx):
    t = mods().conf.DEFAULTS["recall"]["teacher"]
    found = f"p4.json defaults model={t.get('model')}, second_opinion={t.get('second_opinion')}"
    if ctx.has_store() and (ctx.store / "models" / "config.json").exists():
        found += f"; store config.json teacher-model={json.loads(read(ctx.store / 'models' / 'config.json')).get('teacher-model')}"
    return (HOLDS if t.get("model") == "claude-fable-5-1" and t.get("second_opinion") == "claude-opus-5-5" else DRIFTED), found


@claim("env-overrides", "p4conf.py docstring: 'Environment variables a plugin documents (P4_RECALL_INJECT, P4_SCRATCHPAD_ASKS) still win, for one run.'",
       "Both environment variables are read by a plugin's code.")
def _(ctx):
    hits = {}
    sources = [p for p in (P4 / "lbiza" / "plugins").rglob("*.py") if p.name != "p4conf.py" and not p.name.startswith("test_")]
    for name in ("P4_RECALL_INJECT", "P4_SCRATCHPAD_ASKS"):
        hits[name] = sorted(str(p.relative_to(P4)).replace(os.sep, "/") for p in sources
                            if f'"{name}"' in read(p) and "environ" in read(p))     # named, in a file that reads the environment
    missing = [k for k, v in hits.items() if not v]
    return (HOLDS if not missing else DRIFTED), "; ".join(f"{k}: {v or 'read nowhere'}" for k, v in hits.items())


@claim("p4conf-copies", "p4conf.py docstring: 'The same file is copied into each plugin ... test_p4conf.py holds the copies byte-identical.'",
       "Every p4conf.py under the plugins is byte-identical.")
def _(ctx):
    import hashlib
    copies = {str(p.relative_to(P4)): hashlib.sha1(p.read_bytes()).hexdigest()[:8] for p in (P4 / "lbiza" / "plugins").rglob("p4conf.py")}
    return (HOLDS if len(set(copies.values())) == 1 else DRIFTED), f"{len(copies)} copies, {len(set(copies.values()))} distinct: {copies}"


@claim("inject-default", "findings.md 28: 'shipped on: inject: rules-facts'",
       "The recall's default inject mode is rules-facts.")
def _(ctx):
    d = mods().conf.DEFAULTS["recall"]["inject"]
    found = f"p4.json default inject={d!r}"
    if ctx.has_store() and (ctx.store / "models" / "config.json").exists():
        found += f"; store config.json inject={json.loads(read(ctx.store / 'models' / 'config.json')).get('inject')!r}"
    return (HOLDS if d == "rules-facts" else DRIFTED), found


@claim("design-subcommands", "fresh-marketplace.md 3.4 `p4.py` subcommands",
       "The design's subcommand list matches p4.py's parser.")
def _(ctx):
    doc = read(P4 / ".docs" / "p4" / "fresh-marketplace.md")
    m = re.search(r"### 3\.4 .*?\n(.*?)\n### 3\.5", doc, re.S)
    if not m:
        return UNTESTABLE, "section 3.4 not found"
    said, said_batches = set(), set()
    for row in re.findall(r"^\| `([^`]+)`", m.group(1), re.M):      # the table's first column: `name args`
        said |= {w for w in re.split(r"[`,\s]+", row.split()[0]) if re.fullmatch(r"[a-z-]+", w)}
        if row.startswith("teacher "):
            said_batches = set(row.split()[1].replace("\\|", "|").split("|"))
    for row in re.findall(r"^\| `teacher [^`]+` \| ([^|]*)\|", m.group(1), re.M):   # the row's own qualification
        said_batches -= {b for b in re.findall(r"`([a-z-]+)` is accepted and answers \"not built yet\"", row)}
    src = read(SCRIPTS / "p4.py")
    parsed = set(re.findall(r'add_parser\("([a-z-]+)"', src)) | {"train", "eval", "migrate", "teacher"}
    tsrc = read(SCRIPTS / "teacher.py")
    offered = set(re.findall(r'"(edit-labels|dictionary|extract)"', re.search(r'choices=\(([^)]*)\)', tsrc).group(1))) if "choices=(" in tsrc else set()
    stubbed = set(re.findall(r'action == "([a-z-]+)":\s*\n\s*print\([^\n]*not built', tsrc))
    batches = offered - stubbed
    if not said:
        return UNTESTABLE, "section 3.4's table has no `name` first column the regex could read"
    found = (f"design names {sorted(said)}; parser has {sorted(parsed)}; teacher batches designed {sorted(said_batches)}, built {sorted(batches)}"
             + (f" ({sorted(stubbed)} accepted by the parser and answered 'not built yet')" if stubbed else ""))
    gap = (said - parsed) | (said_batches - batches)
    return (HOLDS if not gap else DRIFTED), found + (f"; designed, not built: {sorted(gap)}" if gap else "")


# ---- phi ----
@claim("phi-readme-store", "phi/README.md: 'phi_store reads a .claude/p4 store as training records (read_p4, split, dispatch_rows, pick_pairs; python -m phi_store <p4 root> prints a summary)'",
       "phi_store reads a .claude/p4 store, the one p4-recall writes included.")
def _(ctx):
    ph = phi()
    names = [n for n in ("read_p4", "split", "dispatch_rows", "pick_pairs") if not hasattr(ph.store, n)]
    found = "functions present" if not names else f"missing: {names}"
    if not ctx.has_store():
        return (PARTLY if not names else DRIFTED), found + "; no store to read"
    try:
        n = sum(1 for _ in ph.store.read_p4(str(ctx.store)))
        return HOLDS, found + f"; read_p4 on the p4-recall store: {n} records"
    except (FileNotFoundError, OSError) as e:
        return DRIFTED, found + f"; read_p4 on the p4-recall store raises {type(e).__name__} (it wants sessions/)"


@claim("phi-model-root", "phi_model docstring: 'p4_root: <project>/.claude/p4, or the path itself when it is a store repository (sessions/ and vindex/ at its top ...)'",
       "phi_model accepts a .claude/p4 store.")
def _(ctx):
    ctx.need_store()
    try:
        phi().model.p4_root(str(ctx.store))
        return HOLDS, "p4_root accepts the p4-recall store"
    except SystemExit as e:
        return DRIFTED, f"p4_root refuses the p4-recall store: {str(e).split(' — ')[-1][:100]}"


@claim("phi-model-subcommands", "phi/README.md: 'phi_model ... (check|init|train|test|update|mode)'",
       "phi_model's parser has exactly those six subcommands.")
def _(ctx):
    src = read(PHI / "phi_model" / "__init__.py")
    m = re.search(r"for name in \(([^)]*)\):\s*\n\s*p = sub\.add_parser\(name\)", src)
    got = set(re.findall(r'"([a-z]+)"', m.group(1))) if m else set(re.findall(r'add_parser\("([a-z]+)"', src))
    return (HOLDS if got == {"check", "init", "train", "test", "update", "mode"} else DRIFTED), f"parser: {sorted(got)}"


@claim("phi-model-recall-config", "phi_model docstring: 'Modes, written to the store's recall config (vindex/phi-index.md, ## recall)'",
       "phi_model's mode switch writes the file p4-recall reads its settings from.", fix="code")
def _(ctx):
    ph = phi()
    root = str(ctx.store) if ctx.has_store() else str(WORK / "fake-store")
    if not ctx.has_store():
        os.makedirs(os.path.join(root, "store"), exist_ok=True)
    path = ph.model.recall_path(root)
    conf = mods().conf.FILE
    return (HOLDS if path.endswith(conf) else DRIFTED), f"phi_model writes {os.path.basename(path)} for a p4-recall store; p4-recall reads <p4>/{conf} (recall section)"


@claim("phi-model-retrain-notices", "phi_model docstring: 'Sessions ... append one row to models/retrain-needed.tsv'",
       "The plugin's sessions write models/retrain-needed.tsv.", fix="code")
def _(ctx):
    writers = sorted(p.name for p in SCRIPTS.glob("*.py") if "retrain-needed" in read(p))
    return (HOLDS if writers else DRIFTED), f"p4-recall scripts naming retrain-needed.tsv: {writers or 'none'}"


@claim("phi-model-finds-plugin", "phi_model.is_plugin: a plugin root is one with scripts/pick_model.py",
       "phi_model can find the plugin that writes the store.", fix="code")
def _(ctx):
    ph = phi()
    return (HOLDS if ph.model.is_plugin(str(RECALL)) else DRIFTED), f"is_plugin({RECALL.name}) = {ph.model.is_plugin(str(RECALL))}; p4-recall has no scripts/pick_model.py"


@claim("phi-rank-cli", "phi_rank docstring: 'python -m phi_rank pretrain | train | eval' (and shared)",
       "phi_rank's parser has pretrain, train, eval and shared.")
def _(ctx):
    src = read(PHI / "phi_rank" / "__init__.py")
    got = set(re.findall(r'add_parser\("([a-z]+)"', src))
    m = re.search(r"for name in \(([^)]*)\):\s*\n\s*p = sub\.add_parser\(name\)", src)
    if m:
        got |= set(re.findall(r'"([a-z]+)"', m.group(1)))
    return (HOLDS if {"pretrain", "train", "eval", "shared"} <= got else DRIFTED), f"parser: {sorted(got)}"


@claim("phi-serve-endpoints", "phi_serve docstring: 'GET /health ... POST /score'",
       "phi_serve answers /health and /score.")
def _(ctx):
    src = read(PHI / "phi_serve" / "__init__.py")
    return (HOLDS if '"/health"' in src and '"/score"' in src else DRIFTED), f"/health in code: {'/health' in src}; /score in code: {'/score' in src}"


@claim("phi-rank-labels-out", "phi_rank docstring: 'the store's .gitignore decides what is trainable'; LABEL_FILE_RE and SKIP_DIRS",
       "The plugin's label files never enter phi_rank's pretraining text.")
def _(ctx):
    ph = phi()
    skip = "models" in ph.rank.SKIP_DIRS
    found = f"SKIP_DIRS has models/: {skip} (labels.tsv and tasks.tsv live there); LABEL_FILE_RE names the old vlds files only"
    if ctx.has_store():
        files = [os.path.relpath(p, ctx.store).replace(os.sep, "/") for p in ph.rank.text_files(str(ctx.store))]
        leak = [f for f in files if f.startswith("models/")]
        found += f"; {len(files)} text files read, {len(leak)} under models/"
        return (HOLDS if not leak else DRIFTED), found
    return (HOLDS if skip else DRIFTED), found


@claim("phi-readme-links", "phi/README.md: links to MODEL-ROAD.md in the vlds plugin and to .docs/p4/continual_p4.ipynb",
       "The README's links point at files that exist.")
def _(ctx):
    readme = read(PHI / "README.md")
    out, drift = [], False
    for m in re.finditer(r"\]\((\.docs/[^)]+)\)", readme):
        ok = (PHI / m.group(1)).exists()
        drift |= not ok
        out.append(f"{m.group(1)}: {'exists' if ok else 'missing'}")
    for m in re.finditer(r"github\.com/ezexe/p4/blob/main/([^)\s]+)", readme):
        ok = (P4 / m.group(1)).exists()
        drift |= not ok
        out.append(f"p4/{m.group(1)}: {'exists in the p4 checkout' if ok else 'not in the p4 checkout'}")
    return (DRIFTED if drift else HOLDS), "; ".join(out) or "no such links"


# ---- better than it says: the bar above the docs ----
@claim("usage-note-mode", "served_back B: the served log is deleted at Stop, so shown and shadow lines could not be told apart",
       "Every usage label says whether its line was shown to the model or logged in shadow.", bar="better")
def _(ctx):
    src = read(SCRIPTS / "labels.py")
    has = "def served_mode" in src and "; shown" in src and "; shadow" in src
    found = "labels.usage_labels writes `used; shown` / `unused; shadow` notes" if has else "the usage note is `used` / `unused` only"
    if ctx.has_store():
        m = mods()
        notes = Counter(x.get("note", "") for x in m.lb.read_tsv(m.lb.mpath(str(ctx.store), m.lb.TRAIN)) if x.get("source") == "usage")
        found += f"; notes in the store: {dict(notes)}"
    return (MET if has else NOT_YET), found


@claim("row-recommended", "served_back F: 300 characters of the reply's first paragraph rarely hold the recommendation handed the owner",
       "A closed row keeps the reply's recommending sentence beside its first paragraph.", bar="better")
def _(ctx):
    src = read(SCRIPTS / "rows.py")
    has = "def recommended_sentence" in src and "recommended" in read(SCRIPTS / "store.py")
    found = "rows.row_close writes `recommended:`; store.HEADERS names it" if has else "no recommended: field"
    if ctx.has_store():
        n = sum(bool(e.fields.get("recommended", "").strip()) for rel, es in ctx.files().items() if rel.startswith("messages/") for e in es)
        found += f"; rows with one in the store: {n}"
    return (MET if has else NOT_YET), found


@claim("phi-rank-sidecar-separate", "says-claim ranker-json-shared",
       "phi_rank never writes over the plugin's models/ranker.json.", bar="better")
def _(ctx):
    ph = phi()
    if not hasattr(ph.rank, "sidecar_name"):
        return NOT_YET, "phi_rank has one SIDECAR name for every store"
    name = ph.rank.sidecar_name(str(ctx.store)) if ctx.has_store() else ph.rank.SIDECAR_RECALL
    return (MET if name != "ranker.json" else NOT_YET), f"in a p4-recall store the sidecar is {name!r}; read_sidecar ignores a file of the plugin's kind: {'kind' in read(PHI / 'phi_rank' / '__init__.py').split('def read_sidecar')[1][:600]}"


@claim("phi-store-new-layout", "says-claim phi-readme-store",
       "phi_store reads a p4-recall store into pretrain and pair records, and phi_rank builds pools from its labels without a pairs file.", bar="better")
def _(ctx):
    ph = phi()
    src = read(PHI / "phi_rank" / "__init__.py")
    pools = "labels.tsv" in src and ("def pools_from_labels" in src or "def read_label_pools" in src or "labels_pools" in src or "from_labels" in src)
    found = f"phi_rank builds pools from labels.tsv: {pools}"
    if not ctx.has_store():
        return (MET if pools else NOT_YET), found + "; no store to read"
    try:
        kinds = Counter(r.kind for r in ph.store.read_p4(str(ctx.store)))
        found += f"; read_p4 records {dict(kinds)}"
        return (MET if pools and kinds.get("pair") else NOT_YET), found
    except (FileNotFoundError, OSError) as e:
        return NOT_YET, found + f"; read_p4 raises {type(e).__name__}"


@claim("echo-judged", "says-claim echo-offline",
       "The close hook judges a completed row ECHO or SUPERSEDED when it resembles an earlier row, from the two replies.", bar="better")
def _(ctx):
    writers = echo_writers()
    return (MET if writers else NOT_YET), f"state writers outside migrate.py: {writers or 'none'}"


@claim("first-kind", "says-claim task-kinds",
       "A session's first owner message is recorded as kind first.", bar="better")
def _(ctx):
    live = bool(re.search(r'"first"', read(SCRIPTS / "hook.py") + read(SCRIPTS / "rows.py")))
    history = bool(re.search(r'"first"', read(SCRIPTS / "labels.py")))
    return (MET if live else NOT_YET), f"the live road (hook.py close) writes kind first: {live}; the history road (labels.from_history) does: {history}"


@claim("gitignore-reseed", "says-claim seeded-gitignore",
       "The layout hook adds any seeded .gitignore line a store lacks, under its own marker, never touching the owner's lines.", bar="better")
def _(ctx):
    src = read(SCRIPTS / "hook.py")
    has = "seeded by p4-recall" in src and re.search(r"missing|lacks|not in", src.split("GITIGNORE")[-1][:4000] if "GITIGNORE" in src else "") is not None and "marker" in src
    return (MET if has else NOT_YET), f"hook.py has a re-seed path with a marker: {has}"


@claim("teacher-facts", "served_back F: facts.md is empty on every branch; recall serves facts under inject: rules-facts",
       "A paid teacher batch writes grounded facts.md entries (claim, source URL from a search, verified date) for the owner's asks.", bar="better")
def _(ctx):
    src = read(SCRIPTS / "teacher.py")
    m = re.search(r'choices=\(([^)]*)\)', src)
    offered = set(re.findall(r'"([a-z-]+)"', m.group(1))) if m else set()
    stubbed = set(re.findall(r'action == "([a-z-]+)":\s*\n\s*print\([^\n]*not built', src))
    built = "facts" in offered - stubbed and "web_search" in src
    return (MET if built else NOT_YET), f"teacher actions {sorted(offered)}, stubbed {sorted(stubbed)}; web search used: {'web_search' in src}"


@claim("phi-model-plugin-recall", "says-claims phi-model-recall-config, phi-model-finds-plugin, phi-model-retrain-notices",
       "phi_model finds the p4-recall plugin, writes its modes into p4.json's recall section, and the plugin leaves retrain notices it can answer.", bar="better")
def _(ctx):
    ph = phi()
    finds = ph.model.is_plugin(str(RECALL))
    root = str(ctx.store) if ctx.has_store() else str(WORK / "fake-store")
    if not ctx.has_store():
        os.makedirs(os.path.join(root, "store"), exist_ok=True)
    conf = ph.model.recall_path(root).endswith(mods().conf.FILE)
    notices = any("retrain-needed" in read(p) for p in SCRIPTS.glob("*.py") if not p.name.startswith("test_"))
    return (MET if finds and conf and notices else NOT_YET), f"finds p4-recall: {finds}; writes p4.json: {conf}; plugin writes retrain notices: {notices}"


# ---- running ----
def run_claims(ctx, only=None):
    rows = []
    for c in CLAIMS:
        if only and c.id not in only:
            continue
        try:
            verdict, found = c.check(ctx)
        except LookupError as e:
            verdict, found = UNTESTABLE, str(e)
        except Exception as e:  # noqa: BLE001 — a check that breaks is reported, never hides the others
            verdict, found = UNTESTABLE, f"{type(e).__name__}: {str(e)[:200]}"
        rows.append({"id": c.id, "where": c.where, "says": c.says, "does": found, "verdict": verdict, "bar": c.bar, "fix": c.fix})
    return rows


def report(rows, ctx):
    says = [r for r in rows if r["bar"] == "says"]
    better = [r for r in rows if r["bar"] == "better"]
    counts = Counter(r["verdict"] for r in says)
    cell = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
    L = ["# Says vs does: p4-recall and phi", "",
         f"{len(says)} documented claims and {len(better)} targets above them, checked against the code at p4@{git_head(P4)} and phi@{git_head(PHI)}"
         + (f", and the store at {ctx.store} ({git_head(ctx.store)})" if ctx.has_store() else ", no store given") + ".",
         "", "## What it says", "", "| verdict | claims |", "|---|---|"]
    L += [f"| {v} | {counts.get(v, 0)} |" for v in (HOLDS, PARTLY, DRIFTED, UNTESTABLE)]
    L += ["", "| claim | where it is said | what it says | what was found | verdict | fix |", "|---|---|---|---|---|---|"]
    for r in sorted(says, key=lambda r: ({DRIFTED: 0, PARTLY: 1, UNTESTABLE: 2, HOLDS: 3}[r["verdict"]], r["id"])):
        fix = r["fix"] if r["verdict"] in (DRIFTED, PARTLY) else ""
        L.append(f"| `{r['id']}` | {cell(r['where'])} | {cell(r['says'])} | {cell(r['does'])} | **{r['verdict']}** | {fix} |")
    met = sum(r["verdict"] == MET for r in better)
    L += ["", f"## Better than it says: {met} of {len(better)} targets met", "",
          "| target | set from | what the code should do | what was found | state |", "|---|---|---|---|---|"]
    for r in sorted(better, key=lambda r: (r["verdict"] != NOT_YET, r["id"])):
        L.append(f"| `{r['id']}` | {cell(r['where'])} | {cell(r['says'])} | {cell(r['does'])} | **{r['verdict']}** |")
    L += ["", "Reading it: HOLDS is a sentence the code or the data bears out; DRIFTED one it does not; PARTLY one that is "
          "true with a qualification the sentence lacks; UNTESTABLE one this run could not check (the reason is in the "
          "found column). The fix column says where a drifted sentence is resolved: `code` keeps the promise, `doc` "
          "replaces it with what is true. A target is a bar set above the docs by what served_back measured; MET means "
          "the code now does it, NOT YET that it is still owed. The two live rows, `recall-latency` and `recall-silence`, "
          "change with the store and the machine."]
    return "\n".join(L) + "\n"


def git_head(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "?"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(DEFAULT_STORE) if DEFAULT_STORE.exists() else None,
                    help="a checked-out branch of a p4-recall store (default: served_back's win worktree when present)")
    ap.add_argument("--out", default=str(WORK / "results"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    ctx = Ctx(a.store)
    rows = run_claims(ctx, set(a.only) if a.only else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "claims.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    text = report(rows, ctx)
    (out / "claims.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
