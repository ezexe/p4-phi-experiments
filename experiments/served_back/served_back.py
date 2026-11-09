#!/usr/bin/env python3
"""
served_back.py: what the p4 sessions of a per-machine store were served, how much of it came back, and what phi's
models can train on from the store as the p4-recall plugin now writes it (its current layout), with the paid parts
judged by the Claude API with no Claude Code session in the loop.

Three questions, each on every branch of STORE_REPO:
  A. architecture  phi_store / phi_model were written for the old layout (sessions/*/vlds, mrindex/) and now read the
                   new one too; p4-recall writes store/, memory/, flights/, models/{labels,tasks,eval}.tsv. What share
                   of the store's text each phi reader reaches (an error here is a reader that still refuses the
                   layout), and whether the plugin's labels can train phi_rank's .pt ranker at all (a bridge:
                   labels.tsv + index.json -> phi_rank pools, time-split, trained on the CPU here, free).
  B. served back   the hooks log what recall served each turn and label it `used` or `unused` from the reply, and
                   the close hook completes each owner message's row with the reply's first paragraph. How many owner
                   turns got lines, how many lines came back, per kind, per session; and what the owner had to
                   correct: for each correction, did the store already hold the knowledge, did recall serve it, and
                   was it followed (the judge's class: NEW_KNOWLEDGE / KNOWN_NOT_SERVED / SERVED_NOT_FOLLOWED).
  C. teacher       finding 16 of p4/.docs/p4/findings.md: ranker-against-teacher agreement. For held-out tasks the
                   API picks the entries a session should have been shown; every road (the hooks' labels, BM25, the
                   bridge .pt ranker, recall's own query) is scored against it, and two model families against each
                   other.
  D. epistemic     the recommendations the sessions handed the owner (replies that say should / recommend / use X
                   instead), re-derived two ways: the model's default answer with no tools (what it recommends from
                   its priors), and a grounded answer that must search for the current official documentation and
                   cite it (the URLs are taken from the search results the API returned, never from the model's
                   claims). A judge says which one the session's own recommendation matched, and whether the default
                   and the documented approach differ (the epistemic gap). Every grounded answer is drafted as the
                   store's facts.md entry (claim, source, verified, time): the file recall serves under `inject:
                   rules-facts`, and the one a store's branches often leave empty.

Why the API and not a session: a session answering about its own store leans on the owner's history. Here every
request is a bare Messages call with a neutral system prompt, owner names scrubbed, candidates shuffled per seed.

    python experiments/served_back/served_back.py run --store <url or path>         # FAKE: offline, no key; the free steps are real
    python experiments/served_back/served_back.py run --store <url or path> --paid  # the paid steps too, capped at TOTAL_MAX_USD
    ... --project <checkout>          # the project the store's repo: labels name, so those entries are read too

No store is built in: --store (or P4_STORE_REPO in .env) is a p4 store repository, one branch per machine.

The store under test is one project's data and nothing more: every setting that names it (the repository, the
project checkout) is an argument, the owner's names are read from the store's own committers and scrubbed, and the
findings that carry are the ones about the shared implementation (p4-recall, phi), never about that project.

Steps (each a function the notebook served_back.ipynb beside this file calls, in this order):
  1. layout       clone, one worktree per branch; the new layout's counts; what phi_store, phi_model and phi_rank
                  read of it (free)
  2. served_back  owner turns, served turns, used share by kind and session, corrections per turn, each correction's
                  as-of candidates and recall lines (free)
  3. bridge       labels.tsv -> phi_rank pools; BM25, a fresh .pt ranker and a pretrained one on a time split (free,
                  CPU torch)
  4. teacher      held-out tasks judged by TEACHER_MODELS; every road's precision / recall / F1 against them (paid)
  5. adherence    each correction classified by JUDGE_MODEL from its as-of evidence; a sample of the hooks' lexical
                  used/unused labels checked against the reply excerpt (paid)
  6. epistemic    EPISTEMIC_SAMPLE recommendation turns: default arm, grounded arm with web search, the judge's match
                  and gap; results/facts-draft.md in the store's format (paid)
  7. report       results/report.md and results/summary.json under WORK (free)
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                          # the shared experiments project holding p4/ and phi/
RECALL_SCRIPTS = ROOT / "p4" / "lbiza" / "plugins" / "p4-recall" / "scripts"

# ---- settings (the notebook's settings cell overrides these through configure) ----
FAKE = True
TOTAL_MAX_USD = 10.0
STORE_REPO = ""                                 # the store repository (a URL or a local path): --store, or P4_STORE_REPO
BRANCHES = []                                   # [] = every remote branch that has a store/ directory
PROJECT_REPO = os.environ.get("SERVED_BACK_PROJECT") or os.environ.get("P4_PROJECT", "")   # the checkout the store's `repo:` labels name, when this
                                                           # machine has it (`run --project D`); "" = their paths only
REPO_DEADLINE_S = 180.0                         # repo.py's first read of PROJECT_REPO stops here (incremental after)
HOLDOUT_SHARE = 0.25                            # the newest share of labelled tasks, held out by time
NEGATIVES_K = 10                                # unlabelled BM25 candidates added to each pool as negatives
PRETRAIN_MINUTES = 0.5
EPOCHS = 10
TEACHER_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5"]
TEACHER_TASKS = 12                              # held-out tasks per branch sent to the teachers
CAND_K = 20                                     # candidates a teacher sees per task (plus the hooks' labelled ones)
TOP_N = 5                                       # a road's picks: its top 5, recall's own cap
JUDGE_MODEL = "claude-opus-5-5"
CORRECTIONS_MAX = 30                            # corrections and retractions judged, newest first, both branches
USAGE_SAMPLE = 20                               # usage labels checked against the reply excerpt
EPISTEMIC_SAMPLE = 10                           # recommendation turns re-derived through the API, newest first
RECOMMEND_MODEL = "claude-opus-5-5"             # answers both arms: the same model, with and without the search tool
SEARCH_MAX_USES = 5                             # web searches the grounded arm may run per recommendation
SEARCH_USD = 0.01                               # per search, on top of the tokens
RECOMMEND_RE = re.compile(r"(?i)\b(recommend|suggest|better to|prefer(red|able)?|the right way|instead of|rather than|"
                          r"go with|best (way|option|practice)|i'?d (use|go|pick|keep|pin|set)|the fix is|"
                          r"(you|we) should|should (use|pin|keep|move|set|run|switch|avoid|not|stay|be))\b")
ASK_RE = re.compile(r"(?i)\b(should|better|best|recommend|suggest|which (one|way|approach|option)|what (do|would) you "
                    r"(think|suggest|recommend|do)|or (should|just)|is it (worth|better|ok|fine|right)|do (i|we) need|"
                    r"proper|optimal|correct way)\b")          # an owner message asking for a recommendation (with a ?)
SILENT_TASKS = 6                                # ordinary prompts per branch the teachers should stay silent on
RECIRCULATE_TURNS = 150                         # later owner turns a drafted fact is tried against, newest first
EFFORT = "high"
SEED = 1
SCRUB = {r"\bthe boss's\b": "the owner's", r"\bthe boss\b": "the owner", r"\bboss\b": "owner"}
SCRUB_AUTHORS = True                            # also scrub the store's committers' names and STORE_REPO's owner
WORK = HERE / "out"                             # the clone, worktrees and results/; ignored by the project's .gitignore

PRICES = {"claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-haiku-5-5": (0.10, 0.50)}

NEUTRAL_SYSTEM = (
    "You review the memory store of an AI coding agent working in one software project: the project owner's "
    "rulings, corrections and retractions, their past messages, the project's own files and commits. You have no "
    "history with this project or its owner. Judge from the text given alone.")

PICK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["picks"], "properties": {"picks": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["candidate", "why"],
                               "properties": {"candidate": {"type": "string"}, "why": {"type": "string"}}}}}}

ADHERE_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["known_before", "known_in", "served", "class", "why"],
                 "properties": {"known_before": {"type": "boolean"},
                                "known_in": {"type": "array", "items": {"type": "string"}},
                                "served": {"type": "boolean"},
                                "class": {"type": "string",
                                          "enum": ["NEW_KNOWLEDGE", "KNOWN_NOT_SERVED", "SERVED_NOT_FOLLOWED"]},
                                "why": {"type": "string"}}}

REFLECT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reflected", "why"],
                  "properties": {"reflected": {"type": "boolean"}, "why": {"type": "string"}}}

RECOMMEND_SCHEMA = {"type": "object", "additionalProperties": False,
                    "required": ["recommendation", "basis", "sources", "as_of", "confidence"],
                    "properties": {"recommendation": {"type": "string"},
                                   "basis": {"type": "string", "enum": ["DOCUMENTATION", "EXPERIENCE", "GUESS"]},
                                   "sources": {"type": "array", "items": {"type": "string"}},
                                   "as_of": {"type": "string"},
                                   "confidence": {"type": "number"}}}

EPISTEMIC_SCHEMA = {"type": "object", "additionalProperties": False,
                    "required": ["session_matches", "gap", "documented_is_newer", "fact_claim", "why"],
                    "properties": {"session_matches": {"type": "string",
                                                       "enum": ["GROUNDED", "DEFAULT", "BOTH", "NEITHER", "UNDECIDABLE"]},
                                   "gap": {"type": "boolean"},
                                   "documented_is_newer": {"type": "boolean"},
                                   "fact_claim": {"type": "string"},
                                   "why": {"type": "string"}}}

AGENT_SYSTEM = ("You are an AI coding agent answering a software project's owner. Give the one recommendation you "
                "would hand them, in two or three sentences, and say what it rests on: DOCUMENTATION when you are "
                "citing a source you can name, EXPERIENCE when it is common practice you have seen, GUESS otherwise. "
                "Name the sources you would point them at, if any, and the date your knowledge of them is current to.")

GROUNDED_SYSTEM = ("You are an AI coding agent answering a software project's owner, and you must not answer from "
                   "memory. Search for the current official documentation or primary source for what the message "
                   "asks about (the project's own docs, the vendor's reference, the standard, the release notes), "
                   "read it, and recommend what it says, in two or three sentences. Quote the documented approach's "
                   "date or version. Cite every source you used by URL. When the documentation does not settle it, "
                   "say so instead of recommending.")


def configure(**over):
    for k, v in over.items():
        if k not in globals():
            raise KeyError(k)
        globals()[k] = Path(v) if k == "WORK" else v


def say(*a):
    print(*a, flush=True)


def paths():
    w = Path(WORK)
    p = {"work": w, "repo": w / "store-repo", "branches": w / "branches", "results": w / "results"}
    p["results"].mkdir(parents=True, exist_ok=True)
    return p


_OWNER_TERMS = {}


def owner_terms():
    """The words that name the store's owner, read from the store itself: every committer's name parts and e-mail
    local part (four letters or more) over all branches, and the owner of STORE_REPO's URL. Nothing is typed in."""
    key = str(STORE_REPO)
    if key not in _OWNER_TERMS:
        terms = set()
        m = re.search(r"github\.com[:/]([^/]+)/", key)
        if m:
            terms.add(m.group(1).lower())
        repo = paths()["repo"]
        if (repo / ".git").exists():
            try:
                for line in git(repo, "log", "--all", "--format=%an%x09%ae").splitlines():
                    name, _tab, email = line.partition("\t")
                    for part in re.split(r"[\s.<>_-]+", name) + re.split(r"[._+-]+", email.split("@")[0]):
                        if len(part) >= 4 and part.isalpha():   # never the e-mail's domain
                            terms.add(part.lower())
            except subprocess.CalledProcessError:
                pass
        _OWNER_TERMS[key] = sorted(terms - {"noreply", "users", "github", "gmail", "anthropic", "claude", "fable"}, key=len, reverse=True)
    return _OWNER_TERMS[key]


def scrub(text):
    text = text or ""
    for pat, rep in SCRUB.items():
        text = re.sub(pat, rep, text, flags=re.I)
    if SCRUB_AUTHORS:
        for term in owner_terms():
            text = re.sub(rf"\b{re.escape(term)}\w*", "owner", text, flags=re.I)
    return text


# ---- the plugin's and phi's modules ----
def recall():
    """store, labels, ranker, p4 (query), repo of the p4-recall plugin, imported from the checkout beside this project."""
    s = str(RECALL_SCRIPTS)
    if s not in sys.path:
        sys.path.insert(0, s)
    import labels as lb
    import p4 as cli
    import ranker as rk
    import repo as rp
    import store as st
    return SimpleNamespace(st=st, lb=lb, rk=rk, cli=cli, rp=rp)


def phi():
    os.environ.setdefault("PHI_DEVICE", "cpu")
    import phi_model
    import phi_rank
    import phi_store
    return SimpleNamespace(store=phi_store, rank=phi_rank, model=phi_model)


# ---- spend ----
def spent():
    f = paths()["results"] / "spend.json"
    return json.loads(f.read_text())["usd"] if f.exists() else 0.0


def add_spend(model, usage):
    if FAKE:
        return 0.0
    p_in, p_out = PRICES.get(model, (10.0, 50.0))
    usd = (usage.input_tokens * p_in + usage.output_tokens * p_out) / 1e6
    f = paths()["results"] / "spend.json"
    f.write_text(json.dumps({"usd": round(spent() + usd, 4)}))
    return usd


def guard(step):
    if not FAKE and spent() >= TOTAL_MAX_USD:
        raise RuntimeError(f"budget: ${spent():.2f} spent of TOTAL_MAX_USD = {TOTAL_MAX_USD}; {step} not started")


# ---- the client: bare Messages calls, structured output, refusal-checked ----
class FakeClient:
    """Offline stand-in: a deterministic, well-formed answer per schema, so the notebook and the test run end to end
    without a key. Its verdicts are not a measurement."""

    def __init__(self):
        self.beta = SimpleNamespace(messages=self)

    def create(self, model, messages, system="", output_config=None, **kw):
        prompt = messages[-1]["content"]
        props = (output_config or {}).get("format", {}).get("schema", {}).get("properties", {})
        cands = re.findall(r"^(C\d+) ", prompt, re.M)
        if "picks" in props:
            out = {"picks": [{"candidate": c, "why": "fake"} for c in cands[:2]]}
        elif "known_before" in props:
            out = {"known_before": bool(cands), "known_in": cands[:1], "served": False,
                   "class": "KNOWN_NOT_SERVED" if cands else "NEW_KNOWLEDGE", "why": "fake"}
        elif "basis" in props:
            grounded = "https://" in prompt
            out = {"recommendation": "fake: use the documented default", "basis": "DOCUMENTATION" if grounded else "GUESS",
                   "sources": re.findall(r"https?://\S+", prompt)[:2], "as_of": "2026-01", "confidence": 0.5}
        elif "session_matches" in props:
            msg = re.search(r"Owner's message:\n(.*)", prompt)
            out = {"session_matches": "GROUNDED", "gap": True, "documented_is_newer": True,
                   "fact_claim": "fake fact about: " + (msg.group(1)[:120] if msg else "the documented default"), "why": "fake"}
        else:
            out = {"reflected": "hook label: used" in prompt, "why": "fake"}
        text = json.dumps(out)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4))


def client():
    if FAKE:
        return FakeClient()
    load_env()
    import anthropic
    return anthropic.Anthropic(max_retries=6)


def require_store():
    """STORE_REPO from --store or the notebook, else P4_STORE_REPO (the environment or .env); a run without one stops
    here, before anything is cloned: this project ships no store of its own."""
    global STORE_REPO, PROJECT_REPO
    load_env()
    STORE_REPO = STORE_REPO or os.environ.get("P4_STORE_REPO", "")
    PROJECT_REPO = PROJECT_REPO or os.environ.get("SERVED_BACK_PROJECT") or os.environ.get("P4_PROJECT", "")
    if not STORE_REPO:
        raise SystemExit("served_back: no store. Pass --store <URL or path of a p4 store repository>, or set "
                         "P4_STORE_REPO in .env (see .env.example).")
    return STORE_REPO


def load_env():
    sys.path.insert(0, str(HERE.parent))
    import env
    return env.load()


def ask(c, model, prompt, schema, system=NEUTRAL_SYSTEM, max_tokens=8000):
    """One bare request: no tools, no memory, no history."""
    guard(f"request to {model}")
    kw = dict(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": prompt}],
              output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
              betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if FAKE:
        r = c.beta.messages.create(**kw)
    else:
        with c.beta.messages.stream(**kw) as s:
            r = s.get_final_message()
    add_spend(model, r.usage)
    if r.stop_reason == "refusal":
        raise RuntimeError(f"{model} refused: {getattr(r, 'stop_details', None)}")
    if r.stop_reason == "max_tokens":
        raise RuntimeError(f"{model} hit max_tokens; raise it")
    return json.loads(next(b.text for b in r.content if b.type == "text"))


def search(c, model, prompt, system=GROUNDED_SYSTEM, max_tokens=16000):
    """One request with the web search tool: (the answer text, the URLs the search results actually returned, the
    number of searches run). The URLs come from the tool result blocks, so a source the model names without having
    fetched it is not among them. A pause_turn is continued, up to three times."""
    guard(f"search request to {model}")
    if FAKE:
        return ("fake grounded answer citing https://docs.example.org/current (2026-01).",
                ["https://docs.example.org/current"], 1)
    messages = [{"role": "user", "content": prompt}]
    text, urls, searches = "", [], 0
    for _ in range(3):
        with c.beta.messages.stream(model=model, max_tokens=max_tokens, system=system, messages=messages,
                                    tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": SEARCH_MAX_USES}],
                                    output_config={"effort": EFFORT}, betas=["server-side-fallback-2026-07-01"],
                                    fallbacks="default") as s:
            r = s.get_final_message()
        add_spend(model, r.usage)
        for b in r.content:
            if b.type == "web_search_tool_result":
                content = getattr(b, "content", None)
                if isinstance(content, list):
                    searches += 1
                    urls += [getattr(x, "url", "") for x in content if getattr(x, "url", "")]
            elif b.type == "text":
                text += b.text
                for cit in getattr(b, "citations", None) or []:
                    if getattr(cit, "url", ""):
                        urls.append(cit.url)
        if searches:
            f = paths()["results"] / "spend.json"
            f.write_text(json.dumps({"usd": round(spent() + SEARCH_USD * searches, 4)}))
        if r.stop_reason == "refusal":
            raise RuntimeError(f"{model} refused: {getattr(r, 'stop_details', None)}")
        if r.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": r.content})
    return text, list(dict.fromkeys(urls)), searches


# ---- git: the clone and one worktree per branch ----
def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def worktrees():
    """{branch: worktree path} for every branch of STORE_REPO that holds a store/ directory, each checked out
    detached at origin/<branch> under WORK/branches/."""
    require_store()
    p = paths()
    repo = p["repo"]
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--no-checkout", STORE_REPO, str(repo)], check=True)
    git(repo, "fetch", "-q", "--all", "--prune")
    names = [b.strip().removeprefix("origin/") for b in git(repo, "branch", "-r", "--format=%(refname:short)").split()
             if b.strip().startswith("origin/") and "HEAD" not in b and b.strip() != "origin"]
    out = {}
    for b in (BRANCHES or names):
        if "store" not in git(repo, "ls-tree", "--name-only", f"origin/{b}").split():
            continue
        wt = p["branches"] / b
        if (wt / ".git").exists():
            git(wt, "checkout", "-q", "--detach", f"origin/{b}")
        else:
            wt.parent.mkdir(parents=True, exist_ok=True)
            git(repo, "worktree", "prune")
            git(repo, "worktree", "add", "-q", "--detach", str(wt), f"origin/{b}")
        out[b] = wt
    return out


# ---- the store as the plugin reads it ----
def owner_rows(st, files):
    """[(session s8, Entry)] for every message row of the owner's own words, in file order."""
    out = []
    for rel, entries in files.items():
        if not rel.startswith(st.MESSAGES + "/"):
            continue
        s8 = rel[len(st.MESSAGES) + 1:-3]
        out += [(s8, e) for e in entries if st.owner_message(e)]
    return out


def index_of(wt):
    """The plugin's index of a worktree, built fresh (and the repository's entries when PROJECT_REPO is a git
    tree: repo.json is built once and copied between the branches' worktrees, the project being the same)."""
    r = recall()
    wt = str(wt)
    models = os.path.join(wt, "models")
    os.makedirs(models, exist_ok=True)
    repo_state = os.path.join(models, "repo.json")
    if PROJECT_REPO and os.path.isdir(os.path.join(PROJECT_REPO, ".git")):
        shared = paths()["work"] / "repo.json"
        if shared.exists() and not os.path.exists(repo_state):
            shutil.copy(shared, repo_state)
        stats = r.rp.build(wt, project=PROJECT_REPO, deadline_s=REPO_DEADLINE_S)
        shutil.copy(repo_state, shared)
        say(f"  repo entries: {stats.get('entries')} files, {stats.get('commits')} commits, complete={stats.get('complete')}")
    elif PROJECT_REPO:
        say(f"  PROJECT_REPO {PROJECT_REPO} is not a git checkout here: repo: labels read as their paths only")
    r.st.build_index(wt)
    idx = r.st.load_index(wt)
    return idx


def docs_of(idx):
    return [dict(d) for d in idx["docs"]]


def doc_text(d):
    """What a candidate reads as: the entry's shown text, its path when it is a file, and the time it carries."""
    show = d.get("show") or ""
    path = d.get("path") or ""
    return (f"{path}: {show}" if path and not show.startswith(path) and d.get("kind") in ("file", "commit")
            else show) or path


def bm25_rank(rk, text, docs):
    """docs sorted by BM25 of `text` against each doc's indexed tokens, scores beside them."""
    tok_lists = [(d.get("toks") or "").split() for d in docs]
    idf, _n = rk.idf_table(tok_lists)
    avgdl = sum(len(t) for t in tok_lists) / max(1, len(tok_lists))
    terms = list(dict.fromkeys(rk.tokens(text)))
    scored = [(rk.bm25(terms, t, idf, avgdl), d) for t, d in zip(tok_lists, docs)]
    scored.sort(key=lambda x: -x[0])
    return scored


def before(docs, when, exclude=()):
    """The docs a session at `when` could have been shown: store and memory entries dated before it (an undated
    one counts), repository files always, commits dated before it; never the ids in exclude."""
    out = []
    for d in docs:
        if d["id"] in exclude:
            continue
        t = d.get("time") or ""
        if d.get("kind") == "file" or not t or t < when:
            out.append(d)
    return out


# ---- step 1: layout (free) ----
def step_layout():
    r, ph = recall(), phi()
    wts = worktrees()
    out = {}
    for b, wt in wts.items():
        files = r.st.load_store(str(wt))
        states = Counter()
        chars = Counter()
        for rel, es in files.items():
            group = "messages" if rel.startswith(r.st.MESSAGES + "/") else rel
            for e in es:
                states[f"{group}:{e.state}"] += 1
                chars["store"] += sum(len(v) for _k, v in e.pairs)
        mem = sorted((wt / "memory").glob("*.md")) if (wt / "memory").is_dir() else []
        chars["memory"] = sum(len(p.read_text(encoding="utf-8", errors="replace")) for p in mem)
        flights = sorted(p for p in (wt / "flights").iterdir() if p.is_dir()) if (wt / "flights").is_dir() else []
        landed = 0
        for f in flights:
            for name in ("plan.md", "log", "evidence.md", "rulings.md"):
                if (f / name).exists():
                    chars["flights"] += len((f / name).read_text(encoding="utf-8", errors="replace"))
            log = (f / "log").read_text(encoding="utf-8", errors="replace") if (f / "log").exists() else ""
            landed += any(ln.split("\t")[-1].startswith(("landed", "home")) for ln in log.splitlines())
        labels = r.lb.read_tsv(r.lb.mpath(str(wt), r.lb.TRAIN))
        tasks = r.lb.read_tasks(str(wt))
        evals = r.lb.read_tsv(r.lb.mpath(str(wt), "eval.tsv"))
        last = evals[-1] if evals else {}
        # what phi reads of it
        phi_store = {"records": 0, "pairs": 0, "chars": 0, "error": ""}
        try:
            for rec in ph.store.read_p4(str(wt)):
                phi_store["records"] += 1
                phi_store["pairs"] += rec.kind == "pair"
                phi_store["chars"] += len(rec.text) + len(rec.target)
        except (FileNotFoundError, OSError) as e:
            phi_store["error"] = f"{type(e).__name__}: {e}".split(":")[0] + ": no sessions/ directory"
        try:
            ph.model.p4_root(str(wt))
            phi_model = "accepts the store"
        except SystemExit as e:
            phi_model = str(e).split(" — ", 1)[-1][:120]
        pairs, read = ph.rank.corpus_pairs(str(wt))
        n_text = len(ph.rank.text_files(str(wt)))
        total = sum(chars.values())
        out[b] = {"head": git(wt, "rev-parse", "--short", "HEAD").strip(),
                  "store": {k: v for k, v in sorted(states.items())},
                  "memory_files": len(mem), "flights": len(flights), "flights_landed": landed,
                  "labels": dict(Counter(x.get("source", "") for x in labels)), "tasks": dict(Counter(t["kind"] for t in tasks.values())),
                  "eval_rounds": len(evals), "eval_last": {k: last.get(k, "") for k in ("time", "r_precision", "precision", "promoted", "why")},
                  "chars": dict(chars),
                  "phi_store": phi_store, "phi_model": phi_model,
                  "phi_rank": {"text_files": n_text, "passage_pairs": len(pairs), "chars_read": read},
                  "coverage": {"phi_store": round(phi_store["chars"] / total, 3) if total else 0.0,
                               "phi_rank": round(min(1.0, read / total), 3) if total else 0.0}}
        say(f"layout {b}@{out[b]['head']}: store {sum(states.values())} entries, {len(mem)} memory files, {len(flights)} flights "
            f"({landed} landed), labels {out[b]['labels']}, tasks {out[b]['tasks']}; phi_store reads {phi_store['records']} "
            f"records ({phi_store['error'] or 'ok'}), phi_rank {n_text} text files")
    (paths()["results"] / "layout.json").write_text(json.dumps(out, indent=1))
    return out


# ---- step 2: served back (free) ----
def correction_cases(st, files):
    """One case per correction and retraction: what to look for before it (the owner's words, what was meant or
    the lesson), its time, and its id."""
    cases = []
    for fname in ("corrections.md", "retractions.md"):
        for e in files.get(fname, []):
            f = e.fields
            want = " ".join(st.collapse(f.get(k, "")) for k in ("meant", "omitted", "lesson", "freed", "owner-words"))
            cases.append({"id": e.id, "file": fname, "time": e.time, "kind": st.kind_of(fname, e),
                          "head": st.collapse(e.value)[:300], "match": st.collapse(f.get("match", ""))[:300],
                          "meant": st.collapse(f.get("meant") or f.get("lesson") or f.get("omitted") or "")[:300],
                          "owner_words": st.collapse(f.get("owner-words", ""))[:300], "want": want})
    return cases


def trigger_row(st, rk, rows, case, window_minutes=90):
    """The owner message the correction answers: of the rows in the window before its time, the one sharing most
    tokens with its owner-words (then the nearest in time); None when no row is in the window."""
    t = case["time"]
    if not t:
        return None
    cands = [(s8, e) for s8, e in rows if e.time and e.time <= t and minutes_between(e.time, t) <= window_minutes]
    if not cands:
        return None
    want = set(rk.tokens(case["owner_words"] or case["want"]))
    def key(x):
        s8, e = x
        fp = st.unquote(e.fields.get("fingerprint", e.value))
        return (len(want & set(rk.tokens(fp))), e.time)
    s8, e = max(cands, key=key)
    return {"session": s8, "id": e.id, "time": e.time, "text": st.unquote(e.fields.get("fingerprint", e.value))[:600],
            "addressed": st.collapse(e.fields.get("addressed", ""))[:300]}


def minutes_between(a, b):
    import datetime
    try:
        da, db = (datetime.datetime.strptime(x, "%Y-%m-%d %H:%M") for x in (a, b))
    except ValueError:
        return 1e9
    return (db - da).total_seconds() / 60


def step_served_back():
    r = recall()
    wts = worktrees()
    out = {}
    for b, wt in wts.items():
        files = r.st.load_store(str(wt))
        rows = owner_rows(r.st, files)
        tasks = r.lb.read_tasks(str(wt))
        labels = r.lb.read_tsv(r.lb.mpath(str(wt), r.lb.TRAIN))
        usage = [x for x in labels if x.get("source") == "usage"]
        served_tasks = {x["task"] for x in usage}
        by_kind, by_session, by_day = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        for x in usage:
            kind = x["entry"].split(":", 1)[0]
            kind = "messages" if kind.startswith("messages/") else kind.replace(".md", "")
            by_kind[kind][x["label"]] += 1
            s8 = tasks.get(x["task"], {}).get("session") or (x["task"].split(":")[1][:8] if ":" in x["task"] else "")
            by_session[s8][x["label"]] += 1
            by_day[(x.get("time") or "")[:10]][x["label"]] += 1
        turns_by_session = Counter(s8 for s8, _e in rows)
        completed = sum(r.st.completed(e) for _s8, e in rows)
        resembles = sum(bool(e.fields.get("resembles", "").strip()) for _s8, e in rows)
        states = Counter(e.fields.get("state", "").split(" ")[0] or "(open)" for _s8, e in rows)
        addressed_len = sorted(len(e.fields.get("addressed", "")) for _s8, e in rows if r.st.completed(e))
        labelled_tasks = {x["task"] for x in labels}
        owner_task_ids = {e.id for _s8, e in rows}
        cases = correction_cases(r.st, files)
        # as-of evidence per correction: the store entries and message rows before it, BM25 top 8; recall's lines
        idx = index_of(wt)
        docs = [d for d in docs_of(idx) if d.get("kind") not in ("file", "commit")]
        for c in cases:
            c["trigger"] = trigger_row(r.st, r.rk, rows, c)
            pool = before(docs, c["time"] or "9999", exclude={c["id"]})
            if c["trigger"]:
                pool = [d for d in pool if d["id"] != c["trigger"]["id"]]
            top = bm25_rank(r.rk, c["want"], pool)[:8]
            c["candidates"] = [{"id": d["id"], "kind": d.get("kind"), "time": d.get("time"), "text": doc_text(d)[:300],
                                "bm25": round(s, 2)} for s, d in top if s > 0]
            q_text = (c["trigger"] or {}).get("text") or c["owner_words"] or c["head"]
            q = r.cli.query(str(wt), q_text, (c["trigger"] or {}).get("session"), now=c["time"] or None, idx=idx)
            by_id = {d["id"]: d for d in docs}
            c["recall"] = [{"id": i, "text": doc_text(by_id[i])[:300]} for i in q["ids"]
                           if i in by_id and (not by_id[i].get("time") or by_id[i]["time"] < (c["time"] or "9999"))]
            want_toks = set(r.rk.tokens(c["want"]))
            c["known_lexical"] = any(len(want_toks & set((d.get("toks") or "").split())) >= 3 for _s, d in top)
        # the recommendations handed the owner, newest first (the paid epistemic step re-derives them): a reply whose
        # kept paragraph recommends (source reply), or an owner message that asks for a recommendation (source ask;
        # the reply's first 300 characters rarely show the recommendation, the question always shows the need)
        recs = []
        for s8, e in rows:
            if not r.st.completed(e):
                continue
            addressed = r.st.collapse(e.fields.get("addressed", ""))
            text = (tasks.get(e.id, {}).get("text") or r.st.unquote(e.fields.get("fingerprint", e.value)))
            if text.lstrip().startswith("<") or len(text.split()) < 5 or addressed.startswith("(a notification"):
                continue
            source = ("reply" if RECOMMEND_RE.search(addressed)
                      else "ask" if ASK_RE.search(text) and ("?" in text or re.match(r"(?i)\s*(should|which|what|how|is|do|can|would)\b", text))
                      else "")
            if source:
                recs.append({"id": e.id, "session": s8, "time": e.time, "source": source, "text": text[:1500],
                             "addressed": addressed[:300]})
        recs.sort(key=lambda x: x["time"], reverse=True)
        n_turns = len(rows)
        out[b] = {"owner_turns": n_turns, "sessions": len(turns_by_session), "rows_completed": completed,
                  "recommendation_turns": len(recs), "recommendation_sources": dict(Counter(x["source"] for x in recs)),
                  "facts_entries": sum(e.state == "LIVE" for e in files.get("facts.md", [])),
                  "recommendations": recs[:3 * EPISTEMIC_SAMPLE],
                  "rows_resembling_earlier": resembles, "row_states": dict(states),
                  "addressed_chars_median": addressed_len[len(addressed_len) // 2] if addressed_len else 0,
                  "tasks": len(tasks), "tasks_labelled": len(labelled_tasks),
                  "tasks_from_owner_rows": len(owner_task_ids & set(tasks)),
                  "served_turns": len(served_tasks), "served_share": round(len(served_tasks) / n_turns, 3) if n_turns else 0.0,
                  "served_ids": len(usage), "used": sum(x["label"] == "1" for x in usage),
                  "used_share": round(sum(x["label"] == "1" for x in usage) / len(usage), 3) if usage else 0.0,
                  "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
                  "by_session": {s: {"turns": turns_by_session.get(s, 0), "served_ids": sum(v.values()), "used": v["1"]}
                                 for s, v in sorted(by_session.items())},
                  "by_day": {d: dict(v) for d, v in sorted(by_day.items())},
                  "corrections": sum(c["file"] == "corrections.md" for c in cases),
                  "retractions": sum(c["file"] == "retractions.md" for c in cases),
                  "corrections_per_100_turns": round(100 * len(cases) / n_turns, 1) if n_turns else 0.0,
                  "cases_with_trigger": sum(bool(c["trigger"]) for c in cases),
                  "cases_known_lexical": sum(c["known_lexical"] for c in cases),
                  "cases_recall_served_any": sum(bool(c["recall"]) for c in cases),
                  "cases": cases}
        say(f"served_back {b}: {n_turns} owner turns in {len(turns_by_session)} sessions, {completed} rows completed, "
            f"{len(recs)} recommendation turns {out[b]['recommendation_sources']}, facts.md holds {out[b]['facts_entries']}; "
            f"{len(served_tasks)} turns served ({out[b]['served_share']:.0%}), {len(usage)} ids, {out[b]['used_share']:.0%} used; "
            f"{len(cases)} corrections+retractions, {out[b]['cases_known_lexical']} with a lexical match before them, "
            f"{out[b]['cases_recall_served_any']} where recall had a line")
    (paths()["results"] / "served_back.json").write_text(json.dumps(out, indent=1))
    return out


# ---- step 3: bridge (free, CPU torch) ----
def pools_of(wt, idx=None):
    """phi_rank pools from the plugin's labels: one per task with a positive whose entry the index holds; rows are
    the task's labelled entries plus NEGATIVES_K unlabelled BM25 candidates as negatives. Sorted by task time."""
    r = recall()
    idx = idx or index_of(wt)
    docs = docs_of(idx)
    by_id = {d["id"]: d for d in docs}
    tasks = r.lb.read_tasks(str(wt))
    labels = r.lb.read_tsv(r.lb.mpath(str(wt), r.lb.TRAIN))
    by_task = defaultdict(dict)
    for x in labels:
        if x["entry"] in by_id and x.get("task") in tasks:
            by_task[x["task"]][x["entry"]] = max(by_task[x["task"]].get(x["entry"], 0), int(x["label"] == "1"))
    pools = []
    for tid, labs in by_task.items():
        if not any(labs.values()):
            continue
        task = tasks[tid]
        text = task["text"] or ""
        if not text.strip() or text.lstrip().startswith("<"):
            continue
        rows = [{"file": by_id[e].get("path") or e, "text": doc_text(by_id[e]), "label": v, "id": e} for e, v in labs.items()]
        pool = [d for d in before(docs, task["time"] or "9999", exclude=set(labs) | {tid})]
        for _s, d in bm25_rank(r.rk, text, pool)[:NEGATIVES_K]:
            rows.append({"file": d.get("path") or d["id"], "text": doc_text(d), "label": 0, "id": d["id"]})
        pools.append({"id": tid, "task": text[:2000], "time": task["time"], "session": task["session"], "rows": rows})
    pools.sort(key=lambda p: p["time"])
    return pools


def split_pools(pools):
    cut = int(len(pools) * (1 - HOLDOUT_SHARE))
    return pools[:cut], pools[cut:]


def eval_scores(ph, pools, scorer, threshold=None):
    got = []
    for p in pools:
        sc = scorer(p)
        labels = [x["label"] for x in p["rows"]]
        picked = ph.rank.select([x["file"] for x in p["rows"]], sc, threshold) if threshold is not None else []
        got.append(ph.rank.pool_metrics(labels, sc, picked))
    if not got:
        return {}
    return {k: round(sum(g[k] for g in got) / len(got), 3) for k in ("r-precision", "recall@20", "precision", "recall")}


def step_bridge():
    r, ph = recall(), phi()
    import torch
    wts = worktrees()
    out = {}
    for b, wt in wts.items():
        pools = pools_of(wt)
        train, held = split_pools(pools)
        res = {"pools": len(pools), "train": len(train), "held": len(held),
               "rows_per_pool": round(sum(len(p["rows"]) for p in pools) / len(pools), 1) if pools else 0,
               "positives_per_pool": round(sum(sum(x["label"] for x in p["rows"]) for p in pools) / len(pools), 1) if pools else 0,
               "cutoff": held[0]["time"] if held else ""}
        if len(train) < 3 or not held:
            res["error"] = "too few labelled pools to split"
            out[b] = res
            say(f"bridge {b}: {res}")
            continue
        def bm25_scorer(p):
            toks = [r.rk.tokens(x["text"]) for x in p["rows"]]
            idf, _n = r.rk.idf_table(toks)
            avgdl = sum(len(t) for t in toks) / max(1, len(toks))
            terms = list(dict.fromkeys(r.rk.tokens(p["task"])))
            return [r.rk.bm25(terms, t, idf, avgdl) for t in toks]
        res["bm25"] = eval_scores(ph, held, bm25_scorer)
        torch.manual_seed(SEED)
        fresh = ph.rank.Ranker()
        ph.rank.finetune(fresh, train, epochs=EPOCHS, seed=SEED)
        t_fresh = ph.rank.best_threshold(fresh, train)
        res["pt_fresh"] = {**eval_scores(ph, held, lambda p: ph.rank.score_pool(fresh, p["task"], [x["text"] for x in p["rows"]]), t_fresh),
                           "threshold": t_fresh}
        torch.manual_seed(SEED)
        pre = ph.rank.Ranker()
        pairs, read = ph.rank.corpus_pairs(str(wt))
        steps = ph.rank.pretrain(pre, pairs, PRETRAIN_MINUTES, log=lambda *a: None) if PRETRAIN_MINUTES > 0 else 0
        res["pretrain"] = {"passage_pairs": len(pairs), "chars": read, "steps": steps, "minutes": PRETRAIN_MINUTES}
        ph.rank.finetune(pre, train, epochs=EPOCHS, seed=SEED)
        t_pre = ph.rank.best_threshold(pre, train)
        res["pt_pretrained"] = {**eval_scores(ph, held, lambda p: ph.rank.score_pool(pre, p["task"], [x["text"] for x in p["rows"]]), t_pre),
                                "threshold": t_pre}
        evals = r.lb.read_tsv(r.lb.mpath(str(wt), "eval.tsv"))
        res["json_ranker_last_eval"] = {k: evals[-1].get(k, "") for k in ("round", "r_precision", "precision", "promoted")} if evals else {}
        mdir = paths()["work"] / "models"
        mdir.mkdir(exist_ok=True)
        ph.rank.save(pre, {"nb": pre.bag.num_embeddings, "dim": pre.bag.embedding_dim, "branch": b, "pools": len(train),
                           "epochs": EPOCHS}, str(mdir / f"bridge-{b}.pt"), half=False)
        res["model"] = str(mdir / f"bridge-{b}.pt")
        out[b] = res
        say(f"bridge {b}: {len(pools)} pools ({len(train)} train / {len(held)} held from {res['cutoff']}); held-out r-precision "
            f"bm25 {res['bm25'].get('r-precision')}, .pt fresh {res['pt_fresh'].get('r-precision')}, "
            f".pt pretrained {res['pt_pretrained'].get('r-precision')} (json ranker's own eval: {res['json_ranker_last_eval'].get('r_precision', '-')})")
    (paths()["results"] / "bridge.json").write_text(json.dumps(out, indent=1))
    return out


# ---- step 4: teacher (paid) ----
def teacher_tasks(wt, pools):
    """The held-out pools the teachers judge: owner messages of at least 8 words, kind new, TEACHER_TASKS newest."""
    _train, held = split_pools(pools)
    r = recall()
    tasks = r.lb.read_tasks(str(wt))
    got = [p for p in held if tasks.get(p["id"], {}).get("kind") == "new" and len(p["task"].split()) >= 8]
    return got[-TEACHER_TASKS:]


def candidates_for(wt, idx, pool, docs=None):
    """CAND_K BM25 candidates the task's session could have been shown, plus the hooks' labelled entries, shuffled."""
    r = recall()
    docs = docs or docs_of(idx)
    by_id = {d["id"]: d for d in docs}
    pool_docs = before(docs, pool["time"] or "9999", exclude={pool["id"]})
    top = [d for _s, d in bm25_rank(r.rk, pool["task"], pool_docs)[:CAND_K] if _s > 0]
    ids = {d["id"] for d in top}
    for x in pool["rows"]:
        if x["id"] in by_id and x["id"] not in ids and len(top) < CAND_K + 10:
            top.append(by_id[x["id"]])
            ids.add(x["id"])
    random.Random(f"{SEED}:{pool['id']}").shuffle(top)
    return top


def road_picks(wt, idx, pool, cands, model):
    """Each road's top TOP_N among the candidates: the hooks' positive labels, BM25, the bridge ranker, recall."""
    r, ph = recall(), phi()
    ids = [d["id"] for d in cands]
    hooks = [x["id"] for x in pool["rows"] if x["label"] and x["id"] in ids]
    bm = [d["id"] for _s, d in bm25_rank(r.rk, pool["task"], cands)[:TOP_N]]
    roads = {"hooks": hooks, "bm25": bm}
    if model is not None:
        sc = ph.rank.score_pool(model, pool["task"], [doc_text(d) for d in cands])
        roads["bridge"] = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -sc[i])[:TOP_N]]
    q = r.cli.query(str(wt), pool["task"], (pool.get("session") or "")[:8] or None, now=pool["time"] or None, idx=idx)
    roads["recall"] = [i for i in q["ids"] if i in ids]
    return roads


def prf(picked, truth):
    picked, truth = set(picked), set(truth)
    if not truth and not picked:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    hit = len(picked & truth)
    p = hit / len(picked) if picked else 0.0
    rc = hit / len(truth) if truth else 0.0
    return {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0}


def silent_prompts():
    """Ordinary prompts the store has nothing for, from the plugin's silent-prompts.jsonl (its test split first): a
    teacher, and every road, should pick nothing for them."""
    path = RECALL_SCRIPTS / "silent-prompts.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda x: ("split test" not in x.get("from", ""), x.get("id", "")))
    return [{"id": f"silent:{x['id']}", "task": x["prompt"], "time": "", "session": "", "rows": []} for x in rows[:SILENT_TASKS]]


def pick_ids(got, cands):
    out = []
    for p in got.get("picks", []):
        mm = re.match(r"C(\d+)", str(p.get("candidate", "")).strip())
        if mm and 1 <= int(mm.group(1)) <= len(cands):
            out.append(cands[int(mm.group(1)) - 1]["id"])
    return out


def teacher_prompt(pool, cands):
    listing = "\n".join(f"C{i + 1} [{d.get('kind')} {d.get('time') or 'undated'}] {scrub(doc_text(d))[:300]}"
                        for i, d in enumerate(cands))
    return ("An engineer working in this project just sent the message below to their AI coding agent. From "
            "the store entries listed, pick the ones the agent should be shown with the message: an entry "
            f"that bears on how to do this task right, or that the message is about. At most {TOP_N}; none "
            "when nothing applies (an ordinary message needs no memory). Name candidates by their C-number.\n\n"
            f"Message:\n{scrub(pool['task'])[:2000]}\n\nCandidates:\n{listing}")


def step_teacher(bridge=None):
    r, ph = recall(), phi()
    c = client()
    wts = worktrees()
    out = {}
    for b, wt in wts.items():
        idx = index_of(wt)
        docs = docs_of(idx)
        pools = pools_of(wt, idx)
        chosen = teacher_tasks(wt, pools)
        model = None
        mpath = (bridge or {}).get(b, {}).get("model")
        if mpath and os.path.exists(mpath):
            model, _meta = ph.rank.load(mpath)
        # silence: ordinary prompts, the same candidates road, nothing should be picked
        silent = []
        for pool in silent_prompts():
            cands = candidates_for(wt, idx, pool, docs)
            if not cands:   # nothing in the store shares a word with it: silent by construction, no call
                silent.append({"task": pool["id"], "text": pool["task"][:200], "teacher": {m: [] for m in TEACHER_MODELS},
                               "roads": {"bm25": 0, "bridge": 0, "recall": 0}})
                continue
            prompt = teacher_prompt(pool, cands)
            roads = road_picks(wt, idx, pool, cands, model)
            picks = {m: pick_ids(ask(c, m, prompt, PICK_SCHEMA), cands) for m in TEACHER_MODELS}
            silent.append({"task": pool["id"], "text": pool["task"][:200], "teacher": picks,
                           "roads": {k: len(v) for k, v in roads.items() if k != "hooks"}})
            say(f"silence {b} {pool['id']}: " + ", ".join(f"{m.split('-')[1]} picks {len(picks[m])}" for m in TEACHER_MODELS)
                + f"; recall printed {len(roads['recall'])}  spent ${spent():.2f}")
        rows = []
        for pool in chosen:
            cands = candidates_for(wt, idx, pool, docs)
            if not cands:
                continue
            prompt = teacher_prompt(pool, cands)
            roads = road_picks(wt, idx, pool, cands, model)
            picks = {m: pick_ids(ask(c, m, prompt, PICK_SCHEMA), cands) for m in TEACHER_MODELS}
            rows.append({"task": pool["id"], "time": pool["time"], "text": pool["task"][:300], "n_candidates": len(cands),
                         "teacher": picks, "roads": roads,
                         "scores": {m: {road: prf(ids, picks[m]) for road, ids in roads.items()} for m in TEACHER_MODELS}})
            say(f"teacher {b} {pool['time']}: " + ", ".join(f"{m.split('-')[1]} picks {len(picks[m])}" for m in TEACHER_MODELS)
                + f"; roads {{{', '.join(f'{k}:{len(v)}' for k, v in roads.items())}}}  spent ${spent():.2f}")
        summary = {}
        for m in TEACHER_MODELS:
            summary[m] = {}
            for road in ("hooks", "bm25", "bridge", "recall"):
                vals = [x["scores"][m][road] for x in rows if road in x["scores"][m]]
                if vals:
                    summary[m][road] = {k: round(sum(v[k] for v in vals) / len(vals), 3) for k in ("precision", "recall", "f1")}
            summary[m]["picks_per_task"] = round(sum(len(x["teacher"][m]) for x in rows) / len(rows), 2) if rows else 0
            summary[m]["silent_tasks"] = sum(not x["teacher"][m] for x in rows)
        if len(TEACHER_MODELS) >= 2 and rows:
            a, bb = TEACHER_MODELS[:2]
            jac = [len(set(x["teacher"][a]) & set(x["teacher"][bb])) / len(set(x["teacher"][a]) | set(x["teacher"][bb]))
                   for x in rows if set(x["teacher"][a]) | set(x["teacher"][bb])]
            summary["teacher_agreement_jaccard"] = round(sum(jac) / len(jac), 3) if jac else None
        if silent:
            summary["silence"] = {"tasks": len(silent),
                                  **{m: {"silent": sum(not s["teacher"][m] for s in silent),
                                         "picks": sum(len(s["teacher"][m]) for s in silent)} for m in TEACHER_MODELS},
                                  "roads_silent": {road: sum(s["roads"].get(road, 0) == 0 for s in silent)
                                                   for road in ("bm25", "bridge", "recall") if any(road in s["roads"] for s in silent)}}
        out[b] = {"tasks": len(rows), "summary": summary, "rows": rows, "silent": silent}
    (paths()["results"] / "teacher.json").write_text(json.dumps(out, indent=1))
    return out


# ---- step 5: adherence (paid) ----
def step_adherence(served):
    r = recall()
    c = client()
    wts = worktrees()
    seen, cases = set(), []
    for b in wts:
        for case in served.get(b, {}).get("cases", []):
            if case["id"] not in seen:
                seen.add(case["id"])
                cases.append({**case, "branch": b})
    cases.sort(key=lambda x: x["time"], reverse=True)
    cases = cases[:CORRECTIONS_MAX]
    judged = []
    for case in cases:
        cands = case.get("candidates", [])
        listing = "\n".join(f"C{i + 1} [{d['kind']} {d['time'] or 'undated'}] {scrub(d['text'])}" for i, d in enumerate(cands)) or "(none)"
        recall_lines = "\n".join(f"- {scrub(x['text'])}" for x in case.get("recall", [])) or "(no line)"
        trig = case.get("trigger") or {}
        prompt = ("The project owner corrected their AI coding agent. Below: the correction as the store recorded it, "
                  "the owner message the agent was answering, the store entries that existed before that moment and "
                  "match the correction best, and the memory lines the agent was actually shown with the message.\n\n"
                  "Decide: known_before — did any listed earlier entry already carry what the correction says (name "
                  "them in known_in by C-number)? served — do the shown lines carry it? class — NEW_KNOWLEDGE when "
                  "nothing before it held it, KNOWN_NOT_SERVED when the store held it but the shown lines did not "
                  "carry it, SERVED_NOT_FOLLOWED when a shown line carried it and the agent still had to be corrected.\n\n"
                  f"Correction ({case['kind']}, {case['time']}):\n- what the agent did or read: {scrub(case['match'] or case['head'])}\n"
                  f"- what was meant: {scrub(case['meant'])}\n- the owner's words: {scrub(case['owner_words'])}\n\n"
                  f"The owner message answered ({trig.get('time', 'unknown time')}):\n{scrub(trig.get('text', '(not found)'))}\n\n"
                  f"Earlier store entries:\n{listing}\n\nMemory lines shown with the message:\n{recall_lines}")
        got = ask(c, JUDGE_MODEL, prompt, ADHERE_SCHEMA)
        known_ids = []
        for k in got.get("known_in", []):
            mm = re.match(r"C(\d+)", str(k).strip())
            if mm and 1 <= int(mm.group(1)) <= len(cands):
                known_ids.append(cands[int(mm.group(1)) - 1]["id"])
        judged.append({"id": case["id"], "branch": case["branch"], "time": case["time"], "kind": case["kind"],
                       "head": case["head"][:160], "known_before": got["known_before"], "known_in": known_ids,
                       "served": got["served"], "class": got["class"], "why": got["why"][:400],
                       "known_lexical": case.get("known_lexical"), "recall_lines": len(case.get("recall", []))})
        say(f"adherence {case['time']} {case['kind']}: {got['class']}  spent ${spent():.2f}")
    classes = Counter(j["class"] for j in judged)
    # the hooks' lexical used/unused against the reply excerpt
    usage_rows = []
    for b, wt in wts.items():
        files = r.st.load_store(str(wt))
        rows_by_id = {e.id: e for _s8, e in owner_rows(r.st, files)}
        tasks = r.lb.read_tasks(str(wt))
        idx = index_of(wt)
        by_id = {d["id"]: d for d in docs_of(idx)}
        for x in r.lb.read_tsv(r.lb.mpath(str(wt), r.lb.TRAIN)):
            if x.get("source") != "usage" or x["task"] not in rows_by_id or x["entry"] not in by_id:
                continue
            e = rows_by_id[x["task"]]
            if not r.st.completed(e):
                continue
            usage_rows.append({"branch": b, "task": x["task"], "entry": x["entry"], "label": x["label"],
                               "line": doc_text(by_id[x["entry"]])[:300], "text": (tasks.get(x["task"], {}).get("text") or "")[:600],
                               "addressed": r.st.collapse(e.fields.get("addressed", ""))[:300]})
    rng = random.Random(SEED)
    used = [u for u in usage_rows if u["label"] == "1"]
    unused = [u for u in usage_rows if u["label"] == "0"]
    rng.shuffle(used)
    rng.shuffle(unused)
    sample = used[:USAGE_SAMPLE // 2] + unused[:USAGE_SAMPLE - USAGE_SAMPLE // 2]
    checked = []
    for u in sample:
        prompt = ("An AI coding agent was shown one memory line with the owner's message, and replied. Only the "
                  "reply's first paragraph (at most 300 characters) was kept. Does that excerpt reflect the line: act "
                  "on it, follow it, cite it, or answer in its terms? reflected: true only when the excerpt clearly "
                  f"does.\n\nMessage:\n{scrub(u['text'])}\n\nMemory line shown:\n{scrub(u['line'])}\n\nReply excerpt:\n"
                  f"{scrub(u['addressed'])}\n\n(hook label: {'used' if u['label'] == '1' else 'unused'})")
        got = ask(c, TEACHER_MODELS[-1], prompt, REFLECT_SCHEMA)
        checked.append({**u, "reflected": got["reflected"], "why": got["why"][:300]})
    agree = sum((u["reflected"] and u["label"] == "1") or (not u["reflected"] and u["label"] == "0") for u in checked)
    out = {"judged": judged, "classes": dict(classes), "cases_judged": len(judged),
           "usage_checked": len(checked), "usage_agreement": round(agree / len(checked), 3) if checked else None,
           "usage_used_reflected": sum(u["reflected"] for u in checked if u["label"] == "1"),
           "usage_used_n": sum(u["label"] == "1" for u in checked),
           "usage_unused_reflected": sum(u["reflected"] for u in checked if u["label"] == "0"),
           "usage_unused_n": sum(u["label"] == "0" for u in checked), "usage_rows": checked}
    (paths()["results"] / "adherence.json").write_text(json.dumps(out, indent=1))
    say(f"adherence: {dict(classes)}; hook used/unused agrees with the judge on {agree}/{len(checked)}")
    return out


# ---- step 6: epistemic (paid) ----
_RECIRC = {}


def recirculation(branch, when, claim):
    """Would the drafted fact be served again: among the branch's owner turns after `when` (RECIRCULATE_TURNS newest),
    how many rank the claim in BM25's top TOP_N against every store, memory and message entry — the fact's place in
    the recall road, had it been in facts.md. (free; the index and tasks are read once per branch)"""
    r = recall()
    if branch not in _RECIRC:
        wt = worktrees()[branch]
        idx = index_of(wt)
        docs = [d for d in docs_of(idx) if d.get("kind") not in ("file", "commit")]
        tasks = [t for t in r.lb.read_tasks(str(wt)).values()
                 if t.get("kind") == "new" and (t.get("text") or "").strip() and not t["text"].lstrip().startswith("<")]
        _RECIRC[branch] = (docs, sorted(tasks, key=lambda t: t["time"], reverse=True))
    docs, tasks = _RECIRC[branch]
    later = [t for t in tasks if t["time"] > when][:RECIRCULATE_TURNS]
    fact = {"id": "fact:draft", "kind": "fact", "time": when, "toks": " ".join(dict.fromkeys(r.rk.tokens(claim))), "show": claim}
    hits = []
    for t in later:
        top = bm25_rank(r.rk, t["text"], docs + [fact])[:TOP_N]
        if any(d["id"] == "fact:draft" and s > 0 for s, d in top):
            hits.append(t["id"])
    return {"later_turns": len(later), "later_turns_hit": len(hits), "hit_tasks": hits[:5]}


def step_epistemic(served):
    c = client()
    seen, recs = set(), []
    for b, x in served.items():
        for rec in x.get("recommendations", []):
            if rec["id"] not in seen:
                seen.add(rec["id"])
                recs.append({**rec, "branch": b})
    recs.sort(key=lambda x: x["time"], reverse=True)
    recs = recs[:EPISTEMIC_SAMPLE]
    rows, facts = [], []
    today = __import__("time").strftime("%Y-%m-%d")
    for rec in recs:
        msg = scrub(rec["text"])
        default = ask(c, RECOMMEND_MODEL, f"The owner's message:\n\n{msg}", RECOMMEND_SCHEMA, system=AGENT_SYSTEM)
        text, urls, searches = search(c, RECOMMEND_MODEL, f"The owner's message:\n\n{msg}")
        grounded = ask(c, TEACHER_MODELS[-1], "Below is an agent's answer written after searching documentation. Put it "
                       "into the schema: its recommendation, its basis (DOCUMENTATION only when it cites a source it read), "
                       "the source URLs it cites, the date or version it says the documentation is current to.\n\n" + text,
                       RECOMMEND_SCHEMA, system="You extract structured fields from text; you add nothing.")
        verified = [u for u in grounded["sources"] if u in urls] or urls[:3]
        judge = ask(c, JUDGE_MODEL,
                    "An AI coding agent in a Claude Code session answered the owner's message below; only the first "
                    "paragraph of its reply was kept. The same message was then answered twice more through the API: "
                    "once by the model alone (default), once after it searched the current documentation (grounded), "
                    "with the URLs the search returned.\n\nDecide: session_matches — which of the two the session's "
                    "reply agrees with (GROUNDED, DEFAULT, BOTH when they agree with each other, NEITHER, UNDECIDABLE "
                    "when the excerpt does not show a recommendation). gap — do the default and the grounded "
                    "recommendations differ in a way that would change what the owner does? documented_is_newer — does "
                    "the grounded answer rest on something dated after what the default answer knows? fact_claim — one "
                    "sentence stating the documented fact, written so it holds without this conversation (a claim a "
                    "memory store could keep beside its source URL); empty when nothing was documented.\n\n"
                    f"Owner's message:\n{msg}\n\nThe session's reply (first paragraph):\n{scrub(rec['addressed'])}\n\n"
                    f"Default answer ({default['basis']}, as of {default['as_of'] or 'unstated'}):\n{default['recommendation']}\n"
                    f"Its named sources: {default['sources'] or 'none'}\n\n"
                    f"Grounded answer ({grounded['basis']}, as of {grounded['as_of'] or 'unstated'}):\n{grounded['recommendation']}\n"
                    f"Search results it read: {verified or 'none'}", EPISTEMIC_SCHEMA)
        row = {"id": rec["id"], "branch": rec["branch"], "time": rec["time"], "source": rec.get("source", "reply"),
               "message": rec["text"][:200], "session_reply": rec["addressed"], "default": default,
               "grounded": {**grounded, "searches": searches, "urls_returned": urls[:10], "sources_verified": verified},
               "judge": judge}
        rows.append(row)
        if judge["fact_claim"].strip() and verified:
            row["fact"] = {"claim": judge["fact_claim"].strip(), "source": verified[0],
                           **recirculation(rec["branch"], rec["time"], judge["fact_claim"])}
            facts.append(f"- claim: {judge['fact_claim'].strip()}\n  source: {verified[0]}\n  verified: {today}\n"
                         f"  time: {rec['time']}\n  from: {rec['id']}\n"
                         f"  recirculates: {row['fact']['later_turns_hit']} of {row['fact']['later_turns']} later owner turns\n")
        say(f"epistemic {rec['time']}: session={judge['session_matches']} gap={judge['gap']} default={default['basis']} "
            f"grounded={grounded['basis']} ({searches} searches, {len(verified)} sources)  spent ${spent():.2f}")
    matches = Counter(r["judge"]["session_matches"] for r in rows)
    out = {"rows": rows, "sampled": len(rows), "session_matches": dict(matches),
           "gap": sum(r["judge"]["gap"] for r in rows), "documented_is_newer": sum(r["judge"]["documented_is_newer"] for r in rows),
           "default_basis": dict(Counter(r["default"]["basis"] for r in rows)),
           "grounded_basis": dict(Counter(r["grounded"]["basis"] for r in rows)),
           "grounded_with_sources": sum(bool(r["grounded"]["sources_verified"]) for r in rows),
           "facts_drafted": len(facts),
           "facts_recirculating": sum(r["fact"]["later_turns_hit"] > 0 for r in rows if "fact" in r),
           "later_turns_hit": sum(r["fact"]["later_turns_hit"] for r in rows if "fact" in r),
           "by_source": dict(Counter(r["source"] for r in rows)),
           "recommendation_turns": {b: x.get("recommendation_turns", 0) for b, x in served.items()},
           "facts_entries": {b: x.get("facts_entries", 0) for b, x in served.items()}}
    (paths()["results"] / "epistemic.json").write_text(json.dumps(out, indent=1))
    (paths()["results"] / "facts-draft.md").write_text(
        "Claims checked against a source, with where and when they were checked.\nDrafted by served_back's epistemic "
        "step from the grounded arm's answers; the owner reviews each before it enters store/facts.md.\n"
        "Fields: claim, source, verified, time, from.\n\n" + "\n".join(facts), encoding="utf-8")
    say(f"epistemic: {dict(matches)}, gap in {out['gap']}/{len(rows)}, {len(facts)} facts drafted")
    return out


# ---- step 7: report (free) ----
def pct(x):
    return f"{x:.0%}" if isinstance(x, (int, float)) else str(x)


def step_report(layout, served, bridge, teacher, adherence, epistemic=None):
    epistemic = epistemic or {"rows": [], "sampled": 0, "session_matches": {}, "gap": 0, "documented_is_newer": 0,
                              "default_basis": {}, "grounded_basis": {}, "grounded_with_sources": 0, "facts_drafted": 0,
                              "recommendation_turns": {}, "facts_entries": {}}
    epistemic.setdefault("facts_recirculating", 0)
    epistemic.setdefault("later_turns_hit", 0)
    epistemic.setdefault("by_source", {})
    L = [f"# Served back: {STORE_REPO}", "",
         f"{'FAKE run: the paid steps are offline stand-ins, not a measurement; the free steps are real. ' if FAKE else ''}"
         f"Teachers {TEACHER_MODELS}, judge {JUDGE_MODEL}; spent ${spent():.2f} of ${TOTAL_MAX_USD}. The store's `repo:` "
         f"entries read as {'the files of ' + PROJECT_REPO if PROJECT_REPO else 'their paths only (no --project)'}; the owner's "
         f"names are scrubbed from every request ({len(owner_terms()) if SCRUB_AUTHORS else 0} terms from the store's committers).", ""]
    L += ["## A. What phi reads of the store as p4-recall writes it", "",
          "| branch | store entries | memory | flights (landed) | labels | phi_store | phi_model | phi_rank text files | reach: phi_store / phi_rank |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b, x in layout.items():
        L.append(f"| {b}@{x['head']} | {sum(x['store'].values())} | {x['memory_files']} files | {x['flights']} ({x['flights_landed']}) | "
                 f"{x['labels']} | {x['phi_store']['records']} records ({x['phi_store']['error'] or 'ok'}) | {x['phi_model']} | "
                 f"{x['phi_rank']['text_files']} | {pct(x['coverage']['phi_store'])} / {pct(x['coverage']['phi_rank'])} |")
    L += ["", "## B. Served back: what recall gave the sessions and what came back", "",
          "| branch | owner turns (sessions) | rows completed | turns served | ids served | used | corrections + retractions | per 100 turns | known before (lexical) | recall had a line |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b, x in served.items():
        L.append(f"| {b} | {x['owner_turns']} ({x['sessions']}) | {x['rows_completed']} | {x['served_turns']} ({pct(x['served_share'])}) | "
                 f"{x['served_ids']} | {x['used']} ({pct(x['used_share'])}) | {x['corrections']} + {x['retractions']} | "
                 f"{x['corrections_per_100_turns']} | {x['cases_known_lexical']} | {x['cases_recall_served_any']} |")
    for b, x in served.items():
        L += ["", f"Used share by entry kind, {b}: " + ", ".join(f"{k} {v.get('1', 0)}/{sum(v.values())}" for k, v in x["by_kind"].items())]
    L += ["", "## C. The bridge: phi_rank trained on the plugin's labels (held out by time)", "",
          "| branch | pools (train / held) | rows / positives per pool | BM25 r-prec | .pt fresh r-prec | .pt pretrained r-prec | picks P / R (pretrained) | json ranker's own eval |",
          "|---|---|---|---|---|---|---|---|"]
    for b, x in bridge.items():
        if x.get("error"):
            L.append(f"| {b} | {x['pools']} | | {x['error']} | | | | |")
            continue
        L.append(f"| {b} | {x['pools']} ({x['train']} / {x['held']}) | {x['rows_per_pool']} / {x['positives_per_pool']} | "
                 f"{x['bm25'].get('r-precision')} | {x['pt_fresh'].get('r-precision')} | {x['pt_pretrained'].get('r-precision')} | "
                 f"{x['pt_pretrained'].get('precision')} / {x['pt_pretrained'].get('recall')} | "
                 f"r_precision {x['json_ranker_last_eval'].get('r_precision', '-')} ({x['json_ranker_last_eval'].get('promoted', '-')}) |")
    L += ["", "## D. Teacher agreement: each road against the API's picks (precision / recall / F1)", ""]
    for b, x in teacher.items():
        L.append(f"### {b}: {x['tasks']} held-out tasks")
        L.append("")
        L.append("| teacher | hooks' labels | BM25 top 5 | bridge .pt top 5 | recall's lines | picks per task | silent tasks |")
        L.append("|---|---|---|---|---|---|---|")
        for m in TEACHER_MODELS:
            s = x["summary"].get(m, {})
            cell = lambda road: (f"{s[road]['precision']:.2f} / {s[road]['recall']:.2f} / {s[road]['f1']:.2f}" if road in s else "-")
            L.append(f"| {m} | {cell('hooks')} | {cell('bm25')} | {cell('bridge')} | {cell('recall')} | {s.get('picks_per_task')} | {s.get('silent_tasks')} |")
        L.append(f"\nTeacher-teacher agreement (Jaccard of picks): {x['summary'].get('teacher_agreement_jaccard')}")
        sil = x["summary"].get("silence")
        if sil:
            L.append(f"\nSilence on {sil['tasks']} ordinary prompts: " + ", ".join(
                f"{m} silent on {sil[m]['silent']} ({sil[m]['picks']} picks)" for m in TEACHER_MODELS if m in sil)
                + "; roads silent: " + ", ".join(f"{k} {v}" for k, v in sil["roads_silent"].items()))
        L.append("")
    L += ["## E. Adherence: why the owner had to correct", "",
          f"{adherence['cases_judged']} corrections and retractions judged: " + ", ".join(f"{k} {v}" for k, v in adherence["classes"].items()), ""]
    for j in adherence["judged"]:
        L.append(f"- {j['time']} {j['kind']} ({j['branch']}): **{j['class']}** — {j['head'][:120]}")
    L += ["", f"The hooks' lexical `used` label against the judge on the reply's first paragraph: agreement "
          f"{adherence['usage_agreement']} over {adherence['usage_checked']} rows (used rows reflected "
          f"{adherence['usage_used_reflected']}/{adherence['usage_used_n']}, unused rows reflected "
          f"{adherence['usage_unused_reflected']}/{adherence['usage_unused_n']}). The excerpt is 300 characters, so an "
          "unreflected line may have been followed later in the reply.", "",
          "## F. Epistemic recommendations: what the sessions recommended, against the documentation", "",
          "| branch | owner turns | recommendation turns (reply / ask) | facts.md entries |", "|---|---|---|---|"]
    for b, x in served.items():
        src = x.get("recommendation_sources", {})
        L.append(f"| {b} | {x['owner_turns']} | {x.get('recommendation_turns', 0)} ({src.get('reply', 0)} / {src.get('ask', 0)}) | {x.get('facts_entries', 0)} |")
    L += ["", f"{epistemic['sampled']} recommendations re-derived {epistemic['by_source']} (default arm: {RECOMMEND_MODEL} alone; grounded arm: the "
          f"same model with web search, sources taken from the results it read). The session's reply matched: "
          + (", ".join(f"{k} {v}" for k, v in epistemic["session_matches"].items()) or "none sampled")
          + f". Default and documented answers differ in {epistemic['gap']} of {epistemic['sampled']}; the documentation "
          f"was newer than the default's knowledge in {epistemic['documented_is_newer']}. Default arm's basis: "
          f"{epistemic['default_basis']}; grounded arm's: {epistemic['grounded_basis']}, with verified sources in "
          f"{epistemic['grounded_with_sources']}. {epistemic['facts_drafted']} facts.md entries drafted in `results/facts-draft.md`; "
          f"{epistemic['facts_recirculating']} of them would have been recalled again, for {epistemic['later_turns_hit']} later owner turns "
          f"in all (BM25 top {TOP_N} against the whole store).", ""]
    for r in epistemic["rows"]:
        f = r.get("fact")
        L.append(f"- {r['time']} ({r['branch']}, {r.get('source', 'reply')}): session **{r['judge']['session_matches']}**, gap {r['judge']['gap']}"
                 + (f", recirculates {f['later_turns_hit']}/{f['later_turns']}" if f else "")
                 + f" — {r['message'][:100].replace(chr(10), ' ')}")
    L += ["", "## Reading it", "",
          "- A reach under 100% in A is text a phi reader never sees: an error in the phi_store column is a reader that "
          "still wants the old layout (sessions/, mrindex/); phi_rank follows git and reads everything the store's "
          ".gitignore keeps. A share over 100% counts a message and its reply once each on top of the entry they share.",
          "- B's served share is the share of owner turns where recall had lines (shown or shadow: the served log "
          "does not survive the turn, so the two are not told apart here); the used share is the hooks' lexical "
          "label, checked in E.",
          "- C compares three rankers on the same held-out pools; the json ranker's own eval column is its last "
          "round on its own split, a reference not a comparison.",
          "- D's hooks road is the labels the models train on: its precision against the teacher is the label noise. "
          "The silence line is the other half of precision: a teacher or road that picks for an ordinary prompt "
          "would have printed a wrong line (findings 21, 22 and 30).",
          "- E's KNOWN_NOT_SERVED is a recall failure, SERVED_NOT_FOLLOWED an adherence failure, NEW_KNOWLEDGE "
          "neither.",
          "- F's gap is where the model's default recommendation and the documented one part: a session that matched "
          "DEFAULT there handed the owner a guess. A store whose facts.md stays empty cannot serve the documented "
          "answer next time; the draft is what the grounded arm would have left in it, and its recirculation count "
          "is how often the store's later turns would have recalled it."]
    (paths()["results"] / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = {"fake": FAKE, "spent_usd": spent(),
               "layout": {b: {"coverage": x["coverage"], "phi_store_error": x["phi_store"]["error"]} for b, x in layout.items()},
               "served": {b: {k: x[k] for k in ("owner_turns", "served_turns", "served_share", "served_ids", "used_share",
                                                 "corrections", "retractions", "cases_known_lexical", "cases_recall_served_any")}
                          for b, x in served.items()},
               "bridge": {b: {k: x.get(k) for k in ("pools", "bm25", "pt_fresh", "pt_pretrained", "error")} for b, x in bridge.items()},
               "teacher": {b: x["summary"] for b, x in teacher.items()},
               "adherence": {k: adherence[k] for k in ("classes", "cases_judged", "usage_agreement", "usage_checked")},
               "epistemic": {k: epistemic[k] for k in ("sampled", "by_source", "session_matches", "gap", "documented_is_newer",
                                                       "default_basis", "grounded_basis", "facts_drafted", "facts_recirculating",
                                                       "later_turns_hit", "recommendation_turns", "facts_entries")}}
    (paths()["results"] / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def run(paid=False, work=None, store=None, project=None):
    configure(FAKE=not paid, **({"WORK": work} if work else {}), **({"STORE_REPO": store} if store else {}),
              **({"PROJECT_REPO": project} if project is not None else {}))
    require_store()
    if not paid:
        shutil.rmtree(paths()["results"], ignore_errors=True)
    layout = step_layout()
    served = step_served_back()
    bridge = step_bridge()
    teacher = step_teacher(bridge)
    adherence = step_adherence(served)
    epistemic = step_epistemic(served)
    s = step_report(layout, served, bridge, teacher, adherence, epistemic)
    say((paths()["results"] / "report.md").read_text(encoding="utf-8"))
    return s


def rejudge(paid=False, work=None, store=None, project=None):
    """Steps 4 to 6 again on the last run's free results (for a changed prompt, model or sample); the last results
    are copied to results-<n>/ first and the spend carries over into the cap."""
    configure(FAKE=not paid, **({"WORK": work} if work else {}), **({"STORE_REPO": store} if store else {}),
              **({"PROJECT_REPO": project} if project is not None else {}))
    require_store()
    p = paths()
    load = lambda n: json.loads((p["results"] / n).read_text(encoding="utf-8"))  # noqa: E731
    layout, served, bridge = load("layout.json"), load("served_back.json"), load("bridge.json")
    n = 1 + sum(1 for d in p["work"].glob("results-*") if d.is_dir())
    shutil.copytree(p["results"], p["work"] / f"results-{n}")
    teacher = step_teacher(bridge)
    adherence = step_adherence(served)
    epistemic = step_epistemic(served)
    s = step_report(layout, served, bridge, teacher, adherence, epistemic)
    say((p["results"] / "report.md").read_text(encoding="utf-8"))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rejudge"])
    ap.add_argument("--paid", action="store_true")
    ap.add_argument("--work")
    ap.add_argument("--store", help="the store repository, a URL or a local path (else P4_STORE_REPO from the environment or .env)")
    ap.add_argument("--project", help="the checkout the store's repo: labels name, when this machine has it")
    a = ap.parse_args(argv)
    {"run": run, "rejudge": rejudge}[a.cmd](a.paid, a.work, a.store, a.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
