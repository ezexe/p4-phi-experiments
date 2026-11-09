#!/usr/bin/env python3
"""
keep_or_go.py: which directories of a per-machine p4 store earn their keep, and whether repo-local skills can move
into the p4 marketplace as generic skills, judged by the Claude API with no Claude Code session in the loop.

Why the API and not a session: a Claude Code session loads CLAUDE.md, the auto-memory, p4-recall's lines and the
owner's history, so its keep/go call leans on how the owner has worked. Here every request is a bare Messages call
with a neutral system prompt; the evidence is computed from git, never written by an agent; owner names and
honorifics are scrubbed; the unit order is shuffled per seed; and two model families judge. `ablate` measures the
remaining lean directly: the same evidence unscrubbed, and the verdicts that change are reported.

    python experiments/keep_or_go/keep_or_go.py run --store <url or path> [--skill <dir>]...   # FAKE: offline, no cost
    python experiments/keep_or_go/keep_or_go.py run --store <url or path> --skill <dir> --paid  # capped at TOTAL_MAX_USD
    ... --project <checkout>     # the repository whose .claude/p4/p4.json the marketplace skills' settings read

Nothing project-specific is built in. --store (or P4_STORE_REPO) is a p4 store repository, one branch per machine;
--skill (or P4_SKILL_DIRS, separated by the OS path separator) is a handwritten skill directory to test against the
marketplace (none: steps 4 and 5 are skipped); the owner's names are read from the store's committers and scrubbed.

Steps (each a function the notebook keep_or_go.ipynb beside this file calls, in this order):
  1. inventory  clone the store repo, one row per unit (directory) per branch: files, bytes, commits, last touch,
                identical-across-branches, what references it in the marketplace and phi code (free)
  2. judge      KEEP / MIGRATE / DROP per unit, by each of JUDGE_MODELS x SEEDS, structured output (paid)
  3. ablate     the same, unscrubbed, on the first judge model and seed: the verdicts the owner's wording moves (paid)
  4. rules      each rule of each skill: GENERIC / REPO_SPECIFIC / OBSOLETE, and the generic rewrite (paid)
  5. probe      probe tasks answered with the handwritten skill, the generic draft and no skill, graded by
                PROBE_JUDGE against the rubric each probe carries: does the generic draft keep the behaviour (paid)
  6. report     results/report.md and results/summary.json under WORK (free)
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

# ---- settings (the notebook's settings cell overrides these through configure) ----
FAKE = True
TOTAL_MAX_USD = 6.0
STORE_REPO = ""                                 # --store, or P4_STORE_REPO: a URL or a local path
BRANCHES = []                                   # [] = every remote branch
SKILL_DIRS = []                                 # --skill (repeatable), or P4_SKILL_DIRS; [] skips steps 4 and 5
REFERENCE_ROOTS = [str(ROOT / "p4"), str(ROOT / "phi")]   # code whose mentions of a unit count as a reference
JUDGE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5"]          # two families' calls, so one model's lean shows
SEEDS = [1, 2]
EFFORT = "high"
PROBE_MODEL = "claude-opus-5-5"                 # the day-to-day agent the skill is written for
PROBE_JUDGE = "claude-sonnet-5-5"
PROBE_ARMS = ["handwritten", "generic", "plugin", "none"]
PROBE_REPEATS = 2                               # independent answers per probe and arm (the model takes no seed)
PLUGIN_SKILLS = str(ROOT / "p4" / "lbiza" / "plugins" / "p4-plans" / "skills")   # the marketplace skills under test
PLUGIN_REPO = ""                                # --project, or P4_PROJECT: where the plugin's settings.py runs
GATE_POINTS = 0.05                              # the plugin arm may trail the handwritten one by this much
SCRUB = {r"\bthe boss's\b": "the owner's", r"\bthe boss\b": "the owner", r"\bboss\b": "owner"}
SCRUB_AUTHORS = True                            # also scrub the store's committers' names and STORE_REPO's owner
COLLAPSE_OVER = 12                              # a directory with more children than this is one unit
SAMPLE_CHARS = 600                              # per unit: the head of its newest file, scrubbed
WORK = HERE / "out"                             # the clone and results/; ignored by the project's .gitignore

PRICES = {"claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-haiku-5-5": (0.10, 0.50)}

NEUTRAL_SYSTEM = (
    "You review the layout of a repository that stores an AI coding agent's working records for one project, kept "
    "on one branch per machine. You have no history with this repository or its owner. Judge each unit from the "
    "evidence given alone: what it holds, whether anything reads it, whether it duplicates another branch or "
    "another store, how much it costs to keep (size, churn) and what would be lost without it. Popularity with the "
    "owner is not evidence; a unit is not kept because it exists. Say DROP when nothing reads it and its content is "
    "recoverable or stale, MIGRATE when it should exist but somewhere else (another branch, a shared branch, the "
    "generic marketplace, the code repository, git history), KEEP otherwise.")

UNIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["units"], "properties": {"units": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["unit", "verdict", "where", "confidence", "reason"],
                               "properties": {"unit": {"type": "string"},
                                              "verdict": {"type": "string", "enum": ["KEEP", "MIGRATE", "DROP"]},
                                              "where": {"type": "string"},
                                              "confidence": {"type": "number"},
                                              "reason": {"type": "string"}}}}}}

RULE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["rules", "generic_skill"], "properties": {
    "rules": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                         "required": ["rule", "class", "why"],
                                         "properties": {"rule": {"type": "string"},
                                                        "class": {"type": "string",
                                                                  "enum": ["GENERIC", "REPO_SPECIFIC", "OBSOLETE"]},
                                                        "why": {"type": "string"}}}},
    "generic_skill": {"type": "string"}}}

GRADE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["checks"], "properties": {"checks": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["criterion", "pass"],
                               "properties": {"criterion": {"type": "string"}, "pass": {"type": "boolean"}}}}}}

# Probe tasks: behaviour a generic delegation/planning skill should produce in any repository. The rubric names the
# behaviour, never this owner's vocabulary, so a generic draft is not penalised for dropping it.
PROBES = [
    {"id": "delegate-small", "skill": "sub-agentic",
     "prompt": "Find which file defines the function parse_config in a 40-file Python repo and tell me its line.",
     "rubric": ["does the lookup itself or says it would, rather than spawning a subagent",
                "does not propose more than one agent"]},
    {"id": "delegate-audit", "skill": "sub-agentic",
     "prompt": "Audit all 30 services in this monorepo for unpinned dependencies. Plan the delegation before starting.",
     "rubric": ["states the agent count or its bound before any launch",
                "names a model or effort per role rather than inheriting one model for all",
                "bounds any verify or retry loop",
                "says verifiers or finders are read-only, or that writers get disjoint files"]},
    {"id": "delegate-brief", "skill": "sub-agentic",
     "prompt": "Write the brief you would give a subagent that must find every call site of a deprecated API.",
     "rubric": ["states the objective", "states the output form", "states what the agent may and may not write",
                "states when it is done"]},
    {"id": "plan-where", "skill": "plan-planner",
     "prompt": "Write a migration plan for moving our logging library to structured logs. Where do you save it?",
     "rubric": ["names one fixed location for in-flight plans rather than the repo root or a temp directory",
                "says the plan is written once and its later history lives elsewhere (changelog, git)"]},
    {"id": "plan-tests-first", "skill": "plan-planner",
     "prompt": "Lay out section 1 of a plan that changes how retries back off in the HTTP client.",
     "rubric": ["lays out the tests before the implementation",
                "keeps the implementation to what the tests need",
                "shows edits as complete code fences, not ellipses"]},
    {"id": "plan-close", "skill": "plan-planner",
     "prompt": "The plan's sections have all landed. What do you do to close it?",
     "rubric": ["checks for edits that landed outside the plan's sections",
                "re-anchors the evidence links to the current tree",
                "records what was and was not run",
                "moves or archives the plan rather than editing it further"]},
]


def configure(**over):
    for k, v in over.items():
        if k not in globals():
            raise KeyError(k)
        globals()[k] = Path(v) if k == "WORK" else v


def say(*a):
    print(*a, flush=True)


def paths():
    w = Path(WORK)
    p = {"work": w, "repo": w / "store-repo", "results": w / "results"}
    p["results"].mkdir(parents=True, exist_ok=True)
    return p


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
    """Offline stand-in: answers every schema with a deterministic, well-formed object, so the notebook and the test
    run end to end without a key. Its verdicts are not a measurement."""

    def __init__(self):
        self.beta = SimpleNamespace(messages=self)

    def create(self, model, messages, system="", output_config=None, **kw):
        prompt = messages[-1]["content"]
        fmt = (output_config or {}).get("format")
        if not fmt:   # a probe answer: carries the skill's marker so the fake grader can tell the arms apart
            text = "fake answer" + (" SKILL" if "SKILL:" in system else "")
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                                   usage=SimpleNamespace(input_tokens=len(prompt) // 4, output_tokens=3))
        schema = fmt["schema"]
        if "units" in schema["properties"]:
            names = re.findall(r"^### (\S+)", prompt, re.M)
            out = {"units": [{"unit": n, "verdict": ["KEEP", "MIGRATE", "DROP"][len(n) % 3], "where": "",
                              "confidence": 0.5, "reason": "fake"} for n in names]}
        elif "rules" in schema["properties"]:
            heads = re.findall(r"^\d+\. \*\*(.+?)\*\*", prompt, re.M)[:5] or ["fake rule"]
            out = {"rules": [{"rule": h, "class": "GENERIC", "why": "fake"} for h in heads],
                   "generic_skill": "---\nname: fake\ndescription: fake\n---\n# fake\n"}
        else:
            crit = re.findall(r"^- (.+)$", prompt, re.M)
            out = {"checks": [{"criterion": c, "pass": "SKILL" in prompt} for c in crit]}
        text = json.dumps(out)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4))


def client():
    if FAKE:
        return FakeClient()
    load_env()
    import anthropic
    return anthropic.Anthropic(max_retries=6)


def load_env():
    """The project's .env (experiments/env.py: the nearest one up the tree, never overriding what is set)."""
    sys.path.insert(0, str(HERE.parent))
    import env
    return env.load()


def ask(c, model, prompt, schema=None, system=NEUTRAL_SYSTEM, max_tokens=32000):
    """One bare request: no tools, no memory, no history. Streams when real (long input and output)."""
    guard(f"request to {model}")
    kw = dict(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": prompt}],
              output_config={"effort": EFFORT, **({"format": {"type": "json_schema", "schema": schema}}
                                                   if schema else {})},
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
    text = next(b.text for b in r.content if b.type == "text")
    return json.loads(text) if schema else text


# ---- step 1: inventory (free) ----
def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


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
            _OWNER_TERMS[key] = sorted(terms - {"noreply", "users", "github", "gmail", "anthropic", "claude", "fable"},
                                       key=len, reverse=True)
        else:
            return sorted(terms, key=len, reverse=True)       # before the clone: not cached, so it is read again
    return _OWNER_TERMS[key]


def scrub(text):
    text = text or ""
    for pat, rep in SCRUB.items():
        text = re.sub(pat, rep, text, flags=re.I)
    if SCRUB_AUTHORS:
        for term in owner_terms():
            text = re.sub(rf"\b{re.escape(term)}\w*", "owner", text, flags=re.I)
    return text


def require_store():
    """STORE_REPO, SKILL_DIRS and PLUGIN_REPO from the arguments or the notebook, else P4_STORE_REPO, P4_SKILL_DIRS
    and P4_PROJECT (the environment or .env); a run without a store stops here: this project ships none of its own."""
    global STORE_REPO, SKILL_DIRS, PLUGIN_REPO
    load_env()
    STORE_REPO = STORE_REPO or os.environ.get("P4_STORE_REPO", "")
    SKILL_DIRS = SKILL_DIRS or [d for d in os.environ.get("P4_SKILL_DIRS", "").split(os.pathsep) if d.strip()]
    PLUGIN_REPO = PLUGIN_REPO or os.environ.get("P4_PROJECT", "")
    if not STORE_REPO:
        raise SystemExit("keep_or_go: no store. Pass --store <URL or path of a p4 store repository>, or set "
                         "P4_STORE_REPO in .env (see .env.example).")
    if not SKILL_DIRS:
        say("keep_or_go: no --skill given; the rules and probe steps (4, 5) are skipped")
    return STORE_REPO


def clone():
    require_store()
    repo = paths()["repo"]
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--no-checkout", STORE_REPO, str(repo)], check=True)
    git(repo, "fetch", "-q", "--all", "--prune")
    names = [b.strip().removeprefix("origin/") for b in git(repo, "branch", "-r", "--format=%(refname:short)").split()
             if b.strip().startswith("origin/") and "HEAD" not in b and b.strip() != "origin"]
    return repo, (BRANCHES or names)


def unit_of(path, wide):
    parts = path.split("/")
    if len(parts) == 1:
        return "(root files)"
    return parts[0] if parts[0] in wide or len(parts) == 2 else "/".join(parts[:2])


def reference_needle(unit):
    """The unit's path as code names it: written out (`flights/`, `"store/messages`, `store\\messages`), or built
    from quoted parts (`p4 / "flights"`, `os.path.join(root, "store", "messages")`, `Path(p4, "memory")`)."""
    parts = [re.escape(p) for p in unit.split("/")]
    written = r"[/\\\"'`]" + r"[/\\]".join(parts) + (r"[/\\]" if len(parts) == 1 else r"\b")
    built = r"[/,(]\s*" + r"\s*[/,]\s*".join(rf"[\"']{p}[\"']" for p in parts)
    return re.compile(f"{written}|{built}")


def reference_count(unit):
    """Files under REFERENCE_ROOTS (code and docs, not the store itself) that name the unit's path."""
    leaf = unit.split("/")[-1]
    if unit == "(root files)" or len(leaf) < 4:
        return {}
    needle = reference_needle(unit)
    hits = Counter()
    for root in REFERENCE_ROOTS:
        root = Path(root)
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.suffix not in {".py", ".sh", ".json", ".md", ".toml"} or ".git" in f.parts or "keep-or-go" in f.parts:
                continue
            try:
                if needle.search(f.read_text(encoding="utf-8", errors="ignore")):
                    hits[root.name] += 1
            except OSError:
                pass
    return dict(hits)


def step_inventory():
    repo, branches = clone()
    rows = defaultdict(dict)
    for b in branches:
        ref = f"origin/{b}"
        files = [ln.split(None, 4) for ln in git(repo, "ls-tree", "-r", "-l", ref).splitlines()]
        files = [(f[2], int(f[3]) if f[3].isdigit() else 0, f[4]) for f in files]
        top_children = defaultdict(set)
        for _sha, _sz, p in files:
            parts = p.split("/")
            if len(parts) > 2:
                top_children[parts[0]].add(parts[1])
        wide = {d for d, ch in top_children.items() if len(ch) > COLLAPSE_OVER}
        by_unit = defaultdict(list)
        for f in files:
            by_unit[unit_of(f[2], wide)].append(f)
        for u, fs in by_unit.items():
            log = git(repo, "log", "--format=%cs", ref, "--", *(["."] if u == "(root files)" else [u])).split()
            newest = max(fs, key=lambda f: f[1])[2]
            sample = git(repo, "show", f"{ref}:{newest}")[:SAMPLE_CHARS] if not newest.endswith(".log") else ""
            rows[u][b] = {"files": len(fs), "bytes": sum(f[1] for f in fs), "commits": len(log),
                          "last": log[0] if log else "", "blobs": sorted(f[0] for f in fs),
                          "children": len(top_children.get(u, ())) if u in wide else None,
                          "sample_path": newest, "sample": sample}
    inv = []
    for u in sorted(rows):
        per = rows[u]
        blob_sets = {b: set(v["blobs"]) for b, v in per.items()}
        shared = set.intersection(*blob_sets.values()) if len(blob_sets) > 1 else set()
        allb = set.union(*blob_sets.values())
        inv.append({"unit": u, "branches": {b: {k: v for k, v in d.items() if k != "blobs"} for b, d in per.items()},
                    "absent_on": [b for b in branches if b not in per],
                    "identical_share": round(len(shared) / len(allb), 2) if allb else 0.0,
                    "referenced_by": reference_count(u)})
    out = paths()["results"] / "inventory.json"
    out.write_text(json.dumps({"repo": STORE_REPO, "branches": branches, "units": inv}, indent=1))
    say(f"inventory: {len(inv)} units over {branches} -> {out}")
    return inv


def evidence(inv, scrubbed=True, seed=0):
    units = list(inv)
    random.Random(seed).shuffle(units)
    f = scrub if scrubbed else (lambda s: s)
    out = ["Units of the store, one per section. For each: per branch, files / bytes / commits touching it / last "
           "commit date; the share of its blobs identical on every branch; the code repositories whose files name its "
           "path; the head of its largest file.", ""]
    for u in units:
        out.append(f"### {u['unit']}")
        for b, d in u["branches"].items():
            extra = f", {d['children']} children" if d["children"] else ""
            out.append(f"- {b}: {d['files']} files, {d['bytes']} B, {d['commits']} commits, last {d['last']}{extra}")
        if u["absent_on"]:
            out.append(f"- absent on: {', '.join(u['absent_on'])}")
        out.append(f"- identical across branches: {u['identical_share']:.0%} of blobs")
        out.append(f"- referenced by: {u['referenced_by'] or 'nothing'}")
        if u["branches"]:
            d = next(iter(u["branches"].values()))
            if d["sample"]:
                out += [f"- head of {d['sample_path']}:", "```", f(d["sample"]).strip(), "```"]
        out.append("")
    out.append("Return one verdict per unit, every unit named exactly as its heading.")
    return "\n".join(out)


# ---- step 2/3: judge, ablate (paid) ----
def step_judge(inv):
    c, res = client(), {}
    for m in JUDGE_MODELS:
        for s in SEEDS:
            r = ask(c, m, evidence(inv, True, s), UNIT_SCHEMA)
            res[f"{m}/s{s}"] = {x["unit"]: x for x in r["units"]}
            say(f"judge {m} seed {s}: {Counter(x['verdict'] for x in r['units'])}  spent ${spent():.2f}")
    (paths()["results"] / "judge.json").write_text(json.dumps(res, indent=1))
    return res


def consensus(res):
    units = sorted({u for r in res.values() for u in r})
    table = []
    for u in units:
        votes = Counter(r[u]["verdict"] for r in res.values() if u in r)
        top, n = votes.most_common(1)[0]
        table.append({"unit": u, "verdict": top, "agreement": round(n / sum(votes.values()), 2), "votes": dict(votes),
                      "reasons": [r[u]["reason"] for r in res.values() if u in r][:2]})
    return table


def step_ablate(inv, res):
    """The first judge and seed again with the owner's wording left in: a verdict that moves is lean, not evidence."""
    m, s = JUDGE_MODELS[0], SEEDS[0]
    raw = {x["unit"]: x for x in ask(client(), m, evidence(inv, False, s), UNIT_SCHEMA)["units"]}
    base = res[f"{m}/s{s}"]
    moved = [{"unit": u, "scrubbed": base[u]["verdict"], "raw": raw[u]["verdict"]}
             for u in base if u in raw and base[u]["verdict"] != raw[u]["verdict"]]
    (paths()["results"] / "ablate.json").write_text(json.dumps(moved, indent=1))
    say(f"ablate: {len(moved)} of {len(base)} verdicts move when the owner's wording is left in")
    return moved


# ---- step 4: rules (paid) ----
def skill_text(d):
    d = Path(d)
    parts = [f"## {p.relative_to(d.parent)}\n\n{p.read_text(encoding='utf-8')}" for p in sorted(d.rglob("SKILL.md"))]
    scripts = [str(p.relative_to(d)) for p in d.rglob("*.py") if "__pycache__" not in p.parts]
    return "\n\n".join(parts) + (f"\n\nScripts shipped with it: {scripts}" if scripts else "")


def step_rules():
    c, out = client(), {}
    for d in SKILL_DIRS:
        name = Path(d).name
        prompt = ("Below is a skill written by hand for one repository. Split it into its rules. Classify each: "
                  "GENERIC (holds in any repository that uses an AI coding agent), REPO_SPECIFIC (names this "
                  "project's tree, machines, branches, people, products or vocabulary), OBSOLETE (restates a default "
                  "the agent already follows, or is superseded inside the skill). Then write `generic_skill`: a "
                  "complete SKILL.md (frontmatter name and description, then the body) that keeps every GENERIC rule, "
                  "turns each REPO_SPECIFIC rule that has a generic core into a setting the repository supplies "
                  "(a path, a branch name, a script), and drops the rest. Keep it shorter than the original.\n\n"
                  + scrub(skill_text(d)))
        out[name] = ask(c, JUDGE_MODELS[0], prompt, RULE_SCHEMA, system="You edit agent skills. You have no history "
                        "with this repository or its owner.")
        k = Counter(r["class"] for r in out[name]["rules"])
        draft = paths()["results"] / f"generic-{name}.SKILL.md"
        draft.write_text(out[name]["generic_skill"], encoding="utf-8")
        say(f"rules {name}: {dict(k)}; generic draft -> {draft}")
    (paths()["results"] / "rules.json").write_text(json.dumps(out, indent=1))
    return out


# ---- step 5: probe (paid) ----
def plugin_text(name):
    """The marketplace skill as the agent would load it: every .md of the skill, then its settings as settings.py
    prints them in PLUGIN_REPO (sub-agentic reads plan-planner's settings too)."""
    d = Path(PLUGIN_SKILLS) / name
    if not d.is_dir():
        return None
    text = "\n\n".join(f"## {f.relative_to(d.parent)}\n\n{f.read_text(encoding='utf-8')}" for f in sorted(d.rglob("*.md")))
    script = Path(PLUGIN_SKILLS) / "plan-planner" / "scripts" / "settings.py"
    out = subprocess.run([sys.executable, "-I", str(script)], cwd=PLUGIN_REPO, capture_output=True, text=True,
                         encoding="utf-8").stdout if PLUGIN_REPO and Path(PLUGIN_REPO).is_dir() else "(defaults; no repository)"
    return text + f"\n\n## Output of `python3 -I {script.as_posix()}` in this repository\n\n```text\n{out}```"


def step_probe(rules):
    c, rows = client(), []
    hand = {Path(d).name: scrub(skill_text(d)) for d in SKILL_DIRS}
    for p in PROBES:
        arms = {"handwritten": hand.get(p["skill"]), "generic": rules.get(p["skill"], {}).get("generic_skill"),
                "plugin": plugin_text(p["skill"]), "none": ""}
        for arm in PROBE_ARMS:
            skill = arms[arm]
            if skill is None:
                continue
            system = ("You are a coding agent working in a repository. "
                      + (f"Follow this SKILL:\n\n{skill}" if skill else ""))
            for rep in range(PROBE_REPEATS):
                answer = ask(c, PROBE_MODEL, p["prompt"], system=system, max_tokens=16000)
                grade = ask(c, PROBE_JUDGE, "Grade the answer against each criterion; pass only when the answer "
                            "clearly does it.\n\nTask:\n" + p["prompt"] + "\n\nAnswer:\n" + answer + "\n\nCriteria:\n"
                            + "\n".join(f"- {r}" for r in p["rubric"]), GRADE_SCHEMA,
                            system="You grade answers strictly against stated criteria.", max_tokens=4000)
                score = sum(x["pass"] for x in grade["checks"]) / max(1, len(p["rubric"]))
                rows.append({"probe": p["id"], "skill": p["skill"], "arm": arm, "rep": rep,
                             "score": round(score, 2), "answer": answer[:4000],
                             "missed": [x["criterion"] for x in grade["checks"] if not x["pass"]]})
    (paths()["results"] / "probe.json").write_text(json.dumps(rows, indent=1))
    summary = probe_summary(rows)
    say("probe:", summary)
    return rows, summary


def probe_summary(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["skill"], r["arm"])].append(r["score"])
    return {f"{s}/{a}": round(sum(v) / len(v), 2) for (s, a), v in sorted(by.items())}


def gate(summary):
    """Per skill: the plugin arm within GATE_POINTS of the handwritten arm and above no skill."""
    out = {}
    for skill in sorted({k.split("/")[0] for k in summary}):
        h, g, n = (summary.get(f"{skill}/{a}") for a in ("handwritten", "plugin", "none"))
        out[skill] = None if h is None or g is None else bool(g >= h - GATE_POINTS and (n is None or g > n))
    return out


# ---- step 6: report (free) ----
def step_report(inv, res, moved, rules, probe_summary):
    table = consensus(res)
    L = [f"# Keep or go: {STORE_REPO}", "",
         f"{'FAKE run: offline answers, not a measurement. ' if FAKE else ''}Judges {JUDGE_MODELS} x seeds {SEEDS}, "
         f"scrubbed evidence, shuffled order; spent ${spent():.2f} of ${TOTAL_MAX_USD}.", "",
         "| unit | verdict | agreement | votes | reason |", "|---|---|---|---|---|"]
    for t in table:
        L.append(f"| `{t['unit']}` | {t['verdict']} | {t['agreement']:.0%} | {t['votes']} | "
                 f"{(t['reasons'] or [''])[0].replace('|', '/')[:160]} |")
    L += ["", f"## Lean check: {len(moved)} verdict(s) move with the owner's wording left in", ""]
    L += [f"- `{m['unit']}`: {m['scrubbed']} scrubbed, {m['raw']} raw" for m in moved] or ["- none"]
    L += ["", "## Skills", ""]
    for name, r in rules.items():
        L.append(f"- **{name}**: {dict(Counter(x['class'] for x in r['rules']))}; draft `results/generic-{name}.SKILL.md`")
    L += ["", "## Probe scores (mean share of rubric met)", ""]
    L += [f"- {k}: {v:.0%}" for k, v in probe_summary.items()]
    g = gate(probe_summary)
    L += ["", f"Gate (plugin within {GATE_POINTS:.0%} of handwritten and above none): "
          + ", ".join(f"{k} {'passes' if v else 'fails' if v is False else 'not measured'}" for k, v in g.items())]
    L += ["", "A generic draft is fit to replace the handwritten skill when its score is within a few points of the "
          "handwritten arm and clearly above `none`; a unit with agreement under 75% or a lean-check move is the "
          "owner's call, not the judges'."]
    (paths()["results"] / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = {"fake": FAKE, "spent_usd": spent(), "consensus": table, "lean_moves": moved,
               "probe": probe_summary, "gate": gate(probe_summary)}
    (paths()["results"] / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def run(paid=False, work=None, store=None, skills=None, project=None):
    configure(FAKE=not paid, **({"WORK": work} if work else {}), **({"STORE_REPO": store} if store else {}),
              **({"SKILL_DIRS": list(skills)} if skills else {}), **({"PLUGIN_REPO": project} if project else {}))
    require_store()
    if not paid:
        shutil.rmtree(paths()["results"], ignore_errors=True)
    inv = step_inventory()
    res = step_judge(inv)
    moved = step_ablate(inv, res)
    rules = step_rules()
    _rows, ps = step_probe(rules)
    s = step_report(inv, res, moved, rules, ps)
    say((paths()["results"] / "report.md").read_text(encoding="utf-8"))
    return s


def reprobe(paid=False, work=None, store=None, skills=None, project=None):
    """Step 5 again with the last run's rules (for a changed plugin skill or PROBE_ARMS); the last results are copied
    to results-<n>/ first. Rewrites the report with the last judges' verdicts."""
    configure(FAKE=not paid, **({"WORK": work} if work else {}), **({"STORE_REPO": store} if store else {}),
              **({"SKILL_DIRS": list(skills)} if skills else {}), **({"PLUGIN_REPO": project} if project else {}))
    require_store()
    p = paths()
    rules = json.loads((p["results"] / "rules.json").read_text(encoding="utf-8"))
    n = 1 + sum(1 for d in p["work"].glob("results-*") if d.is_dir())
    shutil.copytree(p["results"], p["work"] / f"results-{n}")
    _rows, ps = step_probe(rules)
    inv = json.loads((p["results"] / "inventory.json").read_text(encoding="utf-8"))["units"]
    res = json.loads((p["results"] / "judge.json").read_text(encoding="utf-8"))
    moved = json.loads((p["results"] / "ablate.json").read_text(encoding="utf-8"))
    s = step_report(inv, res, moved, rules, ps)
    say("gate:", s["gate"])
    return s


def rejudge(paid=False, work=None, store=None, skills=None, project=None):
    """Steps 1-3 again over a fresh inventory, keeping the last run's rules and probes (steps 4-5): for a change to
    the evidence. The last results are copied to results-<n>/ first; the spend carries over into the cap."""
    configure(FAKE=not paid, **({"WORK": work} if work else {}), **({"STORE_REPO": store} if store else {}),
              **({"SKILL_DIRS": list(skills)} if skills else {}), **({"PLUGIN_REPO": project} if project else {}))
    require_store()
    p = paths()
    rules = json.loads((p["results"] / "rules.json").read_text(encoding="utf-8"))
    rows = json.loads((p["results"] / "probe.json").read_text(encoding="utf-8"))
    n = 1 + sum(1 for d in p["work"].glob("results-*") if d.is_dir())
    shutil.copytree(p["results"], p["work"] / f"results-{n}")
    by = defaultdict(list)
    for r in rows:
        by[(r["skill"], r["arm"])].append(r["score"])
    ps = {f"{s}/{a}": round(sum(v) / len(v), 2) for (s, a), v in sorted(by.items())}
    inv = step_inventory()
    res = step_judge(inv)
    moved = step_ablate(inv, res)
    s = step_report(inv, res, moved, rules, ps)
    say((p["results"] / "report.md").read_text(encoding="utf-8"))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rejudge", "reprobe"])
    ap.add_argument("--paid", action="store_true")
    ap.add_argument("--work")
    ap.add_argument("--store", help="the store repository, a URL or a local path (else P4_STORE_REPO)")
    ap.add_argument("--skill", action="append", help="a handwritten skill directory to test (repeatable; else P4_SKILL_DIRS)")
    ap.add_argument("--project", help="the repository whose p4.json the marketplace skills' settings read (else P4_PROJECT)")
    a = ap.parse_args(argv)
    {"run": run, "rejudge": rejudge, "reprobe": reprobe}[a.cmd](a.paid, a.work, a.store, a.skill, a.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
