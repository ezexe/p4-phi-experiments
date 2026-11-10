#!/usr/bin/env python3
"""
dictated.py: where standing instructions decide for the model, and whether the model would have decided right
without them.

A skill, a contract or a store's ruling is written on one day and read on every day after. Some of its lines dictate a
decision the facts settle (which version, which model, which limit, which tool), and the facts move. Others set a
manner (short, never ask, plain words) that the agent then applies where it makes the answer worse. In both cases a
recommendation comes out that rests on the instruction and not on a fact, and nothing in the session says so.

This experiment finds those lines and measures each one, through bare Claude API calls with no session in the loop:

  1. extract   every directive in the instruction files (free): a bullet or sentence that tells the agent what to do,
               with the day it was last written (git blame; a store ruling's own `time:`), classed as
                 datable    it names a value that changes: a version, a number with a unit, a model, a word like
                            "latest" or "current", a URL (the values are kept: they are what the probe aims at)
                 manner     it sets how every answer is given: length, format, voice, asking or not asking
                 named      its only datable part is a name in backticks (a file, a command, an identifier)
                 procedure  the rest
               The last two are not probed through the API, since no documentation outside the project can call
               them wrong; step 2 checks them against the project itself.
  2. ground    (free) what the project and its store say about each directive:
                 MISSING    a path it names in backticks exists nowhere it could be meant (the project, the
                            instruction's own directory, its repository, the store's dependency roots)
                 UNSEEN     an identifier it names appears in none of the project's tracked files other than
                            documents (a dependency may still define it: those trees are too large to search)
                 RETRACTED  the store freed or corrected, on a later day than the line was written, a reading the line still
                            dictates (a retraction's `freed`, a correction's `match`)
                 CONTESTED  a later correction or retraction is about the same thing (its lesson or `meant`)
               and, for the store's own records: each correction and retraction is CARRIED when a live ruling or
               an instruction line states its lesson, ORPHAN when the lesson lives only in the record (so recall has
               to serve it, since nothing dictates it).
  3. probe     for a sample (datable first, strongest evidence and oldest first, then manner), one model writes an
               owner message the directive would decide, without quoting it; for a datable directive the message
               must hinge on whether the value it names is still current. That message is answered three ways by
               the same model: bare (no instruction), dictated (the directive in the system prompt), grounded (it must
               search the current documentation; the URLs are the ones the search returned). A judge reads all three.
               With --store, the store's facts (facts.md) are checked the same way: each claim searched again.
  4. verdict   from the judge's fields, mechanically:
                 TRAP        the directive is wrong now and the bare model is right: it makes a right model wrong
                 STALE       the directive and the bare model are both behind the documentation
                 DATED       the owner's choice stands, but the value it names has moved (4.4 is no longer the latest)
                 NEEDED      the directive is right and the bare model is wrong: it carries what the model lacks
                 REDUNDANT   the directive is right and the bare model decides the same: it adds nothing today
                 PREFERENCE-STEERS  no fact settles it, and the bare model would have done otherwise: the line works
                 PREFERENCE-SAME    no fact settles it, and the bare model does the same unprompted
               and beside it: whether the dictated answer followed the directive, whether it was right, and what its
               justification rested on (INSTRUCTION / FACT / BOTH / NEITHER).
  5. train     what the verdicts give the p4 models, initially and over time:
                 facts-draft.md        a facts.md entry per TRAP, STALE and DATED (claim, source, verified, since, checked,
                                       prior = what the directive dictated), the form p4-recall serves under
                                       `inject: rules-facts` and its teacher marks as newer than its training
                 newer-tasks-draft.jsonl  the probe as a task in p4's newer_tasks.jsonl form (the directive's answer
                                       as the bad one), for the suites that validate a store's facts
                 edits.md              per directive: keep, drop, or the rewrite as a dated fact with how to check
                 ledger.tsv            one row per probe: directive, its text's hash, the answering model, the day,
                                       the verdict. A directive is probed again only when its text changed, the
                                       answering model changed, or RECHECK_DAYS passed; a verdict that differs from
                                       the last one for the same text is reported as a flip (the instruction went
                                       stale, or a new model no longer needs it).

    python experiments/dictated/dictated.py list                      # the directives, free
    python experiments/dictated/dictated.py check [--store S] [--project D]   # steps 1 and 2, free: results/ground.md
    python experiments/dictated/dictated.py run                       # FAKE: offline, no key; extraction is real
    python experiments/dictated/dictated.py run --paid [--sample N]   # the probes, capped at TOTAL_MAX_USD
    ... --instructions FILE_OR_DIR ...   # instruction files to read (default: the p4 checkout's skills and contracts)
    ... --store <project>/.claude/p4     # also that store's live rulings, its corrections, retractions and facts
    ... --project DIR                    # the project the instructions are about (default: the store's, else the
                                         # first instruction file's repository)

Nothing is built in: the instruction files are arguments (or P4_INSTRUCTIONS in .env), and store text is scrubbed of
its owner's names the way served_back does it.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "served_back"))
import served_back as sb  # noqa: E402  (the client, the spend guard, the search call and the scrub)

# ---- settings ----
FAKE = True
TOTAL_MAX_USD = 10.0
INSTRUCTIONS = []                               # files or directories; [] = DEFAULT_GLOBS under the p4 checkout
STORE = ""                                      # a store directory (<project>/.claude/p4): its live rulings are read too
DEFAULT_GLOBS = ["lbiza/plugins/*/skills/**/*.md", "scripts/session/fresh/emission-plain.md",
                 ".docs/p4/rebuild-seed/*/*.md"]
PROBE_SAMPLE = 12                               # directives probed per run
MANNER_SHARE = 1 / 3                            # of the sample; the rest is datable
PROBE_MODEL = "claude-sonnet-5-5"               # writes the owner message a directive decides
ANSWER_MODEL = "claude-opus-5-5"                # answers all three arms
JUDGE_MODEL = "claude-opus-5-5"
RECHECK_DAYS = 30                               # a verdict older than this is measured again
PROBE_VERSION = 3                               # 2: a datable probe hinges on the named value; 3: the value gets a
                                                # search of its own. A verdict of another version is measured again
AGREE_MIN = 3                                   # probes that must agree before a line is dropped or rewritten
ONLY = set()                                    # directive ids (or text hashes) to probe, whatever the ledger says
PROJECT = ""                                    # the project the instructions are about; "" = worked out (project_dir)
FACTS_SAMPLE = 12                               # store facts searched again per run, the longest unchecked first
GREP_MAX = 1500                                 # identifiers looked for in the project per run
OVERLAP_MIN, SHARED_MIN = 0.5, 3                # a record and a line are about the same thing: this share of the
                                                # record's rare words, and at least this many of them
MIN_CHARS, MAX_CHARS = 20, 700                  # a directive shorter or longer than this is not one line of instruction
TODAY = ""                                      # YYYY-MM-DD; "" = the clock (the tests set it)
WORK = HERE / "out"

DIRECTIVE_RE = re.compile(r"(?i)\b(must|never|always|do not|don't|should(n't| not)?|shall|only (when|if|after|once)|"
                          r"prefer|defaults? to|instead of|rather than|avoid|required|forbidden|not allowed|unless)\b")
IMPERATIVE_RE = re.compile(r"(?i)^(use|keep|write|run|say|show|set|put|ask|give|settle|check|invoke|launch|name|state|"
                           r"read|call|add|make|treat|leave|end|start|send|pass|cap|pin|load|skip|stop|report|cite|"
                           r"quote|mark|pick|choose|brief|scale|commit|push|save|link)\b")
WEIGHT = {"model": 3, "version": 3, "recency": 2, "url": 1, "number": 1, "named": 0}   # how likely the world moves it
DATABLE = [("version", re.compile(r"\bv\d+(?:\.\d+)*\b|\b\d+\.\d+\.\d+(?:\.\d+)*\b|"       # v7, 4.4.5, or a two-part
                                  r"(?i:\b(?:version|release|cmake|python|node|clang|gcc|msvc|xcode|macos|ubuntu|"
                                  r"windows|chrome|chromium|cef|juce|c\+\+|std|api|sdk|ndk|jdk|java|swift|rust|go)\s?"
                                  r"v?\d+(?:\.\d+)*\b)")),     # number only beside a name that has versions
           ("number", re.compile(r"(?i)\b\d[\d,.]*\s?(k|kb|mb|gb|ms|sec|seconds?|minutes?|hours?|days?|chars?|characters|"
                                 r"lines?|bytes?|tokens?|words?|usd)\b|\d\s?%|\$\s?\d")),
           ("model", re.compile(r"(?i)\b(opus|sonnet|haiku|fable|mythos|claude-[\w.-]+|gpt-?[\w.]+|gemini)\b")),
           ("recency", re.compile(r"(?i)\b(latest|current(ly)?|newest|deprecated|as of|today|nowadays|no longer|not yet|"
                                  r"still|recent(ly)?|up to date|legacy|modern)\b")),
           ("url", re.compile(r"https?://\S+")),
           ("named", re.compile(r"`[^`\n]{2,60}`"))]      # a name alone is not a dated value: class `named`
PATH_RE = re.compile(r"^[\w.@-]+(?:/[\w.@-]+)+/?$|^[\w.@-]+\.(?:md|py|json|sh|cmake|txt|h|hpp|cc|cpp|js|ts|toml|ya?ml|"
                     r"tsv|gn|gni|ipynb|ps1|bat)$")
IDENT_RE = re.compile(r"^[A-Za-z_][\w:.\-]{3,}$")
STOP = frozenset("""that this with from have will when then than they them their there here what which while where
into onto over under must never always should only does done make made uses used using also each every other more
most some such same been being were your yours ours about after before again since until both either neither
because just like very much many any all not are was but for the and its it's out one two own new old""".split())
MANNER_RE = re.compile(r"(?i)\b(concise|short(er)?|brief|plain(ly)?|tone|voice|format|bullets?|headers?|tables?|length|"
                       r"sentences?|paragraphs?|jargon|narrat\w*|preamble|recap|hedg\w*|caveats?|opinion|stance|"
                       r"philosoph\w*|reasoning|think(ing)?|mode|style|verbos\w*|terse|lead with|"
                       r"first (line|sentence)|never ask|don't ask|do not ask|ask (first|before|only)|confirm\w*)\b")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
FIELD_RE = re.compile(r"^\s+([a-z][a-z-]*):\s*(.*)$")
ENTRY_RE = re.compile(r"^(ruling|claim|key|freed|asked|fingerprint|term|correction):\s+(.*)$")

PROBE_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["question", "dictates", "value", "answerable_without"],
                "properties": {"question": {"type": "string"}, "dictates": {"type": "string"},
                               "value": {"type": "string"}, "answerable_without": {"type": "boolean"}}}

FACT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["still_true", "now", "why"],
               "properties": {"still_true": {"type": "string", "enum": ["TRUE", "CHANGED", "UNKNOWN"]},
                              "now": {"type": "string"}, "why": {"type": "string"}}}

ANSWER_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["recommendation", "basis", "sources", "as_of", "confidence"],
                 "properties": {"recommendation": {"type": "string"},
                                "basis": {"type": "string", "enum": ["INSTRUCTION", "DOCUMENTATION", "EXPERIENCE", "GUESS"]},
                                "sources": {"type": "array", "items": {"type": "string"}},
                                "as_of": {"type": "string"},
                                "confidence": {"type": "number"}}}

JUDGE_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["decidable", "value_current", "current_value", "instruction_right", "bare_right",
                             "bare_follows", "dictated_right", "dictated_followed", "dictated_basis", "fact_claim",
                             "rewrite", "why"],
                "properties": {"decidable": {"type": "string", "enum": ["FACT", "PREFERENCE"]},
                               "value_current": {"type": "string", "enum": ["CURRENT", "CHANGED", "UNKNOWN", "NONE"]},
                               "current_value": {"type": "string"},
                               "bare_follows": {"type": "boolean"},
                               "instruction_right": {"type": "boolean"},
                               "bare_right": {"type": "boolean"},
                               "dictated_right": {"type": "boolean"},
                               "dictated_followed": {"type": "boolean"},
                               "dictated_basis": {"type": "string", "enum": ["INSTRUCTION", "FACT", "BOTH", "NEITHER"]},
                               "fact_claim": {"type": "string"},
                               "rewrite": {"type": "string"},
                               "why": {"type": "string"}}}

AGENT_SYSTEM = ("You are an AI coding agent answering a software project's owner. Give the one recommendation you "
                "would hand them, in two or three sentences, and say what it rests on: INSTRUCTION when it rests on "
                "the project's standing instructions, DOCUMENTATION when you are citing a source you can name, "
                "EXPERIENCE when it is common practice you have seen, GUESS otherwise. Name the sources you would "
                "point them at, if any, and the date your knowledge of them is current to.")

PROBE_SYSTEM = ("You write test messages for an AI coding agent. You are given one directive from the standing "
                "instructions the agent works under. You add no opinion on whether the directive is right.")

ADVICE = {"TRAP": "drop it, or replace it with the dated fact: the model decides this right without it, and it is wrong now",
          "STALE": "replace it with the dated fact: the directive and the model's own knowledge are both behind",
          "DATED": "keep the choice and update the value it names: the value has moved since the line was written",
          "REDUNDANT": "it is right, and asked directly the model answers the same without it; a probe cannot show whether the model would act on it unprompted, so read the line before dropping it",
          "NEEDED": "keep it, with its date and source: it carries what the model lacks",
          "PREFERENCE-STEERS": "keep it: the owner's choice, and without it the model does otherwise",
          "PREFERENCE-SAME": "the owner's choice, and the model does the same unprompted: keeping or dropping it changes nothing today"}
VERDICTS = ("TRAP", "STALE", "DATED", "NEEDED", "REDUNDANT", "PREFERENCE-STEERS", "PREFERENCE-SAME")
WRONG = ("TRAP", "STALE", "DATED")
LEDGER_HEAD = ["id", "sha", "class", "written", "checked", "model", "verdict", "followed", "right", "basis", "source",
               "probe"]


def configure(**over):
    for k, v in over.items():
        if k not in globals():
            raise KeyError(k)
        globals()[k] = Path(v) if k == "WORK" else v


def say(*a):
    print(*a, flush=True)


def today():
    return TODAY or time.strftime("%Y-%m-%d")


def results():
    d = Path(WORK) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sync():
    """served_back's settings follow this module's: one spend file, one budget, the same FAKE."""
    sb.configure(FAKE=FAKE, WORK=WORK, TOTAL_MAX_USD=TOTAL_MAX_USD)


# ---- step 1: extract (free) ----
def sha8(text):
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:8]


def git_top(d):
    try:
        return Path(subprocess.run(["git", "-C", str(d), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", check=True).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def blame_dates(path):
    """{line number: the day that line was last written} from git blame; {} when the file is in no repository."""
    try:
        out = subprocess.run(["git", "-C", str(path.parent), "blame", "--line-porcelain", "--", path.name],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    dates, t, n = {}, 0, 0
    for line in out.splitlines():
        if line.startswith("author-time "):
            t = int(line.split()[1])
        elif line.startswith("\t"):
            n += 1
            dates[n] = time.strftime("%Y-%m-%d", time.gmtime(t))
    return dates


def units(text):
    """[{line, end, section, text, fields}]: each bullet whole, each prose sentence, in file order. Front matter,
    code fences, headings (kept as the section), tables, quotes and comments are not instruction lines. A store entry
    (`- ruling: ...` with indented `key: value` fields) yields its ruling and keeps the fields; the store's other
    entry kinds are records, not directives, and yield nothing."""
    lines = text.splitlines()
    skip = 0
    if lines and lines[0].strip() == "---":
        skip = next((i + 1 for i in range(1, len(lines)) if lines[i].strip() == "---"), 0)
    out, cur, section, fence = [], None, "", False

    def flush():
        nonlocal cur
        if not cur:
            return
        body = " ".join(cur["parts"]).strip()
        base = {"line": cur["line"], "end": cur["end"], "section": section, "fields": cur["fields"]}
        if cur["bullet"]:
            m = ENTRY_RE.match(body)
            if not m:
                out.append({**base, "text": body, "entry": ""})
            elif m.group(1) == "ruling":
                out.append({**base, "text": m.group(2).strip(), "entry": "ruling"})
        else:
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'`(\[*])", body):
                if s.strip():
                    out.append({**base, "text": s.strip(), "entry": ""})
        cur = None

    for n, raw in enumerate(lines, 1):
        if n <= skip:
            continue
        s = raw.strip()
        if s.startswith("```") or s.startswith("~~~"):
            flush()
            fence = not fence
            continue
        if fence:
            continue
        if not s:
            flush()
            continue
        if s.startswith("#"):
            flush()
            section = s.lstrip("#").strip()
            continue
        if s[0] in "|>" or s.startswith("<!--"):
            flush()
            continue
        m = BULLET_RE.match(raw)
        if m:
            flush()
            cur = {"bullet": True, "line": n, "end": n, "parts": [m.group(1).strip()], "fields": {}}
        elif cur and cur["bullet"] and raw[:1] in " \t":
            f = FIELD_RE.match(raw)
            if f and ENTRY_RE.match(cur["parts"][0]):
                cur["fields"][f.group(1)] = f.group(2).strip()
            else:
                cur["parts"].append(s)
            cur["end"] = n
        else:
            if not cur or cur["bullet"]:
                flush()
                cur = {"bullet": False, "line": n, "end": n, "parts": [], "fields": {}}
            cur["parts"].append(s)
            cur["end"] = n
            if s[-1] in ".!?":
                flush()
    flush()
    return out


def classify(text):
    """(class, the datable kinds found, strength, manner?, values): datable when a value that changes is named;
    manner when the line sets how answers are given; named when a backticked name is all it carries; procedure
    otherwise. Strength adds the kinds' weights (a model or a version outweighs a number with a unit, which is as
    often the project's own measurement); values are the dated strings themselves, the heaviest kind first."""
    kinds = [k for k, rx in DATABLE if rx.search(text)]
    strength = sum(WEIGHT[k] for k in kinds)
    manner = bool(MANNER_RE.search(text))
    cls = "datable" if strength else "manner" if manner else "named" if kinds else "procedure"
    values = []
    for k, rx in sorted(DATABLE, key=lambda x: -WEIGHT[x[0]]):
        if k != "named":
            values += [m.group(0).strip() for m in rx.finditer(text)][:3]
    return cls, kinds, strength, manner, list(dict.fromkeys(values))[:6]


def is_directive(u):
    t = re.sub(r"^\*\*[^*]+\*\*:?\s*", "", u["text"]).lstrip("*_ ")      # past a bold label, to the first word
    return bool(u["entry"] or DIRECTIVE_RE.search(u["text"]) or IMPERATIVE_RE.match(t))


def instruction_files():
    """The files to read: INSTRUCTIONS (each a file, or a directory read for *.md), else P4_INSTRUCTIONS, else the
    p4 checkout's skills and contracts; plus STORE's rules.md. README, CHANGELOG and LICENSE files are not
    instructions."""
    given = list(INSTRUCTIONS) or [p for p in os.environ.get("P4_INSTRUCTIONS", "").split(os.pathsep) if p]
    files = []
    if given:
        for g in given:
            p = Path(g).expanduser()
            files += sorted(p.rglob("*.md")) if p.is_dir() else [p]
    else:
        for g in DEFAULT_GLOBS:
            files += sorted((ROOT / "p4").glob(g))
    if STORE:
        files.append(Path(STORE).expanduser() / "store" / "rules.md")
    seen, out = set(), []
    for f in files:
        f = f.resolve()
        if f in seen or not f.is_file() or f.name.upper() in ("README.MD", "CHANGELOG.MD", "LICENSE.MD"):
            continue
        seen.add(f)
        out.append(f)
    return out


def extract(files=None):
    """Every directive of the instruction files: id (`<path in its repository>:<line>:<sha8 of its text>`), text,
    section, class, the datable kinds, the day it was last written, its age in days."""
    out = []
    now = time.mktime(time.strptime(today(), "%Y-%m-%d"))
    for f in (files if files is not None else instruction_files()):
        text = f.read_text(encoding="utf-8", errors="replace")
        top = git_top(f.parent)
        rel = f.relative_to(top).as_posix() if top and top in f.parents else f.name
        dates = blame_dates(f)
        mtime = time.strftime("%Y-%m-%d", time.gmtime(f.stat().st_mtime))
        for u in units(text):
            if not (MIN_CHARS <= len(u["text"]) <= MAX_CHARS) or not is_directive(u):
                continue
            if u["entry"] and u["fields"].get("status", "LIVE").upper() not in ("LIVE", ""):
                continue
            written = (u["fields"].get("time", "")[:10]
                       or max((dates.get(n, "") for n in range(u["line"], u["end"] + 1)), default="") or mtime)
            cls, kinds, strength, manner, values = classify(u["text"])
            try:
                age = int((now - time.mktime(time.strptime(written, "%Y-%m-%d"))) // 86400)
            except ValueError:
                age = 0
            out.append({"id": f"{rel}:{u['line']}:{sha8(u['text'])}", "sha": sha8(u["text"]), "file": rel,
                        "line": u["line"], "section": u["section"], "text": u["text"], "class": cls, "kinds": kinds,
                        "strength": strength, "manner": manner, "values": values, "written": written,
                        "age_days": max(age, 0), "from_store": bool(u["entry"]), "dir": str(f.parent),
                        "top": str(top or "")})
    return out


def step_extract():
    files = instruction_files()
    ds = extract(files)
    (results() / "directives.json").write_text(json.dumps(ds, indent=1), encoding="utf-8")
    by = Counter(d["class"] for d in ds)
    say(f"extract: {len(ds)} directives in {len(files)} files: {dict(by)}")
    return ds


# ---- step 2: ground (free) ----
def project_dir(directives=()):
    """The project the instructions are about: PROJECT, else the store's (two levels above a .claude/p4), else the
    first instruction file's repository."""
    if PROJECT:
        return str(Path(PROJECT).expanduser())
    if STORE:
        s = Path(STORE).expanduser().resolve()
        if s.name == "p4" and s.parent.name == ".claude":
            return str(s.parent.parent)
    return next((d["top"] for d in directives if d.get("top")), "")


def dependency_bases():
    """The store's dependency roots (deps.json, as p4-recall's deps.py found them), each with its directories and
    their src/: where a path a skill about a framework names may live, outside the project."""
    if not STORE:
        return []
    try:
        roots = list(json.loads((Path(STORE).expanduser() / "models" / "deps.json").read_text(encoding="utf-8"))["roots"])
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for r in roots:
        out.append(r)
        try:
            subs = [e.path for e in os.scandir(r) if e.is_dir()]
        except OSError:
            continue
        out += subs + [os.path.join(x, "src") for x in subs if os.path.isdir(os.path.join(x, "src"))]
    return out


def names_of(text):
    """(paths, identifiers) among a line's backticked names. A name with a placeholder, a flag, a machine path or a
    plain word is neither; an identifier has to look like one (an underscore, `::`, a dot, a capital inside, a digit)."""
    paths, idents = [], []
    for raw in re.findall(r"`([^`\n]{2,80})`", text):
        n = raw.strip().split()[0] if " " in raw.strip() else raw.strip()
        if not n or n[0] in "-/~$<.%" or re.search(r"[<>{}*$|=\\]|\.\.\.|^[A-Za-z]:", n) or "://" in n:
            continue
        n = n.rstrip(".,;:)")
        if PATH_RE.match(n):
            paths.append(n.rstrip("/"))
        elif IDENT_RE.match(n) and re.search(r"_|::|\.|[a-z][A-Z]|\d", n):
            idents.append(n)
    return list(dict.fromkeys(paths)), list(dict.fromkeys(idents))


def grep_project(project, idents):
    """The identifiers no tracked file of the project other than a Markdown document contains (git grep -F, one
    word each, eight at a time)."""
    from concurrent.futures import ThreadPoolExecutor
    if not project or not idents:
        return set()

    def absent(n):
        r = subprocess.run(["git", "-C", project, "grep", "-q", "-I", "-F", "-e", n, "--", ".", ":(exclude)*.md"],
                           capture_output=True)               # documents are out: the instruction itself would match
        return n if r.returncode == 1 else None            # 1: no match; anything else (no repository): not judged
    with ThreadPoolExecutor(8) as ex:
        return {n for n in ex.map(absent, idents) if n}


def toks(text):
    return {w for w in re.findall(r"[a-z][a-z0-9_]{3,}", (text or "").lower()) if w not in STOP}


def about(record, line, idf):
    """(share of the record's rare words the line holds, the shared words): the two are about the same thing when
    the share reaches OVERLAP_MIN on at least SHARED_MIN words."""
    shared = record & line
    total = sum(idf.get(t, 1.0) for t in record)
    return (sum(idf.get(t, 1.0) for t in shared) / total if total else 0.0), shared


def store_records():
    """The store's corrections and retractions: [{id, kind, time, wrong (the reading that was freed or corrected),
    lesson (what holds instead)}], and its facts [{id, claim, source, verified, time}]."""
    if not STORE:
        return [], []
    st = sb.recall().st
    p4 = str(Path(STORE).expanduser())
    recs, facts = [], []
    for e in st.load_file(p4, "retractions.md"):
        recs.append({"id": e.id, "kind": "retraction", "time": (e.time or "")[:10], "wrong": e.value,
                     "lesson": e.fields.get("lesson", "")})
    for e in st.load_file(p4, "corrections.md"):
        recs.append({"id": e.id, "kind": "correction", "time": (e.time or "")[:10],
                     "wrong": e.fields.get("match", ""), "lesson": e.fields.get("meant", "") or e.fields.get("omitted", "")
                     or e.fields.get("delta", "")})
    for e in st.load_file(p4, "facts.md"):
        facts.append({"id": e.id, "claim": e.value, "source": e.fields.get("source", ""),
                      "verified": (e.fields.get("checked") or e.fields.get("verified") or e.time or "")[:10],
                      "time": (e.time or "")[:10]})
    return recs, facts


def step_ground(directives):
    """Each directive against the project and the store, and each record against the directives; results/ground.md."""
    project = project_dir(directives)
    for d in directives:
        d["paths"], d["idents"] = names_of(d["text"])
    wanted = list(dict.fromkeys(n for d in directives for n in d["idents"]))[:GREP_MAX]
    unseen = grep_project(project, wanted)
    recs, facts = store_records()
    line_toks = {d["id"]: toks(d["text"]) for d in directives}
    df = {}
    for ts in list(line_toks.values()) + [toks(r["wrong"]) | toks(r["lesson"]) for r in recs]:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    n_docs = max(1, len(line_toks) + len(recs))
    idf = {t: math.log(1 + n_docs / c) for t, c in df.items()}
    for r in recs:
        r["carried_by"] = []
    dep_bases = dependency_bases()
    for d in directives:
        bases = [b for b in (project, d["dir"], d["top"]) if b] + dep_bases
        d["missing"] = [p for p in d["paths"] if not any(os.path.exists(os.path.join(b, p)) for b in bases)]
        d["unseen"] = [n for n in d["idents"] if n in unseen]
        d["retracted_by"], d["contested_by"] = [], []
        lt = line_toks[d["id"]]
        for r in recs:
            share, shared = about(toks(r["wrong"]), lt, idf)
            later = bool(r["time"] and d["written"]) and r["time"] > d["written"]     # a record of the line's own day
                                                                                    # is where it came from, not a challenge
            if share >= OVERLAP_MIN and len(shared) >= SHARED_MIN and later:
                d["retracted_by"].append(r["id"])
                continue
            share, shared = about(toks(r["lesson"]), lt, idf)
            if share >= OVERLAP_MIN and len(shared) >= SHARED_MIN:
                r["carried_by"].append(d["id"])
                if later:
                    d["contested_by"].append(r["id"])
        d["ground"] = ("RETRACTED" if d["retracted_by"] else "MISSING" if d["missing"] else "UNSEEN" if d["unseen"]
                       else "CONTESTED" if d["contested_by"] else "STANDS" if d["paths"] or d["idents"] else "NOTHING NAMED")
    for r in recs:
        r["state"] = "CARRIED" if r["carried_by"] else "ORPHAN"
    by = Counter(d["ground"] for d in directives)
    L = ["# ground: the directives against the project and the store", "",
         f"Project: `{Path(project).name if project else '(none)'}`; {len(directives)} directives; "
         f"{len(wanted)} identifiers looked for in its tracked files.", "",
         "| result | directives | what it means |", "|---|---|---|",
         f"| RETRACTED | {by['RETRACTED']} | the store later freed or corrected the reading this line still dictates |",
         f"| MISSING | {by['MISSING']} | a path it names exists neither in the project nor in a dependency root |",
         f"| UNSEEN | {by['UNSEEN']} | an identifier it names appears in none of the project's own files (a dependency may define it) |",
         f"| CONTESTED | {by['CONTESTED']} | a later correction or retraction is about the same thing |",
         f"| STANDS | {by['STANDS']} | everything it names is there |",
         f"| NOTHING NAMED | {by['NOTHING NAMED']} | no path or identifier to check |", ""]
    for key, head in (("RETRACTED", "Still dictated after the store freed or corrected it"), ("MISSING", "Names a path that is not there"),
                      ("CONTESTED", "A later record is about the same thing"), ("UNSEEN", "Names an identifier the project does not contain")):
        rows = [d for d in directives if d["ground"] == key]
        if rows:
            L += [f"## {head}", ""]
            L += [f"- `{d['id']}` ({d['class']}, written {d['written']}): {scrub(d['text'])[:200]}"
                  + (f" Missing: {', '.join('`' + p + '`' for p in d['missing'][:4])}." if key == "MISSING" else "")
                  + (f" Unseen: {', '.join('`' + p + '`' for p in d['unseen'][:4])}." if key == "UNSEEN" else "")
                  + (f" Record: {', '.join(d['retracted_by'][:2] or d['contested_by'][:2])}." if key in ("RETRACTED", "CONTESTED") else "")
                  for d in rows[:60]]
            L += [f"- ... and {len(rows) - 60} more (results/directives.json)."] if len(rows) > 60 else []
            L += [""]
    if recs:
        orphan = [r for r in recs if r["state"] == "ORPHAN"]
        L += ["## The store's corrections and retractions", "",
              f"{len(recs)} records: {len(recs) - len(orphan)} CARRIED (a live ruling or an instruction line states the "
              f"lesson), {len(orphan)} ORPHAN (the lesson lives only in the record).", ""]
        L += [f"- ORPHAN {r['kind']} {r['time']}: {scrub(r['lesson'] or r['wrong'])[:180]}" for r in orphan[:40]] + [""]
    out = {"project": project, "by": dict(by), "identifiers": len(wanted), "records": len(recs),
           "orphans": sum(r["state"] == "ORPHAN" for r in recs), "facts": len(facts)}
    (results() / "ground.md").write_text("\n".join(L), encoding="utf-8")
    (results() / "directives.json").write_text(json.dumps(directives, indent=1), encoding="utf-8")
    (results() / "records.json").write_text(json.dumps({"records": recs, "facts": facts}, indent=1), encoding="utf-8")
    say(f"ground: {dict(by)}; records {out['records']} ({out['orphans']} orphan), facts {out['facts']}")
    return {**out, "facts_list": facts}


# ---- the ledger: what was measured, with which model, when ----
def ledger_path():
    """A FAKE run keeps its own ledger: a placeholder verdict must never make a paid run skip a directive."""
    return results() / ("ledger-fake.tsv" if FAKE else "ledger.tsv")


def read_ledger():
    f = ledger_path()
    if not f.exists():
        return []
    rows = [line.split("\t") for line in f.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    return [dict(zip(LEDGER_HEAD, r + [""] * (len(LEDGER_HEAD) - len(r)))) for r in rows]


def append_ledger(rows):
    f = ledger_path()
    head = "" if f.exists() else "\t".join(LEDGER_HEAD) + "\n"
    with f.open("a", encoding="utf-8", newline="\n") as h:
        h.write(head + "".join("\t".join(str(r[k]).replace("\t", " ").replace("\n", " ") for k in LEDGER_HEAD) + "\n"
                               for r in rows))


def days_between(a, b):
    return int((time.mktime(time.strptime(b, "%Y-%m-%d")) - time.mktime(time.strptime(a, "%Y-%m-%d"))) // 86400)


def is_fresh(ledger, d, model=None, now=None):
    """A verdict for this exact text, by this answering model, younger than RECHECK_DAYS: nothing to measure again."""
    model, now = model or ANSWER_MODEL, now or today()
    return any(r["sha"] == d["sha"] and r["model"] == model and r.get("probe") == str(PROBE_VERSION)
               and days_between(r["checked"], now) < RECHECK_DAYS for r in ledger)


def pick(directives, ledger):
    """The run's sample: datable directives by evidence strength then age (oldest first), manner directives by age,
    MANNER_SHARE of the sample for manner and what either lacks filled from the other. Fresh verdicts are skipped."""
    if ONLY:                                    # the lines a decision is waiting on: probed again, fresh or not
        due = [d for d in directives if d["id"] in ONLY or d["sha"] in ONLY]
        return due[:PROBE_SAMPLE], len(due)
    due = [d for d in directives if d["class"] in ("datable", "manner") and not is_fresh(ledger, d)]
    dat = sorted((d for d in due if d["class"] == "datable"), key=lambda d: (-d["strength"], d["written"], d["id"]))
    man = sorted((d for d in due if d["class"] == "manner"), key=lambda d: (d["written"], d["id"]))
    n_man = min(len(man), math.floor(PROBE_SAMPLE * MANNER_SHARE))
    n_dat = min(len(dat), PROBE_SAMPLE - n_man)
    n_man = min(len(man), PROBE_SAMPLE - n_dat)
    return dat[:n_dat] + man[:n_man], len(due)


# ---- step 2 and 3: probe, verdict (paid) ----
class FakeClient:
    """Offline stand-in: one well-formed answer per schema, so the run and its test go end to end without a key.
    Its verdicts are not a measurement."""

    def __init__(self):
        self.beta = SimpleNamespace(messages=self)

    def create(self, model, messages, system="", output_config=None, **kw):
        prompt = messages[-1]["content"]
        props = (output_config or {}).get("format", {}).get("schema", {}).get("properties", {})
        d = re.search(r"Directive:\n(.*)", prompt)
        d = d.group(1)[:120] if d else "the directive"
        if "dictates" in props:
            v = re.search(r"Values it names: (.*)", prompt)
            out = {"question": f"fake: what should I do here, given: {d[:60]}?", "dictates": f"fake: do as told: {d}",
                   "value": v.group(1).split(" | ")[0] if v else "", "answerable_without": True}
        elif "still_true" in props:
            out = {"still_true": "CHANGED", "now": "fake: the documentation says otherwise now", "why": "fake"}
        elif "decidable" in props:
            out = {"decidable": "FACT", "value_current": "CHANGED", "current_value": "fake-newer", "bare_follows": False,
                   "instruction_right": False, "bare_right": True, "dictated_right": False,
                   "dictated_followed": True, "dictated_basis": "INSTRUCTION", "fact_claim": f"fake fact replacing: {d}",
                   "rewrite": f"fake: as of 2026-01, check the documentation before: {d}", "why": "fake"}
        else:
            told = "standing instructions" in system
            enum = props.get("basis", {}).get("enum", [])
            out = {"recommendation": "fake: do as the instructions say" if told else "fake: do what the documentation says",
                   "basis": "INSTRUCTION" if told and "INSTRUCTION" in enum else "DOCUMENTATION" if "https://" in prompt else "GUESS",
                   "sources": re.findall(r"https?://[^\s)]+", prompt)[:2], "as_of": "2026-01", "confidence": 0.5}
        text = json.dumps(out)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4))


def client():
    return FakeClient() if FAKE else sb.client()


def verdict(j):
    if j["decidable"] != "FACT":
        if j.get("value_current") == "CHANGED":
            return "DATED"
        return "PREFERENCE-SAME" if j.get("bare_follows") else "PREFERENCE-STEERS"
    if j["instruction_right"]:
        return "REDUNDANT" if j["bare_right"] else "NEEDED"
    return "TRAP" if j["bare_right"] else "STALE"


_OWNER = {}


def scrub(text):
    """served_back's scrub, plus the names of STORE's own committers (four letters or more), read from its git."""
    text = sb.scrub(text)
    if STORE and STORE not in _OWNER:
        terms = set()
        try:
            log = subprocess.run(["git", "-C", str(Path(STORE).expanduser()), "log", "--all", "--format=%an%x09%ae"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            log = ""
        for line in log.splitlines():
            name, _tab, email = line.partition("\t")
            for part in re.split(r"[\s.<>_-]+", name) + re.split(r"[._+-]+", email.split("@")[0]):
                if len(part) >= 4 and part.isalpha():
                    terms.add(part.lower())
        _OWNER[STORE] = sorted(terms - {"noreply", "users", "github", "gmail", "anthropic", "claude", "fable"},
                               key=len, reverse=True)
    for term in _OWNER.get(STORE, []):
        text = re.sub(rf"\b{re.escape(term)}\w*", "owner", text, flags=re.I)
    return text


def probe(c, d):
    text, where = scrub(d["text"]), d["section"] or d["file"]
    p = sb.ask(c, PROBE_MODEL,
               "Write one realistic message a project owner could send their coding agent, for which the directive "
               "below would decide what the agent answers or does. The message must not quote, name or hint at the "
               "directive: an agent that never saw it has to decide from the facts. `dictates`: what the directive "
               "makes the agent answer or do for that message, in one or two sentences. `answerable_without`: true "
               "when a careful agent with no such directive could settle the message from current documentation or "
               "from the message itself."
               + (" The directive names values that change with time (a version, a model, a limit, a price, what is "
                  "\"latest\"). Write the message so that its right answer depends on whether one of those values is "
                  "still the current one today, not on the policy around it; `value`: the one the message hinges on, "
                  "as the directive writes it." if d["class"] == "datable" else " `value`: leave it empty.")
               + f"\n\nSection: {where}\nWritten: {d['written']}\n"
               + (f"Values it names: {' | '.join(d.get('values') or [])}\n" if d["class"] == "datable" else "")
               + f"Directive:\n{text}", PROBE_SCHEMA, system=PROBE_SYSTEM)
    q = f"The owner's message:\n\n{p['question']}"
    bare = sb.ask(c, ANSWER_MODEL, q, ANSWER_SCHEMA, system=AGENT_SYSTEM)
    dictated = sb.ask(c, ANSWER_MODEL, q, ANSWER_SCHEMA,
                      system=AGENT_SYSTEM + f"\n\nThe project's standing instructions:\n[{where}] {text}")
    found, urls, searches = sb.search(c, ANSWER_MODEL, q)
    value, vtext, vurls = (p.get("value") or "").strip(), "", []
    if d["class"] == "datable" and value:
        # the grounded arm answers the owner's message; whether the named value is still the current one is its own
        # question, and gets its own search
        vtext, vurls, n = sb.search(
            c, ANSWER_MODEL, f"An instruction written on {d['written']} says:\n\n{text}\n\nIt names `{value}`. From the "
            "primary source (the vendor's release page, reference or price list), find whether that is still the "
            "current one today, and state the current one with its date. If it is the project's own measurement or "
            "choice, which no public source could know, say so in one line and stop.",
            system="You check one dated value against its primary source. You must search; you do not answer from "
                   "memory. Cite the URL you read.")
        searches += n
    grounded = sb.ask(c, PROBE_MODEL, "Below is an agent's answer written after searching documentation. Put it into "
                      "the schema: its recommendation, its basis (DOCUMENTATION only when it cites a source it read), "
                      "the source URLs it cites, the date or version it says the documentation is current to.\n\n" + found,
                      sb.RECOMMEND_SCHEMA, system="You extract structured fields from text; you add nothing.")
    verified = [u for u in grounded["sources"] if u in urls] or urls[:3]
    j = None
    j = sb.ask(c, JUDGE_MODEL,
               "An AI coding agent works under standing instructions. One directive from them is below, with the day "
               "it was written. An owner's message that the directive decides was answered three times by the same "
               "model: bare (no instructions), dictated (the directive in its instructions), and grounded (after "
               "searching the current documentation; the URLs the search returned are listed).\n\n"
               "decidable: FACT when current documentation or the message itself settles what the right answer is; "
               "PREFERENCE when only the owner's taste settles it. value_current: for the value the directive "
               "names (given below), CURRENT when the grounded answer's sources show it is still the current one "
               "today, CHANGED when they show a newer or different one has replaced it (current_value: what it is "
               "now, with the source's date), UNKNOWN when the search does not settle it or the value is the "
               "project's own, NONE when no value is named. A directive that says \"the latest X\" and names a "
               "version is CHANGED once a later version exists, even if the owner's policy still stands. For FACT, "
               "judge against the grounded answer: "
               "instruction_right (what the directive dictates here is right today), bare_right, dictated_right. "
               "bare_follows: the bare answer does what the directive dictates, though it never saw it. "
               "dictated_followed: the dictated answer did what the directive dictates. dictated_basis: what the "
               "dictated answer's justification rests on, INSTRUCTION (it says so because it was told), FACT (a "
               "checkable fact it states), BOTH or NEITHER. fact_claim: one sentence stating the documented fact so "
               "that it holds without this conversation; empty when nothing was documented. rewrite: the directive "
               "reworded so it cannot go stale silently, one line that gives the fact, the date it was checked and "
               "how to check it again, and leaves the decision to the agent; empty when the directive should stay "
               "as it is.\n\n"
               f"Directive (written {d['written']}):\n{text}\n\nWhat it dictates here: {p['dictates']}\n"
               f"The value it names: {value or 'none'}\n"
               + (f"A separate search about that value alone (judge value_current from this):\n{vtext[:2500]}\n"
                  f"URLs it read: {vurls[:4] or 'none'}\n" if vtext else "") + "\n"
               f"Owner's message:\n{p['question']}\n\n"
               f"Bare answer ({bare['basis']}, as of {bare['as_of'] or 'unstated'}):\n{bare['recommendation']}\n\n"
               f"Dictated answer ({dictated['basis']}, as of {dictated['as_of'] or 'unstated'}):\n{dictated['recommendation']}\n\n"
               f"Grounded answer ({grounded['basis']}, as of {grounded['as_of'] or 'unstated'}):\n{grounded['recommendation']}\n"
               f"Search results it read: {verified or 'none'}", JUDGE_SCHEMA)
    return {"id": d["id"], "sha": d["sha"], "class": d["class"], "kinds": d["kinds"], "written": d["written"],
            "age_days": d["age_days"], "directive": d["text"], "section": d["section"], "probe": p, "bare": bare,
            "dictated": dictated, "grounded": {**grounded, "searches": searches, "sources_verified": verified},
            "value_check": {"text": vtext[:3000], "urls": vurls[:5]},
            "judge": j, "verdict": verdict(j), "said_basis": dictated["basis"]}


def step_probe(directives):
    sync()
    ledger = read_ledger()
    sample, due = pick(directives, ledger)
    c, rows = client(), []
    for d in sample:
        r = probe(c, d)
        rows.append(r)
        append_ledger([{"id": r["id"], "sha": r["sha"], "class": r["class"], "written": r["written"], "checked": today(),
                        "model": ANSWER_MODEL, "verdict": r["verdict"], "followed": int(r["judge"]["dictated_followed"]),
                        "right": int(r["judge"]["dictated_right"]), "basis": r["judge"]["dictated_basis"],
                        "source": (r["grounded"]["sources_verified"] or [""])[0], "probe": PROBE_VERSION}])
        say(f"probe {r['id']}: {r['verdict']} followed={r['judge']['dictated_followed']} "
            f"right={r['judge']['dictated_right']} basis={r['judge']['dictated_basis']}  spent ${sb.spent():.2f}")
    (results() / "probes.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return {"rows": rows, "due": due, "ledger_before": ledger}


def step_facts(facts):
    """The store's facts searched again (paid): each claim TRUE, CHANGED or UNKNOWN today; results/facts-recheck.md,
    a CHANGED one drafted as the entry that supersedes it. The longest-unchecked FACTS_SAMPLE of them."""
    sync()
    c, rows = client(), []
    for f in sorted(facts, key=lambda f: f["verified"])[:FACTS_SAMPLE]:
        text, urls, searches = sb.search(c, ANSWER_MODEL, "Is this claim still true today? Search the current "
                                         "documentation or primary source and say what it says now.\n\nClaim (checked "
                                         f"{f['verified'] or 'at an unknown date'}): {scrub(f['claim'])}")
        j = sb.ask(c, JUDGE_MODEL, "A memory store keeps the claim below, with the day it was last checked. An agent "
                   "searched for it again; its answer and the URLs the search returned follow. still_true: TRUE when "
                   "the sources confirm it today, CHANGED when they show it no longer holds (now: what holds instead, "
                   "one sentence that stands on its own), UNKNOWN when the search does not settle it or the claim is "
                   f"about the project's own state.\n\nClaim (checked {f['verified'] or 'unknown'}): {scrub(f['claim'])}\n\n"
                   f"The search's answer:\n{text}\n\nURLs returned: {urls[:5] or 'none'}", FACT_SCHEMA)
        rows.append({**f, **j, "urls": urls[:5]})
        say(f"fact {f['id']}: {j['still_true']}  spent ${sb.spent():.2f}")
    L = ["Facts of the store, searched again.", f"Checked {today()} by dictated.py; the owner reviews each before it enters store/facts.md.", ""]
    for r in rows:
        L.append(f"- {r['still_true']}: {scrub(r['claim'])[:240]}")
        if r["still_true"] == "CHANGED" and r["urls"]:
            L.append(f"  - draft: `- claim: {r['now']}` / `source: {r['urls'][0]}` / `verified: {today()}` / `supersedes: {r['id']}`")
    (results() / "facts-recheck.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"checked": len(rows), "by": dict(Counter(r["still_true"] for r in rows)), "of": len(facts)}


# ---- step 5: train (free, from the probes) ----
def flips(before, rows):
    """A verdict that differs from the last one recorded for the same text: [(row, the earlier ledger row)]."""
    last = {}
    for r in before:
        last[r["sha"]] = r
    return [(r, last[r["sha"]]) for r in rows if r["sha"] in last and last[r["sha"]]["verdict"] != r["verdict"]]


def step_train(probed):
    rows, facts, tasks, edits = probed["rows"], [], [], []
    for r in rows:
        j, v = r["judge"], r["verdict"]
        src = (r.get("value_check", {}).get("urls") if v == "DATED" else None) or r["grounded"]["sources_verified"]
        if v in WRONG and j["fact_claim"].strip() and src:
            facts.append(f"- claim: {j['fact_claim'].strip()}\n  source: {src[0]}\n  verified: {today()}\n"
                         f"  time: {today()} 00:00\n  since: {r['grounded']['as_of'] or 'unstated'}\n  checked: {today()}\n"
                         f"  prior: {(r['probe'].get('value') or r['probe']['dictates']).strip()}\n  from: {r['id']}\n")
            bad = r["dictated"]["recommendation"] if not j["dictated_right"] else r["probe"]["dictates"]
            tasks.append({"id": f"d-{r['sha']}", "category": "dictated", "split": "train", "fact": f"d-{r['sha']}",
                          "prompt": r["probe"]["question"],
                          "grader": {"type": "judge", "rubric": f"Passes if it agrees with: {j['fact_claim'].strip()} Or if it "
                                     "says it cannot know and how to check. Fails if it answers as the outdated "
                                     f"instruction did: {r['probe']['dictates'].strip()}"},
                          "reference": r["grounded"]["recommendation"], "bad": [bad]})
        line = f"- `{r['id']}` {v}: {ADVICE[v]}.\n  - was: {r['directive']}\n"
        if j.get("value_current") == "CHANGED":
            line += f"  - value: {r['probe'].get('value') or '(unnamed)'} -> {j.get('current_value') or '(see the source)'}\n"
        if v in WRONG and j["rewrite"].strip():
            line += f"  - rewrite: {j['rewrite'].strip()}\n"
        edits.append(line)
    out = results()
    (out / "facts-draft.md").write_text(
        "Claims checked against a source, with where and when they were checked.\nDrafted by dictated.py from the "
        "directives the documentation no longer agrees with; the owner reviews each before it enters store/facts.md.\n"
        "Fields: claim, source, verified, time, since, checked, prior, from.\n\n" + "\n".join(facts), encoding="utf-8")
    (out / "newer-tasks-draft.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    (out / "edits.md").write_text("# What each probed directive should become\n\n" + "".join(edits), encoding="utf-8")
    return {"facts": len(facts), "tasks": len(tasks), "edits": len(edits)}


# ---- step 6: report (free) ----
def step_report(directives, probed, trained, ground=None, facts=None):
    ground = ground or {"by": {}, "records": 0, "orphans": 0}
    facts = facts or {"checked": 0, "by": {}, "of": 0}
    rows = probed["rows"]
    by_class = Counter(d["class"] for d in directives)
    verdicts = Counter(r["verdict"] for r in rows)
    wrong_now = [r for r in rows if r["verdict"] in WRONG]
    valued = [r for r in rows if r["judge"].get("value_current") in ("CURRENT", "CHANGED")]
    trapped = [r for r in wrong_now if r["judge"]["dictated_followed"] and not r["judge"]["dictated_right"]]
    from_instruction = [r for r in rows if r["judge"]["dictated_basis"] == "INSTRUCTION"]
    fl = flips(probed["ledger_before"], rows)
    s = {"fake": FAKE, "day": today(), "answer_model": ANSWER_MODEL, "directives": len(directives),
         "by_class": dict(by_class), "due": probed["due"], "probed": len(rows), "verdicts": dict(verdicts),
         "by_class_verdict": {c: dict(Counter(r["verdict"] for r in rows if r["class"] == c))
                              for c in ("datable", "manner")},
         "wrong_now": len(wrong_now), "trapped": len(trapped), "resisted": len(wrong_now) - len(trapped),
         "justified_by_instruction_alone": len(from_instruction),
         "said_instruction": sum(r["said_basis"] == "INSTRUCTION" for r in rows),
         "values_checked": len(valued), "values_changed": sum(r["judge"]["value_current"] == "CHANGED" for r in valued),
         "values_unknown": sum(r["judge"].get("value_current") == "UNKNOWN" for r in rows),
         "ground": ground["by"], "records": ground["records"], "orphans": ground["orphans"], "facts": facts,
         "flips": [{"id": r["id"], "was": o["verdict"], "was_model": o["model"], "was_checked": o["checked"],
                    "now": r["verdict"]} for r, o in fl],
         **{f"{k}_drafted": v for k, v in trained.items()}, "spent_usd": sb.spent()}
    L = ["# dictated: where instructions decide for the model", "",
         ("**FAKE run: the verdicts below are placeholders, not a measurement.**" if FAKE else
          f"Answering model {ANSWER_MODEL}, judge {JUDGE_MODEL}, {today()}; ${s['spent_usd']:.2f} spent."), "",
         f"{len(directives)} directives: {by_class['datable']} datable, {by_class['manner']} manner, "
         f"{by_class['named']} named and {by_class['procedure']} procedure (those two checked against the project in "
         f"`ground.md`, not probed): " + (", ".join(f"{k} {v}" for k, v in sorted(ground["by"].items())) or "not checked") + ".",
         f"The store's records: {ground['records']} corrections and retractions, {ground['orphans']} of them ORPHAN; "
         f"{facts['checked']} of {facts['of']} facts searched again ({facts['by'] or 'none'}).",
         f"{probed['due']} were due a measurement; {len(rows)} were probed this run.", "",
         "| verdict | datable | manner | what to do |", "|---|---|---|---|"]
    for v in VERDICTS:
        L.append(f"| {v} | {s['by_class_verdict']['datable'].get(v, 0)} | {s['by_class_verdict']['manner'].get(v, 0)} | {ADVICE[v]} |")
    L += ["", f"Of the {len(wrong_now)} directives that are wrong now, the dictated answer followed {len(trapped)} into the "
          f"wrong answer and resisted {len(wrong_now) - len(trapped)}.",
          f"The dictated answer's justification rested on the instruction alone in {len(from_instruction)} of {len(rows)} "
          f"(the model itself said INSTRUCTION in {s['said_instruction']}).",
          f"The value a datable directive names was settled by the search in {len(valued)} probes: "
          f"{s['values_changed']} changed, {len(valued) - s['values_changed']} still current; {s['values_unknown']} not settled.",
          f"Drafted: {trained['facts']} facts, {trained['tasks']} tasks, {trained['edits']} edits.", ""]
    if fl:
        L += ["## Verdicts that changed since the last measurement", ""]
        L += [f"- `{r['id']}`: {o['verdict']} ({o['model']}, {o['checked']}) → {r['verdict']}" for r, o in fl] + [""]
    L += ["## Probes", ""]
    for r in rows:
        j = r["judge"]
        L += [f"### {r['verdict']}: `{r['id']}`", "",
              f"- directive ({r['class']}, written {r['written']}, {r['age_days']} days): {r['directive']}",
              f"- value: {r['probe'].get('value') or 'none'}: {j.get('value_current', 'NONE')}"
              + (f", now {j.get('current_value')}" if j.get("value_current") == "CHANGED" else ""),
              f"- message: {r['probe']['question']}",
              f"- bare ({r['bare']['basis']}, {'right' if j['bare_right'] else 'wrong'}): {r['bare']['recommendation']}",
              f"- dictated ({r['dictated']['basis']}, {'right' if j['dictated_right'] else 'wrong'}, "
              f"{'followed' if j['dictated_followed'] else 'did not follow'}, rests on {j['dictated_basis']}): {r['dictated']['recommendation']}",
              f"- grounded ({', '.join(r['grounded']['sources_verified'][:2]) or 'no source'}): {r['grounded']['recommendation']}",
              f"- judge: {j['why']}", ""]
    (results() / "report.md").write_text("\n".join(L), encoding="utf-8")
    (results() / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    say(f"report: {results() / 'report.md'}")
    return s


def run(paid=False, work=None, instructions=None, store=None, sample=None, project=None):
    configure(FAKE=not paid, **({"WORK": work} if work else {}), **({"INSTRUCTIONS": instructions} if instructions else {}),
              **({"STORE": store} if store else {}), **({"PROBE_SAMPLE": sample} if sample else {}),
              **({"PROJECT": project} if project else {}))
    sync()
    directives = step_extract()
    ground = step_ground(directives)
    probed = step_probe(directives)
    facts = step_facts(ground["facts_list"])
    trained = step_train(probed)
    return step_report(directives, probed, trained, ground, facts)


def decision(verdicts):
    """What a line's verdicts so far allow, and how many more probes a decision needs. One probe is one message a
    model wrote to make the line matter, so a line is changed only when AGREE_MIN probes say the same and none says
    the line works: `same when asked` (every probe REDUNDANT or PREFERENCE-SAME; never an automatic drop: a probe asks
    the question outright, so it cannot show that the model would take a procedure's step unprompted), `rewrite` (TRAP, STALE or DATED in most),
    `keep` (NEEDED or PREFERENCE-STEERS in most). A verdict from before the preference split counts for nothing."""
    v = [x for x in verdicts if x != "PREFERENCE"]
    n = len(v)
    idle = sum(x in ("REDUNDANT", "PREFERENCE-SAME") for x in v)
    wrong = sum(x in WRONG for x in v)
    works = sum(x in ("NEEDED", "PREFERENCE-STEERS") for x in v)
    if n >= AGREE_MIN and idle == n:
        return "same when asked", 0
    if n >= AGREE_MIN and wrong > n / 2:
        return "rewrite", 0
    if n >= AGREE_MIN and works > n / 2:
        return "keep", 0
    if n >= AGREE_MIN + 2:                       # five probes and no majority: the line depends on the situation
        return "keep (the probes disagree)", 0
    return "undecided", max(1, AGREE_MIN - n)


def step_decide(dirs):
    """Every probed line's verdicts over the paid ledgers of `dirs` (each a WORK directory), with what they allow:
    results/decisions.md under WORK. Free."""
    rows, text = {}, {}
    for d in dirs:
        f = Path(d) / "results" / "ledger.tsv"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines()[1:]:
            r = dict(zip(LEDGER_HEAD, line.split("\t")))
            rows.setdefault(r["id"], []).append(r)
        try:
            for x in json.loads((Path(d) / "results" / "directives.json").read_text(encoding="utf-8")):
                text[x["id"]] = x["text"]
        except (OSError, ValueError):
            pass
    out = []
    for i, rs in rows.items():
        what, more = decision([r["verdict"] for r in rs])
        out.append({"id": i, "verdicts": [r["verdict"] for r in rs], "decision": what, "more": more, "text": text.get(i, "")})
    order = {"rewrite": 0, "same when asked": 1, "undecided": 2}
    out.sort(key=lambda r: (order.get(r["decision"], 3), r["more"], r["id"]))
    L = ["# What the probes so far allow", "",
         f"A line is dropped or rewritten only when {AGREE_MIN} probes agree and none says it works.", "",
         "| decision | lines |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in Counter(r["decision"] for r in out).most_common()] + [""]
    for r in out:
        L += [f"- **{r['decision']}**" + (f" ({r['more']} more to decide)" if r["more"] else "") + f": `{r['id']}` "
              f"{', '.join(r['verdicts'])}", f"  - {scrub(r['text'])[:220]}"]
    (results() / "decisions.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (results() / "decisions.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["list", "check", "run", "decide"])
    ap.add_argument("--ledgers", nargs="*", default=[], help="decide: the WORK directories whose ledgers to read")
    ap.add_argument("--only", nargs="*", default=[], help="run: probe exactly these directive ids, fresh or not")
    ap.add_argument("--project", default="", help="the project the instructions are about")
    ap.add_argument("--instructions", nargs="*", default=[], help="instruction files or directories")
    ap.add_argument("--store", default="", help="a store directory (<project>/.claude/p4): its live rulings too")
    ap.add_argument("--paid", action="store_true", help="real API calls, capped at TOTAL_MAX_USD")
    ap.add_argument("--sample", type=int, default=0, help=f"directives probed (default {PROBE_SAMPLE})")
    ap.add_argument("--work", default="", help="the output directory (default: out/ beside this file)")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        configure(**({"WORK": a.work} if a.work else {}), INSTRUCTIONS=a.instructions, STORE=a.store)
        for d in sorted(step_extract(), key=lambda d: (d["class"], -d["strength"], d["written"])):
            say(f"{d['class']:9} {d['written']} {','.join(d['kinds']) or '-':28} {d['id']}  {d['text'][:110]}")
        return 0
    if a.cmd == "decide":
        configure(**({"WORK": a.work} if a.work else {}), STORE=a.store)
        got = step_decide(a.ledgers or [WORK])
        for r in got:
            say(f"{r['decision']:28} {('+' + str(r['more'])) if r['more'] else '  ':3} {','.join(r['verdicts'])[:60]:60} {r['id'][-56:]}")
        say(f"decide: {results() / 'decisions.md'}")
        return 0
    configure(ONLY=set(a.only))
    if a.cmd == "check":
        configure(**({"WORK": a.work} if a.work else {}), INSTRUCTIONS=a.instructions, STORE=a.store, PROJECT=a.project)
        sync()
        step_ground(step_extract())
        say(f"check: {results() / 'ground.md'}")
        return 0
    s = run(paid=a.paid, work=a.work or None, instructions=a.instructions or None, store=a.store or None,
            sample=a.sample or None, project=a.project or None)
    say(json.dumps({k: s[k] for k in ("fake", "directives", "by_class", "ground", "records", "orphans", "facts", "due",
                                      "probed", "verdicts", "values_checked", "values_changed", "trapped",
                                      "justified_by_instruction_alone", "spent_usd")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
