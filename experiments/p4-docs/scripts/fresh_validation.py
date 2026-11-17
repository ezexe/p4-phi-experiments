#!/usr/bin/env python3
"""fresh_validation.py: the fresh p4 marketplace's paid validation (fresh-marketplace.md 8.2-8.5, build step 11),
step by step, and the upgrades that follow it. The notebook (.docs/p4/fresh_validation.ipynb, written by
make_fresh_validation_notebook.py) runs these same steps one cell each; this script runs them all.

  python3 scripts/session/fresh_validation.py run            # FAKE: offline, no key, no cost (the default)
  python3 scripts/session/fresh_validation.py run --paid     # the paid run, capped at TOTAL_MAX_USD

Steps:
  1. tasks     the short-turn set (8.3) and the newer-than-the-model set (8.4) from the migrated stores
  2. contexts  recall_contexts.py: =recall@<store>, =recall+pinned@<store>, =emission-plain, after one training
               round on each migrated store
  3. judge     --check-graders --judge-check on every judged task with the first of JUDGE_MODELS (Haiku 5.5, then
               Sonnet 5.5): every reference passes and every bad answer fails, or the next judge is tried
  4. validate  8.2: the suite's general tasks and the lesson tasks, builtin against the three pathways, every seed
  5. short     8.3: the short turns, builtin against =recall, every seed
  6. newer     8.4: the newer tasks, builtin against =recall, every seed
  7. findings  each pass rule of 8.2-8.4 read off the results: results/summary.json and results/findings.md
  8. upgrades  .docs/p4/upgrades/index.json: each upgrade tried on a scratch copy, kept only when its own test and
               every free test pass, skipped when it conflicts with one kept; the free measures re-run; the kept set
               committed (APPLY_UPGRADES), never pushed

Every paid step counts what it spent from the answers it added to the cache, and none starts once TOTAL_MAX_USD is
reached; the suite itself refuses a run whose estimate is above what is left. Everything is kept under WORK, so a run
cut short resumes and nothing is asked twice.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

# --- settings (the notebook's settings cell sets the same names) ----------------------------------------------
FAKE = True                               # offline: no key, no call, no cost
TOTAL_MAX_USD = 42.0                      # the design's step-9-to-11 budget (5.7, about $42) as the cap of this run;
#                                           step 11 itself is estimated at about $25, the rest is its reserve
ANSWER_MODEL = "claude-opus-5-5"
EFFORT = "high"
SEEDS = [0, 1]
JUDGE_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]   # 8.2: Haiku if it passes the grader check, else Sonnet
WORK = Path(os.environ.get("P4_FRESH_WORK", Path.home() / ".cache" / "p4-fresh-validation"))
STORES = ""                               # NAME=PATH,...; "" is the four clones under P4_FIXTURES
GENERAL_REPO = "all"                      # general tasks' recall from each store in turn, or one store's name
LESSON_TASKS = os.environ.get("P4_LESSON_TASKS", "/Users/ez/gritz/p4/.claude/p4/scratchpad/.out/p4-colab/overrides-1/"
                                                 "lesson_tasks.jsonl")
CACHE_FROM = os.environ.get("P4_CACHE_FROM", "/Users/ez/gritz/p4/.claude/p4/scratchpad/.out/p4-colab/overrides-1/"
                                             "cache")   # the audit's answers: the built-in's are read, never asked again
NOW = ""                                  # the date the queries run at; "" is now
TRAIN = True                              # one training round on each migrated store before the queries
N_SHORT = 150
WORKERS = 8
UPGRADES_DIR = REPO / ".docs" / "p4" / "upgrades"
APPLY_UPGRADES = True                     # commit the kept upgrades on the worktree's branch (never pushed)
UPGRADES_PAID = False                     # re-measure the kept set with the paid steps too, within the cap
FREE_TESTS = [                            # the whole free suite, run from a root; {root} is replaced
    "python3 {root}/lbiza/plugins/p4-memory/hooks/test_p4_memory.py",
    "python3 {root}/lbiza/plugins/p4-scratchpad/hooks/test_scratchpad.py",
    "python3 {root}/lbiza/plugins/p4-recall/scripts/test_store.py",
    "python3 {root}/lbiza/plugins/p4-recall/scripts/test_query.py",
    "python3 {root}/lbiza/plugins/p4-recall/scripts/test_labels.py",
    "python3 {root}/lbiza/plugins/p4-recall/scripts/test_migrate.py",
    "python3 {root}/lbiza/plugins/p4-recall/scripts/test_train.py",
    "python3 {root}/lbiza/plugins/p4-recall/scripts/test_hooks.py",
    "python3 {root}/scripts/session/test_overrides_suite.py",
]
FREE_MEASURES = {                         # re-run after the upgrades; each prints its number
    "floor": "P4_SLOW=1 python3 {root}/lbiza/plugins/p4-recall/scripts/test_train.py AudioPluginFloor",
    "silence and the regression queries": "python3 {root}/lbiza/plugins/p4-recall/scripts/test_query.py Regressions",
    "latency": "python3 {root}/lbiza/plugins/p4-recall/scripts/test_hooks.py RecallLatency",
}
COST_TABLE = [                            # the first markdown cell's per-step estimates (5.7, 8.2-8.4)
    ("3. judge check", "Haiku 5.5 grades every judged task's reference and bad answer (~200 tasks x 2)", 0.30),
    ("4. validation (8.2)", "Opus 5.5 answers, 3 pathways x 2 seeds over 64 tasks; the built-in from the audit's "
                            "cache, silent tasks free", 20.00),
    ("5. short turns (8.3)", "Opus 5.5 answers ~150 turns x 2 pathways x 2 seeds, Haiku judge", 4.00),
    ("6. newer than the model (8.4)", "Opus 5.5 answers 12 tasks x 2 pathways x 2 seeds, Haiku judge", 0.30),
]
PASS = {"noise": 0.02, "noise_min_tasks": 8, "simple_acc": 0.8, "simple_tokens": 37 * 1.5, "jargon": 0.10,
        "lessons": 0.42, "short_lines": 2.0}


def settings():
    return {k: globals()[k] for k in ("FAKE", "TOTAL_MAX_USD", "ANSWER_MODEL", "EFFORT", "SEEDS", "JUDGE_MODELS",
                                      "WORK", "STORES", "GENERAL_REPO", "LESSON_TASKS", "CACHE_FROM", "NOW", "TRAIN",
                                      "N_SHORT", "WORKERS", "UPGRADES_DIR", "APPLY_UPGRADES", "UPGRADES_PAID")}


def configure(**over):
    """Set the module's settings (the notebook's settings cell)."""
    for k, v in over.items():
        if k not in globals():
            raise KeyError(k)
        globals()[k] = Path(v) if k in ("WORK", "UPGRADES_DIR") else v


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def paths(work=None):
    w = Path(work or WORK)
    return {"work": w, "tasks": w / "tasks", "ctx": w / "ctx", "cache": w / "cache", "results": w / "results",
            "spend": w / "spend.json", "short": w / "tasks" / "short.jsonl", "newer": w / "tasks" / "newer.jsonl",
            "lessons": w / "tasks" / "lessons.jsonl"}


# --- spend -----------------------------------------------------------------------------------------------------

def spent(work=None):
    p = paths(work)["spend"]
    return round(sum(json.loads(p.read_text())["steps"].values()), 4) if p.exists() else 0.0


def add_spend(step, usd, work=None):
    """Record what a step spent; a FAKE run's answers cost nothing (the fake client's token counts are not dollars)."""
    if FAKE:
        usd = 0.0
    p = paths(work)["spend"]
    d = json.loads(p.read_text()) if p.exists() else {"steps": {}}
    d["steps"][step] = round(d["steps"].get(step, 0.0) + usd, 6)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1))


def left(step, work=None):
    rest = round(TOTAL_MAX_USD - spent(work), 2)
    if rest <= 0:
        raise RuntimeError(f"budget: ${spent(work)} spent of TOTAL_MAX_USD = {TOTAL_MAX_USD}; {step} not started")
    return rest


def cache_usd(cache, before):
    """Dollars of the answers added to the cache since `before` (a set of file names), judge calls included."""
    usd = 0.0
    for f in Path(cache).glob("*.json"):
        if f.name not in before:
            r = json.loads(f.read_text(encoding="utf-8"))
            usd += (r.get("usd") or 0.0) + (r.get("judge_usd") or 0.0)
    return usd


# --- steps -----------------------------------------------------------------------------------------------------

def step_tasks(work=None):
    import fresh_tasks as ft
    import recall_contexts as rc
    p = paths(work)
    p["tasks"].mkdir(parents=True, exist_ok=True)
    stores = rc.parse_stores(STORES)
    short, info = ft.short_tasks(stores, p["work"], N_SHORT, NOW or None)
    ft.write(p["short"], short)
    ft.write(p["newer"], ft.newer_tasks())
    lessons = []
    if LESSON_TASKS and os.path.exists(LESSON_TASKS):
        lessons = [json.loads(l) for l in open(LESSON_TASKS, encoding="utf-8") if l.strip()]
    ft.write(p["lessons"], lessons)
    say(f"tasks: {len(short)} short turns ({', '.join(f'{n} {i['rows']}' for n, i in info.items())}), "
        f"12 newer, {len(lessons)} lesson tasks")
    return {"short": len(short), "short_by_store": info, "lessons": len(lessons)}


def step_contexts(work=None):
    import recall_contexts as rc
    p = paths(work)
    files = ",".join(str(x) for x in (HERE / "overrides_tasks.jsonl", p["lessons"], p["short"], p["newer"]))
    m = rc.build(rc.load(files), rc.parse_stores(STORES), p["ctx"], p["work"], GENERAL_REPO, NOW or None,
                 train=TRAIN)
    for n, x in sorted(m["pathways"].items()):
        if "tasks" in x:
            say(f"{n}: {x['tasks']} tasks, {x['silent']} silent, {x['lines']} lines")
    return m


def suite(argv):
    import overrides_suite as osu
    return osu.main([str(a) for a in argv])


def step_judge(work=None):
    """The first of JUDGE_MODELS that grades every judged task as its reference and bad answers say."""
    p = paths(work)
    p["results"].mkdir(parents=True, exist_ok=True)
    extra = ",".join(str(x) for x in (p["lessons"], p["short"], p["newer"]))
    for model in JUDGE_MODELS:
        out = p["results"] / f"judge-check-{model}.json"
        if out.exists() and not json.loads(out.read_text())["faults"]:
            say(f"judge: {model} (checked before)")
            return model
        argv = ["--tasks", HERE / "overrides_tasks.jsonl", "--extra-tasks", extra, "--check-graders", "--judge-check",
                "--judge", model, "--out", out, "--max-usd", left("judge check", work)]
        rc = suite(argv + (["--fake"] if FAKE else []))
        d = json.loads(out.read_text())
        add_spend(f"judge check {model}", d["usd"], work)
        if rc == 0:
            say(f"judge: {model}, every reference passes and every bad answer fails")
            return model
        say(f"judge: {model} has {len(d['faults'])} faults; the next judge is tried")
    raise RuntimeError(f"no judge of {JUDGE_MODELS} passes the grader check; nothing more is spent")


def pathway_names(work=None):
    m = json.loads((paths(work)["ctx"] / "manifest.json").read_text())
    return sorted(n for n in m["pathways"] if n.startswith("=recall"))


def run_set(name, tasks_file, extra, only, judge, work=None, categories=""):
    """One suite run a seed; returns {seed: out path}. Spend counted from the cache's new answers."""
    p = paths(work)
    cache = p["cache"] / f"{ANSWER_MODEL}-{EFFORT}"
    cache.mkdir(parents=True, exist_ok=True)
    outs = {}
    for seed in SEEDS:
        out = p["results"] / f"{name}-{ANSWER_MODEL}-s{seed}.json"
        before = {f.name for f in cache.glob("*.json")}
        argv = ["--p4", REPO, "--tasks", tasks_file, "--extra-tasks", extra, "--model", ANSWER_MODEL,
                "--effort", EFFORT, "--seed", seed, "--pathways", "builtin", "--only", ",".join(only),
                "--context-dir", p["ctx"], "--judge", judge, "--workers", WORKERS, "--cache", cache, "--out", out,
                "--max-usd", left(f"{name} seed {seed}", work)]
        if categories:
            argv += ["--categories", categories]
        rc = suite(argv + (["--fake"] if FAKE else []))
        add_spend(f"{name} s{seed}", cache_usd(cache, before), work)
        if rc not in (0, 3):
            raise RuntimeError(f"{name} seed {seed}: the suite returned {rc}")
        outs[seed] = str(out)
    say(f"{name}: spent so far ${spent(work):.2f} of ${TOTAL_MAX_USD}")
    return outs


def seed_cache(work=None):
    """The audit's answers (CACHE_FROM/<model>-s<seed>/) copied into this run's cache, so the built-in pathway is read,
    not asked: the key is the model, effort, stand-in, context, turns, prompt and seed, so only identical requests
    match."""
    p = paths(work)
    cache = p["cache"] / f"{ANSWER_MODEL}-{EFFORT}"
    cache.mkdir(parents=True, exist_ok=True)
    n = 0
    if CACHE_FROM and os.path.isdir(CACHE_FROM):
        for d in Path(CACHE_FROM).glob(f"{ANSWER_MODEL}-s*"):
            for f in d.glob("*.json"):
                if not (cache / f.name).exists():
                    shutil.copyfile(f, cache / f.name)
                    n += 1
    return n


def step_validate(judge, work=None):
    p = paths(work)
    names = pathway_names(work)
    only = ["builtin", "=emission-plain"] + names
    return run_set("validate", HERE / "overrides_tasks.jsonl", p["lessons"], only, judge, work)


def step_short(judge, work=None):
    p = paths(work)
    first = [n for n in pathway_names(work) if not n.startswith("=recall+pinned")][:1]
    return run_set("short", "", p["short"], ["builtin"] + first, judge, work)


def step_newer(judge, work=None):
    p = paths(work)
    first = [n for n in pathway_names(work) if not n.startswith("=recall+pinned")][:1]
    return run_set("newer", "", p["newer"], ["builtin"] + first, judge, work)


# --- findings ---------------------------------------------------------------------------------------------------

def load_outs(outs):
    return {int(s): json.loads(Path(f).read_text()) for s, f in outs.items()}


def verdict_validate(outs, tasks):
    """8.2's rules per pathway: {pathway: {rule: holds?, ...}}, and the categories each loses."""
    runs = load_outs(outs)
    size = {}
    for t in tasks:
        size[t["category"]] = size.get(t["category"], 0) + 1
    out = {}
    names = sorted({n for r in runs.values() for n in r["pathways"]} - {"builtin"})
    for n in names:
        lost, notes = [], []
        for s, r in runs.items():
            b, x = r["pathways"]["builtin"]["by_category"], r["pathways"][n]["by_category"]
            for c, v in x.items():
                if c == "protocol":
                    continue
                d = v - b.get(c, 0.0)
                if size.get(c, 0) >= PASS["noise_min_tasks"] and d < -PASS["noise"]:
                    lost.append(f"{c} (seed {s}: {v:.2f} against {b.get(c, 0.0):.2f})")
                elif d < 0:
                    notes.append(f"{c} {d:+.2f} (seed {s}, {size.get(c, 0)} tasks: within noise)")
        p = {s: r["pathways"][n] for s, r in runs.items()}
        simple_ok = all((v["by_category"].get("simple", 1.0) >= PASS["simple_acc"]) and
                        ((v["simple_output_tokens"] or 0) <= PASS["simple_tokens"]) for v in p.values())
        jargon_ok = all((v["jargon_rate"] or 0) <= PASS["jargon"] for v in p.values())
        lessons_ok = all(v["by_category"].get("lessons", 1.0) >= PASS["lessons"] for v in p.values()) \
            if size.get("lessons") else None
        out[n] = {"no category lost": not lost, "lost": lost, "within noise": notes, "simple": simple_ok,
                  "jargon": jargon_ok, "lessons >= 0.42": lessons_ok,
                  "accuracy": {s: v["accuracy"] for s, v in p.items()},
                  "builtin": {s: r["pathways"]["builtin"]["accuracy"] for s, r in runs.items()},
                  "passes": not lost and simple_ok and jargon_ok and lessons_ok is not False}
    return out


def verdict_pair(outs, max_lines=None):
    """8.3 / 8.4: the recall pathway answers more right than builtin on every seed, and never fewer."""
    runs = load_outs(outs)
    rows = {}
    for s, r in runs.items():
        names = [n for n in r["pathways"] if n != "builtin"]
        n = names[0] if names else None
        b = sum(t["passed"] for t in r["pathways"]["builtin"]["tasks"].values())
        x = sum(t["passed"] for t in r["pathways"][n]["tasks"].values()) if n else b
        rows[s] = {"pathway": n, "builtin_right": b, "recall_right": x, "of": r["pathways"]["builtin"]["answered"],
                   "lines_mean": r["pathways"][n]["lines_mean"] if n else 0.0}
    more = all(v["recall_right"] > v["builtin_right"] for v in rows.values())
    never_fewer = all(v["recall_right"] >= v["builtin_right"] for v in rows.values())
    lines_ok = max_lines is None or all(v["lines_mean"] <= max_lines for v in rows.values())
    return {"seeds": rows, "more on every seed": more, "never fewer": never_fewer, "lines <= 2": lines_ok,
            "passes": more and never_fewer and lines_ok}


def step_findings(outs, work=None):
    p = paths(work)
    tasks = [json.loads(l) for f in (HERE / "overrides_tasks.jsonl", p["lessons"]) if Path(f).exists()
             for l in open(f, encoding="utf-8") if l.strip()]
    summary = {"model": ANSWER_MODEL, "seeds": SEEDS, "fake": FAKE, "spent_usd": spent(work),
               "cap_usd": TOTAL_MAX_USD, "validate": verdict_validate(outs["validate"], tasks),
               "short": verdict_pair(outs["short"], PASS["short_lines"]), "newer": verdict_pair(outs["newer"])}
    recall = [n for n in summary["validate"] if n.startswith("=recall") and "+pinned" not in n]
    rp = [n for n in summary["validate"] if n.startswith("=recall+pinned")]
    gate = all(summary["validate"][n]["passes"] for n in recall)
    summary["verdict"] = {
        "8.2 =recall never below the built-in": gate,
        "8.2 =recall+pinned ships": all(summary["validate"][n]["passes"] and
                                        mean_acc(summary["validate"][n]) > mean_acc(summary["validate"][n.replace(
                                            "+pinned", "")]) for n in rp) if rp else False,
        "8.2 =emission-plain ships": bool(summary["validate"].get("=emission-plain", {}).get("passes")) and
        mean_acc(summary["validate"]["=emission-plain"]) > max([mean_acc(summary["validate"][n]) for n in recall]
                                                               or [0.0]),
        "8.3 short turns read right": summary["short"]["passes"],
        "8.4 newer than the model": summary["newer"]["passes"],
    }
    summary["verdict"]["release (8.5's parts 1 and 3; part 2 is measured apart)"] = \
        summary["verdict"]["8.2 =recall never below the built-in"] and summary["verdict"]["8.3 short turns read right"]
    p["results"].mkdir(parents=True, exist_ok=True)
    (p["results"] / "summary.json").write_text(json.dumps(summary, indent=1))
    (p["results"] / "findings.md").write_text(findings_md(summary))
    say("findings:", json.dumps(summary["verdict"]))
    return summary


def mean_acc(v):
    a = [x for x in v["accuracy"].values() if x is not None]
    return sum(a) / len(a) if a else 0.0


def findings_md(s):
    L = [f"# Fresh marketplace validation, {time.strftime('%Y-%m-%d')}", "",
         f"{s['model']}, seeds {s['seeds']}{' (FAKE: offline answers, not a measurement)' if s['fake'] else ''}; "
         f"spent ${s['spent_usd']:.2f} of the ${s['cap_usd']:.0f} cap.", "", "## Verdict", ""]
    L += [f"- {k}: {'holds' if v else 'does not hold'}" for k, v in s["verdict"].items()]
    L += ["", "## 8.2, by pathway", "", "| pathway | accuracy by seed | built-in | lost | simple | jargon | lessons |",
          "|---|---|---|---|---|---|---|"]
    for n, v in s["validate"].items():
        L.append(f"| `{n}` | {v['accuracy']} | {v['builtin']} | {'; '.join(v['lost']) or 'none'} | {v['simple']} | "
                 f"{v['jargon']} | {v['lessons >= 0.42']} |")
    for k in ("short", "newer"):
        L += ["", f"## {'8.3 short turns' if k == 'short' else '8.4 newer than the model'}", ""]
        for seed, r in s[k]["seeds"].items():
            L.append(f"- seed {seed}: built-in {r['builtin_right']}/{r['of']}, recall {r['recall_right']}/{r['of']}, "
                     f"{r['lines_mean']:.2f} lines a turn")
    return "\n".join(L) + "\n"


# --- upgrades --------------------------------------------------------------------------------------------------

def sh(cmd, cwd, timeout=1800, env=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, **(env or {})})
    return r.returncode, r.stdout + r.stderr


def upgrade_list(upgrades_dir):
    """[(id, dir)] in index.json's order: a list of names or of {"id"/"name", "dir"/"path"} objects, or an object
    holding such a list under "upgrades"."""
    d = Path(upgrades_dir)
    idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
    items = idx.get("upgrades", []) if isinstance(idx, dict) else idx
    out = []
    for it in items:
        if isinstance(it, str):
            out.append((it, d / it))
        else:
            name = it.get("id") or it.get("name") or it.get("dir") or it.get("path")
            out.append((name, d / (it.get("dir") or it.get("path") or name)))
    return out


def scratch_copy(src, dest):
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".out", "node_modules")
    shutil.copytree(src, dest, ignore=ignore, symlinks=True)
    return dest


def applies(patch, root):
    rc, _ = sh(f"git apply --check {json.dumps(str(patch))}", root)
    return rc == 0


def test_line(text):
    for l in reversed(text.strip().splitlines()):
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if isinstance(d, dict) and "pass" in d:
            return d
    return None


def free_tests(root, tests=None):
    """[(command, ok)] of the free suite run from `root`."""
    out = []
    for c in (tests if tests is not None else FREE_TESTS):
        cmd = c.format(root=root)
        rc, text = sh(cmd, root)
        out.append((cmd, rc == 0, text[-400:]))
    return out


def step_upgrades(repo=None, upgrades_dir=None, tests=None, measures=None, work=None, apply=None, paid=None):
    """Try each upgrade of index.json on a scratch copy, keep the ones that pass, re-run the free measures, and
    commit the kept set with results.md (APPLY_UPGRADES). Returns the record."""
    repo = Path(repo or REPO)
    udir = Path(upgrades_dir or UPGRADES_DIR)
    apply = APPLY_UPGRADES if apply is None else apply
    paid = UPGRADES_PAID if paid is None else paid
    rec = {"upgrades": [], "kept": [], "measures": {}, "committed": None}
    if not (udir / "index.json").exists():
        say(f"upgrades: no {udir / 'index.json'}; nothing to try")
        return rec
    tmp = Path(tempfile.mkdtemp(prefix="p4-upgrades-"))
    pristine = scratch_copy(repo, tmp / "pristine")
    kept_root = scratch_copy(repo, tmp / "kept")
    for n, (uid, d) in enumerate(upgrade_list(udir)):
        patch, test = d / "patch.diff", d / "test.sh"
        row = {"id": uid, "dir": str(d)}
        rec["upgrades"].append(row)
        if not patch.exists() or not test.exists():
            row.update(kept=False, why="no patch.diff or test.sh")
            continue
        if not applies(patch, kept_root):
            row.update(kept=False, why=("conflicts with an upgrade already kept: " + ", ".join(rec["kept"]))
                       if applies(patch, pristine) else "does not apply to the worktree")
            continue
        trial = scratch_copy(kept_root, tmp / f"trial-{n}")
        rc, text = sh(f"git apply {json.dumps(str(patch))}", trial)
        if rc:
            row.update(kept=False, why=f"git apply failed: {text[-200:]}")
            continue
        rc, text = sh(f"sh {json.dumps(str(test))}", trial, env={"P4_ROOT": str(trial)})
        res = test_line(text)
        row["test"] = res
        if not res or not res.get("pass"):
            row.update(kept=False, why="its test did not pass" + ("" if res else " (no JSON line)"))
            continue
        broke = [c for c, ok, _t in free_tests(trial, tests) if not ok]
        if broke:
            row.update(kept=False, why="breaks " + ", ".join(broke))
            continue
        row.update(kept=True, why="its test passes and every free test passes")
        rec["kept"].append(uid)
        shutil.rmtree(kept_root)
        kept_root = trial
    for name, c in (measures if measures is not None else FREE_MEASURES).items():
        rc, text = sh(c.format(root=kept_root), kept_root)
        rec["measures"][name] = {"ok": rc == 0, "tail": text.strip().splitlines()[-3:]}
    if paid and rec["kept"] and not FAKE:
        try:
            rec["paid"] = subprocess.run([sys.executable, str(kept_root / "scripts" / "session" / "fresh_validation.py"),
                                          "run", "--paid", "--no-upgrades", "--work", str(paths(work)["work"] /
                                                                                         "upgraded"),
                                          "--max-usd", str(left("upgrades re-measure", work))],
                                         capture_output=True, text=True).returncode
        except RuntimeError as e:
            rec["paid"] = str(e)
    results = udir / "results.md"
    results.write_text(results_md(rec), encoding="utf-8")
    if apply and rec["kept"] and FAKE and repo.resolve() == REPO.resolve():
        say("upgrades: FAKE run, so the kept set is not committed on this worktree (a paid run commits it)")
    elif apply and rec["kept"]:
        for uid in rec["kept"]:
            d = dict(upgrade_list(udir))[uid]
            rc, text = sh(f"git apply {json.dumps(str(d / 'patch.diff'))}", repo)
            if rc:
                raise RuntimeError(f"upgrade {uid} applied on the copy but not on the worktree: {text[-300:]}")
        msg = ("p4 upgrades: " + ", ".join(rec["kept"]) + "\n\nKept by the fresh validation notebook's upgrades "
               "section: each upgrade's own test passed on a scratch copy and so did every free test. Numbers in "
               ".docs/p4/upgrades/results.md.\n")
        touched = {str(results.resolve().relative_to(repo.resolve()))} if results.resolve().is_relative_to(
            repo.resolve()) else set()
        for uid in rec["kept"]:
            d = dict(upgrade_list(udir))[uid]
            _rc, text = sh(f"git apply --numstat {json.dumps(str(d / 'patch.diff'))}", repo)
            touched |= {l.split("\t", 2)[2] for l in text.splitlines() if l.count("\t") >= 2}
        # only what the kept patches touched and results.md: the worktree's other changes stay out of the commit
        sh("git add -- " + " ".join(json.dumps(t) for t in sorted(touched)), repo)
        rc, text = sh(f"git commit -q -m {json.dumps(msg)}", repo)
        rec["committed"] = rc == 0 and sh("git rev-parse --short HEAD", repo)[1].strip()
    shutil.rmtree(tmp, ignore_errors=True)
    say(f"upgrades: kept {rec['kept'] or 'none'}; committed {rec['committed']}")
    return rec


def results_md(rec):
    L = ["# Upgrades tried after the fresh validation", "", "| upgrade | kept | why | metric | before | after |",
         "|---|---|---|---|---|---|"]
    for r in rec["upgrades"]:
        t = r.get("test") or {}
        L.append(f"| {r['id']} | {'yes' if r.get('kept') else 'no'} | {r.get('why', '')} | {t.get('metric', '')} | "
                 f"{t.get('before', '')} | {t.get('after', '')} |")
    L += ["", "## The free measures on the kept set", ""]
    L += [f"- {k}: {'pass' if v['ok'] else 'FAIL'} — {' / '.join(v['tail'])}" for k, v in rec["measures"].items()]
    return "\n".join(L) + "\n"


# --- the whole run -------------------------------------------------------------------------------------------------

def run(work=None, upgrades=True):
    p = paths(work)
    p["work"].mkdir(parents=True, exist_ok=True)
    say(f"fresh validation: {'FAKE (offline)' if FAKE else 'PAID'}, cap ${TOTAL_MAX_USD}, work {p['work']}")
    step_tasks(work)
    step_contexts(work)
    if not FAKE:
        say(f"cache: {seed_cache(work)} answers of the audit copied in")
    judge = step_judge(work)
    outs = {"validate": step_validate(judge, work), "short": step_short(judge, work),
            "newer": step_newer(judge, work)}
    summary = step_findings(outs, work)
    if upgrades:
        summary["upgrades"] = step_upgrades(work=work)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("run", "upgrades"))
    ap.add_argument("--paid", action="store_true", help="call the API (default: FAKE, offline)")
    ap.add_argument("--work", default="")
    ap.add_argument("--max-usd", type=float, default=None, help="override TOTAL_MAX_USD")
    ap.add_argument("--no-upgrades", action="store_true")
    a = ap.parse_args(argv)
    configure(FAKE=not a.paid)
    if a.work:
        configure(WORK=a.work)
    if a.max_usd is not None:
        configure(TOTAL_MAX_USD=a.max_usd)
    if a.cmd == "upgrades":
        step_upgrades()
        return 0
    run(upgrades=not a.no_upgrades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
