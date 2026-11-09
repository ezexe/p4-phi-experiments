#!/usr/bin/env python3
"""
Generates keep_or_go.ipynb beside it (without outputs). Every cell calls one step of keep_or_go.py,
so the notebook and `keep_or_go.py run` are the same run, and test_keep_or_go.py runs these cells offline.

    python experiments/keep_or_go/make_notebook.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import keep_or_go as kg  # noqa: E402

OUT = HERE / "keep_or_go.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def cells():
    return [
        md(f"""
# Keep or go: a store's directories and its handwritten skills, judged without the owner's history

A Claude Code session reads CLAUDE.md, the auto-memory and p4-recall's lines before it answers, so its call on which
directories to keep leans on how the owner has worked. This notebook asks the Claude API directly: bare Messages
requests, a neutral system prompt, evidence computed from git (never written by an agent), owner names scrubbed, the
unit order shuffled per seed, and two model families judging. Step 3 measures what lean is left: the same evidence
unscrubbed, and every verdict that moves.

Then each handwritten skill is split into rules (GENERIC / REPO_SPECIFIC / OBSOLETE) with a generic SKILL.md drafted
from them, and probe tasks are answered with the handwritten skill, the generic draft and no skill: the draft can
replace the handwritten one in the p4 marketplace when it scores like the handwritten arm and above `none`.

`FAKE = True` (the default) runs every cell offline, with no key and no cost. The paid run is `FAKE = False` here, or
`python experiments/keep_or_go/keep_or_go.py run --store <store> --skill <dir> --paid` (the project's `.venv`, and `ANTHROPIC_API_KEY` in `.env` or
`ant auth login`). It stops before any request once `TOTAL_MAX_USD` (${kg.TOTAL_MAX_USD:.0f}) is spent; one run is
about $1.50 to $3 at the defaults.
"""),
        md("## Settings"),
        code(f"""
import os, sys, json, pathlib
HERE = pathlib.Path(os.environ.get("KEEP_OR_GO_DIR", ".")).resolve()   # this folder, inside it, or the project root
DIR = next((p for p in [HERE, *HERE.parents, HERE / "experiments" / "keep_or_go"] if (p / "keep_or_go.py").exists()),
           None)
sys.path.insert(0, str(DIR))
import keep_or_go as kg

FAKE = True                                   # False: the paid run
TOTAL_MAX_USD = {kg.TOTAL_MAX_USD}
STORE_REPO = os.environ.get("P4_STORE_REPO", "")   # required: a p4 store repository, a URL or a local path
BRANCHES = []                                 # [] = every remote branch
SKILL_DIRS = [d for d in os.environ.get("P4_SKILL_DIRS", "").split(os.pathsep) if d]   # handwritten skills; [] skips 4-5
PLUGIN_REPO = os.environ.get("P4_PROJECT", "")   # the repository the marketplace skills' settings read
JUDGE_MODELS = {kg.JUDGE_MODELS}
SEEDS = {kg.SEEDS}
PROBE_MODEL, PROBE_JUDGE = {kg.PROBE_MODEL!r}, {kg.PROBE_JUDGE!r}
WORK = os.environ.get("KEEP_OR_GO_WORK", str(kg.WORK))
kg.configure(FAKE=FAKE, TOTAL_MAX_USD=TOTAL_MAX_USD, STORE_REPO=STORE_REPO, BRANCHES=BRANCHES, SKILL_DIRS=SKILL_DIRS, PLUGIN_REPO=PLUGIN_REPO,
             JUDGE_MODELS=JUDGE_MODELS, SEEDS=SEEDS, PROBE_MODEL=PROBE_MODEL, PROBE_JUDGE=PROBE_JUDGE, WORK=WORK)
print("FAKE" if FAKE else "PAID", "| cap $", TOTAL_MAX_USD, "| work", WORK)
"""),
        md("## 1. Inventory (free)\nOne unit per directory (a directory with more than "
           f"{kg.COLLAPSE_OVER} children is one unit), per branch: files, bytes, commits, last touch, the share of "
           "blobs identical on every branch, and which code repositories name its path."),
        code("inv = kg.step_inventory()\nfor u in inv:\n"
             "    print(f\"{u['unit']:40} {sorted(u['branches'])} same={u['identical_share']:.0%} refs={u['referenced_by']}\")"),
        md("The exact text every judge receives (seed 1's order):"),
        code("print(kg.evidence(inv, True, SEEDS[0])[:4000])"),
        md("## 2. Judge (paid)"),
        code("res = kg.step_judge(inv)\nfor t in kg.consensus(res):\n"
             "    print(f\"{t['verdict']:8} {t['agreement']:.0%}  {t['unit']}\")"),
        md("## 3. Lean check (paid)\nThe first judge and seed with the owner's wording left in."),
        code("moved = kg.step_ablate(inv, res)\nmoved"),
        md("## 4. Skill rules and generic drafts (paid)"),
        code("rules = kg.step_rules()\nfor n, r in rules.items():\n    print(n)\n"
             "    for x in r['rules']:\n        print(f\"  {x['class']:13} {x['rule'][:100]}\")"),
        md("## 5. Probes: handwritten vs generic vs none (paid)"),
        code("probe_rows, probe_summary = kg.step_probe(rules)\nprobe_summary"),
        md("## 6. Report"),
        code("summary = kg.step_report(inv, res, moved, rules, probe_summary)\n"
             "print((kg.paths()['results'] / 'report.md').read_text(encoding='utf-8'))"),
    ]


def notebook():
    return {"cells": cells(), "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                          "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
