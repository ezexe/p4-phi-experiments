#!/usr/bin/env python3
"""
Generates served_back.ipynb beside it (without outputs). Every cell calls one step of served_back.py, so the
notebook and `served_back.py run` are the same run, and test_served_back.py runs these cells offline.

    python experiments/served_back/make_notebook.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import served_back as sb  # noqa: E402

OUT = HERE / "served_back.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def cells():
    return [
        md(f"""
# Served back: what the p4 sessions were served, what came back, and what phi can train on

Every branch of a per-machine p4 store (one machine per branch) is read three ways. **A** asks what phi's readers
(`phi_store`, `phi_model`, `phi_rank`) reach of the layout p4-recall now writes, and bridges the plugin's own labels
into `phi_rank` pools so the `.pt` ranker trains on them, held out by time against BM25. **B** counts what recall served
the sessions (the hooks' `usage` labels), how much of it the replies used, and what the owner had to correct; the API
then classifies each correction from the evidence that existed before it: new knowledge, known but not served, or
served and not followed. **C** is finding 16's ranker-against-teacher agreement: for held-out tasks two model
families pick the entries a session should have been shown, and every road (the hooks' labels, BM25, the bridge
ranker, recall's own lines) is scored against them. **D** takes the recommendations the sessions handed the owner and
re-derives each two ways, the model alone and the model made to search and cite the current documentation (sources
taken from the search results it read, never from its claims); the judge says which one the session matched and
whether the two differ, and every grounded answer is drafted as the store's `facts.md` entry, the file recall serves
under `inject: rules-facts` and the one every branch leaves empty.

Every request is a bare Messages call with a neutral system prompt, owner names scrubbed, candidates shuffled per
seed. `FAKE = True` (the default) runs every cell offline: the free steps (1 to 3) are real measurements, the paid ones
(4 and 5) return stand-ins. The paid run is `FAKE = False` here, or `python experiments/served_back/served_back.py run --store <store>
--paid` (the project's `.venv`, and `ANTHROPIC_API_KEY` in `.env` or `ant auth login`). It stops before any request
once `TOTAL_MAX_USD` (${sb.TOTAL_MAX_USD:.0f}) is spent; one run is about $2 to $3 at the defaults.
"""),
        md("## Settings"),
        code(f"""
import os, sys, json, pathlib
HERE = pathlib.Path(os.environ.get("SERVED_BACK_DIR", ".")).resolve()   # this folder, inside it, or the project root
DIR = next((p for p in [HERE, *HERE.parents, HERE / "experiments" / "served_back"] if (p / "served_back.py").exists()),
           None)
sys.path.insert(0, str(DIR))
import served_back as sb

FAKE = True                                   # False: the paid run
TOTAL_MAX_USD = {sb.TOTAL_MAX_USD}
STORE_REPO = os.environ.get("P4_STORE_REPO", "")   # required: a p4 store repository, a URL or a local path
BRANCHES = []                                 # [] = every remote branch with a store/
PROJECT_REPO = os.environ.get("SERVED_BACK_PROJECT") or os.environ.get("P4_PROJECT", "")   # the checkout the store's repo: labels name; "" = paths only
TEACHER_MODELS = {sb.TEACHER_MODELS}
JUDGE_MODEL = {sb.JUDGE_MODEL!r}
TEACHER_TASKS, CORRECTIONS_MAX, USAGE_SAMPLE, EPISTEMIC_SAMPLE = {sb.TEACHER_TASKS}, {sb.CORRECTIONS_MAX}, {sb.USAGE_SAMPLE}, {sb.EPISTEMIC_SAMPLE}
PRETRAIN_MINUTES, EPOCHS = {sb.PRETRAIN_MINUTES}, {sb.EPOCHS}
WORK = os.environ.get("SERVED_BACK_WORK", str(sb.WORK))
sb.configure(FAKE=FAKE, TOTAL_MAX_USD=TOTAL_MAX_USD, STORE_REPO=STORE_REPO, BRANCHES=BRANCHES, PROJECT_REPO=PROJECT_REPO,
             TEACHER_MODELS=TEACHER_MODELS, JUDGE_MODEL=JUDGE_MODEL, TEACHER_TASKS=TEACHER_TASKS,
             CORRECTIONS_MAX=CORRECTIONS_MAX, USAGE_SAMPLE=USAGE_SAMPLE, EPISTEMIC_SAMPLE=EPISTEMIC_SAMPLE,
             PRETRAIN_MINUTES=PRETRAIN_MINUTES, EPOCHS=EPOCHS, WORK=WORK)
print("FAKE" if FAKE else "PAID", "| cap $", TOTAL_MAX_USD, "| work", WORK)
"""),
        md("## 1. Layout (free)\nOne worktree per branch; the store's counts as p4-recall reads them, and what each phi "
           "reader reaches of it."),
        code("layout = sb.step_layout()\nfor b, x in layout.items():\n"
             "    print(b, x['coverage'], '| phi_store:', x['phi_store']['error'] or 'ok', '| phi_model:', x['phi_model'])"),
        md("## 2. Served back (free)\nOwner turns, served turns, the used share by kind and session, and each correction's "
           "as-of evidence (the entries that existed before it, and the lines recall would have printed)."),
        code("served = sb.step_served_back()\nfor b, x in served.items():\n"
             "    print(b, {k: x[k] for k in ('owner_turns', 'served_turns', 'served_ids', 'used_share', 'corrections', 'retractions')})"),
        md("## 3. The bridge (free, CPU)\nThe plugin's labels as phi_rank pools, split by time: BM25, a fresh `.pt` "
           "ranker, and one pretrained on the branch's text."),
        code("bridge = sb.step_bridge()\nfor b, x in bridge.items():\n"
             "    print(b, {k: x.get(k) for k in ('pools', 'bm25', 'pt_fresh', 'pt_pretrained', 'error')})"),
        md("## 4. Teacher agreement (paid)"),
        code("teacher = sb.step_teacher(bridge)\nfor b, x in teacher.items():\n    print(b, json.dumps(x['summary'], indent=1))"),
        md("## 5. Adherence (paid)\nEach correction classified from its as-of evidence; the hooks' lexical used/unused "
           "labels checked against the reply excerpt."),
        code("adherence = sb.step_adherence(served)\nprint(adherence['classes'], 'usage agreement', adherence['usage_agreement'])"),
        md("## 6. Epistemic recommendations (paid)\nThe replies that recommended something, re-derived by the model "
           "alone and by the model made to search and cite current documentation; the judge says which one the session "
           "matched, and `results/facts-draft.md` holds the facts.md entries the grounded answers would have left."),
        code("epistemic = sb.step_epistemic(served)\nprint(epistemic['session_matches'], 'gap', epistemic['gap'], 'of', epistemic['sampled'])"),
        md("## 7. Report"),
        code("summary = sb.step_report(layout, served, bridge, teacher, adherence, epistemic)\n"
             "print((sb.paths()['results'] / 'report.md').read_text(encoding='utf-8'))"),
    ]


def notebook():
    return {"cells": cells(), "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                          "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    OUT.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
