#!/usr/bin/env python3
"""
Generates .docs/p4/fresh_validation.ipynb (without outputs): the fresh p4 marketplace's paid validation, build step 11
of .docs/p4/fresh-marketplace.md (8.2's first gate, 8.3's short turns, 8.4's newer-than-the-model test), then the
upgrade cells the owner asked for after the findings (18:36-18:37: "after findings a cell for improvements / run
upgrade suggestion cells based on tested results").

A separate, small notebook rather than new sections of make_colab_notebook.py: that notebook measures the old
contracts on Colab with the old store layout, and none of its sections is needed here. Every cell calls one step of
scripts/session/fresh_validation.py, so the notebook and `fresh_validation.py run` are the same run, and the FAKE
end-to-end test (test_overrides_suite.py) runs these cells.

    python3 scripts/session/make_fresh_validation_notebook.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_validation as fv  # noqa: E402

OUT = HERE.parents[1] / ".docs" / "p4" / "fresh_validation.ipynb"
PAID_COMMAND = "python3 scripts/session/fresh_validation.py run --paid"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def cost_rows():
    rows = [f"| {s} | {what} | ${usd:.2f} |" for s, what, usd in fv.COST_TABLE]
    rows.append("| 7. findings, 8. upgrades | free (the upgrades' paid re-measure is off: `UPGRADES_PAID = False`) "
                "| $0.00 |")
    total = sum(u for _s, _w, u in fv.COST_TABLE)
    rows.append(f"| **total** | | **${total:.2f}** (cap `TOTAL_MAX_USD` = ${fv.TOTAL_MAX_USD:.0f}) |")
    return "\n".join(rows), total


def cells():
    rows, total = cost_rows()
    out = [md(f"""
# The fresh p4 marketplace, validated

**The paid run is one command**, from the repository root, once the owner says go (it needs `pip install anthropic`
and `ANTHROPIC_API_KEY`):

```
{PAID_COMMAND}
```

Estimated cost, by step (fresh-marketplace.md 5.7; Opus 5.5 answers, Haiku 5.5 judges, two seeds):

| step | what is paid for | estimate |
|---|---|---|
| 1. tasks, 2. contexts | free: the short-turn and newer task sets, the migrated stores, a training round, the recall lines | $0.00 |
{rows}

About ${total:.0f} in all. No step starts once `TOTAL_MAX_USD` is spent, and the suite refuses any run whose estimate
is above what is left. Run the notebook itself instead with `FAKE = False` in the settings cell; with `FAKE = True`
(the default) every cell runs offline, with no key and no cost, as `python3 scripts/session/fresh_validation.py run`
does.

What it settles (fresh-marketplace.md 8.2-8.5):

| pathway | what it is |
|---|---|
| `builtin` | the built-in stand-in alone, answered from the audit's cache where the request is the same |
| `=recall@<store>` | plus the lines `p4.py query` prints for the task over the migrated store (none for most general tasks) |
| `=recall+pinned@<store>` | plus each store's `form:` and pinned rulings, at most 1,500 characters |
| `=emission-plain` | plus the 10-line plain rewrite of the emission gate's testable rules |
"""),
           md("## Settings"),
           code(f"""
import os, sys, json, pathlib
REPO = next((p for p in [pathlib.Path(os.environ.get("P4_REPO", ".")).resolve(), *pathlib.Path.cwd().resolve().parents,
             pathlib.Path.cwd().resolve()] if (p / "scripts" / "session" / "fresh_validation.py").exists()), None)
sys.path.insert(0, str(REPO / "scripts" / "session"))
import fresh_validation as fv

FAKE = True                       # False: the paid run (the API key in ANTHROPIC_API_KEY)
TOTAL_MAX_USD = {fv.TOTAL_MAX_USD}              # the whole run's cap: the design's ~$42, step 11's ~${total:.0f} and its reserve
ANSWER_MODEL = "{fv.ANSWER_MODEL}"
SEEDS = {fv.SEEDS}
JUDGE_MODELS = {fv.JUDGE_MODELS}   # the first that passes the grader check judges
WORK = os.environ.get("P4_FRESH_WORK", str(fv.WORK))
STORES = ""                       # NAME=PATH,...; "" is the four clones under P4_FIXTURES
GENERAL_REPO = "all"              # general tasks' recall from each store in turn, or one store
APPLY_UPGRADES = True             # commit the kept upgrades on this branch (never pushed)
UPGRADES_PAID = False             # re-measure the kept upgrades with the paid steps too
fv.configure(FAKE=FAKE, TOTAL_MAX_USD=TOTAL_MAX_USD, ANSWER_MODEL=ANSWER_MODEL, SEEDS=SEEDS, JUDGE_MODELS=JUDGE_MODELS,
             WORK=WORK, STORES=STORES, GENERAL_REPO=GENERAL_REPO, APPLY_UPGRADES=APPLY_UPGRADES,
             UPGRADES_PAID=UPGRADES_PAID)
print("FAKE" if FAKE else "PAID", "| cap $", TOTAL_MAX_USD, "| work", WORK)
"""),
           md("## 1. Tasks (free)\nThe short turns held out of each migrated store (8.3) and the twelve newer-than-the-model tasks (8.4)."),
           code("tasks_info = fv.step_tasks()"),
           md("## 2. Contexts (free)\nOne training round on each migrated store, then `p4.py query` for every task: the per-task context files of the recall pathways."),
           code("manifest = fv.step_contexts()"),
           md("## 3. The judge\nEvery judged task's reference must pass and every bad answer fail, on Haiku 5.5 first, Sonnet 5.5 if Haiku does not."),
           code("if not FAKE:\n    print('cache:', fv.seed_cache(), 'answers of the audit copied in')\nJUDGE = fv.step_judge()"),
           md("## 4. The first gate (8.2)"),
           code("outs = {'validate': fv.step_validate(JUDGE)}"),
           md("## 5. Short turns, held out (8.3)"),
           code("outs['short'] = fv.step_short(JUDGE)"),
           md("## 6. Newer than the model (8.4)"),
           code("outs['newer'] = fv.step_newer(JUDGE)"),
           md("## 7. Findings\nEach pass rule of 8.2-8.4, read off the results; `results/summary.json` and `results/findings.md` under WORK."),
           code("summary = fv.step_findings(outs)\nprint(open(fv.paths()['results'] / 'findings.md').read())"),
           md("## Verdict"),
           code("for k, v in summary['verdict'].items():\n    print(('holds   ' if v else 'FAILS   ') + k)\nprint(f\"spent ${fv.spent():.2f} of ${fv.TOTAL_MAX_USD}\")"),
           md("""
## 8. Upgrades, from the tested results

`.docs/p4/upgrades/index.json` lists the upgrades the precision upgrader wrote, one directory each (`patch.diff`,
`test.sh` printing one JSON line `{"metric", "before", "after", "pass"}`, a README). In order, each is applied to a
scratch copy of the worktree with the upgrades already kept; it is kept only when its `test.sh` passes and every free
test passes there; one that no longer applies over the kept ones is skipped as a conflict. Then the free measures
run again on the kept set (the floor, silence and the regression queries, latency). With `UPGRADES_PAID`, the kept
set is re-measured with the paid steps too, within what is left of the cap. With `APPLY_UPGRADES`, the kept set is
one commit on this branch, with `.docs/p4/upgrades/results.md`; nothing is pushed.
"""),
           code("upgrades = fv.step_upgrades()\nprint(json.dumps({k: upgrades[k] for k in ('kept', 'committed')}, indent=1))"),
           ]
    return out


def notebook():
    return {"cells": cells(), "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                          "name": "python3"},
                                           "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
