# p4-docs

The measurements and notebooks that ezexe/p4 kept in its folder `.docs/p4/`, with the scripts that ran them and wrote them.
ezexe/p4 keeps the plugins, their fast suite and the pages written from its code, and these files sit here.
A path written inside one of them is a path of an ezexe/p4 checkout, and its `.docs/p4/` is this folder.
Six pages of `.docs/p4/` that these files cite are not here: `fresh-marketplace.md`, `overrides.md`, `phi-concept.md`, `phi-prompt-cap.md`, `precision-upgrades.md` and `recall-node.md`, which ezexe/p4 holds at a66d0bf.
Every file equals its blob at ezexe/p4 a66d0bf except `upgrades/run.sh`, which carries one more line: it stops with a message naming `P4_ROOT` and `P4_SUITE` when the checkout it measures holds no `scripts/session/overrides_tasks.jsonl`, the suite with the held-out split that `measure.py` counts.
ezexe/p4 at ed929b5 holds that suite, and its `main` does not; `lbiza/plugins/p4-phi/scripts/silent-prompts.jsonl` there cannot stand in for it, since it keeps only the train split's prompts and no `split` field.
The repository's own [README](../../README.md) says how this checkout sits beside ezexe/p4 and ezexe/phi.

## What they need

Each run but the last of the list under Run measures plugins that ezexe/p4's `main` lacks, so it runs from a checkout of ezexe/p4 at the commit its line names, made by `git clone https://github.com/ezexe/p4 p4-at` and `git -C p4-at checkout <commit>`.

- **b34874a**, 2026-11-11: the last commit holding p4-memory, p4-recall and p4-scratchpad, the marketplace the fresh validation checks, and its child b01f260 removes p4-memory and p4-scratchpad.
- **ed929b5**, 2026-11-01: the commit upgrade 01 was written against, where its `patch.diff` applies with `git apply`.
- **f2323b2**, 2026-11-01: the last commit holding the vlds, src-fragger and emission-discipline plugins whose SessionStart contracts the overrides notebook measures, and its child 1750fa2 removes them.

The measured stores are AudioPlugin-p4, ap-p4_mac, p4-p4 and phi-p4 in the layout of their time, held only by the private repositories of those names under ezexe.
The fresh validation reads a clone of each from `$P4_FIXTURES/<name>`, and upgrade 01 reads AudioPlugin-p4 and ap-p4_mac from `$P4_FIXTURES/stores/<name>`.
A paid run needs `pip install anthropic` and `ANTHROPIC_API_KEY`, and it spends money within the cap it names.

## Run

- [`scripts/fresh_validation.py`](scripts/fresh_validation.py): from a checkout at b34874a, `python3 scripts/session/fresh_validation.py run` runs the fresh validation offline against a fake client, and `run --paid` is the paid run, capped at `TOTAL_MAX_USD`; its answers, contexts and `results/summary.json` and `results/findings.md` go under `P4_FRESH_WORK`, by default `~/.cache/p4-fresh-validation`.
- Unless it is given `--no-upgrades`, every run of `fresh_validation.py` ends with its upgrades step, which rewrites the checkout's `.docs/p4/upgrades/results.md`, and on a paid run applies each upgrade it keeps to the checkout and commits it there, since `APPLY_UPGRADES` is `True`.
- That step runs upgrade 01's `test.sh` with the run's environment, so a run that sets `P4_FIXTURES` to the folder of the four clones hands `test.sh` a folder with no `stores/` in it; `--no-upgrades` skips the step.
- [`scripts/fresh_tasks.py`](scripts/fresh_tasks.py), [`scripts/recall_contexts.py`](scripts/recall_contexts.py) and [`scripts/fresh/`](scripts/fresh): no command of their own; `fresh_validation.py` imports the two scripts, they read the three fixtures, and the checkout holds all of them in `scripts/session/`, beside `overrides_suite.py` and `overrides_tasks.jsonl`, which `fresh_validation.py` also imports and reads.
- [`scripts/make_fresh_validation_notebook.py`](scripts/make_fresh_validation_notebook.py): from a checkout at b34874a, `python3 scripts/session/make_fresh_validation_notebook.py` writes `.docs/p4/fresh_validation.ipynb`, equal to `fresh_validation.ipynb` here once its line ends are read as LF; it needs only the standard library.
- [`scripts/make_colab_notebook.py`](scripts/make_colab_notebook.py): from a checkout at f2323b2, `python3 scripts/session/make_colab_notebook.py` writes `.docs/p4/overrides_colab.ipynb`, equal to `overrides_colab.ipynb` here once its line ends are read as LF; it needs only the standard library.
- [`upgrades/01-silence-tasks/test.sh`](upgrades/01-silence-tasks/test.sh): in a checkout at ed929b5, apply the upgrade with `git apply <this folder>/upgrades/01-silence-tasks/patch.diff`, then from this folder `P4_ROOT=<checkout> P4_FIXTURES=<dir> sh upgrades/01-silence-tasks/test.sh` prints one JSON line with the held-out silent share before and after the patch and whether its pass rule holds; it is free and offline, and it runs [`upgrades/run.sh`](upgrades/run.sh) and [`upgrades/measure.py`](upgrades/measure.py) on copies of p4-recall and of AudioPlugin-p4 and ap-p4_mac.
- [`prompty_eval.py`](https://github.com/ezexe/p4/blob/main/lbiza/plugins/p4-phi/scripts/prompty_eval.py): ezexe/p4 keeps this script, and from a checkout of its `main` `python -I -X utf8 lbiza/plugins/p4-phi/scripts/prompty_eval.py --p4 <store>` measures again what `phi-prompty-scores.md` reports.

## The notebooks

- [`fresh_validation.ipynb`](fresh_validation.ipynb): the fresh validation, one cell a step of `fresh_validation.py`, with `FAKE = True` by default; it finds the script from a working folder inside a checkout at b34874a or from `P4_REPO` naming one, and its last section runs the upgrades step on what `.docs/p4/upgrades/index.json` of that checkout lists.
- [`overrides_colab.ipynb`](overrides_colab.ipynb): the SessionStart contracts of f2323b2 measured through the Claude API, then rebuilt from blank pages and validated on a held-out split, on Google Colab with its state on Google Drive; ezexe/p4's `main` lacks what it measures, so locally `P4_PATH` names a checkout at f2323b2, and on Colab the run's `state.json` gets `"pins": {"p4": "f2323b246bf3f383b04de561151851428a1eb227"}` before **Run all**.
- `PUSH_RESULTS`, `MIGRATE` and `MIGRATE_PUSH` start as `True` in `overrides_colab.ipynb`, so a run that sets none of them pushes a branch to ezexe/p4 and upgrades and pushes each store repository; set them to `False` for a run that only measures.
- `overrides_colab.ipynb` clones the four store repositories, so it needs a `GITHUB_TOKEN` that reads them, and `ANTHROPIC_API_KEY` unless `FAKE` is on.
- [`rebuild-seed/`](rebuild-seed): the two blank-page designs, `budget/` and `core/`, that section 7 of `overrides_colab.ipynb` cuts into candidates; [its README](rebuild-seed/README.md) says what each holds, and a run reads them from `.docs/p4/rebuild-seed/` of its checkout.

## Results

The runs kept their result files on the machines and the Drive that ran them, so these pages are what was written down from them.

- [`upgrades/results.md`](upgrades/results.md): what the fresh validation's upgrades step wrote of each upgrade it tried and of the free measures on the set it kept; its one row reads that upgrade 01 did not apply to the tree it was tried on.
- [`upgrades/index.json`](upgrades/index.json): the list of upgrades that step reads, each with its metric, its numbers before and after, and the files it touches.
- [`upgrades/01-silence-tasks/README.md`](upgrades/01-silence-tasks/README.md): what upgrade 01 changes in p4-recall's training, why, what it measured, and its pass rule.
- [`findings.md`](findings.md): the findings of 2026-11-01 from building the overrides notebook and from the fresh validation's paid run, each with what shipped from it.
- [`teacher-run.md`](teacher-run.md): the paid commands of p4-recall's `teacher.py` with the estimates that `p4.py teacher … --fake --estimate` gave for each and what each writes; none had run against the API when it was written, and p4-recall at b34874a runs them.
- [`phi-prompty-scores.md`](phi-prompty-scores.md): prompty's route and owner picks, measured on 2026-11-16 from the session history of the store `E:/projects/p4-phi/.claude/p4`, and the settings of p4-phi set by those numbers.
