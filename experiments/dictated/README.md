# dictated

Where standing instructions decide for the model, and whether the model would have decided right without them.

A skill, a contract or a store's ruling is written on one day and read on every day after.
Some of its lines dictate a decision the facts settle (which version, which model, which limit, which tool), and the facts move.
Others set a manner (short, never ask, plain words) that the agent then applies where it makes the answer worse.
Either way a recommendation comes out that rests on the instruction and not on a fact, and nothing in the session says so.
`served_back`'s epistemic step re-derives the recommendations sessions already gave; this experiment starts one step earlier, at the instruction lines that produce them.

## Run

```bash
.venv/bin/python -I experiments/dictated/dictated.py list
```

```bash
.venv/bin/python -I experiments/dictated/dictated.py run
```

```bash
.venv/bin/python -I experiments/dictated/dictated.py run --paid --sample 12
```

- `list` prints every directive with its class, the day it was last written and its id; free.
- `run` without `--paid` uses a fake client: extraction is real, the verdicts are placeholders.
- `--instructions FILE_OR_DIR ...` names the instruction files (or `P4_INSTRUCTIONS` in `.env`); the default is the p4 checkout's skills, `scripts/session/fresh/emission-plain.md` and the rebuild-seed contracts.
- `--store <project>/.claude/p4` adds that store's live rulings (`store/rules.md`), each dated by its own `time:` field and scrubbed of the store's committers' names.

## What it measures

1. **extract** (free): each bullet or sentence that tells the agent what to do, dated by `git blame`, classed `datable` (it names a version, a number with a unit, a model, a word like "latest", a URL, or an identifier in backticks), `manner` (it sets length, format, voice, asking or not asking) or `procedure` (listed, not probed).
2. **probe** (paid): one model writes an owner message the directive would decide, without quoting it; the same answering model answers it bare, with the directive in its system prompt, and after searching the current documentation; a judge reads all three.
3. **verdict**: computed from the judge's fields, not asked for.

| verdict | the directive today | the bare model | what to do |
|---|---|---|---|
| TRAP | wrong | right | drop it or replace it with the dated fact |
| STALE | wrong | wrong | replace it with the dated fact |
| REDUNDANT | right | right | candidate to drop |
| NEEDED | right | wrong | keep it, with its date and source |
| PREFERENCE | no fact settles it | | keep it: the owner's choice |

Beside the verdict: whether the dictated answer followed the directive, whether it was right, and whether its justification rested on the instruction alone.

## What it gives the models

- `results/facts-draft.md`: a `facts.md` entry per TRAP and STALE, with `prior:` holding what the directive dictated; p4-recall serves facts under `inject: rules-facts`, and its teacher marks the ones newer than its training.
- `results/newer-tasks-draft.jsonl`: each such probe as a task in p4's `newer_tasks.jsonl` form, the directive's answer as the bad one.
- `results/edits.md`: per probed directive, keep, drop, or the rewrite as a dated fact with how to check it.
- `results/ledger.tsv`: one row per probe (directive, text hash, answering model, day, verdict); a directive is probed again only when its text changed, the answering model changed, or `RECHECK_DAYS` (30) passed, and a verdict that differs from the last one for the same text is reported as a flip.

The drafts are for the owner to review; nothing is written to a store or a skill.

## Limits

- The probe is written by a model that sees the directive, so the message is built to make the directive matter; the share of TRAP verdicts says nothing about how often a real session meets such a message.
- The grounded arm is right only as far as the documentation it found; a directive about the project's own conventions has no outside source and comes back PREFERENCE.
- A single API call has no tools and no session, so a directive that only acts through a hook or a tool call is not exercised.
- The class is a regular expression's guess used to order the sample; "current tree" counts as a recency word. The verdict does not depend on it.

## Test

```bash
.venv/bin/python -I experiments/dictated/test_dictated.py
```
