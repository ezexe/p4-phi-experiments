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
.venv/bin/python -I experiments/dictated/dictated.py check --store <project>/.claude/p4
```

```bash
.venv/bin/python -I experiments/dictated/dictated.py run
```

```bash
.venv/bin/python -I experiments/dictated/dictated.py run --paid --sample 12
```

- `list` prints every directive with its class, the day it was last written and its id; free.
- `check` runs the two free steps (extract, ground) and writes `results/ground.md`.
- `run` without `--paid` uses a fake client: extraction and the ground check are real, the verdicts are placeholders.
- `run --only ID ...` probes exactly those directives, whatever the ledger says: the way to give a line the further probes a decision needs.
- `decide --ledgers DIR ...` reads the paid ledgers of those output directories and writes `results/decisions.md`: per line, its verdicts so far and what they allow; free.
- `--instructions FILE_OR_DIR ...` names the instruction files (or `P4_INSTRUCTIONS` in `.env`); the default is the p4 checkout's skills, `scripts/session/fresh/emission-plain.md` and the rebuild-seed contracts.
- `--store <project>/.claude/p4` adds that store's live rulings (`store/rules.md`, each dated by its own `time:` field), its corrections, retractions and facts, all scrubbed of the store's committers' names.
- `--project DIR` names the project the instructions are about; the default is the store's project, else the first instruction file's repository.

## What it measures

1. **extract** (free): each bullet or sentence that tells the agent what to do, dated by `git blame`, classed `datable` (it names a version, a number with a unit, a model, a word like "latest", a URL; the values are kept), `manner` (it sets length, format, voice, asking or not asking), `named` (a backticked name is all it carries) or `procedure`.
2. **ground** (free): every directive against the project and the store, and the store's records against the directives.
3. **probe** (paid): one model writes an owner message the directive would decide, without quoting it; for a datable directive the message must hinge on whether the value it names is still current. The same answering model answers it bare, with the directive in its system prompt, and after searching the current documentation; a judge reads all three. The store's facts are searched again the same way.
4. **verdict**: computed from the judge's fields, not asked for.

| ground result | what it means |
|---|---|
| RETRACTED | the store freed or corrected, on a later day, a reading the line still dictates |
| MISSING | a path it names exists neither in the project, the instruction's own directory, nor a dependency root of the store |
| UNSEEN | an identifier it names appears in none of the project's tracked files other than documents |
| CONTESTED | a later correction or retraction is about the same thing |
| STANDS | everything it names is there |
| CARRIED / ORPHAN (a record) | a live ruling or an instruction line states the correction's or retraction's lesson, or the lesson lives only in the record |

| verdict | the directive today | the bare model | what to do |
|---|---|---|---|
| TRAP | wrong | right | drop it or replace it with the dated fact |
| STALE | wrong | wrong | replace it with the dated fact |
| DATED | the owner's choice stands, the value it names has moved | | keep the choice, update the value |
| NEEDED | right | wrong | keep it, with its date and source |
| REDUNDANT | right | right | read it before dropping: see Limits |
| PREFERENCE-STEERS | no fact settles it | would do otherwise | keep it: the line works |
| PREFERENCE-SAME | no fact settles it | does the same unprompted | keeping or dropping it changes nothing today |

Beside the verdict: whether the value a datable directive names is still current, whether the dictated answer followed the directive, whether it was right, and whether its justification rested on the instruction alone.

## What it gives the models

- `results/facts-draft.md`: a `facts.md` entry per TRAP, STALE and DATED, with `prior:` holding the value or the answer the directive dictated; p4-recall serves facts under `inject: rules-facts`, and its teacher marks the ones newer than its training.
- `results/facts-recheck.md`: each store fact searched again, a changed one drafted as the entry that supersedes it.
- `results/newer-tasks-draft.jsonl`: each such probe as a task in p4's `newer_tasks.jsonl` form, the directive's answer as the bad one.
- `results/edits.md`: per probed directive, keep, drop, or the rewrite as a dated fact with how to check it.
- `results/ground.md`: the free check's lists, and the store's orphan records (the lessons recall has to serve, since nothing dictates them).
- `results/ledger.tsv`: one row per probe (directive, text hash, answering model, day, verdict, probe version); a directive is probed again only when its text changed, the answering model changed, the probe itself changed, or `RECHECK_DAYS` (30) passed, and a verdict that differs from the last one for the same text is reported as a flip.

The drafts are for the owner to review; nothing is written to a store or a skill.

## Deciding

One probe is one message a model wrote to make the line matter, and the same line has come back NEEDED from one probe and TRAP from the next.
So `decide` changes nothing on fewer than three probes: `rewrite` when most say TRAP, STALE or DATED, `keep` when most say NEEDED or PREFERENCE-STEERS, `same when asked` when every one says REDUNDANT or PREFERENCE-SAME, `undecided` with the number of probes still needed otherwise.
`same when asked` is never a drop by itself.
On the first store measured, five lines reached it, and four of them were steps of a procedure (parse the arguments, run this command, walk this decision tree): the model answers the same when the question is put to it, which does not show it would take the step unprompted.

## Limits

- REDUNDANT measures the answer to a direct question, not whether the model would act on the line unasked; a line that works as a step or a reminder can be REDUNDANT every time and still be doing its job.
- The probe is written by a model that sees the directive, so the message is built to make the directive matter; the share of each verdict says nothing about how often a real session meets such a message.
- The grounded arm is right only as far as the documentation it found; a value that is the project's own comes back UNKNOWN.
- A single API call has no tools and no session, so a directive that only acts through a hook or a tool call is not exercised.
- UNSEEN is weak for a skill about an external framework: the identifier may be the framework's, and those trees are too large to search for each name.
- "About the same thing" is word overlap (half of a record's rare words, at least three), so CONTESTED, CARRIED and ORPHAN are leads to read, not verdicts.
- The class is a regular expression's guess used to order the sample; the verdict does not depend on it.

## Test

```bash
.venv/bin/python -I experiments/dictated/test_dictated.py
```
