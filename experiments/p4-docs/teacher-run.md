# The teacher's paid run (build step 9), 2026-11-01

These are the paid commands for `scripts/teacher.py` (fresh-marketplace.md 5.2, 5.3, 5.6, 5.7), to run after the validation run. Nothing here has been run against the API. Every number below comes from `p4.py teacher … --fake --estimate` over copies of the validation's migrated stores. The fake client counts one token per four characters. The real run counts with `client.messages.count_tokens` before it spends anything, refuses when its estimate is above `--max-usd`, and stops before any call that would take the real spend past that cap.

The estimate formula is the one `teacher.py` applies: counted input tokens × the model's input price, plus `OUT_EST` output tokens per call × its output price. `OUT_EST` is 3,000 tokens for edit-labels and 4,000 for dictionary, adaptive thinking included. This is an assumption. The design's 500 and 1,200 left thinking out. Output is about 80% of every estimate. The "light" column assumes 1,000 output tokens per call for edit-labels and 2,000 for dictionary. Run the same command with `--estimate` first; it reads the counter's real input numbers and sends nothing to the model. Real spend goes into each store's `models/spend.json`.

Prices per million tokens: Fable 5.1 $10 / $50, Opus 5.5 $4 / $20.

```sh
PY=/Users/ez/gritz/p4/.venv/bin/python          # has the anthropic SDK (2.3.0)
P4=/Users/ez/gritz/p4-fresh/lbiza/plugins/p4-recall/scripts/p4.py
ST=~/.cache/p4-fresh-validation/stores
```

## 0. Free: the history tasks (no API call)

As the stores stand, they hold only their pool tasks: AudioPlugin-p4 has 24, p4-p4 has 8, ap-p4_mac and phi-p4 have none. To get "the 100 newest AudioPlugin-p4 tasks" you first need step 8's history labels. They write `tasks.tsv` and `labels-raw.tsv` (source `store`):

```sh
for s in AudioPlugin-p4 p4-p4 ap-p4_mac phi-p4; do $PY $P4 labels --p4 $ST/$s --from-history; done
```

On copies this gave AudioPlugin-p4 126 tasks (125 the teacher edits, 1 short), p4-p4 86, ap-p4_mac 9 and phi-p4 0. phi-p4 has no completed message rows, so its edit-labels batch is empty either way.

## 1. The commands and their estimates

| # | command | calls | input tokens | estimate | light | without step 0 |
|---|---|---|---|---|---|---|
| 1 | `$PY $P4 teacher edit-labels --p4 $ST/AudioPlugin-p4 --limit 100 --max-usd 22` | 100 Fable | 357,869 | **$18.58** | $8.58 | 24 tasks, $4.49 |
| 2 | `$PY $P4 teacher edit-labels --p4 $ST/p4-p4 --max-usd 18` | 86 Fable | 217,120 | **$15.07** | $6.47 | 8 tasks, $1.47 |
| 3 | `$PY $P4 teacher edit-labels --p4 $ST/ap-p4_mac --max-usd 2` | 9 Fable | 8,157 | **$1.43** | $0.53 | 0 tasks, $0 |
| 4 | `$PY $P4 teacher edit-labels --p4 $ST/phi-p4 --max-usd 1` | 0 | 0 | **$0** | $0 | $0 |
| 5 | `$PY $P4 teacher edit-labels --p4 $ST/AudioPlugin-p4 --limit 0 --second-opinion 20 --max-usd 2` | 20 Opus 5.5 | 69,085 | **$1.48** | $0.68 | $1.50 |
| 6 | `$PY $P4 teacher dictionary --p4 $ST/AudioPlugin-p4 --max-usd 3` | 10 Fable (182 entries) | 23,202 | **$2.23** | $1.23 | same |
| 7 | `$PY $P4 teacher dictionary --p4 $ST/p4-p4 --max-usd 1` | 2 Fable (25 entries) | 3,316 | **$0.43** | $0.23 | same |
| 8 | `$PY $P4 teacher dictionary --p4 $ST/ap-p4_mac --max-usd 0.5` | 1 Fable (5 entries) | 845 | **$0.21** | $0.11 | same |
| 9 | `for s in AudioPlugin-p4 p4-p4 ap-p4_mac phi-p4; do $PY $P4 eval --p4 $ST/$s; done` | none | | **$0** | | |
| | **total** | 228 calls | | **≈ $39.4** | ≈ $17.8 | ≈ $10.3 |

phi-p4 holds no vocabulary entries, so it gets no dictionary command.

The design's 5.7 budgeted about $14 for this step: 156 edit tasks with 500 output tokens each. At 3,000 output tokens per call, the request as written costs about $39. To stay closer to the design, give command 2 `--limit 40` (its 40 newest tasks, estimated $7.24, or ≈ $3.2 light). That brings the total to about $31.6, or about $14.6 light.

Command 5 runs after command 1. `--limit 0` edits nothing new, so the second opinion takes the newest 20 tasks of the last batch. It reports the agreement F1 and holds the batch (`models/teacher-batches.json`, `status: held`) when F1 is below 0.65. A held batch is not trained on. To release it, set its `status` back to `ok`.

Command 9 is free. For a store with teacher edits, `eval` now also prints `raw against edited on held-out raw labels: raw R, edited E … edited at or above raw` (5.2 step 3). It writes nothing. Only `train` holds the batches when the edited labels score below raw.

## 2. What each run writes (per store, under `models/`)

- `labels.tsv`: `store+edited` rows. A kept or added entry is label 1, a dropped one is label 0. Each row has `teacher` (the `response.model` that answered), `edited` (the date) and `note` (`batch:<id>`, `from:pool` for a pool task, the decision).
- `label-edits.md`: one section per task, with every decision, its reason, the teacher id and the date. It also records refusals (`- refusal: category … (task skipped)`) and the second opinion's F1 line.
- `teacher-batches.json`: each batch's tasks, its teacher, its cutoff, the second opinion and its status.
- `spend.json`: `steps` (dollars per step), `calls` (every call's input and output tokens and dollars), `total`.
- `store/vocabulary.md` (dictionary): a `means:` line and a `means-by: <model> <date> <hash>` line, written only on entries still pending. An entry whose source is the owner is never touched. A meaning edited by hand is never overwritten, because the hash no longer matches.

## 3. Behaviour worth knowing before paying

- Teacher cutoff: the table only knows Claude Opus 5.5's (2026-06). Fable 5.1's is not recorded anywhere in this repo, so it resolves to 2026-06, which errs toward sending more entries as newer. If you know Fable's cutoff, set `"teacher-cutoff": "YYYY-MM"` in `models/config.json` or pass `--cutoff`.
- No server-side fallback: a refused task is recorded and skipped, so no label is made by a model that is not the teacher. Retries come from the SDK (`max_retries=6`).
- Calls are sequential and non-streaming (`max_tokens` 16,000, effort `high`, adaptive thinking). A Fable call can take minutes, so the 195 edit calls may run for hours. A stopped run can be restarted with the same command; tasks already edited are skipped.
- A teacher change later (design step 13): change `teacher-model:` and run `teacher edit-labels --re-edit --max-usd <approved>`. It redoes only tasks another teacher edited, newest first, as many as the cap allows.
