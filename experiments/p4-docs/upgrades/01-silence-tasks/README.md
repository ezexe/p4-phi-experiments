# 01 silence tasks: the ranker learns what no match looks like

**What it changes.** `train.py` adds silence tasks to every round: ordinary messages with no positive, whose
candidates are exactly what `p4.py query` hands the ranker (the top 200 of BM25 over the whole index, the superseded
out), every one a negative at weight 1.0 (`SOURCE_WEIGHT["silent"]`). They come from `models/silent.jsonl`
(`{"prompt": ...}` per line) when the store has one, else from the plugin's new `scripts/silent-prompts.jsonl`, the
26 general prompts of the suite's *train* split (`scripts/session/overrides_tasks.jsonl`); the 12 of its *test* split
are never trained on and are the held-out measure. They train every fold and are held out of none; the threshold is
still chosen on the labelled tasks alone. `eval`/`train` print `N silence tasks, X% printed for (fitted)`.

**Why.** Every labelled task has an answer in the store, so a ranker fit on them alone never sees a message the store
has nothing for. Its features are relative (BM25 over the top candidate's, the rank), so the top candidate of an
ordinary message scores like the top candidate of a real one. Measured on AudioPlugin-p4's 24 pools (leave one task
out, three seeds), with the ranker the round would write:

| | pool R-precision | lines printed on the suite's general prompts: train split (fitted) / **test split (held out)** |
|---|---|---|
| before | 0.371 | 88% / **92%** |
| after | 0.372 | 27% / **8%** (1 of 12: "Rename `parse_row` to `parse_line`…", a correction at 0.71) |

Held-out hits / wrong lines over the 24 pools: 35 / 44 before, 42 / 50 after (more lines at the same precision per
line; the threshold moves 0.85 → 0.65 because the scores now spread). The two ap-p4_mac regression queries still
recall their ruling (they run on the no-ranker road, which this does not touch).

**Pass rule (test.sh).** On the held-out test split the trained ranker prints nothing for ≥ 90% of prompts, more
than before, and the pool R-precision is within 0.02 of before.

**What it touches.** `lbiza/plugins/p4-recall/scripts/train.py` (build_sets, trim, evaluate, run_round, fmt) and
the new `lbiza/plugins/p4-recall/scripts/silent-prompts.jsonl`. No change to `p4.py`, the hooks, the index or the
fallback road, so the other builder's silence tuning (`FALLBACK_*`), the 300 ms budget and the floor are untouched;
a round fits about 5,200 more examples (26 × 200), so `train` takes longer in the async Stop hook (roughly 1.6×
on AudioPlugin-p4), which `async: true` absorbs. The silence share of the no-ranker road (100% after the other
builder's tuning) is what a store without a ranker gets; this upgrade is what keeps that silence once a ranker is
promoted, which today would otherwise print lines for about nine of ten ordinary messages.

**Interaction with the paid teacher batch.** None: the silence tasks carry no labels and `labels.tsv` is not read
for them. When the owner adds ordinary messages of their own to `models/silent.jsonl` (for example the short turns
the history line already answers), they replace the plugin's list for that store.
