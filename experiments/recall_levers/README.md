# recall_levers

What moves p4-recall's JSON ranker, the model recall serves its lines from, measured with the plugin's own `train.evaluate` (grouped 5-fold by time; a task whose labelled positives the candidate stage never draws counts as zero).

```bash
cp -r <project>/.claude/p4 experiments/recall_levers/out/<name>     # a copy: the index is rebuilt in place
.venv/bin/python -I experiments/recall_levers/recall_levers.py --store experiments/recall_levers/out/<name>
.venv/bin/python -I experiments/recall_levers/test_recall_levers.py                 # offline, ~20 s
```

Results land in `out/levers-<name>.json`.
Free, CPU, a few minutes on a store of a few thousand labels.

| Lever | What it changes |
|---|---|
| candidates | `train.CANDIDATES` 200 / 1000 / 5000: how many labelled positives BM25 draws at all |
| features | the base feature set without the per-file one-hot, the kind/state one-hots, age, or the BM25 group |
| weights | the hand-set label weights (none measured in the docs): no stated negatives, no commit labels, no usage labels, usage ×3, all 1.0, cited-only or edited-only positives |
| curve | held-out numbers at 10, 20, 40, 60 and all labelled tasks, three draws each |

`drawn_r-precision` and `drawn_baseline` are the same numbers over the tasks whose positives were drawn: what the ranker does with what it was handed.

## Findings, 2026-11-09

Two stores: a 97-pool store branch with git history (243 labelled tasks, 6,852 repo labels and 129 usage labels; called W below) and this project's own small store (19 labelled tasks, 10 drawn).

1. **The ranker ranked below BM25 on both.** On W, held-out R-precision 0.0057 against BM25's 0.029; over the drawn tasks 0.015 against 0.079.
2. **The cause was the negatives a round trains on.** `train.trim` kept every positive and the 60 negatives BM25 ranked *highest*. With positives kept wherever BM25 put them, a high BM25 score was mostly seen on negatives, and the ranker learned it as one: its `bm25` weight fit at −1.05 on W (−0.98 on the small store), against +0.28 when every candidate is kept.

   | W, held out | R-precision | drawn | `bm25` weight |
   |---|---|---|---|
   | top-60 BM25 negatives (as shipped) | 0.0057 | 0.015 | −1.05 |
   | every drawn negative | 0.0103 | 0.028 | +0.28 |
   | 60 random negatives | 0.0117 | 0.032 | −0.11 |
   | 60 random negatives, no silence tasks | 0.0114 | 0.031 | +0.63 |

   The fix (random negatives, seeded by the task) is in p4-recall's `train.trim`; with it the shipped evaluation reads 0.0102 on W and 0.054 on the small store (from 0.047).
   It roughly doubles the ranker on W but does not lift it past BM25 (0.029); the store's promotion floor already keeps such a ranker from being served.
3. **The BM25 baseline was overstated by 1.5×.** `train.evaluate` scaled the ranker's R-precision by the share of labelled positives drawn but not the baseline's, which ranks the same candidates. On W that read 0.044 for a fair 0.029. Fixed in the same file, with a test.
4. **Most of the loss is before the ranker.** 153 of 243 labelled tasks on W have no positive among the 200 BM25 candidates; 1,000 candidates leave 134, 5,000 leave 131. The labelled positives (files the reply cited or edited) share few words with the prompt.
5. **No feature group or label weight closes the gap.** Without the per-file, kind/state or age features: within ±0.001. Without the BM25 group: 0.011, better than with it, the clue to 2. Weights: no commit labels raised drawn R-precision to 0.030 on fewer tasks; no stated negatives, usage off or ×3, all-1.0, cited-only: 0.005–0.009.
6. **More tasks do not help yet.** The learning curve is flat or falling: drawn R-precision 0.026 at 10 tasks, 0.038 at 20, 0.019 at 40, 0.025 at 60, 0.017 at 90. With these labels the ranker is short of signal, not of tasks.

Not tested here: the ranker on candidates that always hold the positives (phi_rank's pools do; there the `.pt` ranker scores 0.50 against BM25's 0.21, see `../phi_depth`), and teacher-edited labels (paid).
