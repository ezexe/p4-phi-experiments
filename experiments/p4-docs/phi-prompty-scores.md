# prompty's route and owner picks, measured

This page holds the measurement that ezexe/p4#41 asks for.
It was taken on 2026-11-16 from the session history of the store `E:/projects/p4-phi/.claude/p4`, with `lbiza/plugins/p4-phi/scripts/prompty_eval.py`.
The script reads a store and writes nothing into it, unless `--train` finds a scorer worth keeping.

## The labelled set

- A case starts from a kept turn whose prompt is ezexe's own message and is work as prompty reads work.
- Handed briefs, relays between sessions, app events and conversation summaries are left out, and a message repeated in one session counts once.
- The store held 70 sessions and 467 kept turns, and 25 such messages reached an orchestrator, in 7 sessions.
- The owner label is the session that ended up doing the message's work.
- No message was relayed to another live session with its words quoted, 4 went to a session started for them, and 21 were done by the session that received them.
- Those 25 are the "others" cases: the candidates are the other sessions live at that time, and the right pick is no owner.
- 18 of the 21 give a second case, the "doer" case: the session that did the work stands among the live sessions, and the right pick is that session.
- A doer case is left out when the message is the session's own task text, since the scorer would be reading the answer.
- That makes 43 cases, 18 with an owner to find and 25 with none.

## Owners

| Scorer | Right of 43 | Wrong owners named | Owners missed of 18 |
| --- | --- | --- | --- |
| Name no owner by words | 25 | 0 | 18 |
| Word overlap at 4 words, the rule before | 11 | 25 | 16 |
| Word overlap at 6 words | 17 | 12 | 17 |
| Word overlap at 8 words | 21 | 6 | 17 |
| Word overlap at 12 words | 22 | 5 | 17 |
| BM25 over the ranker's tokens, threshold learned | 25 | 0 | 18 |
| Ranker trained on the history, tested on held-out sessions | 24 | 4 | 17 |

- The two trained rows are cross-validated: each session's cases are scored by a scorer that never saw that session.
- Word overlap at 4 words named a wrong owner on 25 of 43 cases and found 2 of the 18 owners.
- No scorer found the owner from the words: the best found 2 of 18.
- The BM25 row is right only by learning never to name an owner, so it is the first row under another name.
- The trained ranker got 24 right, one fewer than naming no owner.

## Routes

- The store holds 1 registered route.
- Word overlap matched it to 1 of the 25 messages, and that message's work was done by another session.
- One route is too few to measure a scorer on.

## What neither store has

- No store on this machine keeps a trained recall ranker: `phi/models/ranker.json` is absent in the p4-phi store and in AudioPlugin's.
- Every training round in both `eval.tsv` files ends "kept old", with precision under 0.6.
- torch is not installed here, so the `.pt` ranker was not run either.
- AudioPlugin's store holds 2 session folders and gave no case.
- So the ranker measured above is the ranker's own code, `ranker.candidate_features` and `ranker.fit`, trained on this history, and not a model trained for recall.

## What was changed, by the numbers

- **Owners stay on word overlap, at 8 words.** The trained ranker did not beat naming no owner, so no owner model is kept, and `OWNER_WORDS` went from 4 to 8, which cut wrong owners from 25 to 6 for one owner fewer found.
- **A shard still names an owner first.** A message that names a path inside a live session's shard is that session's, as before, and that rule was not measured here.
- **Routes stay on word overlap, unchanged.** `ROUTE_MIN` 4 and `ROUTE_CLOSE` 0.6 stand until the store holds enough routes to measure.
- **The ranker path is in place for both.** `puppeteer.score_routes` and `puppeteer.prompty` score through `rank_texts` when the store keeps `phi/models/prompty.json`, and fall back to word overlap when it does not.
- **A model is kept only on evidence.** `prompty_eval.py --p4 <store> --train` keeps an owner scorer only when, on held-out sessions, it beats both word overlap and naming no owner; here it printed "no owner scorer kept".

## Measuring again

```bash
python -I -X utf8 lbiza/plugins/p4-phi/scripts/prompty_eval.py --p4 E:/projects/p4-phi/.claude/p4
```

Run it when the store holds more routes or when messages have been relayed to live owners, since those are the cases this history lacks.
