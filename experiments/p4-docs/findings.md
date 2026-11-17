# Findings for the p4 rework, 2026-11-01

Everything found while building the overrides notebook, with where each one is used. "Measured" means the notebook (`.docs/p4/overrides_colab.ipynb`) tests it; "code" means it changes hooks, scripts or the store, which a single API call cannot test and the plugins' own tests must cover.

## The fresh marketplace's paid validation, and what ships (2026-11-01, $9.94)

Opus 5.5 answering, two seeds, Haiku 5.5 judging after it passed the grader check on every hand-written key (`scripts/session/fresh_validation.py`; results in `~/.cache/p4-fresh-validation/results/`).

| measure | built-in | recall, every line | recall, rules and facts only |
|---|---|---|---|
| general and lesson tasks (accuracy, both seeds) | 0.672 / 0.672 | 0.656 / 0.656 | the same as the built-in (silent) |
| held-out short turns | 0.393 / 0.373 | 0.373 / 0.360 | the same as the built-in (silent) |
| newer than the model | 0.50 / 0.50 | 0.75 / 0.75 | 0.667 / 0.667, nothing lost |

| # | finding | what ships |
|---|---|---|
| 26 | Every recall line together is never better than the built-in on the general tasks and costs the same lesson task on both seeds (`api-removal-readers`: the correction it was made from, recalled cut at 160 characters). | not shipped |
| 27 | On held-out short turns, recall's history and repeat lines make the reading worse: 1–2 right where the built-in was wrong, 4 wrong where it was right, on each seed. Goal 1 (short turns read right) is not met by recall lines; the rows keep being recorded and labelled for a later measurement. | not shipped |
| 28 | Recall helps where the purpose says it should: facts newer than the model. Rule and fact lines alone keep that gain (0.667 against 0.50, both seeds) and are silent everywhere else. | **shipped on: `inject: "rules-facts"`** |
| 29 | The plain-words emission rewrite loses the same two simple questions on both seeds. | not shipped |
| 30 | The trained ranker printed lines for 88–92% of ordinary prompts; training on silence tasks takes it to 92% silent with R-precision 0.371 → 0.372. | shipped (upgrade 01) |
| 31 | The judge prompt never showed a task's earlier turns, so the Opus audit graded m2 and m3 without them; short-turn answer keys made from their own rubric cannot validate a judge. | fixed in the suite |
| 32 | Fable's label edits on the migrated repos ($6.07): on held-out raw labels the edited sets score slightly below raw overall (AudioPlugin-p4 0.246 vs 0.258, p4-p4 0.404 vs 0.418, ap-p4_mac 0.389 vs 0.478) but above raw on the tasks Fable edited (0.359 vs 0.344, 0.331 vs 0.312). The gate holds every batch, and no repo's ranker is promoted, so recall runs on the silent fallback everywhere. Each call took about 1.5 minutes, so the runs were stopped at 15, 18 and 17 tasks; `teacher edit-labels` resumes them, skipping what is done, and finishing them is what would tell whether the per-task gain is real. | measured; batches held |
| 33 | The old mind-reader plugin's `mind_reader.py watch` processes kept running for up to two hours after the plugin was uninstalled and its files deleted (finding 20's other face: the watcher detaches from its hook entirely). | to stop by hand |

## From the session that ported dow.sh and compdb.py to macOS

| # | finding | status |
|---|---|---|
| 1 | Words the contracts invent ("frag", "ruling", "pour", "tombstone", "derivation", "operator") leak into replies; the user called it "talking in riddles". | measured: every answer's jargon rate; a piece that raises it is never added back; the splitter and the designs are written in plain words |
| 2 | About 45k characters load at every session start, and the session followed them only in part, which is worse than all or none. | measured: the rebuild has a 3,000-character budget and starts from an empty frame |
| 3 | The operator launch should be one exact line, its brief in a file, its reply to the main context capped at 5 plain lines. **Corrected the same day:** the ceremony itself stays whole. The operator, the pool and its children, and the judged closes produce the phi models' training labels: pool picks, barrier verdicts (with `match:` and `addressed:`), teacher audits, and completed dispatch rows. The owner calls this one of the most important pieces. Only the ceremony's cost to the main context shrinks. | measured: both designs carry the launch as one exact line, as mechanics the rebuild always keeps. Code: the first stop is blocked until the operator ran, on every road, not just the conductor's |
| 4 | ~~The Stop hook writes the turn's record from the transcript.~~ **Corrected:** a hook cannot judge, so it cannot produce the labels. A row the hook marked FRESH can be completed mechanically. A row it flagged as resembling an earlier one keeps a judged close, by the operator in the background, never the main context. The store's field names (`state:`, `match:`, `addressed:`) stay as they are, for training continuity; plain words apply only at the boundary to the main context and the owner. | code; until then, `MECHANICS_FROM = "budget"` keeps the record.py step |
| 5 | The prompt hook's output on every message should be 0 to 2 lines, silent when the message is new. | code |
| 6 | R22 and the acts-popup rule ("popup last, final message after it") collide with the desktop app's own wrap-up and question panel. Keep one closing protocol per host. | measured in part: the `core` design drops both and keeps only the departures line (protocol task p1); the per-host switch is code |
| 7 | A worktree edit hook (not p4's) blocks writes to `.claude/p4/**` from worktree sessions. | code, outside p4: whitelist `.claude/p4/**`, or resolve the store from the main checkout |
| 8 | Scope as the owner corrected it at 15:29: "not just scripts both need its scripts the repo code uses / its for when a frag turns into code the repo works/relies/depends/works better with". Code the repo works with, relies on, depends on, or works better with moves into the repo itself, wherever the repo keeps that kind of code. p4 `src/` keeps only what the repo doesn't use: session-private probes and one-offs. The src-fragger's "worth keeping" trigger should check whether the repo now uses a file, not only whether a later session runs it. | measured: both designs say so; protocol task p2 accepts `scripts/` or `src/`; this suite lives in `scripts/session/`. Code: the trigger, sent to the rework session |
| 9 | One store, one format per kind (rules, corrections, scripts, retractions), each entry with the owner's words, a date and a status; no per-session trees for the model to navigate. | code |
| 10 | The benefits must survive the rework: retraction, repeat, stale rule, runnable code, settle before asking, script kept, plain language, cost. | measured: tasks m1, m2, m3, the fences tasks, j6, j7, p2, p3 and the jargon rate; character counts per pathway |

## From the four p4 repos (section 4 of the notebook)

| repo | stores | corrections | retractions | missed details | repeats | live rules | the old hook prints |
|---|---|---|---|---|---|---|---|
| ezexe/AudioPlugin-p4 | 142 | 86 | 26 | 20 | 44 | 102 (26,799 chars) | 0 chars |
| ezexe/ap-p4_mac | 2 | 5 | 6 | 0 | 3 | 9 (874 chars) | 0 chars |
| ezexe/p4-p4 | 27 | 19 | 0 | 7 | 2 | 56 (12,494 chars) | 12,292 chars |
| ezexe/phi-p4 | 2 | 3 | 0 | 0 | 0 | 16 (2,512 chars) | 0 chars |

| # | finding | status |
|---|---|---|
| 11 | `phi.py standing` prints only rulings with a `form:` field and every-turn index rules. Three of the four repos record their rulings with `scope:`, so their owners' live rules never reach a session start whole; they arrive only if the operator runs and pools them. A session that skips the operator (finding 2) works without them. | measured: validation compares `old+<repo>` (the old hook's block), `rebuilt+<repo>` (only the contracts changed) and `rules+<repo>` (every live rule, verbatim); fixing what the hook prints is code |
| 12 | Every live rule verbatim does not fit a 3k budget on a big store: 26,799 characters on AudioPlugin-p4. The always-on text needs the rules that bear on the task, which is the checker agent's job (finding 3), not all of them. | measured: `rules+AudioPlugin-p4` shows the cost; scoping is code |
| 13 | The stores hold 113 corrections, 32 retractions, 27 missed details and 49 caught repeats across the four repos: the old contracts' real failures. | measured: section 4 turns them into judged train tasks and lesson-driven candidates for the rebuild (needs the API) |
| 15 | The labels are thin because most sessions skip the ceremony. The operator ran in 25 of 142 sessions on AudioPlugin-p4 (18%), 7 of 27 on p4-p4 (26%), and none on ap-p4_mac or phi-p4. Dispatch rows completed: 3,216 of 4,239 (76%), 251 of 321 (78%), 105 of 108 (97%), 0 of 1. The pick model on AudioPlugin-p4 was trained on 24 pools (held out 0.379 without the `.pt`, 0.368 with it); the dispatch model on 244 rows. A design that lowers launch or completion starves the models, whatever it saves in tokens. Session counts include short and abandoned sessions. | measured: section 4 reports launch rate, completion rate, judged rows and the models' baselines per repo; section 13 refuses to push an upgrade that loses a row, a completion or a judged verdict. Code: the enforced launch (finding 3) is what raises the launch rate |
| 16 | Not measurable here: the operator's reply length to the main context, and ranker-against-teacher agreement after a rework. | the first needs a live session; the second is ezexe/phi's directions notebook (its teacher leg), run on the stores after the upgrade |
| 14 | Many session stores are on the φ register with an empty keyspace (`index-engine: idb-migrate-gentle`, "the store: empty"); every check reports 0 corruption and 0 debt. | section 13 migrates them with the published p4, dry by default |

## From the operator catch-up in the macOS-port session (ap-p4_mac, 15:20)

The operator ran its open (the pool on the children road) and its close, the owner's words being "the entire point of the session open hook ceremony is also integral fo the phi models to work properly". ap-p4_mac's launch count went from 0 of 2 to 1. The run completed 13 dispatch rows, added 2 rulings and 3 corrections, and found three plugin defects:

| # | defect | status |
|---|---|---|
| 17 | **A correction disables the ruling it produced.** The barrier's mask matches tombstones on shared owner-words, so a tombstone that freed a wrong reading also masks the right ruling with the same words. Six LIVE rulings in ap-p4_mac show as FREED; two bore on that session's task and reached it only as "surfaced", not applied. The highest-impact item here. | code: sent to the rework session |
| 18 | **record.py cannot seed ledger.md** from a peer session's ledger that has no `---` separator, and refused all three corrections as "file absent". The operator created the file by hand. | code: sent to the rework session |
| 19 | **The post-write hook calls a write inside the store "OUTSIDE the store"** (`…/2c09be2c/vlds/ledger.md`), most likely by resolving a relative path against the wrong directory. | code: sent to the rework session |

## The first audit: Opus 5.5, seeds 0 and 1, 64 tasks (run overrides-1, stopped 16:13)

Accuracy on the general tasks, seed 0 (seed 1's 704 answers are cached, its summary not yet written):

| pathway | accuracy | answers with p4 vocabulary | output tokens on simple questions |
|---|---|---|---|
| built-in stand-in | 0.672 | 5% | 37 |
| all four old contracts | 0.639 | 74% | 360 |
| neutral text of the same length | 0.672 | 7% | 53 |
| all but the memory override | 0.656 | 23% | 230 |
| the memory override alone | 0.672 | 89% | 302 |
| all but the emission gate | 0.574 | 62% | 325 |

| # | finding | status |
|---|---|---|
| 21 | Together, the old contracts lower Opus's accuracy below the built-in stand-in. Length alone does not: neutral text of the same length scores the same as no text at all. The cost is the content. | measured, one seed so far |
| 22 | Simple questions go from 0.8 to 0.0 under the old contracts, because answers grow tenfold (37 to 360 tokens) and miss the "right and short" bound. | measured |
| 23 | The memory override drives the vocabulary bleed: 89% of answers use p4 words with it alone, 23% without it. | measured |
| 24 | Not everything costs. On the lesson tasks made from the four repos' own mistakes, the old contracts score 0.42 against 0.35 for the built-in. Taking the emission gate out of the set drops accuracy from 0.639 to 0.574, so some of its rules do work. | measured |
| 25 | My cost estimate left out the judge: 40 of the 64 tasks are judged, and an Opus judge call costs about as much as the answer. Spend so far: about $35 in Opus answers over two seeds, plus an estimated $20 in judge calls the suite did not yet record. | fixed in the suite: judge cost is now recorded and estimated |

## From the plugins' own tests (run 1, section 4)

| # | finding | status |
|---|---|---|
| 20 | 13 of 14 plugin test files pass. `lbiza/plugins/mind-reader/hooks/test_mind_reader.py` fails on macOS with "the watcher <pid> died with its hook", both inside the notebook and from a plain shell, so the instrument is ruled out. The detached watcher the hook starts does not outlive it here. | code: sent to the rework session |

## Found in the notebook itself, fixed

- The standing-rules capture copied the tree without `.git`, so the layout hook re-initialized the repo and its notices were captured as rules. The copy keeps `.git` now, and notice lines are filtered out.
- The jargon scan flagged "pour-over"; a hyphenated compound no longer matches.
