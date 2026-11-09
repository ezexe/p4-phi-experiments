# p4-phi

Reusable experiments across [ezexe/p4](https://github.com/ezexe/p4) (the p4 marketplace, whose p4-recall plugin keeps a per-project store at `.claude/p4/`) and [ezexe/phi](https://github.com/ezexe/phi) (the phi models that train on such a store).
Both repositories are cloned beside this file and keep their own git; this repository ignores them and tracks only what spans the two.
No store, project or person is built in: every experiment reads the store you point it at.

## Layout

| Path | What it is | Tracked here |
|---|---|---|
| `p4/` | the p4 marketplace (plugins, session scripts, notebooks) | no, its own repo |
| `phi/` | the phi models (phi_prune, phi_rank, phi_store, phi_serve, phi_model) | no, its own repo |
| `.claude/p4/` | this project's own session store, when p4-recall runs here; its own git repository | no |
| `.venv/` | the one Python environment both repos build and test under | no |
| `.env` | API keys and the store under test, copied from `.env.example` | no |
| `experiments/` | experiments that use both repos; `env.py` loads `.env` | yes |
| `experiments/*/out/` | each experiment's clones, models and results | no |
| `requirements.txt`, `requirements.lock` | the environment's needs, and every pin as built | yes |

## Setup

```bash
git clone https://github.com/ezexe/p4 p4
git clone https://github.com/ezexe/phi phi
python -m venv .venv            # 3.14; on Windows: py -3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock -e ./phi[dev]     # Windows: .venv/Scripts/python
cp .env.example .env
```

Then fill in `.env`:

- `ANTHROPIC_API_KEY` for the paid steps, or leave it empty and run `ant auth login`; every experiment's free steps run without it.
- `P4_STORE_REPO`: the store to study, a git URL or a local path of a p4 store repository (p4-recall writes one per project at `<project>/.claude/p4`, one branch per machine when it is pushed). `--store` overrides it.
- `P4_PROJECT`: the project checkout that store belongs to, optional; its `repo:` labels name files there.
- `P4_SKILL_DIRS`: for `keep_or_go`, handwritten skill directories to test against the marketplace, separated by `;` on Windows and `:` elsewhere.

The owner's names are never typed in anywhere: the experiments that send store text to the API read them from the store's own committers and scrub them.

## Tests

All offline, no key, no store needed (each builds its own fixture):

```bash
.venv/bin/python -m pytest -q phi
.venv/bin/python -I experiments/keep_or_go/test_keep_or_go.py
.venv/bin/python -I experiments/served_back/test_served_back.py
.venv/bin/python -I experiments/says_vs_does/test_says_vs_does.py
.venv/bin/python -I experiments/phi_depth/test_phi_depth.py
.venv/bin/python -I experiments/recall_levers/test_recall_levers.py
```

## Experiments

- `experiments/keep_or_go/` decides which directories of a per-machine p4 store to keep and whether handwritten skills can move into the p4 marketplace as generic ones, through bare Claude API calls with no session context; `keep_or_go.ipynb` (kernel `.venv`) and `keep_or_go.py run --store <store> [--skill DIR]... [--project DIR] [--paid]` are the same run, `make_notebook.py` regenerates the notebook, results land in its ignored `out/`.
- `experiments/served_back/` reads every branch of a store three ways: what phi's readers (`phi_store`, `phi_model`, `phi_rank`) reach of the layout p4-recall writes, with the plugin's labels bridged into `phi_rank` pools and trained on the CPU against BM25; what recall served each session's turns and how much of it the replies used, with each owner correction classified by the API from the evidence that existed before it (new knowledge, known but not served, served but not followed); the ranker-against-teacher agreement on held-out tasks; and the recommendations the sessions handed the owner, each re-derived by the model alone and by the model made to search and cite current documentation, with the grounded answers drafted as the store's `facts.md` entries. `served_back.ipynb` and `served_back.py run --store <store> [--project DIR] [--paid]` are the same run, `make_notebook.py` regenerates the notebook, the free steps measure even without a key, results land in its ignored `out/`; `session_check.py --store <project>/.claude/p4 --session <s8>` (or `--latest`) is the same reading for one running session, free.
- `experiments/says_vs_does/` is a claims ledger over p4-recall and phi: every sentence their docstrings, store headers, design findings and READMEs state as fact gets one mechanical check against the code and, where the sentence is about a store, against a checked-out branch of one (by default the worktree served_back leaves), plus two live measurements (recall's query latency against its 300 ms budget, its silence on the plugin's ordinary prompts); `says_vs_does.py [--store DIR]` writes `out/results/claims.md` with HOLDS / PARTLY / DRIFTED / UNTESTABLE per claim; free and offline.
- `experiments/phi_depth/` grows `phi_rank.tree`'s Zeckendorf node tree (addresses 1, 101, 1001, …: p4 = 1, .claude = 0, no 11) from 1 to 13 nodes on a store and measures what each level buys, and `phi_open.py` tests the questions `phi/.docs/p4/nested-store.md` leaves open (seeds, pretraining, the mask as the route, φ over ordered features, the single stream of every node's rounds); free, CPU. See its [README](experiments/phi_depth/README.md).
- `experiments/recall_levers/` measures what moves p4-recall's JSON ranker (the one recall serves) with the plugin's own `train.evaluate`: how many positives the candidate stage draws, feature groups, the hand-set label weights, the learning curve over labelled tasks; free, CPU, on a copy of a store. See its [README](experiments/recall_levers/README.md).
