# phi-p4-models

The measurements behind ezexe/phi's page [`.docs/p4/nested-store.md`](https://github.com/ezexe/phi/blob/main/.docs/p4/nested-store.md): a tree of small rankers nested as a p4 store nests, measured against one large ranker, with its controls, the Colab notebooks that ran them, and the result files those runs wrote.
These files moved here from ezexe/phi, which keeps the package, its fast tests and its pages.
The repository's own [README](../../README.md) says how this checkout sits beside ezexe/p4 and ezexe/phi and how its environment is set up.

## What they need

phi is the package `phi-prune` of ezexe/phi: `pip install "git+https://github.com/ezexe/phi#subdirectory=lbiza"`, or `pip install -e <checkout of ezexe/phi>/lbiza`, with torch beside it.
Run each command from this folder.
The three suites read a pairs file that only the vlds plugin's `scripts/pick_model.py pairs --jsonl` writes, and `nested_suite.py` and `directions_suite.py` also need the plugin itself and a store of its layout, with a `sessions/<id>/vlds` folder.
ezexe/p4 holds that plugin up to commit f2323b2, and its child 1750fa2 deletes it; a p4-phi store has no such folder, so a run needs a store and a p4 checkout from that time.

## Run

- [`scripts/nested_suite.py`](scripts/nested_suite.py): `python scripts/nested_suite.py --p4 <store> --pairs <pairs.jsonl> --plugin <p4 checkout>/lbiza/plugins/vlds --k 4 --minutes 6 --seeds 0 1 --cache <dir> --out <file>` measures every arm of the nested tree against one large ranker; it needs phi and the vlds plugin, and it wrote `results/suite-k4-s0.json`, `suite-k4-s1.json` and `suite-k8-s0.json` and each `suite-k*-s*.json` of `results/colab-q-matrix-t4/`.
- [`scripts/pathways_suite.py`](scripts/pathways_suite.py): `python scripts/pathways_suite.py --p4 <store> --pairs <pairs.jsonl> --k 4 --minutes 6 --check-s 30 --seeds 0 --cache <dir> --out <file>` runs the tree's controls, a binary tree, flat nodes and four masks as routes; it needs phi and imports `nested_suite.py` and `nested_vs_one.py` beside it, and it wrote the `pathways-k4-s*.json` of `results/colab-pathways-1/`.
- [`scripts/directions_suite.py`](scripts/directions_suite.py): `python scripts/directions_suite.py --p4 <store> --pairs <pairs.jsonl> --plugin <vlds> --teacher fable=<dir> --teacher opus=<dir> --seeds 0 1 --cache <dir> --out <file>` measures the JSON ranker, pretraining budgets and teacher label sets; it needs phi and the vlds plugin, and it wrote `results/colab-directions-1/directions.json`.
- [`scripts/nested_vs_one.py`](scripts/nested_vs_one.py): no command of its own; it holds `first_added`, `shards`, `pairs_of` and `zmean`, which the three suites import.
- [`scripts/teacher.py`](scripts/teacher.py): `python scripts/teacher.py label --pairs <pairs.jsonl> --out-dir <dir> --model claude-fable-5-1 --max-usd 25` labels every pool of a pairs file through the Anthropic API and `report` drafts a report from a summary; it needs `pip install anthropic` and `ANTHROPIC_API_KEY`, and it spends money: `label` counts the input with the free token counter first and stops before any paid call when its estimate is above `--max-usd`.
- [`scripts/stream_logic.py`](scripts/stream_logic.py): `python scripts/stream_logic.py --items 2000 --maxlen 12` measures how Fibonacci, Elias-gamma, fixed-width and length-field codes delimit a stream of store addresses under single-bit flips, with the standard library alone, in about two minutes; with its default 3,000 flips it gives the `stream_logic` block of `results/colab-q-matrix-t4/proofs.json`, and with `--flips 1500` it gives `results/stream_logic-flips1500.json`, the table of section 5 of the page.
- [`scripts/make_colab_notebook.py`](scripts/make_colab_notebook.py): `python scripts/make_colab_notebook.py` writes the six notebooks under `notebooks/`; it needs nbformat.
- [`tests/`](tests): `python -m pytest tests` checks, offline in about seven seconds, that the notebooks' git cells keep the GitHub token out of every command line, address and error, and that the store they clone gives `first_added` its whole history; it needs pytest and git.

## The notebooks

Each opens in Colab from its badge, and keeps its state, logs and results on Google Drive, so **Run all** after a disconnect resumes.
They clone ezexe/phi for the package, ezexe/p4-phi-experiments for the scripts and committed results, ezexe/p4 for the vlds plugin, and the private store repository ezexe/AudioPlugin-p4, so they need a Colab secret `GITHUB_TOKEN` that reads that store.
`P4_REF` takes a branch name and ezexe/p4's `main` no longer holds the vlds plugin, so a new run reaches the plugin only through a `pins` entry written into its `state.json` by hand.

- [`notebooks/q_matrix_colab.ipynb`](notebooks/q_matrix_colab.ipynb) and [`notebooks/q_matrix_colab_t4.ipynb`](notebooks/q_matrix_colab_t4.ipynb): the page's exact claims checked up to a bound, the nested suite at k = 2, 3 and 4, and a teacher leg through the Anthropic API that needs `ANTHROPIC_API_KEY`; the T4 copy wrote `results/colab-q-matrix-t4/`.
- [`notebooks/pathways_colab.ipynb`](notebooks/pathways_colab.ipynb) and [`notebooks/pathways_colab_t4.ipynb`](notebooks/pathways_colab_t4.ipynb): `pathways_suite.py` on three seeds; the CPU copy wrote `results/colab-pathways-1/`.
- [`notebooks/directions_colab.ipynb`](notebooks/directions_colab.ipynb) and [`notebooks/directions_colab_t4.ipynb`](notebooks/directions_colab_t4.ipynb): `directions_suite.py`, with two teachers labelling through `teacher.py` and a report it drafts, so they need `ANTHROPIC_API_KEY` and spend money; the CPU copy wrote `results/colab-directions-1/`.

## Results

Each run folder holds what its notebook copied from Drive, and its `summary.json` and `summary.md` record the run's pinned commits, the device it trained on, its figures and its verdicts.

- [`results/suite-k4-s0.json`](results/suite-k4-s0.json), [`suite-k4-s1.json`](results/suite-k4-s1.json) and [`suite-k8-s0.json`](results/suite-k8-s0.json): the CPU runs behind the stage 3 tables of the page.
- [`results/stream_logic-flips1500.json`](results/stream_logic-flips1500.json): `python scripts/stream_logic.py --items 2000 --maxlen 12 --flips 1500`, run on 2026-11-16 with Python 3.14, equal in every cell to the table of section 5 of the page.
- [`results/colab-q-matrix-t4/`](results/colab-q-matrix-t4): phi at d039f00, p4 at 0157bbd and the store at bdd6983, pretrained on a T4; its `summary.json` also holds the teacher leg and the JSON ranker's weights for each `.pt` score.
- [`results/colab-pathways-1/`](results/colab-pathways-1): phi at be6e94f, p4 at 04eee31 and the store at e5df5fa, on a 2-core CPU.
- [`results/colab-directions-1/`](results/colab-directions-1): phi at fd61ea6, p4 at 04eee31 and the store at e5df5fa.
