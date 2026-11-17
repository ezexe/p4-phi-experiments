# Blank-page drafts for the rebuild

Two designs, written from nothing rather than edited from the old contracts. The Colab notebook's rebuild (`.docs/p4/overrides_colab.ipynb`, sections 6 to 9) uses them as candidates, next to the pieces Claude writes fresh from each old contract. A piece goes into the rebuilt contracts only when it measures better.

Each file is the whole text of one contract. Its first line is the old contract's heading, word for word: the hooks and the plugin tests key on it.

- `budget/`: one short contract per plugin, in plain words, under 3k characters in total. Procedure is left to the plugins' skills, and the steps the hooks need are written as exact calls.
- `core/`: the design the macOS-port session proposed on 2026-11-01, as corrected the same day. It has five behavior lines in place of R1 to R23, one line on where scripts go, and one exact call that launches the operator. The operator keeps all its store work, because its pool picks, barrier verdicts and completed rows are the phi models' training labels; only its reply to the main context gets short. The gc's judgment runs inside the operator, so the gc contract keeps only its heading. The design's hook and store changes (an enforced launch, a 5-line reply, judged closes only for flagged rows) are code, and this measurement does not cover them.

File names are `<plugin>__<contract>.md`, the same as the suite's `--edits` files.
