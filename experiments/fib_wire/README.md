# fib_wire

A fibinary wire between the programs around a p4 store, and what it does to a store's real labels.

p4-recall's hooks, its trainer and the phi models pass each other labels as rows of text.
This writes the same statements as a stream of Fibonacci codes (`phi_prune.gaps`: a number's Zeckendorf digits closed by a 1, so the first `11` ends each code and the stream needs no separators), and measures the trade.
It is a wire for programs: a language model does not read or write it, and a person needs the decoder.

## Run

```bash
.venv/bin/python -I experiments/fib_wire/fib_wire.py run --store <project>/.claude/p4
```

```bash
.venv/bin/python -I experiments/fib_wire/test_fib_wire.py
```

Free, offline, seconds; the store is only read, and results land in the ignored `out/`.

## The record

Four codes: the act (edited, cited, dropped, used, ..., and three a session could state itself: applied, set-aside, lacked), the task's number as a gap from the previous record's, the entry's number as a gap from the previous entry of that task, and a check, `((3 act + 5 task + 7 entry) mod 89) + 1`.
The stream's first code is the store's node number in `phi_rank.tree`'s addresses.
Task and entry names are not on the wire: a table beside it lists each once.

## What it did, 2026-11-10

One store: 10,274 statements over 664 tasks and 1,803 entries.

| | bytes |
|---|---|
| the statements as text rows | 959,489 |
| the name table (needed once, by any numbered form) | 123,294 |
| the same numbers at fixed width | 32,106 |
| the wire with its check code | 26,469 |
| the wire without the check | 15,654 |

- Lossless: the records read back are the records written, and two streams end to end read as one.
- Of 2,000 random single-bit flips, the stream's own structure caught 64% and 36% read as a valid stream saying something else. With the check code 99.6% were caught and 0.4% were silent. The check is 41% of the stream.
- 6,000 records a second written, 358,000 read.

## What that means

- It carries exactly what the rows carry, so a model trained from it is the model trained from the rows. It changes how statements travel, not what is known: it does nothing for the missing evidence about which rule applied to which task.
- Almost all the saving over text is from numbering the names (the table), which any format could do. Fibonacci coding over sorted gaps then halves the fixed-width size.
- The "free" integrity of fibinary is partial in a stream: a flip that makes a `11` just ends a code early, and a third of flips pass unnoticed until a check code is paid for.
