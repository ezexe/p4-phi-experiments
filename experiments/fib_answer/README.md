# fib_answer

A model that answers a question in fibinary, and a teacher that speaks to it and reads it back.

p4-recall's rankers score entries; nothing around a store produced an answer.
Here a pupil does, in a language of Fibonacci codes, and it is told no word and no path: the teacher turns words into numbers, the pupil replies in numbers, and the teacher reads the reply back as files, symbols and lines with the source at each address.
Nothing is added on the way back.

## The language

- **names**: for each entry of the store's index, its kind, the file it is part of, and the numbers of the words in its name. Taught once.
- **question**: the store's node number, a count, the numbers of the question's words.
- **answer**: records of two codes, an act (2 edited, 3 invoked, 4 cited) and an entry's number, closed by the act 1 and a check code over the question's words and the records.
- **lesson**: a question and the answer a session really gave it, from `models/tasks.tsv` and `models/labels.tsv`.

## Run

```bash
.venv/bin/python experiments/fib_answer/fib_answer.py run --store <project>/.claude/p4
.venv/bin/python experiments/fib_answer/fib_answer.py ask "<question>" [--bits]
.venv/bin/python experiments/fib_answer/fib_answer.py compare
.venv/bin/python -I experiments/fib_answer/test_fib_answer.py
```

Free, CPU, seconds; the store is only read; everything written lands in the ignored `out/`.

## Result, 2026-11-11, one project's store (19,719 names, 346 lessons)

Trained on the 234 earliest lessons, its stop line set on the next 42, graded on the 70 latest, which it never learned from.

| | precision | recall | a right record in |
|---|---|---|---|
| the pupil | 0.282 | 0.415 | 39 of 70 |
| the entries the latest answers named | 0.302 | 0.445 | 39 of 70 |
| words in common with the question | 0.023 | 0.033 | 8 of 70 |

- The pupil answers twelve times better than word overlap, and gets the act right on 92% of its right records.
- It does not beat repeating what the latest answers named: nearly all it has learned is that a session keeps working on the same files. The question's words add nothing measurable yet.
- Asked a question no lesson resembles, it names the right area of the project and not the files that answer it.

## Fibinary against prose

The same 346 lessons written as lines of words (`compare`):

| | fibinary | prose |
|---|---|---|
| the lessons the pupil is told | the same, number for number | the same |
| size | 41,215 bytes | 283,997 bytes (6.9 times) |
| time to read (Python) | 49 ms | 6 ms |
| one bit flipped: caught | 99.3% | 99.6% |
| one bit flipped: read as something else, no sign | 0.7% | 0.2% |

The pupil learns the same from either, by construction: both say the same numbers.
So fibinary is what the pupil is spoken to in: it says the same, in a seventh of the bytes, and catches damage as well as prose does.
That last part had to be earned: with a check code over the records alone it let 38% of flips through, over the question's words too 3.2%, and with the check taken modulo 1,597 instead of 89, 0.7%.
Prose is still faster to read in Python, which matters to nothing at this size.

## What it cannot do

Write text that is not in the project.
Its answer is a chain of things that exist, with what was done to each.
