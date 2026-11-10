#!/usr/bin/env python3
"""
fib_answer.py: a pupil that answers a question in fibinary, a teacher that speaks to it and reads it back.

p4-recall's rankers score entries; nothing around a store produces an answer. Here a model does, in a language of
Fibonacci codes (phi_prune.gaps: a number's Zeckendorf digits closed by a 1, so `11` ends each code), and a teacher
stands between it and the people: the teacher turns words into numbers, the pupil sees only numbers and replies in
numbers, the teacher reads the reply back as files, symbols and lines. Nothing is added on the way back.

The language (every item one Fibonacci code):
    names      what the pupil is told a thing is called: for each entry of the store's index, its kind, its group
               (the file it is part of), and the numbers of the words in its name. Taught once.
    question   the store's node number, a count, then the numbers of the question's words
    answer     records of two codes, an act (2 edited, 3 invoked, 4 cited) and an entry's number, then the act 1,
               which closes the answer, then a check code over the question's words and the records, so a
               flipped bit that still parses is caught
    lesson     a question and the answer a session really gave it: the entries its turn edited, ran and read,
               as p4-recall's labels already hold them (models/tasks.tsv, models/labels.tsv)

The pupil learns, from lessons alone, which words of a question go with which words of a name (a weight for each
hashed pair of word numbers), how much being named in earlier answers counts, which act an entry gets, and when to
stop. Its reply is the entries it scores over its stop line, edits first, at most MAX_RECORDS.

    teach   --store <project>/.claude/p4      writes out/names.fib, out/lessons.fib and the teacher's tables
    train                                     the pupil reads the two streams, learns on the earlier lessons, and
                                              is graded on the later ones against the words-in-common baseline
    ask "<question>"                          the teacher speaks the question, the pupil replies, the teacher reads
                                              the reply back with the source at each address
    compare                                   the same lessons spoken as prose: whether the pupil is told the
                                              same thing, the size, the reading time, and what a damaged
                                              lesson becomes in each
    run     --store ...                       teach, train and the report

Free, CPU, the store only read. What it cannot do: write text that is not in the project. Its answer is a chain of
things that exist, with what was done to each.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
sys.path.insert(0, str(ROOT / "phi"))
sys.path.insert(0, str(ROOT / "p4" / "lbiza" / "plugins" / "p4-recall" / "scripts"))

STOP, EDITED, INVOKED, CITED = 1, 2, 3, 4
ACT_NAMES = {EDITED: "edited", INVOKED: "invoked", CITED: "cited"}
KINDS = ("store", "file", "symbol", "commit")
CHECK_MOD = 89
MAX_RECORDS = 8            # an answer's records
PER_GROUP = 2              # of one file
CANDIDATES = 300           # entries the pupil weighs for one question
Q_WORDS, E_WORDS = 10, 12  # words of a question and of a name that are paired
PAIR_BITS = 18             # the pair weights' table is 2 ** PAIR_BITS
DENSE = 14
RECENT = 5                 # lessons back whose answers' entries are weighed whatever the words
EPOCHS, RATE, DECAY = 6, 0.05, 1e-6
HELD_OUT = 0.2             # the latest lessons grade the pupil


def gaps():
    from phi_prune import gaps as g
    return g


def code(values):
    g = gaps()
    return "".join(g.fib_code(int(v)) for v in values)


def read(bits, start=0, count=None):
    return gaps().read_codes(bits, start, count)


def check_of(acts, entries, words):
    """Over the question's words too: a lesson whose question was damaged is not a lesson about another question."""
    return (3 * sum(acts) + 5 * sum(entries) + 7 * sum(words) + len(words)) % CHECK_MOD + 1


# --- the teacher: words to numbers and back ------------------------------------------------------------------

def tokens(text):
    import ranker
    return list(dict.fromkeys(ranker.tokens(text)))


def read_tsv(path):
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def load_index(store):
    """[{id, path, line, kind, words, show}] of the store's index as it stands (never rebuilt here)."""
    with open(Path(store) / "models" / "index.json", encoding="utf-8") as f:
        d = json.load(f)
    fields = d["doc_fields"]
    out = []
    for row in d["docs"]:
        e = dict(zip(fields, row))
        i = e["id"]
        kind = ("commit" if i.startswith("commit:") else "symbol" if i.startswith("repo:") and "#" in i
                else "file" if i.startswith("repo:") else "store")
        out.append({"id": i, "path": e["path"], "line": e["line"] or 1, "kind": kind, "show": e["show"],
                    "words": list(dict.fromkeys((e["toks"] or "").split()))})
    return out


def act_of(row):
    word = (row.get("note") or "").split(" ")[0].strip("();")
    return {"edited": EDITED, "invoked": INVOKED}.get(word, CITED)


def lessons_of(store, entry_no):
    """[(task id, time, question text, [(act, entry number)])] in time order: each task with what its turn did."""
    tasks = {r["id"]: r for r in read_tsv(Path(store) / "models" / "tasks.tsv")}
    got = defaultdict(dict)
    for r in read_tsv(Path(store) / "models" / "labels.tsv"):
        if r.get("label") == "1" and r.get("entry") in entry_no and r.get("task") in tasks:
            a, e = act_of(r), entry_no[r["entry"]]
            got[r["task"]][e] = min(a, got[r["task"]].get(e, CITED))
    out = [(t, tasks[t]["time"], tasks[t]["text"], sorted(((a, e) for e, a in v.items())))
           for t, v in got.items() if tokens(tasks[t]["text"])]
    return sorted(out, key=lambda x: (x[1], x[0]))


def teach(store, out=OUT, node=1):
    """Write the two streams the pupil learns from and the tables only the teacher keeps."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = load_index(store)
    entry_no = {e["id"]: n + 1 for n, e in enumerate(index)}
    lessons = lessons_of(store, entry_no)
    count = Counter(w for e in index for w in e["words"])
    count.update(w for _t, _at, text, _a in lessons for w in tokens(text))
    words = [w for w, _n in sorted(count.items(), key=lambda x: (-x[1], x[0]))]     # a common word: a short code
    word_no = {w: n + 1 for n, w in enumerate(words)}
    groups = {}
    names = [code([node, len(index)])]
    for e in index:
        g = groups.setdefault(e["path"], len(groups) + 1)
        ws = [word_no[w] for w in e["words"]]
        names.append(code([KINDS.index(e["kind"]) + 1, g, len(ws) + 1] + ws))
    stream = [code([node, len(lessons)])]
    for _t, _at, text, answer in lessons:
        q = [word_no[w] for w in tokens(text)]
        stream.append(question_bits(q, node, head=False) + answer_bits(answer, q))
    (out / "names.fib").write_text("".join(names), encoding="ascii")
    (out / "lessons.fib").write_text("".join(stream), encoding="ascii")
    table = {"store": str(store), "node": node, "words": words,
             "entries": [[e["id"], e["path"], e["line"], e["show"]] for e in index],
             "lessons": [[t, at] for t, at, _x, _a in lessons]}
    (out / "teacher.json").write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    return {"entries": len(index), "words": len(words), "lessons": len(lessons),
            "names_bits": sum(map(len, names)), "lessons_bits": sum(map(len, stream))}


def question_bits(word_numbers, node=1, head=True):
    return code(([node] if head else []) + [len(word_numbers) + 1] + list(word_numbers))


def answer_bits(answer, words):
    flat = [x for a, e in answer for x in (a, e)]
    return code(flat + [STOP, check_of([a for a, _e in answer], [e for _a, e in answer], words)])


def read_answer(bits, words, start=0):
    """([(act, entry number)], index past the answer); ValueError when the stream or its check is wrong."""
    answer, i = [], start
    while True:
        (a,), i = read(bits, i, 1)
        if a == STOP:
            break
        if a not in ACT_NAMES:
            raise ValueError(f"no such act: {a}")
        (e,), i = read(bits, i, 1)
        answer.append((a, e))
    (c,), i = read(bits, i, 1)
    if c != check_of([a for a, _e in answer], [e for _a, e in answer], words):
        raise ValueError("the answer's check code does not match")
    return answer, i


def read_back(answer, out=OUT, lines=6):
    """The teacher's reading of a reply: each record's act, its entry as the index shows it, and the source there."""
    table = json.loads((Path(out) / "teacher.json").read_text(encoding="utf-8"))
    project = Path(table["store"]).resolve().parents[1]
    deps_path = Path(table["store"]) / "models" / "deps.json"
    outside = {}
    if deps_path.exists():
        for root, v in json.loads(deps_path.read_text(encoding="utf-8")).get("roots", {}).items():
            if v.get("prefix", "").startswith("@"):
                outside[v["prefix"]] = root
    text = []
    for act, e in answer:
        eid, path, line, show = table["entries"][e - 1]
        text.append(f"{ACT_NAMES[act]:8} {show}" + (f"  (line {line})" if eid.startswith("repo:") else ""))
        full = None
        if eid.startswith("repo:"):
            pre = max((p for p in outside if path == p or path.startswith(p + "/")), key=len, default=None)
            full = Path(outside[pre]) / path[len(pre) + 1:] if pre else project / path
        if full is not None and full.is_file() and lines:
            try:
                src = full.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                src = []
            for n in range(max(0, line - 1), min(len(src), line - 1 + lines)):
                text.append(f"    {n + 1:5}  {src[n][:150]}")
    return text


# --- the pupil: numbers in, numbers out ------------------------------------------------------------------------

class Pupil:
    """What it knows is what the two streams told it; it holds no word and no path."""

    def __init__(self):
        self.kind, self.group, self.words = [], [], []      # of entry n (0-based), from the names stream
        self.post = defaultdict(list)                       # word number -> entries named with it
        self.idf = {}
        self.pair = np.zeros(1 << PAIR_BITS, dtype=np.float32)
        self.dense = np.zeros(DENSE, dtype=np.float32)
        self.act_w = np.zeros(DENSE, dtype=np.float32)
        self.named = Counter()                              # entry -> lessons that named it
        self.edited = Counter()
        self.last = {}                                      # entry -> the lesson that last named it
        self.clock = 0                                      # lessons heard
        self.line = 0.0                                     # the stop line
        self.node = 1

    def learn_names(self, bits):
        (self.node, n), i = read(bits, 0, 2)
        for e in range(n):
            (k, g, c), i = read(bits, i, 3)
            ws, i = read(bits, i, c - 1)
            self.kind.append(k - 1)
            self.group.append(g)
            self.words.append(ws)
            for w in ws:
                self.post[w].append(e)
        total = len(self.words)
        self.idf = {w: math.log(1 + total / len(p)) for w, p in self.post.items()}
        self.unseen = math.log(1 + total)

    def read_lessons(self, bits):
        """[(question's word numbers, [(act, entry)])] of a lessons stream."""
        (_node, n), i = read(bits, 0, 2)
        out = []
        for _ in range(n):
            (c,), i = read(bits, i, 1)
            q, i = read(bits, i, c - 1)
            answer, i = read_answer(bits, q, i)
            out.append((q, answer))
        return out

    def candidates(self, q):
        score = Counter()
        for w in q:
            for e in self.post.get(w, ()):
                score[e] += self.idf[w]
        out = [e for e, _s in sorted(score.items(), key=lambda x: (-x[1], x[0]))[:CANDIDATES]]
        have = set(out)
        out += sorted(e for e, at in self.last.items() if self.clock - at <= RECENT and e not in have)
        return out, score

    def heard(self, answer):
        """An answer a session gave, after the pupil's own: what it named, and when."""
        self.clock += 1
        for a, e in answer:
            self.named[e - 1] += 1
            self.edited[e - 1] += a == EDITED
            self.last[e - 1] = self.clock

    def features(self, q, cands, score):
        """(dense [n, DENSE], pair indices [n, Q_WORDS * E_WORDS], pair mask) of a question's candidates."""
        qs = set(q)
        total = sum(self.idf.get(w, self.unseen) for w in q) or 1.0
        top_q = sorted(qs, key=lambda w: -self.idf.get(w, self.unseen))[:Q_WORDS]
        dense = np.zeros((len(cands), DENSE), dtype=np.float32)
        pairs = np.zeros((len(cands), Q_WORDS * E_WORDS), dtype=np.int64)
        mask = np.zeros((len(cands), Q_WORDS * E_WORDS), dtype=np.float32)
        for r, e in enumerate(cands):
            ws = self.words[e]
            shared = [w for w in ws if w in qs]
            d = dense[r]
            d[0] = 1.0
            d[1] = score[e] / total
            d[2] = len(shared) / max(1, len(ws))
            d[3] = min(len(shared), 5) / 5.0
            d[4 + self.kind[e]] = 1.0
            d[8] = math.log1p(self.named[e]) / 3.0
            d[9] = math.log1p(self.edited[e]) / 3.0
            d[10] = len(shared) / max(1.0, len(ws)) ** 0.5 / 2.0
            d[11] = min(len(ws), 30) / 30.0
            if e in self.last:
                d[12] = 1.0 / (self.clock - self.last[e] + 1)
                d[13] = 1.0 if self.clock == self.last[e] else 0.0
            k = 0
            for a in top_q:
                for b in ws[:E_WORDS]:
                    pairs[r, k] = (a * 1000003 + b * 7919) & ((1 << PAIR_BITS) - 1)
                    mask[r, k] = 1.0
                    k += 1
        return dense, pairs, mask

    def scores(self, dense, pairs, mask):
        return dense @ self.dense + (self.pair[pairs] * mask).sum(axis=1) / math.sqrt(Q_WORDS)

    def learn(self, lessons, epochs=EPOCHS, seed=0):
        """Learn from lessons in the order given; the counts of what earlier answers named move with the lessons, so
        a lesson is never told its own answer."""
        rng = np.random.default_rng(seed)
        prepared = []
        for q, answer in lessons:
            cands, score = self.candidates(q)
            if cands:
                want = {e - 1: a for a, e in answer}
                dense, pairs, mask = self.features(q, cands, score)
                y = np.array([1.0 if e in want else 0.0 for e in cands], dtype=np.float32)
                ed = np.array([1.0 if want.get(e) == EDITED else 0.0 for e in cands], dtype=np.float32)
                prepared.append((dense, pairs, mask, y, ed))
            self.heard(answer)
        for _ in range(epochs):
            for j in rng.permutation(len(prepared)):
                dense, pairs, mask, y, ed = prepared[j]
                p = 1.0 / (1.0 + np.exp(-self.scores(dense, pairs, mask)))
                pos = max(1.0, y.sum())
                w = np.where(y > 0, (len(y) - pos) / pos, 1.0).astype(np.float32)      # a right entry is rare
                g = (p - y) * w / len(y)
                self.dense -= RATE * (dense.T @ g + DECAY * self.dense)
                np.subtract.at(self.pair, pairs, RATE * (g[:, None] * mask) / math.sqrt(Q_WORDS))
                if y.sum():
                    rows = y > 0
                    pa = 1.0 / (1.0 + np.exp(-(dense[rows] @ self.act_w)))
                    self.act_w -= RATE * (dense[rows].T @ (pa - ed[rows]) / rows.sum())
        return len(prepared)

    def set_line(self, lessons):
        """The stop line that would have answered these lessons best (F1 over them)."""
        scored = []
        for q, answer in lessons:
            cands, score = self.candidates(q)
            if not cands:
                continue
            s = self.scores(*self.features(q, cands, score))
            scored.append((s, cands, {e - 1 for _a, e in answer}))
            self.heard(answer)
        best = (-1.0, 0.0)
        for line in np.quantile(np.concatenate([s for s, _c, _w in scored]), np.linspace(0.5, 0.999, 60)) if scored else []:
            hit = said = want = 0
            for s, cands, w in scored:
                got = self.pick(s, cands, line)
                hit += len(set(got) & w)
                said += len(got)
                want += min(len(w), MAX_RECORDS)
            f1 = 2 * hit / (said + want) if said + want else 0.0
            if f1 > best[0]:
                best = (f1, float(line))
        self.line = best[1]
        return best

    def pick(self, s, cands, line):
        out, per = [], Counter()
        for j in np.argsort(-s):
            if s[j] < line or len(out) >= MAX_RECORDS:
                break
            g = self.group[cands[j]]
            if per[g] < PER_GROUP:
                per[g] += 1
                out.append(cands[j])
        return out

    def reply(self, bits):
        """A question stream in, an answer stream out."""
        (_node, c), i = read(bits, 0, 2)
        q, _ = read(bits, i, c - 1)
        cands, score = self.candidates(q)
        answer = []
        if cands:
            dense, pairs, mask = self.features(q, cands, score)
            s = self.scores(dense, pairs, mask)
            row = {e: r for r, e in enumerate(cands)}
            for e in self.pick(s, cands, self.line):
                act = EDITED if dense[row[e]] @ self.act_w > 0 else CITED
                answer.append((act, e + 1))
            answer.sort(key=lambda x: x[0])                 # what it would change, then what it would read
        return answer_bits(answer, q)

    def save(self, path):
        np.savez_compressed(path, pair=self.pair, dense=self.dense, act_w=self.act_w, line=self.line,
                            named=np.array(sorted(self.named.items()), dtype=np.int64).reshape(-1, 2),
                            edited=np.array(sorted(self.edited.items()), dtype=np.int64).reshape(-1, 2),
                            last=np.array(sorted(self.last.items()), dtype=np.int64).reshape(-1, 2), clock=self.clock)

    def load(self, path):
        d = np.load(path)
        self.pair, self.dense, self.act_w, self.line = d["pair"], d["dense"], d["act_w"], float(d["line"])
        self.named = Counter({int(a): int(b) for a, b in d["named"]})
        self.edited = Counter({int(a): int(b) for a, b in d["edited"]})
        self.last, self.clock = {int(a): int(b) for a, b in d["last"]}, int(d["clock"])


# --- grading ------------------------------------------------------------------------------------------------

def grade(pupil, lessons, baseline=False):
    """The pupil's replies to lessons it has not learned from, against what the sessions did; after each reply it
    hears what the session did, as it would in use (its weights do not move). The baselines say the same number of
    entries: by words in common alone, and the ones the latest answers named."""
    hit = said = want = found = spoke = act_ok = 0
    for q, answer in lessons:
        truth = {e: a for a, e in answer}
        bits = pupil.reply(question_bits(q, pupil.node))
        got, _ = read_answer(bits, q)
        if baseline == "words":
            cands, _score = pupil.candidates(q)
            got = [(CITED, e + 1) for e in cands[:len(got)]]
        elif baseline == "latest":
            late = sorted(pupil.last, key=lambda e: (-pupil.last[e], -pupil.named[e], e))
            got = [(CITED, e + 1) for e in late[:len(got)]]
        right = [(a, e) for a, e in got if e in truth]
        hit += len(right)
        said += len(got)
        want += min(len(truth), MAX_RECORDS)
        found += bool(right)
        spoke += bool(got)
        act_ok += sum((a == EDITED) == (truth[e] == EDITED) for a, e in right)
        pupil.heard(answer)
    p, r = hit / max(1, said), hit / max(1, want)
    return {"lessons": len(lessons), "spoke": spoke, "records": said, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "a_right_record": found, "acts_right": act_ok / max(1, hit)}


def train(out=OUT, held_out=HELD_OUT):
    out = Path(out)
    pupil = Pupil()
    pupil.learn_names((out / "names.fib").read_text(encoding="ascii"))
    lessons = pupil.read_lessons((out / "lessons.fib").read_text(encoding="ascii"))
    cut = int(len(lessons) * (1 - held_out))
    dev = int(cut * 0.85)
    t0 = time.time()
    pupil.learn(lessons[:dev])
    line = pupil.set_line(lessons[dev:cut])
    res = {"learned": dev, "line_set_on": cut - dev, "seconds": round(time.time() - t0, 1), "stop_line_f1": line[0],
           "pupil": None}
    state = (Counter(pupil.named), Counter(pupil.edited), dict(pupil.last), pupil.clock)
    for name, base in (("pupil", False), ("words_in_common", "words"), ("latest_named", "latest")):
        pupil.named, pupil.edited, pupil.last, pupil.clock = Counter(state[0]), Counter(state[1]), dict(state[2]), state[3]
        res[name] = grade(pupil, lessons[cut:], baseline=base)
    pupil.save(out / "pupil.npz")
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def ask(text, out=OUT, lines=6):
    """(question stream, reply stream, the teacher's reading of the reply)."""
    out = Path(out)
    table = json.loads((out / "teacher.json").read_text(encoding="utf-8"))
    word_no = {w: n + 1 for n, w in enumerate(table["words"])}
    unknown = len(word_no) + 1                      # a word the pupil was never told: one number for all of them
    q = [word_no.get(w, unknown) for w in tokens(text)]
    pupil = Pupil()
    pupil.learn_names((out / "names.fib").read_text(encoding="ascii"))
    pupil.load(out / "pupil.npz")
    question = question_bits(q, table["node"])
    reply = pupil.reply(question)
    answer, _ = read_answer(reply, q)
    return question, reply, read_back(answer, out, lines)


# --- the same lessons as prose --------------------------------------------------------------------------------

def prose_lessons(out=OUT):
    """The lessons stream said in words: a line a lesson, the question's words, then each record's act and entry id."""
    out = Path(out)
    table = json.loads((out / "teacher.json").read_text(encoding="utf-8"))
    pupil = Pupil()
    lessons = pupil.read_lessons((out / "lessons.fib").read_text(encoding="ascii"))
    rows = []
    for q, answer in lessons:
        rows.append(" ".join(table["words"][w - 1] for w in q) + " | " +
                    " ; ".join(f"{ACT_NAMES[a]} {table['entries'][e - 1][0]}" for a, e in answer))
    return "\n".join(rows) + "\n"


def read_prose(text, word_no, entry_no):
    """[(question's word numbers, [(act, entry)])] of prose lessons; ValueError for a line that does not read."""
    acts = {v: k for k, v in ACT_NAMES.items()}
    out = []
    for line in text.splitlines():
        q, _, rest = line.partition(" | ")
        answer = []
        for rec in rest.split(" ; ") if rest else []:
            a, _, e = rec.partition(" ")
            answer.append((acts[a], entry_no[e]))            # KeyError: a word that is no act, an id that is no entry
        out.append(([word_no[w] for w in q.split()], answer))
    return out


def compare(out=OUT, flips=1000, seed=0):
    """The lessons as fibinary and as prose: are they the same lessons, how large, how fast to read, and what
    becomes of a damaged one (caught, read as something else without a sign, or unharmed)."""
    import random
    out = Path(out)
    table = json.loads((out / "teacher.json").read_text(encoding="utf-8"))
    word_no = {w: n + 1 for n, w in enumerate(table["words"])}
    entry_no = {e[0]: n + 1 for n, e in enumerate(table["entries"])}
    bits = (out / "lessons.fib").read_text(encoding="ascii")
    prose = prose_lessons(out)
    pupil = Pupil()
    t0 = time.perf_counter()
    from_bits = pupil.read_lessons(bits)
    t1 = time.perf_counter()
    from_prose = read_prose(prose, word_no, entry_no)
    t2 = time.perf_counter()
    same = [(list(q), list(a)) for q, a in from_bits] == [(list(q), list(a)) for q, a in from_prose]
    rng = random.Random(seed)
    fate = {"fib": Counter(), "prose": Counter()}
    for _ in range(flips):
        i = rng.randrange(len(bits))
        try:
            got = pupil.read_lessons(bits[:i] + ("1" if bits[i] == "0" else "0") + bits[i + 1:])
            fate["fib"]["unharmed" if got == from_bits else "silently different"] += 1
        except (ValueError, IndexError):
            fate["fib"]["caught"] += 1
        raw = bytearray(prose.encode("utf-8"))
        j = rng.randrange(len(raw))
        raw[j] ^= 1 << rng.randrange(7)
        try:
            got = read_prose(raw.decode("utf-8", errors="strict"), word_no, entry_no)
            fate["prose"]["unharmed" if got == from_prose else "silently different"] += 1
        except (KeyError, ValueError):
            fate["prose"]["caught"] += 1
    return {"same_lessons": same, "lessons": len(from_bits),
            "fib_bytes": (len(bits) + 7) // 8, "prose_bytes": len(prose.encode("utf-8")),
            "fib_read_ms": round((t1 - t0) * 1000, 1), "prose_read_ms": round((t2 - t1) * 1000, 1),
            "one_bit_flipped": {k: dict(v) for k, v in fate.items()}, "flips": flips}


def compare_report(c):
    f, p = c["one_bit_flipped"]["fib"], c["one_bit_flipped"]["prose"]
    row = lambda name, d: (f"  {name:9} caught {d.get('caught', 0) / c['flips']:5.1%}   read as something else "
                           f"{d.get('silently different', 0) / c['flips']:5.1%}   unharmed {d.get('unharmed', 0) / c['flips']:5.1%}")
    return "\n".join([
        f"the {c['lessons']} lessons as fibinary and as prose: " + ("the same lessons, number for number" if c["same_lessons"] else "NOT the same lessons"),
        f"  size      fibinary {c['fib_bytes']:,} bytes, prose {c['prose_bytes']:,} bytes ({c['prose_bytes'] / max(1, c['fib_bytes']):.1f} times)",
        f"  reading   fibinary {c['fib_read_ms']} ms, prose {c['prose_read_ms']} ms",
        f"one bit flipped, {c['flips']} times each:", row("fibinary", f), row("prose", p)])


def report(taught, res):
    rows = ["", f"taught: {taught['entries']} names, {taught['words']} words, {taught['lessons']} lessons "
                f"({taught['names_bits'] // 8000} kB of names, {taught['lessons_bits'] // 8000} kB of lessons)" if taught else "",
            f"learned {res['learned']} lessons, stop line set on {res['line_set_on']}, in {res['seconds']} s",
            f"graded on the {res['pupil']['lessons']} latest lessons, never learned from:",
            f"  {'':18} {'spoke':>6} {'records':>8} {'precision':>10} {'recall':>7} {'f1':>6} {'a right record':>15} {'acts right':>11}"]
    for name in ("pupil", "words_in_common", "latest_named"):
        g = res[name]
        rows.append(f"  {name.replace('_', ' '):18} {g['spoke']:6d} {g['records']:8d} {g['precision']:10.3f} {g['recall']:7.3f} "
                    f"{g['f1']:6.3f} {g['a_right_record']:15d} {g['acts_right']:11.2f}")
    return "\n".join(r for r in rows if r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("teach", "run"):
        s = sub.add_parser(name)
        s.add_argument("--store", default=os.environ.get("P4_STORE"))
        s.add_argument("--out", default=str(OUT))
    for name in ("train", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(OUT))
    s = sub.add_parser("ask")
    s.add_argument("text")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--lines", type=int, default=6)
    s.add_argument("--bits", action="store_true", help="print the two streams too")
    a = ap.parse_args(argv)
    if a.cmd in ("teach", "run"):
        if not a.store:
            ap.error("--store <project>/.claude/p4 (or P4_STORE)")
        taught = teach(a.store, a.out)
        if a.cmd == "teach":
            print(json.dumps(taught))
            return 0
        print(report(taught, train(a.out)))
    elif a.cmd == "train":
        print(report(None, train(a.out)))
    elif a.cmd == "compare":
        print(compare_report(compare(a.out)))
    else:
        question, reply, text = ask(a.text, a.out, a.lines)
        if a.bits:
            print("question:", question)
            print("reply:   ", reply)
        print("\n".join(text) if text else "(the pupil closed its answer with no record)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
