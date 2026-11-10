#!/usr/bin/env python3
"""fib_answer.py on a store built here: the streams read back, the pupil sees numbers only and learns, a damaged
answer is refused.

    .venv/bin/python -I experiments/fib_answer/test_fib_answer.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fib_answer as fa  # noqa: E402

FIELDS = ["id", "path", "line", "file", "kind", "time", "state", "show", "plain", "dl", "toks", "head", "match",
          "supersedes", "fkind"]


def doc(i, path, line, toks):
    return [i, path, line, "repo", "file", "2026-01-01", "LIVE", path, True, len(toks.split()), toks, toks, None, None, None]


class FibAnswer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proj = Path(self.tmp.name) / "proj"
        self.store = self.proj / ".claude" / "p4"
        (self.store / "models").mkdir(parents=True)
        (self.proj / "src").mkdir()
        (self.proj / "src" / "ring.cc").write_text("// the ring\nvoid Ring::Write() {}\n", encoding="utf-8")
        docs = [doc("repo:src/ring.cc", "src/ring.cc", 1, "src ring"),
                doc("repo:src/ring.cc#aaaa", "src/ring.cc", 2, "src ring write"),
                doc("repo:src/clock.cc", "src/clock.cc", 1, "src clock"),
                doc("repo:docs/build.md", "docs/build.md", 1, "docs build"),
                doc("repo:src/paint.cc", "src/paint.cc", 1, "src paint colour")]
        (self.store / "models" / "index.json").write_text(json.dumps({"doc_fields": FIELDS, "docs": docs}), encoding="utf-8")
        tasks, labels = ["id\tsession\ttime\ttext\tkind"], ["task\tentry\tlabel\tsource\tweight\ttime\tteacher\tedited\tnote"]
        for n in range(40):
            # a question about audio is answered in the ring, one about the timer in the clock: neither shares a word
            topic, entries = (("audio samples lane", [("repo:src/ring.cc#aaaa", "edited (symbol)"), ("repo:src/ring.cc", "cited")])
                              if n % 2 == 0 else ("timer tick late", [("repo:src/clock.cc", "cited")]))
            tasks.append(f"t{n:02d}\ts\t2026-01-01 10:{n:02d}\tfix the {topic} problem {n}\tnew")
            for e, note in entries:
                labels.append(f"t{n:02d}\t{e}\t1\trepo\t0.7\t2026-01-01 10:{n:02d}\t\t\t{note}")
        (self.store / "models" / "tasks.tsv").write_text("\n".join(tasks) + "\n", encoding="utf-8")
        (self.store / "models" / "labels.tsv").write_text("\n".join(labels) + "\n", encoding="utf-8")
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_answer_reads_back_and_a_damaged_one_is_refused(self):
        q, answer = [5, 9, 2], [(fa.EDITED, 3), (fa.CITED, 12)]
        bits = fa.answer_bits(answer, q)
        self.assertEqual(fa.read_answer(bits, q), (answer, len(bits)))
        self.assertEqual(fa.read_answer(fa.answer_bits([], q), q)[0], [], "an answer with no record is an answer")
        with self.assertRaises(ValueError):
            fa.read_answer(bits, [5, 9, 3])                     # the answer to another question
        refused = 0
        for i in range(len(bits)):
            try:
                got, _ = fa.read_answer(bits[:i] + ("1" if bits[i] == "0" else "0") + bits[i + 1:], q)
                self.assertNotEqual(got, answer)
            except (ValueError, IndexError):
                refused += 1
        self.assertGreater(refused / len(bits), 0.8)

    def test_the_pupil_learns_from_numbers_alone_and_answers_in_them(self):
        taught = fa.teach(self.store, self.out)
        self.assertEqual((taught["entries"], taught["lessons"]), (5, 40))
        for name in ("names.fib", "lessons.fib"):
            self.assertEqual(set((self.out / name).read_text(encoding="ascii")), {"0", "1"}, "the pupil is told no word")
        res = fa.train(self.out)
        self.assertGreater(res["pupil"]["precision"], res["words_in_common"]["precision"],
                           "no word of the question is in the right entry's name: only the lessons say")
        self.assertGreater(res["pupil"]["precision"], 0.8)
        question, reply, text = fa.ask("the audio lane drops samples", self.out, lines=1)
        self.assertEqual(set(question + reply), {"0", "1"})
        self.assertTrue(text[0].startswith("edited   src/ring.cc"), text)
        self.assertIn("        2  void Ring::Write() {}", text, "the source at the address, as it is in the project")
        self.assertFalse([t for t in text if "clock" in t or "paint" in t])

    def test_prose_says_the_same_lessons(self):
        fa.teach(self.store, self.out)
        c = fa.compare(self.out, flips=200)
        self.assertTrue(c["same_lessons"])
        self.assertLess(c["fib_bytes"], c["prose_bytes"])
        self.assertEqual(sum(c["one_bit_flipped"]["fib"].values()), 200)


if __name__ == "__main__":
    unittest.main()
