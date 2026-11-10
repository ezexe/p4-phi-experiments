#!/usr/bin/env python3
"""Offline tests of fib_wire.py on a small made-up store.

    python experiments/fib_wire/test_fib_wire.py
"""

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fib_wire as fw  # noqa: E402

ROWS = [("t1", "repo:a.py", "1", "edited"), ("t1", "repo:b.py", "0", "dropped"), ("t1", "rules.md:2026-11-01 10:00:abcd1234", "1", "used"),
        ("t2", "repo:a.py", "1", "cited (symbol)"), ("t2", "repo:a.py", "0", "unopened"), ("t3", "repo:zz/deep/file.cc", "1", ""),
        ("t3", "commit:deadbeef", "0", "")]


def store(root):
    m = Path(root) / "models"
    m.mkdir(parents=True)
    head = "task\tentry\tlabel\tsource\tweight\ttime\tteacher\tedited\tnote\n"
    (m / "labels.tsv").write_text(head + "".join(f"{t}\t{e}\t{lab}\trepo\t1.0\t2026-11-01 10:00\t\t\t{note}\n" for t, e, lab, note in ROWS)
                                  + f"{ROWS[0][0]}\t{ROWS[0][1]}\t1\trepo\t1.0\t2026-11-01 10:00\t\t\tedited\n", encoding="utf-8")
    return str(root)


class FibWire(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.statements = fw.read_labels(store(Path(self.tmp.name) / "p4"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_statements_and_acts(self):
        self.assertEqual(len(self.statements), 7, "a row written twice is one statement")
        acts = {(t, e): a for t, e, a in self.statements if t != "t2"}
        self.assertEqual(acts[("t1", "repo:a.py")], "edited")
        self.assertEqual(acts[("t3", "repo:zz/deep/file.cc")], "positive", "no act in the note: the label's own")
        self.assertEqual(acts[("t3", "commit:deadbeef")], "negative")
        self.assertIn(("t2", "repo:a.py", "cited"), self.statements)
        self.assertIn(("t2", "repo:a.py", "unopened"), self.statements, "the same entry twice in a task, under two acts")

    def test_round_trip_and_the_stream_is_fibinary(self):
        bits, tasks, entries = fw.encode(self.statements, node=3)
        self.assertEqual(set(bits), {"0", "1"})
        node, back = fw.decode(bits, tasks, entries)
        self.assertEqual((node, sorted(back)), (3, sorted(self.statements)))
        codes = bits.split("11")
        self.assertEqual(codes[-1], "", "the stream ends on a closing 11")
        self.assertEqual(len(codes) - 1, 1 + fw.FIELDS * len(self.statements), "a node code, then four codes a record")
        self.assertTrue(all("11" not in c for c in codes), "no code holds two adjacent 1s before its close")
        self.assertEqual(fw.unpack(fw.pack(bits)), bits)
        self.assertEqual(fw.unpack(fw.pack("")), "")
        plain, _, _ = fw.encode(self.statements, with_check=False)
        self.assertLess(len(plain), len(bits))
        self.assertEqual(sorted(fw.decode(plain, tasks, entries, with_check=False)[1]), sorted(self.statements))

    def test_two_streams_against_one_table_read_as_one(self):
        names = fw.tables(self.statements)
        a = fw.encode(self.statements[:3], names=names)[0]
        b = fw.encode(self.statements[3:], names=names)[0]
        self.assertEqual(sorted(fw.decode_many([a, b], *names)), sorted(self.statements))

    def test_corruption_is_caught(self):
        bits, tasks, entries = fw.encode(self.statements)
        with self.assertRaises(fw.Corrupt):
            fw.decode(bits[:-1], tasks, entries)                      # a code cut short
        with self.assertRaises(fw.Corrupt):
            fw.decode(bits + "011", tasks, entries)                   # a stray code: not whole records
        with self.assertRaises(fw.Corrupt):
            fw.decode(bits, tasks[:1], entries)                       # the wrong table: names past its end
        got = fw.flip_test(bits, tasks, entries, self.statements, n=300, seed=1)
        self.assertEqual(sum(got.values()), 300)
        self.assertEqual(got["harmless"], 0, "every bit of the stream means something")
        plain, _, _ = fw.encode(self.statements, with_check=False)
        bare = fw.flip_test(plain, tasks, entries, self.statements, n=300, seed=1, with_check=False)
        self.assertLess(got["silent"], bare["silent"], "the check code catches flips the structure lets through")
        self.assertEqual(bare["check"], 0)

    def test_run_and_report(self):
        res = fw.run(str(Path(self.tmp.name) / "p4"), flips=200)
        self.assertTrue(res["lossless"] and res["append_reads_as_one"])
        self.assertEqual((res["statements"], res["tasks"], res["entries"]), (7, 3, 5))
        self.assertLess(res["wire_bytes"], res["text_bytes"])
        text = fw.report(res, Path(self.tmp.name) / "out" / "report.md")
        self.assertIn("Read back exactly: yes", text)
        self.assertNotIn(self.tmp.name, text)


if __name__ == "__main__":
    unittest.main()
