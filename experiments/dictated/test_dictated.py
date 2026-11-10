#!/usr/bin/env python3
"""Offline tests of dictated.py: a skill file and a store's rules.md in a temporary git repository, FAKE client.

    python experiments/dictated/test_dictated.py
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dictated as dc  # noqa: E402

SKILL = """---
name: x
description: Use the thing, always.
---
# X skill

## Models

- Always use claude-opus-4-1 for the writer role; it is the latest and most capable model.
- Keep replies under three sentences, and never ask the owner a question.
- **Why:** the archive is slow.

Run the tests before you report.
The archive lives on its own branch.

```bash
never run this line
```

| never | a table row |
|---|---|
> never a quoted line
"""

RULES = """The owner's rulings.
Written when the owner rules.
Fields: ruling, owner-words, status, time.

- ruling: the build needs cmake 4.4 everywhere, with no fallback to 3.x
  owner-words: "boss said latest cmake"
  status: LIVE
  time: 2026-10-29 11:05

- ruling: the playhead follows the host clock in every mode
  status: FREED
  time: 2026-10-20 09:00

- claim: a fact entry is a record, with a version 1.2.3 in it, that must not be read as a directive
  time: 2026-10-21 09:00
"""


def sh(cwd, *a):
    subprocess.run(a, cwd=cwd, check=True, capture_output=True)


class Dictated(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.repo = root / "repo"
        (self.repo / "skills" / "x").mkdir(parents=True)
        (self.repo / "store").mkdir()
        (self.repo / "skills" / "x" / "SKILL.md").write_text(SKILL, encoding="utf-8")
        (self.repo / "skills" / "README.md").write_text("Always read me, never.\n", encoding="utf-8")
        (self.repo / "store" / "rules.md").write_text(RULES, encoding="utf-8")
        sh(self.repo, "git", "init", "-q")
        sh(self.repo, "git", "add", "-A")
        sh(self.repo, "git", "-c", "user.name=Morgan Tester", "-c", "user.email=morgan@example.org",
           "-c", "commit.gpgsign=false", "commit", "-q", "-m", "seed", "--date", "2026-09-01T10:00:00")
        self.work = root / "out"
        dc.configure(FAKE=True, WORK=self.work, INSTRUCTIONS=[str(self.repo / "skills")], STORE=str(self.repo),
                     TODAY="2026-11-10", PROBE_SAMPLE=12, ANSWER_MODEL="claude-opus-5-5", RECHECK_DAYS=30)
        dc.sync()

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract(self):
        ds = {d["text"][:28]: d for d in dc.extract()}
        self.assertEqual(len(ds), 4, list(ds))                      # no front matter, fence, table, quote, README
        m = ds["Always use claude-opus-4-1 f"]
        self.assertEqual((m["class"], m["written"], m["age_days"]), ("datable", "2026-09-01", 70))
        self.assertIn("model", m["kinds"])
        self.assertIn("recency", m["kinds"])
        self.assertEqual(m["id"], f"skills/x/SKILL.md:9:{m['sha']}")
        self.assertEqual(ds["Keep replies under three sen"]["class"], "manner")
        self.assertEqual(ds["Run the tests before you rep"]["class"], "procedure")
        r = ds["the build needs cmake 4.4 ev"]                      # the LIVE ruling, dated by its own time: field
        self.assertEqual((r["class"], r["written"], r["from_store"]), ("datable", "2026-10-29", True))
        self.assertFalse(any("playhead" in d["text"] or "fact entry" in d["text"] for d in ds.values()))
        self.assertFalse(any("archive" in d["text"] for d in ds.values()))     # a statement, and a label of reasons

    def test_verdict(self):
        j = {"decidable": "FACT", "instruction_right": True, "bare_right": True}
        self.assertEqual(dc.verdict(j), "REDUNDANT")
        self.assertEqual(dc.verdict({**j, "bare_right": False}), "NEEDED")
        self.assertEqual(dc.verdict({**j, "instruction_right": False}), "TRAP")
        self.assertEqual(dc.verdict({**j, "instruction_right": False, "bare_right": False}), "STALE")
        self.assertEqual(dc.verdict({**j, "decidable": "PREFERENCE", "instruction_right": False}), "PREFERENCE")

    def test_pick(self):
        ds = dc.extract()
        dc.configure(PROBE_SAMPLE=2)
        sample, due = dc.pick(ds, [])
        self.assertEqual(due, 3)                                    # procedure lines are never due
        self.assertEqual([d["class"] for d in sample], ["datable", "datable"])     # floor(2/3) = 0 manner
        self.assertEqual(sample[0]["written"], "2026-09-01")        # two kinds of evidence, before the ruling's one
        dc.configure(PROBE_SAMPLE=3)
        self.assertEqual([d["class"] for d in dc.pick(ds, [])[0]], ["datable", "datable", "manner"])

    def test_run_and_recheck(self):
        s = dc.run(work=self.work)
        self.assertTrue(s["fake"])
        self.assertEqual((s["directives"], s["due"], s["probed"]), (4, 3, 3))
        self.assertEqual(s["verdicts"], {"TRAP": 3})                # the fake judge's one answer
        self.assertEqual((s["trapped"], s["justified_by_instruction_alone"], s["said_instruction"]), (3, 3, 3))
        res = self.work / "results"
        facts = (res / "facts-draft.md").read_text(encoding="utf-8")
        self.assertEqual(facts.count("- claim: "), 3)
        for field in ("source: https://docs.example.org/current", "verified: 2026-11-10", "prior: fake: do as told",
                      "from: skills/x/SKILL.md:9:"):
            self.assertIn(field, facts)
        self.assertNotIn("boss", (res / "probes.json").read_text(encoding="utf-8").split('"directive"')[0])
        tasks = [json.loads(x) for x in (res / "newer-tasks-draft.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(tasks), 3)
        self.assertEqual({t["category"] for t in tasks}, {"dictated"})
        self.assertTrue(all(t["bad"] and t["grader"]["type"] == "judge" for t in tasks))
        self.assertIn("rewrite: fake: as of 2026-01", (res / "edits.md").read_text(encoding="utf-8"))
        self.assertEqual(len(dc.read_ledger()), 3)
        self.assertFalse((res / "ledger.tsv").exists())             # a fake verdict never enters the paid ledger
        dc.configure(FAKE=False)
        self.assertEqual(dc.pick(dc.extract(), dc.read_ledger())[1], 3)     # so a paid run still owes all three
        dc.configure(FAKE=True)

        self.assertEqual(dc.run(work=self.work)["probed"], 0)       # same text, same model, same day: nothing due
        dc.configure(TODAY="2026-12-15")                            # past RECHECK_DAYS
        self.assertEqual(dc.run(work=self.work)["probed"], 3)
        dc.configure(ANSWER_MODEL="claude-sonnet-5-5")              # a new answering model
        self.assertEqual(dc.run(work=self.work)["probed"], 3)
        self.assertEqual(len(dc.read_ledger()), 9)
        p = self.repo / "skills" / "x" / "SKILL.md"                 # a changed directive is a new one
        p.write_text(SKILL.replace("claude-opus-4-1", "claude-opus-5-5"), encoding="utf-8")
        s = dc.run(work=self.work)
        self.assertEqual((s["due"], s["probed"]), (1, 1))

    def test_flips(self):
        s = dc.run(work=self.work)
        self.assertEqual(s["flips"], [])
        rows = dc.read_ledger()
        before = [{**rows[0], "verdict": "NEEDED", "model": "claude-opus-4-1", "checked": "2026-05-01"}]
        probed = json.loads((self.work / "results" / "probes.json").read_text(encoding="utf-8"))
        fl = dc.flips(before, probed)
        self.assertEqual([(r["id"], o["verdict"], r["verdict"]) for r, o in fl], [(rows[0]["id"], "NEEDED", "TRAP")])

    def test_scrub(self):
        self.assertEqual(dc.scrub("Morgan said the boss wants cmake"), "owner said the owner wants cmake")


if __name__ == "__main__":
    unittest.main()
