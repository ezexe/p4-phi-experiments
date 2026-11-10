#!/usr/bin/env python3
"""Offline tests of dictated.py: a skill file and a store (rulings, a retraction, a correction, a fact) in a
temporary git repository, FAKE client.

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
- Always make the playhead follow the host clock in every mode.
- Read `docs/gone.md` before editing, and call `frobnicate_widget` after.
- Check `skills/x/SKILL.md` for the steps, then call `real_helper`.

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

RETRACTIONS = """What the owner freed.
Written when the owner takes something back.
Fields: freed, time, cause, owner-words, lesson.

- freed: the playhead follows the host clock in every mode
  time: 2026-11-05 10:00
  cause: superseded
  lesson: the transport position belongs to the plugin itself
"""

CORRECTIONS = """Readings that were wrong.
Written when the owner corrects.
Fields: key, time, match, meant, delta.

- key: cmake-floor
  time: 2026-11-06 12:00
  match: lower the compiler floor for the older toolchain
  meant: the toolchain floor never moves down for an older compiler
"""

FACTS = """Claims checked against a source.
Written when verified.
Fields: claim, source, verified, time.

- claim: the latest cmake release is 4.4
  source: https://cmake.org/download/
  verified: 2026-10-01
  time: 2026-10-01 09:00
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
        (self.repo / "src").mkdir()
        (self.repo / "skills" / "x" / "SKILL.md").write_text(SKILL, encoding="utf-8")
        (self.repo / "skills" / "README.md").write_text("Always read me, never.\n", encoding="utf-8")
        (self.repo / "src" / "helper.py").write_text("def real_helper():\n    return 1\n", encoding="utf-8")
        for name, text in (("rules.md", RULES), ("retractions.md", RETRACTIONS), ("corrections.md", CORRECTIONS),
                           ("facts.md", FACTS)):
            (self.repo / "store" / name).write_text(text, encoding="utf-8")
        sh(self.repo, "git", "init", "-q")
        sh(self.repo, "git", "add", "-A")
        sh(self.repo, "git", "-c", "user.name=Morgan Tester", "-c", "user.email=morgan@example.org",
           "-c", "commit.gpgsign=false", "commit", "-q", "-m", "seed", "--date", "2026-09-01T10:00:00")
        self.work = root / "out"
        dc.configure(FAKE=True, WORK=self.work, INSTRUCTIONS=[str(self.repo / "skills")], STORE=str(self.repo),
                     PROJECT="", TODAY="2026-11-10", PROBE_SAMPLE=12, ANSWER_MODEL="claude-opus-5-5", RECHECK_DAYS=30)
        dc.sync()

    def tearDown(self):
        self.tmp.cleanup()

    def by_text(self, ds):
        return {d["text"][:28]: d for d in ds}

    def test_extract(self):
        ds = self.by_text(dc.extract())
        self.assertEqual(len(ds), 7, list(ds))                      # no front matter, fence, table, quote, README
        m = ds["Always use claude-opus-4-1 f"]
        self.assertEqual((m["class"], m["written"], m["age_days"]), ("datable", "2026-09-01", 70))
        self.assertEqual(m["values"], ["claude-opus-4-1", "latest"])
        self.assertEqual(dc.classify("`GetServiceForContext` was called 1,196 times for 23.7 ms cumulative")[2], 1,
                         "a measurement is a number with a unit, never a version")
        self.assertEqual(dc.classify("pin CMake 4.4 and actions/checkout v7 in CI")[4], ["CMake 4.4", "v7"])
        self.assertEqual(m["id"], f"skills/x/SKILL.md:9:{m['sha']}")
        self.assertEqual(ds["Keep replies under three sen"]["class"], "manner")
        self.assertEqual(ds["Run the tests before you rep"]["class"], "procedure")
        self.assertEqual(ds["Read `docs/gone.md` before e"]["class"], "named", "a backticked name alone is not a dated value")
        r = ds["the build needs cmake 4.4 ev"]                      # the LIVE ruling, dated by its own time: field
        self.assertEqual((r["class"], r["written"], r["from_store"], r["values"]), ("datable", "2026-10-29", True, ["cmake 4.4"]))
        texts = [d["text"] for d in ds.values()]
        self.assertNotIn("the playhead follows the host clock in every mode", texts, "a FREED ruling is not live")
        self.assertFalse(any("fact entry" in t or "archive" in t for t in texts))

    def test_verdict(self):
        j = {"decidable": "FACT", "instruction_right": True, "bare_right": True, "value_current": "CURRENT"}
        self.assertEqual(dc.verdict(j), "REDUNDANT")
        self.assertEqual(dc.verdict({**j, "bare_right": False}), "NEEDED")
        self.assertEqual(dc.verdict({**j, "instruction_right": False}), "TRAP")
        self.assertEqual(dc.verdict({**j, "instruction_right": False, "bare_right": False}), "STALE")
        p = {**j, "decidable": "PREFERENCE", "instruction_right": False}
        self.assertEqual(dc.verdict({**p, "bare_follows": False}), "PREFERENCE-STEERS")
        self.assertEqual(dc.verdict({**p, "bare_follows": True}), "PREFERENCE-SAME")
        self.assertEqual(dc.verdict({**p, "bare_follows": False, "value_current": "CHANGED"}), "DATED",
                         "the owner's choice stands; the value it names has moved")
        self.assertEqual(set(dc.ADVICE), set(dc.VERDICTS))

    def test_names_and_about(self):
        self.assertEqual(dc.names_of("Read `docs/gone.md`, run `p4.py index --deps`, call `frobnicate_widget`, set `--force`, "
                                     "see `<name>-src`, `main`, `E:/x/y.md` and `base::span`."),
                         (["docs/gone.md", "p4.py"], ["frobnicate_widget", "base::span"]))
        idf = {}
        share, shared = dc.about(dc.toks("the playhead follows the host clock"), dc.toks("make the playhead follow the host clock"), idf)
        self.assertEqual(shared, {"playhead", "host", "clock"})
        self.assertGreaterEqual(share, dc.OVERLAP_MIN)

    def test_ground(self):
        ds = dc.extract()
        g = dc.step_ground(ds)
        got = {k: d["ground"] for k, d in self.by_text(ds).items()}
        self.assertEqual(got["Always make the playhead fol"], "RETRACTED", "freed on 2026-11-05, still dictated")
        self.assertEqual(got["Read `docs/gone.md` before e"], "MISSING")
        self.assertEqual(got["Check `skills/x/SKILL.md` fo"], "STANDS", "the path exists and git grep finds the helper")
        self.assertEqual(got["Run the tests before you rep"], "NOTHING NAMED")
        d = self.by_text(ds)["Read `docs/gone.md` before e"]
        self.assertEqual((d["missing"], d["unseen"]), (["docs/gone.md"], ["frobnicate_widget"]))
        self.assertEqual((g["records"], g["orphans"], g["facts"]), (2, 2, 1), "no line or ruling states either lesson")
        text = (self.work / "results" / "ground.md").read_text(encoding="utf-8")
        for want in ("| RETRACTED | 1 |", "| MISSING | 1 |", "Missing: `docs/gone.md`", "ORPHAN correction 2026-11-06"):
            self.assertIn(want, text)
        self.assertNotIn(self.tmp.name, text, "no local path in the report")

    def test_pick(self):
        ds = dc.extract()
        dc.configure(PROBE_SAMPLE=2)
        sample, due = dc.pick(ds, [])
        self.assertEqual(due, 4, "named and procedure lines are never due a probe")
        self.assertEqual([d["class"] for d in sample], ["datable", "datable"])     # floor(2/3) = 0 manner
        self.assertEqual(sample[0]["written"], "2026-09-01")        # two kinds of evidence, before the ruling's one
        dc.configure(PROBE_SAMPLE=3)
        self.assertEqual([d["class"] for d in dc.pick(ds, [])[0]], ["datable", "datable", "manner"])

    def test_run_and_recheck(self):
        s = dc.run(work=self.work)
        self.assertTrue(s["fake"])
        self.assertEqual((s["directives"], s["due"], s["probed"]), (7, 4, 4))
        self.assertEqual(s["verdicts"], {"TRAP": 4})                # the fake judge's one answer
        self.assertEqual((s["trapped"], s["justified_by_instruction_alone"], s["said_instruction"]), (4, 4, 4))
        self.assertEqual((s["values_checked"], s["values_changed"]), (4, 4))
        self.assertEqual(s["ground"]["RETRACTED"], 1)
        self.assertEqual(s["facts"], {"checked": 1, "by": {"CHANGED": 1}, "of": 1})
        res = self.work / "results"
        probes = json.loads((res / "probes.json").read_text(encoding="utf-8"))
        self.assertEqual({p["probe"]["value"] for p in probes if p["class"] == "datable"}, {"claude-opus-4-1", "cmake 4.4"},
                         "a datable probe hinges on the value the line names")
        self.assertTrue(all(p["value_check"]["urls"] for p in probes if p["class"] == "datable"), "and the value is searched for itself")
        facts = (res / "facts-draft.md").read_text(encoding="utf-8")
        self.assertEqual(facts.count("- claim: "), 4)
        for field in ("source: https://docs.example.org/current", "verified: 2026-11-10", "prior: claude-opus-4-1",
                      "from: skills/x/SKILL.md:9:"):
            self.assertIn(field, facts)
        self.assertIn("supersedes: facts.md:", (res / "facts-recheck.md").read_text(encoding="utf-8"))
        tasks = [json.loads(x) for x in (res / "newer-tasks-draft.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(tasks), 4)
        self.assertEqual({t["category"] for t in tasks}, {"dictated"})
        self.assertTrue(all(t["bad"] and t["grader"]["type"] == "judge" for t in tasks))
        edits = (res / "edits.md").read_text(encoding="utf-8")
        self.assertIn("rewrite: fake: as of 2026-01", edits)
        self.assertIn("value: cmake 4.4 -> fake-newer", edits)
        self.assertEqual(len(dc.read_ledger()), 4)
        self.assertFalse((res / "ledger.tsv").exists())             # a fake verdict never enters the paid ledger
        dc.configure(FAKE=False)
        self.assertEqual(dc.pick(dc.extract(), dc.read_ledger())[1], 4)     # so a paid run still owes all four
        dc.configure(FAKE=True)

        self.assertEqual(dc.run(work=self.work)["probed"], 0)       # same text, same model, same day: nothing due
        dc.configure(TODAY="2026-12-15")                            # past RECHECK_DAYS
        self.assertEqual(dc.run(work=self.work)["probed"], 4)
        dc.configure(ANSWER_MODEL="claude-sonnet-5-5")              # a new answering model
        self.assertEqual(dc.run(work=self.work)["probed"], 4)
        self.assertEqual(len(dc.read_ledger()), 12)
        dc.configure(PROBE_VERSION=dc.PROBE_VERSION + 1)            # a new kind of probe: every verdict is owed again
        self.assertEqual(dc.pick(dc.extract(), dc.read_ledger())[1], 4)
        dc.configure(PROBE_VERSION=dc.PROBE_VERSION - 1)
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

    def test_decision_needs_agreeing_probes(self):
        d = dc.decision
        self.assertEqual(d(["REDUNDANT", "REDUNDANT"]), ("undecided", 1), "two probes decide nothing")
        self.assertEqual(d(["REDUNDANT", "PREFERENCE-SAME", "REDUNDANT"]), ("same when asked", 0))
        self.assertEqual(d(["REDUNDANT", "REDUNDANT", "NEEDED"])[0], "undecided", "one probe found it working: not dropped")
        self.assertEqual(d(["TRAP", "DATED", "REDUNDANT"]), ("rewrite", 0))
        self.assertEqual(d(["NEEDED", "PREFERENCE-STEERS", "REDUNDANT"]), ("keep", 0))
        self.assertEqual(d(["PREFERENCE", "NEEDED", "TRAP"]), ("undecided", 1), "a verdict from before the split is not counted")
        self.assertEqual(d(["TRAP", "NEEDED", "REDUNDANT", "TRAP", "NEEDED"])[0], "keep (the probes disagree)")

    def test_only_and_decide(self):
        ds = dc.extract()
        named = next(d for d in ds if d["class"] == "named")
        dc.configure(ONLY={named["id"]})
        self.assertEqual([d["id"] for d in dc.pick(ds, [])[0]], [named["id"]], "asked for by id, whatever its class")
        dc.configure(ONLY=set(), FAKE=False)                       # the paid ledger's name, written by hand
        dc.append_ledger([{**dict.fromkeys(dc.LEDGER_HEAD, ""), "id": "a.md:1:x", "verdict": v} for v in ("REDUNDANT",) * 3]
                         + [{**dict.fromkeys(dc.LEDGER_HEAD, ""), "id": "b.md:2:y", "verdict": "TRAP"}])
        got = {r["id"]: (r["decision"], r["more"]) for r in dc.step_decide([self.work])}
        dc.configure(FAKE=True)
        self.assertEqual(got, {"a.md:1:x": ("same when asked", 0), "b.md:2:y": ("undecided", 2)})
        self.assertIn("**same when asked**: `a.md:1:x`", (self.work / "results" / "decisions.md").read_text(encoding="utf-8"))

    def test_scrub(self):
        self.assertEqual(dc.scrub("Morgan said the boss wants cmake"), "owner said the owner wants cmake")


if __name__ == "__main__":
    unittest.main()
