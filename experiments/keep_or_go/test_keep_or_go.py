#!/usr/bin/env python3
"""Offline tests of keep_or_go.py and its notebook: a local two-branch store repo and a local skill, FAKE client.

    python experiments/keep_or_go/test_keep_or_go.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import keep_or_go as kg  # noqa: E402
import make_notebook as mk  # noqa: E402
sys.path.insert(0, str(HERE.parent))
import env  # noqa: E402


def sh(cwd, *a):
    subprocess.run(a, cwd=cwd, check=True, capture_output=True)


def make_store(root):
    src = root / "src-store"
    src.mkdir()
    sh(src, "git", "init", "-q", "-b", "mac")
    sh(src, "git", "config", "user.email", "t@t")
    sh(src, "git", "config", "user.name", "t")
    for rel, text in {"memory/a.md": "the boss said keep\n", "store/rules.md": "rule\n",
                      "scratchpad/x/notes.md": "n\n"}.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    sh(src, "git", "add", "-A")
    sh(src, "git", "commit", "-qm", "mac")
    sh(src, "git", "checkout", "-qb", "win")
    (src / "memory/b.md").write_text("win only\n")
    sh(src, "git", "add", "-A")
    sh(src, "git", "commit", "-qm", "win")
    skill = root / "skills" / "sub-agentic"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: sub-agentic\ndescription: d\n---\n1. **Do it here.** x\n")
    return src, skill


class KeepOrGo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        src, skill = make_store(self.tmp)
        kg.configure(FAKE=True, STORE_REPO=str(src), BRANCHES=[], SKILL_DIRS=[str(skill)], REFERENCE_ROOTS=[],
                     WORK=self.tmp / "work")

    def test_inventory_units_and_sharing(self):
        inv = {u["unit"]: u for u in kg.step_inventory()}
        self.assertEqual(set(inv["memory"]["branches"]), {"mac", "win"})
        self.assertEqual(inv["memory"]["identical_share"], 0.5)
        self.assertIn("scratchpad/x", inv)

    def test_evidence_is_scrubbed_and_shuffled(self):
        inv = kg.step_inventory()
        self.assertNotIn("boss", kg.evidence(inv, True, 1))
        self.assertIn("boss", kg.evidence(inv, False, 1))
        heads = lambda s: [ln for ln in kg.evidence(inv, True, s).splitlines() if ln.startswith("### ")]
        self.assertEqual(sorted(heads(1)), sorted(heads(2)))

    def test_no_store_stops_before_anything_is_cloned(self):
        old = os.environ.pop("P4_STORE_REPO", None)
        try:
            kg.configure(STORE_REPO="")
            with mock.patch.object(kg, "load_env", lambda: []):
                with self.assertRaises(SystemExit):
                    kg.step_inventory()
            self.assertFalse((self.tmp / "work" / "store-repo").exists())
        finally:
            if old is not None:
                os.environ["P4_STORE_REPO"] = old

    def test_committers_names_are_scrubbed_without_being_typed_in(self):
        sh(self.tmp / "src-store", "git", "-c", "user.name=Quillon Varga", "-c", "user.email=qvarga@example.org",
           "commit", "-q", "--allow-empty", "-m", "by the owner")
        kg._OWNER_TERMS.clear()
        kg.step_inventory()
        self.assertIn("quillon", kg.owner_terms())
        self.assertEqual(kg.scrub("Quillon asked qvarga to keep it"), "owner asked owner to keep it")

    def test_run_end_to_end_offline(self):
        s = kg.run(paid=False, work=str(self.tmp / "work"))
        self.assertTrue(s["fake"])
        self.assertEqual(s["spent_usd"], 0.0)
        self.assertEqual({t["unit"] for t in s["consensus"]}, {"memory", "store", "scratchpad/x"})
        self.assertGreater(s["probe"]["sub-agentic/handwritten"], s["probe"]["sub-agentic/none"])
        self.assertIn("sub-agentic", s["gate"])
        self.assertTrue((kg.paths()["results"] / "generic-sub-agentic.SKILL.md").exists())

    def test_budget_guard_stops_paid_requests(self):
        kg.configure(FAKE=False, TOTAL_MAX_USD=0.0)
        try:
            with self.assertRaises(RuntimeError):
                kg.guard("x")
        finally:
            kg.configure(FAKE=True, TOTAL_MAX_USD=6.0)

    def test_reference_needle_finds_built_and_written_paths(self):
        hit = lambda unit, text: bool(kg.reference_needle(unit).search(text))
        self.assertTrue(hit("flights", 'for f in (p4 / "flights").iterdir():'))
        self.assertTrue(hit("flights", 'f"flights/{flight.name}/plan.md"'))
        self.assertTrue(hit("flights", "os.path.join(root, 'flights')"))
        self.assertTrue(hit("store/messages", 'os.path.join(root, "store", "messages")'))
        self.assertTrue(hit("store/messages", r"C:\p4\store\messages\x.md"))
        self.assertFalse(hit("flights", 'counts = {"flights": 3}'))
        self.assertFalse(hit("flights", "the flights we flew"))

    def test_gate_needs_plugin_near_handwritten_and_above_none(self):
        self.assertEqual(kg.gate({"a/handwritten": 1.0, "a/plugin": 0.96, "a/none": 0.5}), {"a": True})
        self.assertEqual(kg.gate({"a/handwritten": 1.0, "a/plugin": 0.9, "a/none": 0.5}), {"a": False})
        self.assertEqual(kg.gate({"a/handwritten": 0.5, "a/plugin": 0.5, "a/none": 0.5}), {"a": False})
        self.assertEqual(kg.gate({"a/handwritten": 1.0, "a/none": 0.5}), {"a": None})

    def test_reprobe_keeps_judges(self):
        kg.run(paid=False, work=str(self.tmp / "work"))
        s = kg.reprobe(paid=False, work=str(self.tmp / "work"))
        self.assertTrue((self.tmp / "work" / "results-1" / "probe.json").exists())
        self.assertEqual({t["unit"] for t in s["consensus"]}, {"memory", "store", "scratchpad/x"})

    def test_rejudge_keeps_rules_and_probes(self):
        kg.run(paid=False, work=str(self.tmp / "work"))
        s = kg.rejudge(paid=False, work=str(self.tmp / "work"))
        self.assertTrue((self.tmp / "work" / "results-1" / "report.md").exists())
        self.assertIn("sub-agentic/handwritten", s["probe"])

    def test_load_env_finds_a_parent_env_and_keeps_what_is_set(self):
        import os
        (self.tmp / ".env").write_text("# c\nP4_TEST_A='a'\nP4_TEST_B=\nexport P4_TEST_C=c\n", encoding="utf-8")
        (self.tmp / "deep" / "er").mkdir(parents=True)
        os.environ["P4_TEST_C"] = "kept"
        try:
            self.assertEqual(env.find(self.tmp / "deep" / "er"), self.tmp / ".env")
            self.assertEqual(env.load(env.find(self.tmp / "deep" / "er")), ["P4_TEST_A"])
            self.assertEqual((os.environ["P4_TEST_A"], os.environ["P4_TEST_C"]), ("a", "kept"))
            self.assertNotIn("P4_TEST_B", os.environ)
        finally:
            for k in ("P4_TEST_A", "P4_TEST_C"):
                os.environ.pop(k, None)

    def test_notebook_cells_run(self):
        nb = mk.notebook()
        ns = {}
        cfg = (f"kg.configure(STORE_REPO={kg.STORE_REPO!r}, SKILL_DIRS={kg.SKILL_DIRS!r}, REFERENCE_ROOTS=[], "
               f"WORK={str(kg.WORK)!r})")
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                src = "".join(c["source"])
                exec(compile(src, "cell", "exec"), ns)
                if "kg.configure(FAKE=FAKE" in src:
                    exec(cfg, ns)
        self.assertIn("summary", ns)
        json.dumps(nb)


if __name__ == "__main__":
    unittest.main(verbosity=2)
