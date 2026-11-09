#!/usr/bin/env python3
"""Tests of says_vs_does.py: the harness never hides a broken check, every registered claim runs against the real
checkouts (with and without a store), and the readings the checks make of the code are the ones the code supports.

    python experiments/says_vs_does/test_says_vs_does.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import says_vs_does as svd  # noqa: E402


class Harness(unittest.TestCase):
    def test_a_broken_check_is_reported_not_raised(self):
        saved = list(svd.CLAIMS)
        try:
            svd.CLAIMS[:] = []

            @svd.claim("boom", "here", "it never breaks")
            def _(ctx):
                raise ValueError("it broke")

            @svd.claim("needs-store", "here", "the data says so")
            def _(ctx):
                ctx.need_store()
                return svd.HOLDS, "ok"

            rows = svd.run_claims(svd.Ctx(None))
            self.assertEqual([r["verdict"] for r in rows], [svd.UNTESTABLE, svd.UNTESTABLE])
            self.assertIn("ValueError", rows[0]["does"])
            self.assertIn("--store", rows[1]["does"])
            text = svd.report(rows, svd.Ctx(None))
            self.assertIn("| UNTESTABLE | 2 |", text)
            self.assertIn("## Better than it says: 0 of 0", text)
        finally:
            svd.CLAIMS[:] = saved

    def test_every_claim_has_a_distinct_id_and_a_source(self):
        ids = [c.id for c in svd.CLAIMS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 40)
        for c in svd.CLAIMS:
            self.assertTrue(c.where and c.says.endswith("."), c.id)
            self.assertIn(c.bar, ("says", "better"))
            self.assertIn(c.fix, ("", "code", "doc"))
        self.assertGreaterEqual(sum(c.bar == "better" for c in svd.CLAIMS), 8)


class RealCode(unittest.TestCase):
    """The checks run on the checkouts beside this file; a store is optional, so the data claims may be UNTESTABLE."""

    @classmethod
    def setUpClass(cls):
        cls.no_store = svd.run_claims(svd.Ctx(None))
        cls.by_id = {r["id"]: r for r in cls.no_store}

    def test_code_claims_resolve_without_a_store(self):
        for id_ in ("hooks-registered", "recall-max-lines", "pinned-chars", "p4py-subcommands", "labels-columns",
                    "usage-weight", "train-if-owed", "promotion-floor", "ranker-json-shared", "teacher-models",
                    "env-overrides", "p4conf-copies", "inject-default", "phi-model-subcommands",
                    "phi-model-recall-config", "phi-model-retrain-notices", "phi-model-finds-plugin",
                    "phi-rank-cli", "phi-serve-endpoints", "phi-readme-links"):
            self.assertNotEqual(self.by_id[id_]["verdict"], svd.UNTESTABLE, (id_, self.by_id[id_]["does"]))

    def test_data_claims_say_why_they_wait(self):
        for id_ in ("store-fields", "entry-ids", "seeded-gitignore", "close-completes", "recall-latency", "recall-silence"):
            self.assertEqual(self.by_id[id_]["verdict"], svd.UNTESTABLE)
            self.assertIn("--store", self.by_id[id_]["does"])

    def test_known_readings(self):
        # these follow from the code as it stands; a change there should change them, and this test with it
        self.assertEqual(self.by_id["hooks-registered"]["verdict"], svd.HOLDS)
        self.assertEqual(self.by_id["recall-max-lines"]["verdict"], svd.HOLDS)
        self.assertEqual(self.by_id["p4conf-copies"]["verdict"], svd.HOLDS)
        self.assertEqual(self.by_id["phi-serve-endpoints"]["verdict"], svd.HOLDS)
        for id_ in ("usage-note-mode", "row-recommended", "phi-rank-sidecar-separate", "echo-judged", "first-kind",
                    "gitignore-reseed", "teacher-facts", "phi-model-plugin-recall"):
            self.assertIn(self.by_id[id_]["verdict"], (svd.MET, svd.NOT_YET), id_)   # a target is met or owed, never untestable
        self.assertEqual(self.by_id["usage-note-mode"]["verdict"], svd.MET)
        self.assertEqual(self.by_id["phi-rank-sidecar-separate"]["verdict"], svd.MET)

    def test_report_writes_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            svd.main(["--out", d, "--only", "recall-max-lines", "pinned-chars"])
            rows = json.loads((Path(d) / "claims.json").read_text(encoding="utf-8"))
            self.assertEqual({r["id"] for r in rows}, {"recall-max-lines", "pinned-chars"})
            self.assertIn("# Says vs does", (Path(d) / "claims.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
