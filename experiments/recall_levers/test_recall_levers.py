"""Offline tests for recall_levers.py on p4-recall's own synthetic store (test_train.synthetic), in under a minute.

    .venv/Scripts/python -I experiments/recall_levers/test_recall_levers.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recall_levers as rl  # noqa: E402  (puts the plugin's scripts on sys.path)
from test_train import synthetic  # noqa: E402


class RecallLevers(unittest.TestCase):
    def test_reweight_rules(self):
        rows = [{"label": 1, "source": "repo", "weight": 0.7, "note": "cited"},
                {"label": 1, "source": "repo", "weight": 0.35, "note": "cited (its last commit)"},
                {"label": 0, "source": "repo", "weight": 0.3, "note": "dropped"},
                {"label": 1, "source": "usage", "weight": 0.3, "note": "used"}]
        got = lambda name: [(r["note"], r["weight"]) for r in rl.reweight(rows, rl.WEIGHTS[name])]  # noqa: E731
        self.assertEqual(got("no stated negatives"), [("cited", 0.7), ("cited (its last commit)", 0.35), ("used", 0.3)])
        self.assertNotIn("cited (its last commit)", [n for n, _ in got("no commit labels")])
        self.assertEqual({w for _, w in got("uniform 1.0")}, {1.0})
        self.assertAlmostEqual(dict(got("usage x3 (0.9)"))["used"], 0.9)
        self.assertEqual(rows[0]["weight"], 0.7, "the caller's rows are not changed")

    def test_strip_features_leaves_the_named_groups_out(self):
        orig, wrapped = rl.strip_features({"bm25", "file="})
        cands = [{"file": "rules.md", "kind": "rule", "state": "LIVE", "time": "", "toks": ["relay", "gauge"],
                  "head": ["relay"]}]
        full, _, _ = orig(["relay"], cands, {"relay": 1.0, "gauge": 1.0}, 2.0, "")
        cut, _, _ = wrapped(["relay"], cands, {"relay": 1.0, "gauge": 1.0}, 2.0, "")
        h = rl.ranker.h
        for k in ("bm25", "overlap", "file=rules.md"):
            self.assertIn(h(k), full[0])
            self.assertNotIn(h(k), cut[0])
        self.assertIn(h("kind=rule"), cut[0], "groups not named stay")

    def test_run_measures_every_lever(self):
        with tempfile.TemporaryDirectory() as tmp:
            p4 = synthetic(tmp)
            out = Path(tmp) / "levers.json"
            rep = rl.run(p4, str(out), draws=1)
            self.assertIn("base (as shipped)", rep["variants"])
            self.assertTrue(any(k.startswith("weights: ") for k in rep["variants"]))
            self.assertTrue(any(k.startswith("without ") for k in rep["variants"]))
            self.assertIn("curve", rep)
            base = rep["variants"]["base (as shipped)"]
            self.assertGreater(base["r-precision"], base["baseline"] - 0.2, "a learnable store ranks near or above BM25")
            self.assertEqual(json.loads(out.read_text())["store"], p4)


if __name__ == "__main__":
    unittest.main(verbosity=1)
