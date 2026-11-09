"""Offline tests for phi_depth.py and phi_open.py on a small labelled store (phi's own test fixture), in seconds.

    .venv/Scripts/python -I experiments/phi_depth/test_phi_depth.py
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "phi" / "tests"))
import torch  # noqa: E402

import phi_depth as pd  # noqa: E402
import phi_open as po  # noqa: E402
import phi_prune.masks as masks_mod  # noqa: E402
import phi_rank as pr  # noqa: E402
import phi_rank.tree as tree  # noqa: E402
from phi_prune import delta, gaps  # noqa: E402
from test_p4rank_labels import labelled_store  # noqa: E402

quiet = lambda *a, **k: None  # noqa: E731


class PhiDepth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        labelled_store(cls.tmp)
        cls.p4 = str(cls.tmp)
        cls.pools = pr.read_label_pools(cls.p4, negatives=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_sweep_reports_every_address_and_valid_masks(self):
        r = pd.sweep_one(pr, tree, torch, self.p4, 3, 0.0, 1, self.pools, quiet)
        self.assertEqual(r["addresses"], ["1001", "101", "1"], "deepest (oldest shard) first")
        for a, b in r["bits"].items():
            self.assertEqual(b["adjacent_ones"], 0, f"{a}: phi_mask never keeps two adjacent rows")
            self.assertLessEqual(b["kept_share"], 0.5)
        self.assertIsNone(r["bits"]["1"]["jaccard_with_parent"], "the root has no parent")
        self.assertIsNotNone(r["bits"]["101"]["jaccard_with_parent"])
        h = r["heldout"]
        self.assertGreaterEqual(h["oracle"], max(h["medium"], r["best_solo"][1]) - 1e-9, "the oracle is the ceiling")
        self.assertAlmostEqual(sum(b["route_share"] for b in r["bits"].values()), 1.0, places=3)

    def test_open_masks_keep_half_and_phi_stays_non_adjacent(self):
        w = torch.randn(64, 8)
        for kind in po.MASKS:
            m = po.row_mask(torch, masks_mod, w, kind, 0)
            self.assertEqual(m.shape, (64,))
            if kind != "none":
                self.assertLessEqual(float(m.mean()), 0.5 + 1e-9, kind)
            if kind == "phi":
                self.assertEqual(int((m[1:] * m[:-1]).sum()), 0, kind)
        # the ordered mask is non-adjacent in its own order, not in the hashed one
        x = w - w.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        order = (x @ vh[0]).argsort()
        m = po.row_mask(torch, masks_mod, w, "phi_ordered", 0)[order]
        self.assertEqual(int((m[1:] * m[:-1]).sum()), 0)

    def test_open_evaluate_scores_every_mask(self):
        order, states, _ = po.states_for(pr, tree, torch, self.p4, 2, 0.0, 0)
        res = po.evaluate(pr, tree, torch, masks_mod, order, states, self.pools, 1, 0)
        self.assertEqual(set(res), set(po.MASKS))
        for kind, r in res.items():
            self.assertGreaterEqual(r["oracle"], r["medium"] - 1e-9, kind)

    def test_stream_round_trips_exactly(self):
        r = po.stream_test(pr, tree, torch, delta, gaps, self.p4, self.pools, 0.0, 1, 2, quiet)
        self.assertTrue(r["round_trip_exact"])
        self.assertTrue(r["decoded_deltas_equal"])
        self.assertLess(r["stream_framing_bits_per_round"], r["phid_header_bits_per_round"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
