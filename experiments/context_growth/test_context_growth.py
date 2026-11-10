#!/usr/bin/env python3
"""Offline test of context_growth.py's fit, prediction and scoring on made-up rows.

    python experiments/context_growth/test_context_growth.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_growth as cg  # noqa: E402


class ContextGrowth(unittest.TestCase):
    def test_fit_learns_a_growth_that_follows_its_inputs(self):
        rows = [["s", {"start": 100000 + 1000 * k, "baseline": 80000, "turn": k, "prev": 5000 * (1 + k % 3), "own": 5000},
                 5000 * (1 + k % 3)] for k in range(60)]                # each turn grows as the one before it did
        w = cg.fit(rows)
        got = cg.predict(w, rows[7][1])
        self.assertLess(max(got, rows[7][2]) / min(got, rows[7][2]), 1.2)
        self.assertEqual(cg.score([(100, 100), (100, 250), (100, 40)]), (2.46, 0.333))
        self.assertEqual(cg.score([]), (0, 0.0))
        self.assertEqual(cg.median([3, 1, 2]), 2)
        self.assertEqual(len(cg.vec(rows[0][1])), len(w))


if __name__ == "__main__":
    unittest.main()
