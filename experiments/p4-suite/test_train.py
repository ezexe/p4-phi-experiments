#!/usr/bin/env python3
"""Training on a real store: the AudioPlugin-p4 store's imported pools must reach train.FIRST_FLOOR held out, above
the BM25 baseline. It migrates the fixture that test_store.fixture finds under P4_FIXTURES into a temp directory and
takes about a minute and a half, so it runs by hand and never in the plugin's tests, with P4_SLOW=1:

    P4_SLOW=1 .venv/Scripts/python -X utf8 -m pytest experiments/p4-suite/test_train.py -q -s

It tests the scripts of the p4 checkout beside this repository, at `p4/lbiza/plugins/p4-phi/scripts`, or at the folder
P4_SCRIPTS names.
"""

import os
import sys
import tempfile
import unittest

HERE = os.environ.get("P4_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                                    "p4", "lbiza", "plugins", "p4-phi", "scripts")
HERE = os.path.normpath(HERE)
sys.path.insert(0, HERE)
import labels as lb  # noqa: E402
import store as st  # noqa: E402
import train  # noqa: E402
from test_store import NOW, fixture  # noqa: E402


class AudioPluginFloor(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("P4_SLOW") == "1" and fixture("AudioPlugin-p4"),
                         "slow: P4_SLOW=1 and the AudioPlugin-p4 fixture")
    def test_imported_pools_reproduce_the_floor(self):
        import migrate
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "p4")
            migrate.run(fixture("AudioPlugin-p4"), out=out, apply=True, now=NOW)
            st.build_index(out)
            ev = train.evaluate(train.build_sets(out, sources={"pool"}))
            print(f"\nAudioPlugin-p4 imported pools: {train.fmt(ev)}")
            self.assertEqual(ev["n"], 24)
            labs = [r for r in lb.read_tsv(lb.mpath(out, lb.RAW)) if r["source"] == "pool"]
            self.assertEqual(len(labs), 274, "274 of the 280 steering lines match an entry")
            self.assertEqual(sum(1 for r in labs if r["task"] == "pool:a3ac5de6"), 1, "1 of its 4 picks matches")
            self.assertGreater(ev["r-precision"], ev["baseline"], "above the BM25 baseline")
            self.assertGreaterEqual(ev["r-precision"], train.FIRST_FLOOR)


if __name__ == "__main__":
    unittest.main(verbosity=1)
