#!/usr/bin/env python3
"""The migration on real stores: the clones AudioPlugin-p4, ap-p4_mac, p4-p4 and phi-p4 under P4_FIXTURES, each
migrated dry into a temp directory with every check passing and no entry unplaced, its counts equal to those of
lbiza/plugins/p4-phi/scripts/store_lessons.py, and ap-p4_mac also applied with its archives byte-identical. A clone that is not on
this machine is skipped, and the clones are only read. They measure those stores, so they run by hand and never in
the plugin's tests:

    .venv/Scripts/python -X utf8 -m pytest experiments/p4-suite/test_migrate.py -q

It tests the scripts of the p4 checkout beside this repository, at `p4/lbiza/plugins/p4-phi/scripts`, or at the folder
P4_SCRIPTS names, and reads the store_lessons.py beside that checkout's test_migrate.py.
"""

import importlib.util
import os
import sys
import tempfile
import unittest

HERE = os.environ.get("P4_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                                    "p4", "lbiza", "plugins", "p4-phi", "scripts")
HERE = os.path.normpath(HERE)
sys.path.insert(0, HERE)
import migrate  # noqa: E402

_spec = importlib.util.spec_from_file_location("plugin_test_migrate", os.path.join(HERE, "test_migrate.py"))
_tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)
NOW, fixture, store_lessons = _tm.NOW, _tm.fixture, _tm.store_lessons


class RealRepos(unittest.TestCase):
    """The four clones, migrated dry into a temp dir; read only."""

    def check_repo(self, name):
        src = fixture(name)
        if not src:
            self.skipTest(f"the {name} clone is not on this machine")
        with tempfile.TemporaryDirectory() as d:
            r = migrate.run(src, os.path.join(d, "new"), apply=False, now=NOW)
            fails = [c for c in r["checks"] if not c[1]]
            self.assertEqual(fails, [])
            self.assertEqual(r["plan"].unplaced, [])
            sl = store_lessons()
            if sl:
                got = sl.lessons(src)["counts"]
                self.assertEqual(got["correction"], r["counts"]["correction"])
                self.assertEqual(got["retraction"], r["counts"]["retraction"])
                self.assertEqual(got["missed"], r["counts"]["missed"])
                self.assertEqual(got["repeat"], r["counts"]["repeat"])
                self.assertEqual(len(sl.live_rules(src)), len(r["live_new"]) + len(r["live_lost"]))
                lab = sl.labels(src)
                self.assertEqual(lab["rows"], r["labels"]["rows"] - sum(k - 1 for _w, k, kind in r["plan"].multi
                                                                         if kind == "row"))
                self.assertGreaterEqual(r["labels"]["completed"], lab["completed"])
                self.assertGreaterEqual(r["labels"]["judged"], lab["judged"])
            return r

    def test_audioplugin(self):
        r = self.check_repo("AudioPlugin-p4")
        self.assertEqual(r["counts"]["correction"], 86)
        self.assertEqual(r["pools"]["labelled"], 24)

    def test_ap_mac_finding17(self):
        r = self.check_repo("ap-p4_mac")
        self.assertEqual(len(r["finding17"]), 6)
        self.assertEqual([s for _i, _t, s in r["finding17"]], ["LIVE"] * 6)
        self.assertEqual(len(r["live_new"]), 9)

    def test_p4p4(self):
        self.check_repo("p4-p4")

    def test_phi(self):
        self.check_repo("phi-p4")

    def test_ap_mac_apply_byte_identical(self):
        src = fixture("ap-p4_mac")
        if not src:
            self.skipTest("the ap-p4_mac clone is not on this machine")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "new")
            r = migrate.run(src, out, apply=True, now=NOW)
            names = [n for n, _ok, _d in r["checks"]]
            self.assertIn("archives byte-identical", names)
            self.assertIn("memory/ tree identical", names)
            self.assertEqual([c for c in r["checks"] if not c[1]], [])
            self.assertFalse(os.path.exists(os.path.join(src, "store")), "the clone is never written")


if __name__ == "__main__":
    unittest.main(verbosity=1)
