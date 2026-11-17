#!/usr/bin/env python3
"""The recall hook on a real store: on a store migrated from the AudioPlugin-p4 fixture, the hook run as Claude Code
runs it answers within the budget h_recall in hook.py states, and the index's cache follows index.json. It runs only
when test_store.fixture finds the fixture under P4_FIXTURES, and measures that one store, so it runs by hand and never
in the plugin's tests:

    .venv/Scripts/python -X utf8 -m pytest experiments/p4-suite/test_hooks.py -q

It tests the scripts of the p4 checkout beside this repository, at `p4/lbiza/plugins/p4-phi/scripts`, or at the folder
P4_SCRIPTS names.
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


def _plugin(name):
    """The plugin's own test module `name`, loaded under another name, since this file carries the same one."""
    spec = importlib.util.spec_from_file_location("plugin_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

import store as st  # noqa: E402
_th = _plugin("test_hooks")
git, run_hook, SESSION = _th.git, _th.run_hook, _th.SESSION


@unittest.skipUnless(__import__("test_store").fixture("AudioPlugin-p4"), "the AudioPlugin-p4 clone is not here")
class RecallLatency(unittest.TestCase):
    """The `recall` hook, run as Claude Code runs it by `sh hooks/recall-hook.sh` in a fresh interpreter, answers
    within the budget h_recall in hook.py states on the AudioPlugin-p4 fixture, about 5,000 documents, when the index
    is fresh, as it is after the Stop hook's rebuild. The best of 5 runs is asserted, so another process's burst does
    not fail it."""

    @classmethod
    def setUpClass(cls):
        import migrate
        from test_store import NOW, fixture
        cls.tmp = tempfile.TemporaryDirectory()
        cls.proj = os.path.join(cls.tmp.name, "proj")
        os.makedirs(cls.proj)
        git(cls.proj, "init", "-q")
        cls.p4 = os.path.join(cls.proj, ".claude", "p4")
        migrate.run(fixture("AudioPlugin-p4"), cls.p4, apply=True, now=NOW,
                    report_path=os.path.join(cls.tmp.name, "report.md"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_warm_recall_within_300ms(self):
        import time
        text = "the CEF build keeps failing on the proprietary codecs patch, how do I rebuild incrementally?"
        times = []
        for _ in range(5):
            st.build_index(self.p4)               # what the Stop hook leaves: a fresh index and its cache
            self.assertTrue(os.path.exists(st.cache_path(self.p4)))
            t = time.perf_counter()
            r = run_hook("recall", {"hook_event_name": "UserPromptSubmit", "session_id": SESSION, "prompt": text,
                                    "cwd": self.proj}, self.proj)
            times.append(time.perf_counter() - t)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("systemMessage", r.stdout)
        self.assertLess(min(times), 0.300, f"recall hook took {[round(x, 3) for x in times]} s")

    def test_cache_follows_the_index(self):
        """The cache is keyed on index.json: an index rebuilt with other documents is never read from a stale cache."""
        idx = st.build_index(self.p4)
        n = idx["n"]
        got = st.load_index(self.p4)
        self.assertEqual(len(got["docs"]), n)
        self.assertEqual(got["docs"][n - 1]["id"], idx["docs"][n - 1]["id"])
        with open(st.cache_path(self.p4), "r+b") as f:     # a corrupt cache falls back to the JSON
            f.write(b"\0\0\0\0")
        self.assertEqual(len(st.load_index(self.p4)["docs"]), n)
        os.utime(st.index_path(self.p4), ns=(1, 1))        # a cache keyed on another index.json is not used
        self.assertEqual(len(st.load_index(self.p4)["docs"]), n)


if __name__ == "__main__":
    unittest.main()
