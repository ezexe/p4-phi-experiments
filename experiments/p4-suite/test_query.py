#!/usr/bin/env python3
"""The retrieval query on real stores: a store migrated from the AudioPlugin-p4 fixture, and a copy of the store
P4_MAC_STORE names. Each case runs only when its store is on this machine, and measures that one store, so they run
by hand and never in the plugin's tests.

They test the scripts of the p4 checkout beside this repository, at `p4/lbiza/plugins/p4-phi/scripts`, or at the
folder P4_SCRIPTS names, and import that folder's test_query.py and test_store.py for their helpers:

    .venv/Scripts/python -X utf8 -m pytest experiments/p4-suite/test_query.py -q

On the AudioPlugin-p4 store: how long one `p4.py query` process takes, silence for ordinary messages, and the same
lines for the same message within their bounds. On the P4_MAC_STORE copy: the no-ranker road's regression cases.
"""

import json
import os
import sys
import tempfile
import time
import unittest

HERE = os.environ.get("P4_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                                    "p4", "lbiza", "plugins", "p4-phi", "scripts")
HERE = os.path.normpath(HERE)
sys.path.insert(0, HERE)
import p4 as cli  # noqa: E402
import store as st  # noqa: E402
import importlib.util  # noqa: E402

# the plugin's own test_query.py, under another name, since this file carries the same one
_spec = importlib.util.spec_from_file_location("plugin_test_query", os.path.join(HERE, "test_query.py"))
_tq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tq)
LINE_RE, run_cli = _tq.LINE_RE, _tq.run_cli
from test_store import NOW, fixture  # noqa: E402


@unittest.skipUnless(fixture("AudioPlugin-p4"), "the AudioPlugin-p4 clone is not on this machine")
class RealAudioPlugin(unittest.TestCase):
    """A store migrated from the AudioPlugin-p4 fixture under P4_FIXTURES, when it is on this machine."""

    @classmethod
    def setUpClass(cls):
        import migrate
        cls.tmp = tempfile.TemporaryDirectory()
        cls.p4 = os.path.join(cls.tmp.name, "p4")
        migrate.run(fixture("AudioPlugin-p4"), cls.p4, apply=True, now=NOW,
                    report_path=os.path.join(cls.tmp.name, "report.md"))
        st.build_index(cls.p4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_latency_under_300ms(self):
        """The fastest of five `p4.py query` processes on this store, each timed from its start to its exit: the
        interpreter's start, the imports, the index's load and the query, as the prompt hook's own process pays them.
        The prompt hook's budget is stated at `h_recall` in hook.py."""
        text = "the CEF build keeps failing on the proprietary codecs patch, how do I rebuild incrementally?"
        times = []
        for _ in range(5):            # the best of 5: another process's burst on a loaded machine does not fail it
            t = time.perf_counter()
            r = run_cli("query", "--p4", self.p4, "--text", text, "--now", NOW)
            times.append(time.perf_counter() - t)
            self.assertEqual(r.returncode, 0, r.stderr)
        self.assertLess(min(times), 0.300, f"query took {min(times) * 1000:.0f} ms")

    def test_hi_and_ordinary(self):
        self.assertEqual(run_cli("query", "--p4", self.p4, "--text", "hi").stdout, "")
        self.assertEqual(run_cli("query", "--p4", self.p4, "--text", "what is the weather like today").stdout, "")

    def test_deterministic_and_bounded(self):
        for text in ("the CEF build keeps failing on the proprietary codecs patch", "push it",
                     "move the ring writer comment lines and the playhead inject mapping into the plan"):
            a = cli.query(self.p4, text, None, NOW)
            b = cli.query(self.p4, text, None, NOW)
            self.assertEqual(a["lines"], b["lines"])
            self.assertLessEqual(len(a["lines"]), 5)
            for line in a["lines"]:
                self.assertNotRegex(line, st.JARGON)
                m = LINE_RE.match(line)
                if m:
                    self.assertLessEqual(len(m.group(3)), 160)
                else:
                    self.assertTrue(line.startswith(("- you have said", "- an earlier message like this one")), line)


# a real store in the flat layout: P4_MAC_STORE names it, and the default is a path on another machine
MAC_STORE = os.environ.get("P4_MAC_STORE", "/Users/ez/gritz/p4-repos/ap-p4_mac")


class Regressions(unittest.TestCase):
    """The no-ranker road on a copy of a real store, taken without its ranker: "which branch do mac commits go on"
    and "where does the p4 repo live" must each recall the ruling that answers it in the top 5, and ordinary
    messages and the suite's general prompts must print nothing. fallback_order's gates are held to these cases."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.p4 = None
        if os.path.isdir(os.path.join(MAC_STORE, "store")):
            import shutil
            cls.p4 = os.path.join(cls.tmp.name, "p4")
            # A flat store's folders of the index's sources, without its scratchpad, built index, ranker or served
            # folder. build_index reads those sources at their places in the layout, under phi/ and temportal/, and
            # nothing here moves the copy there.
            for sub in ("store", "memory", "src", "models"):
                if os.path.isdir(os.path.join(MAC_STORE, sub)):
                    shutil.copytree(os.path.join(MAC_STORE, sub), os.path.join(cls.p4, sub),
                                    ignore=shutil.ignore_patterns("index.json", "ranker.json", "served"))
            st.build_index(cls.p4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        if not self.p4:
            self.skipTest("ap-p4_mac store not present")

    def top5(self, text):
        return cli.query(self.p4, text, None, NOW, model={})["lines"][:5]

    def test_mac_branch(self):
        got = self.top5("which branch do mac commits go on")
        self.assertTrue(any("on this Mac the work is committed on _mac" in l or
                            "the Mac's branch is 0.0.1-apple&win&mac_mac" in l for l in got), got)

    def test_p4_repo(self):
        got = self.top5("where does the p4 repo live")
        self.assertTrue(any("the p4 repo is nested in the project" in l for l in got), got)

    def test_ordinary_messages_silent(self):
        for text in ("hi", "keep going", "thanks", "ok", "what is the weather like today"):
            self.assertEqual(self.top5(text), [], text)


    def test_suite_prompts_silent(self):
        """At least 95% of the suite's general prompts, those of every category but lessons, print nothing."""
        suite = os.path.join(HERE, "..", "..", "..", "..", "scripts", "session", "overrides_tasks.jsonl")
        if not os.path.exists(suite):
            self.skipTest("the suite's tasks are not beside this checkout")
        with open(suite, encoding="utf-8") as f:
            prompts = [json.loads(l) for l in f if l.strip()]
        prompts = [p for p in prompts if p.get("category") != "lessons"]
        loud = [p["id"] for p in prompts if self.top5(p["prompt"])]
        self.assertGreaterEqual(1 - len(loud) / len(prompts), 0.95, loud)


if __name__ == "__main__":
    unittest.main()
