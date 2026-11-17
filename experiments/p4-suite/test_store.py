#!/usr/bin/env python3
"""The store on a real one: the ap-p4_mac store, migrated into a temp directory, keeps the six rulings a retraction
with the same owner-words would free LIVE. It runs only when test_store.fixture finds the clone under P4_FIXTURES,
and measures that one store, so it runs by hand and never in the plugin's tests:

    .venv/Scripts/python -X utf8 -m pytest experiments/p4-suite/test_store.py -q

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
_ts = _plugin("test_store")
NOW, fixture = _ts.NOW, _ts.fixture


@unittest.skipUnless(fixture("ap-p4_mac"), "the ap-p4_mac clone is not on this machine")
class RealFinding17(unittest.TestCase):
    def test_six_rulings_live_after_migration(self):
        import migrate
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "p4")
            migrate.run(fixture("ap-p4_mac"), out, apply=True, now=NOW, report_path=os.path.join(d, "r.md"))
            files = st.load_store(out)
            ms = st.masks(files["retractions.md"])
            masked = [e for e in files["rules.md"]
                      if st.status_of("rules.md", e.value, e.fields, ms, rule="old")[0] == "FREED"]
            self.assertEqual(len(masked), 6)
            self.assertEqual([e.state for e in masked], ["LIVE"] * 6)


if __name__ == "__main__":
    unittest.main()
