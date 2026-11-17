#!/usr/bin/env python3
"""Labels from real transcripts: the AudioPlugin repository's transcripts, read into a copy of the ap-p4_mac store, give
repo labels edited and cited and a re-exploration measure. It needs three paths, the store MAC_STORE names, which
P4_MAC_STORE sets when it is given, the repository AP_PROJECT and its transcripts under AP_SLUG, and is skipped where
one is missing. It measures that one store, so it runs by hand and never in the plugin's tests:

    .venv/Scripts/python -X utf8 -m pytest experiments/p4-suite/test_labels.py -q

It tests the scripts of the p4 checkout beside this repository, at `p4/lbiza/plugins/p4-phi/scripts`, or at the folder
P4_SCRIPTS names, and writes only into a temporary copy of the store.
"""

import importlib.util
import os
import shutil
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

import labels as lb  # noqa: E402
import repo  # noqa: E402
import store as st  # noqa: E402


MAC_STORE = os.environ.get("P4_MAC_STORE", "/Users/ez/gritz/p4-repos/ap-p4_mac")
AP_PROJECT = "/Users/ez/code/AudioPlugin"
AP_SLUG = os.path.expanduser("~/.claude/projects/-Users-ez-code-AudioPlugin")


def copy_store(src, dst):
    for sub in ("store", "memory", "src", "models"):
        if os.path.isdir(os.path.join(src, sub)):
            shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub),
                            ignore=shutil.ignore_patterns("index.json", "repo.json", "served"))
    return dst


class RealTranscripts(unittest.TestCase):
    def test_repo_labels_from_the_audioplugin_transcripts(self):
        if not (os.path.isdir(os.path.join(MAC_STORE, "store")) and os.path.isdir(AP_SLUG)
                and repo.is_repo(AP_PROJECT)):
            self.skipTest("the ap-p4_mac store, the AudioPlugin repo or its transcripts are not present")
        with tempfile.TemporaryDirectory() as t:
            p4 = copy_store(MAC_STORE, os.path.join(t, "p4"))
            repo.build(p4, AP_PROJECT)
            st.build_index(p4)
            s = lb.from_transcripts(p4, AP_PROJECT, [AP_SLUG])
            self.assertGreater(s["transcripts"], 0)
            self.assertGreater(s["labels"].get("edited", 0), 0, s)
            self.assertGreater(s["labels"].get("cited", 0), 0, s)
            r = lb.reexploration(p4, AP_PROJECT, [AP_SLUG])
            self.assertGreater(r["turns"], 0)
            self.assertLessEqual(r["cf_calls"], r["calls"])


if __name__ == "__main__":
    unittest.main()
