#!/usr/bin/env python3
"""Whether the lines p4-phi's `agent` hook adds to an Agent call's prompt reach the subagent, in a real session.

Run: P4_LIVE=1 python experiments/p4-suite/test_live_subagent.py
It spends the signed-in account's usage, one or two `claude -p` runs on Haiku, so without P4_LIVE=1 it is skipped.
P4_REPO names a checkout of ezexe/p4, by default the folder `p4` beside `experiments`, and P4_CLAUDE names the CLI,
by default the `claude` on PATH.
ANTHROPIC_API_KEY is removed from the CLI's environment, since the CLI is signed in to a claude.ai account and a key
there breaks that, and `--setting-sources project,local` keeps ezexe's own settings, hooks and installed plugins out,
so only p4-phi, loaded with `--plugin-dir`, runs, on a fixture git project in a temp folder.
P4_PHI=0 keeps p4-phi's session hooks out of the run, so only retrieval adds lines.

The run asks the main agent to launch exactly one subagent on a task the fixture's corpus has lines for, worded so
the ranker serves them: a longer tail on the task's words left it with no line at all.
The evidence read afterwards:
- the hook fired: a served record `agent-*.json` in the store's `phi/models/served/`, or an `agent:` row in
  `phi/models/labels.tsv` once the SubagentStop hook has turned the record into rows;
- the subagent saw the lines: its first user message, in its transcript under
  `~/.claude/projects/<project>/<session>/subagents/`, holds "Notes from this repo's memory:".
P4_LIVE_ROAD=subagentstart runs the SubagentStart road instead: a copy of the plugin whose hooks.json runs
`recall-hook.sh subagent-start` on SubagentStart in place of the PreToolUse `agent` entry, with the same evidence,
the notes then arriving as additional context.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.abspath(os.environ.get("P4_REPO") or os.path.join(HERE, "..", "..", "p4"))
ROOT = os.path.join(P4, "lbiza", "plugins", "p4-phi")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import p4conf  # noqa: E402
from test_hooks import make_project  # noqa: E402

CLI = os.environ.get("P4_CLAUDE") or shutil.which("claude") or "claude"
NOTES = "Notes from this repo's memory:"
TASK = "check whether the quokka harness runs the zebrafish plotter golden images on every push. Use no tools."
PROMPT = (f"Launch exactly one subagent with the Agent tool, subagent_type general-purpose. Its prompt must be exactly: "
          f"\"{TASK}\" Then reply with its answer in one line. Do not use any other tool.")
SUBAGENT_START = {"type": "command", "command": "sh \"${CLAUDE_PLUGIN_ROOT}/hooks/recall-hook.sh\" subagent-start",
                  "timeout": 10}


def plugin_for(road, tmp):
    """The plugin folder the run loads: the checkout's own, or for the SubagentStart road a copy whose hooks.json
    swaps the PreToolUse `agent` entry for a SubagentStart one and keeps every other entry."""
    if road == "pretooluse":
        return ROOT
    dst = os.path.join(tmp, "plugin")
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("__pycache__", ".python"))
    path = os.path.join(dst, "hooks", "hooks.json")
    with open(path, encoding="utf-8") as f:
        h = json.load(f)
    h["hooks"]["PreToolUse"] = [g for g in h["hooks"]["PreToolUse"]
                                if not any(x["command"].endswith(" agent") for x in g["hooks"])]
    h["hooks"]["SubagentStart"] = [{"matcher": ".*", "hooks": [dict(SUBAGENT_START)]}]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(h, f, indent=2)
    return dst


def evidence_paths(proj):
    """Where the hook leaves its evidence in the fixture's store: the served records, and the labels file."""
    store = os.path.join(proj, ".claude", "p4")
    return (os.path.join(p4conf.sub(store, "models", "served"), "agent-*.json"),
            p4conf.sub(store, "models", "labels.tsv"))


def first_user_text(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            if d.get("type") != "user":
                continue
            c = (d.get("message") or {}).get("content")
            if isinstance(c, list):
                c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            if isinstance(c, str) and c.strip():
                return c
    return ""


def whole_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def run(road):
    tmp = tempfile.mkdtemp(prefix="p4-live-")
    proj = make_project(tmp)
    plugin = plugin_for(road, tmp)
    sid = str(uuid.uuid4())
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "CLAUDE_PROJECT_DIR",
                                                            "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    env["P4_PHI"] = "0"
    cmd = [CLI, "-p", PROMPT, "--plugin-dir", plugin, "--model", "haiku", "--setting-sources", "project,local",
           "--session-id", sid, "--output-format", "json"]
    r = subprocess.run(cmd, cwd=proj, env=env, capture_output=True, text=True, timeout=600)
    served_glob, labels_path = evidence_paths(proj)
    served = glob.glob(served_glob)
    try:
        with open(labels_path, encoding="utf-8") as f:
            agent_labels = [l.rstrip("\n") for l in f if l.startswith("agent:")]
    except OSError:
        agent_labels = []
    subs = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}/subagents/*.jsonl"))
    mains = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))
    ev = {"road": road, "returncode": r.returncode, "session": sid, "project": proj,
          "stdout": r.stdout[-1500:], "stderr": r.stderr[-1500:], "served_files": served,
          "agent_labels": agent_labels,
          "subagent_transcripts": subs, "main_transcripts": mains}
    if subs:
        first = first_user_text(subs[0])
        ev["subagent_first_user_message"] = first[:3000]
        ev["notes_in_first_message"] = NOTES in first
        ev["notes_anywhere_in_subagent_transcript"] = NOTES in whole_text(subs[0])
    if mains:
        ev["notes_in_main_transcript"] = NOTES in whole_text(mains[0])
    return ev


@unittest.skipUnless(os.environ.get("P4_LIVE") == "1", "live: set P4_LIVE=1, which spends the account's usage")
class LiveSubagent(unittest.TestCase):
    def test_road(self):
        road = os.environ.get("P4_LIVE_ROAD", "pretooluse")
        ev = run(road)
        out = os.environ.get("P4_LIVE_OUT")
        if out:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=1)
        print(json.dumps({k: v for k, v in ev.items() if k not in ("stdout", "stderr")}, indent=1)[:4000])
        self.assertEqual(ev["returncode"], 0, ev["stderr"])
        self.assertTrue(ev["subagent_transcripts"], "no subagent was launched")
        self.assertTrue(ev["served_files"] or ev["agent_labels"], "the hook served no lines")
        if road == "pretooluse":
            self.assertTrue(ev.get("notes_in_first_message"), "updatedInput's prompt did not reach the subagent")
        else:
            self.assertTrue(ev.get("notes_anywhere_in_subagent_transcript"),
                            "SubagentStart's additionalContext is not in the subagent's transcript")


if __name__ == "__main__":
    unittest.main(verbosity=2)
