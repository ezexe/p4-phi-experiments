#!/usr/bin/env python3
"""test_live_memory_limit.py: measures how much of MEMORY.md Claude Code loads, against p4-temportal's constants.

Run: P4_LIVE=1 python experiments/p4-suite/test_live_memory_limit.py
It spends the signed-in account's usage, one `claude -p` run on Haiku with one short answer, so without P4_LIVE=1 it
is skipped. It moved here from ezexe/p4's lbiza/plugins/p4-temportal/hooks, since a paid run leaves that repository.
P4_REPO names a checkout of ezexe/p4, by default the folder `p4` beside `experiments`, whose p4_memory.py holds the
constants it checks, and P4_CLAUDE names the CLI, by default the `claude` on PATH.

The docs at https://code.claude.com/docs/en/memory say "The first 200 lines of `MEMORY.md`, or the first 25KB,
whichever comes first", and not whether 25KB is 25,000 or 25,600 bytes. This test asks a real session. Its auto memory
directory is a temporary folder, set by `autoMemoryDirectory` through `--settings` as the same page documents, whose
MEMORY.md has 200 numbered lines of 150 bytes each. A 25,000-byte cut ends after line 166, a 25,600-byte cut after
line 170, and no byte cut at all after line 200. The session is asked for the highest line number it can see.
The session runs with `--bare`, so no installed plugin or hook writes into its folder beside the measurement, and with
CLAUDE_CODE_DISABLE_AUTO_MEMORY=0, which keeps auto memory on under `--bare`. ANTHROPIC_API_KEY is removed from its
environment, so the run is billed to the signed-in account and not to a key.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.abspath(os.environ.get("P4_REPO") or os.path.join(HERE, "..", "..", "p4"))
sys.path.insert(0, os.path.join(P4, "lbiza", "plugins", "p4-temportal", "hooks"))
import p4_memory  # noqa: E402

LINE_BYTES = 150


def claude_bin():
    found = os.environ.get("P4_CLAUDE") or shutil.which("claude")
    if found:
        return found
    apps = sorted(glob.glob(os.path.expanduser(
        "~/Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude")))
    return apps[-1] if apps else None


def memory_index():
    lines = []
    for i in range(1, 201):
        head = f"- line {i:03d} of the measurement index, marker M{i:03d} "
        lines.append(head + "x" * (LINE_BYTES - len(head) - 1) + "\n")
    assert all(len(line.encode()) == LINE_BYTES for line in lines)
    return "".join(lines)


def expected_last(byte_cap):
    return min(200, byte_cap // LINE_BYTES)


def main():
    if os.environ.get("P4_LIVE") != "1":
        print("test_live_memory_limit.py: skipped: set P4_LIVE=1 to run a live session")
        return 0
    binary = claude_bin()
    if not binary:
        print("test_live_memory_limit.py: skipped: no claude binary found")
        return 0
    base = tempfile.mkdtemp(prefix="p4-live-memory-")
    try:
        mem = os.path.join(base, "memory")
        proj = os.path.join(base, "proj")
        os.makedirs(mem)
        os.makedirs(proj)
        with open(os.path.join(mem, "MEMORY.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(memory_index())
        settings = os.path.join(base, "settings.json")
        with open(settings, "w", encoding="utf-8") as f:
            json.dump({"autoMemoryDirectory": mem, "autoMemoryEnabled": True}, f)
        prompt = ("Your auto memory index, MEMORY.md, was loaded at the start of this session; its lines carry markers "
                  "M001, M002 and so on. Without reading any file, reply with only the highest marker number you can "
                  "see in it, as digits.")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "0"
        r = subprocess.run([binary, "-p", prompt, "--bare", "--settings", settings, "--model", "haiku",
                            "--disallowedTools", "Read,Bash,Glob,Grep"],
                           cwd=proj, capture_output=True, text=True, timeout=300, env=env)
        got = re.findall(r"\d+", r.stdout)
        if r.returncode != 0 or not got:
            print("test_live_memory_limit.py: the session did not answer:", (r.stdout + r.stderr)[-400:])
            return 1
        seen = int(got[-1])
        readings = {25_000: expected_last(25_000), 25_600: expected_last(25_600), "no byte cap": 200}
        match = [k for k, v in readings.items() if v == seen]
        ours = expected_last(p4_memory.BUILTIN_BYTES)
        print(f"test_live_memory_limit.py: the session saw up to M{seen:03d}; readings {readings}; "
              f"p4_memory.BUILTIN_BYTES = {p4_memory.BUILTIN_BYTES} expects M{ours:03d}; matches {match or 'none'}")
        assert seen == ours, (f"p4_memory.BUILTIN_BYTES expects M{ours:03d}, Claude Code showed M{seen:03d}: set "
                              f"BUILTIN_BYTES and the copy in test_p4_memory.py to match")
        print("test_live_memory_limit.py: green")
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
