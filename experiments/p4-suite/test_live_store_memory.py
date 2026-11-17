#!/usr/bin/env python3
"""test_live_store_memory.py: whether a live session loads the store's MEMORY.md natively with p4-temportal running.

Run: P4_LIVE=1 python experiments/p4-suite/test_live_store_memory.py
It spends the signed-in account's usage, one `claude -p` run on Haiku with one short answer, so without P4_LIVE=1 it
is skipped. It decides ezexe/p4#92: whether `autoMemoryDirectory` naming the store's memory folder can replace the
pointer MEMORY.md, the mirror and the sweep. P4_REPO names a checkout of ezexe/p4, by default the folder `p4` beside
`experiments`, whose p4-temportal is loaded with `--plugin-dir`, and P4_CLAUDE names the CLI, by default the `claude`
on PATH. P4_LIVE_LOG names a file that keeps the run's whole stream-json output.

The fixture is a git repository with a store whose `.claude/p4/temportal/memory/MEMORY.md` holds one marker line
found nowhere else. `autoMemoryDirectory` names that folder through `--settings` only, so no settings file changes.
A hook cannot read `--settings`, so P4_MEMORY_APPDATA tells p4-temportal's hooks the same folder, the variable
p4_memory.appdata_memory reads first. `--setting-sources project,local` keeps the user's own plugins and hooks out of
the run, and the fixture has neither file. The session is asked for the marker with Read, Bash, Glob and Grep
forbidden, so its answer can only come from what it loaded. ANTHROPIC_API_KEY is removed from its environment, so
the run is billed to the signed-in account and not to a key.

It passes when:
1. the answer names the marker, so Claude Code loaded the store's index;
2. the store's MEMORY.md is unchanged, and no MEMORY.md was written in the profile's own memory folder for the
   fixture, so no pointer was written anywhere;
3. the hook events show p4-temportal's SessionStart hooks ran, and none of their output holds the marker, so the
   plugin printed no copy of the index.
The profile folder Claude Code makes for the fixture's transcript is removed afterwards.
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
PLUGIN = os.path.join(P4, "lbiza", "plugins", "p4-temportal")
MARKER = "ZQX-4417"
INDEX = ("- [Build notes](build.md) — how the fixture builds\n"
         f"- [Store marker](marker.md) — the store marker is {MARKER}\n"
         "- [Style](style.md) — one sentence per line\n")


def claude_bin():
    found = os.environ.get("P4_CLAUDE") or shutil.which("claude")
    if found:
        return found
    apps = sorted(glob.glob(os.path.expanduser(
        "~/Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude")))
    return apps[-1] if apps else None


def profile_projects():
    home = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
    return os.path.join(home, "projects")


def hook_events(lines):
    """The stream-json events --include-hook-events adds: every event whose type or subtype names a hook."""
    out = []
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = f"{ev.get('type', '')} {ev.get('subtype', '')}".lower()
        if "hook" in kind:
            out.append(ev)
    return out


def judge_hooks(hooks):
    """-> (whether p4-temportal's SessionStart hooks all ran and exited 0, the hook responses holding the marker). A
    hook event names only its event, as `SessionStart:startup`, never the plugin or the command. The run leaves user
    settings out with --setting-sources, so every SessionStart hook in it is one of the plugin's: they ran when as
    many SessionStart responses came back as the plugin's hooks.json lists, each with exit code 0."""
    with open(os.path.join(PLUGIN, "hooks", "hooks.json"), encoding="utf-8") as f:
        listed = sum(len(g["hooks"]) for g in json.load(f)["hooks"].get("SessionStart", []))
    starts = [ev for ev in hooks if ev.get("subtype") == "hook_response" and ev.get("hook_event") == "SessionStart"]
    ran = len(starts) == listed and all(ev.get("exit_code") == 0 for ev in starts)
    printed = [ev for ev in hooks if ev.get("subtype") == "hook_response" and MARKER in json.dumps(ev)]
    return ran, printed


def replay(log):
    """Judges the hook check of a run kept in P4_LIVE_LOG, with no new run: the paid run happens once."""
    text = open(log, encoding="utf-8").read()
    out = text.split("--- stdout\n", 1)[1].split("\n--- stderr\n", 1)[0]
    lines = out.splitlines()
    ran, printed = judge_hooks(hook_events(lines))
    answer = ""
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            answer = str(ev.get("result") or "")
    print("test_live_store_memory.py: replay of", log, json.dumps(
        {"answer": answer.strip()[:200], "p4-temportal SessionStart hooks ran": ran,
         "hook responses holding the marker": len(printed)}))
    return 0 if ran and not printed and MARKER in answer else 1


def main():
    if os.environ.get("P4_LIVE_REPLAY"):
        return replay(os.environ["P4_LIVE_REPLAY"])
    if os.environ.get("P4_LIVE") != "1":
        print("test_live_store_memory.py: skipped: set P4_LIVE=1 to run a live session")
        return 0
    binary = claude_bin()
    if not binary or not os.path.isdir(PLUGIN):
        print(f"test_live_store_memory.py: skipped: no claude binary or no plugin at {PLUGIN}")
        return 0
    base = tempfile.mkdtemp(prefix="p4-live-store-")
    projects = profile_projects()
    before = set(os.listdir(projects)) if os.path.isdir(projects) else set()
    try:
        proj = os.path.join(base, "proj")
        mem = os.path.join(proj, ".claude", "p4", "temportal", "memory")
        os.makedirs(mem)
        subprocess.run(["git", "init", "-q", proj], check=True)
        with open(os.path.join(mem, "MEMORY.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(INDEX)
        with open(os.path.join(mem, "marker.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"The store marker is {MARKER}.\n")
        settings = os.path.join(base, "settings.json")
        with open(settings, "w", encoding="utf-8") as f:
            json.dump({"autoMemoryDirectory": mem, "autoMemoryEnabled": True}, f)
        prompt = ("Your auto memory index, MEMORY.md, was loaded at the start of this session. One of its lines names "
                  "a store marker of the form ZQX- followed by digits. Without reading any file, reply with only that "
                  "marker, or NONE if you cannot see it.")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY="0", P4_MEMORY_APPDATA=mem)
        r = subprocess.run([binary, "-p", prompt, "--plugin-dir", PLUGIN, "--settings", settings,
                            "--setting-sources", "project,local", "--model", "haiku",
                            "--disallowedTools", "Read,Bash,Glob,Grep",
                            "--output-format", "stream-json", "--verbose", "--include-hook-events"],
                           cwd=proj, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=300, env=env)
        log = os.environ.get("P4_LIVE_LOG")
        if log:
            with open(log, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"exit {r.returncode}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}\n")
        lines = r.stdout.splitlines()
        answer = ""
        for line in lines:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "result":
                answer = str(ev.get("result") or "")
        hooks = hook_events(lines)
        ran, printed = judge_hooks(hooks)
        with open(os.path.join(mem, "MEMORY.md"), encoding="utf-8") as f:
            index_now = f.read()
        own = os.path.join(projects, re.sub(r"[^A-Za-z0-9]", "-", os.path.normpath(proj)), "memory")
        pointer = os.path.isfile(os.path.join(own, "MEMORY.md"))
        report = {"exit": r.returncode, "answer": answer.strip()[:200], "hook events": len(hooks),
                  "p4-temportal SessionStart hooks ran": ran, "hook responses holding the marker": len(printed),
                  "store index unchanged": index_now == INDEX, "pointer in the profile folder": pointer}
        print("test_live_store_memory.py:", json.dumps(report))
        assert r.returncode == 0, f"the session failed: {(r.stdout + r.stderr)[-600:]}"
        assert MARKER in answer, "the session did not name the marker, so it did not load the store's index"
        assert index_now == INDEX, "the store's MEMORY.md changed"
        assert not pointer, f"a MEMORY.md was written in {own}"
        assert ran, "no hook event shows p4-temportal's hooks ran, so the print check is not known"
        assert not printed, "a p4-temportal hook printed a copy of the index"
        print("test_live_store_memory.py: green")
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)
        if os.path.isdir(projects):
            for name in set(os.listdir(projects)) - before:
                if "p4-live-store-" in name:
                    shutil.rmtree(os.path.join(projects, name), ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
