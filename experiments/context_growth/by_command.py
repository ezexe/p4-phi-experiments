#!/usr/bin/env python3
"""
by_command.py: how much a session's context grows after each kind of action, and whether that predicts the next one.

context_growth.py found a turn's growth cannot be told from the size history. This looks at the action instead.
Each API call that asks for tools is followed by a call whose context is larger by what those tools returned; the
difference is the action's cost. An action's kind is its tool, and for a shell command the program it runs (its
first two words: `git status`, `python`, `cmake --build`). Names and numbers only.

The model is a table that learns as it goes: an action is predicted to cost the median of the earlier actions of
its kind, falling back to its tool's median, then to the median of everything.

    python experiments/context_growth/by_command.py run --sessions <folder of session records>

Measured on the newest fifth of the sessions, against the median of everything and the tool's median alone.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "p4" / "lbiza" / "plugins" / "p4-recall" / "scripts"))
sys.path.insert(0, str(HERE))
HOLDOUT = 0.2
WORD = re.compile(r"[A-Za-z_][\w.+-]*")


SUBCOMMANDS = ("git", "gh", "npm", "cargo", "docker", "claude", "pip", "cmake")
TAKES_VALUE = ("-C", "-c", "--git-dir", "--work-tree", "--repo", "-R")


def kind(name, inp):
    """A tool call's kind: the tool, and for a shell command the program its first step runs that is not `cd`,
    with its subcommand for the programs that have them (`git status`, `npm run`)."""
    if name not in ("Bash", "PowerShell"):
        return name
    for step in re.split(r"&&|;|\|", str((inp or {}).get("command", ""))):
        words = [w.strip("\"'") for w in step.split()]
        words = [w for w in words if "=" not in w or w.startswith("-")]          # VAR=value before the program
        if not words or words[0] in ("cd", "sudo", "time", "set", "export"):
            continue
        prog = words[0].replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")
        if not WORD.fullmatch(prog):
            continue
        if prog in SUBCOMMANDS:
            skip = False
            for w in words[1:]:
                if skip:
                    skip = False
                elif w in TAKES_VALUE:
                    skip = True
                elif not w.startswith("-") and WORD.fullmatch(w):
                    return f"{name}: {prog} {w.lower()}"
        return f"{name}: {prog}"
    return f"{name}: ?"


def actions_of(path):
    """[(kind, tool, tokens the context grew by after it)] of one session, in order."""
    import transcript as tr
    seen, calls = set(), []
    for d in tr.read_lines(path):
        m = d.get("message") or {}
        if d.get("type") != "assistant":
            continue
        mid = m.get("id") or d.get("uuid")
        u = m.get("usage") or {}
        size = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        tools = [(b.get("name", ""), b.get("input") or {}) for b in m.get("content") or []
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        if mid in seen:
            if calls:
                calls[-1][1].extend(tools)
            continue
        seen.add(mid)
        if size:
            calls.append([size, tools])
    out = []
    for (size, tools), (nxt, _t) in zip(calls, calls[1:]):
        if tools and nxt > size:
            out += [(kind(n, i), n, (nxt - size) // len(tools)) for n, i in tools]
    return out


def run(folder):
    from context_growth import median, score
    names = sorted((os.path.getmtime(os.path.join(folder, n)), n) for n in os.listdir(folder) if n.endswith(".jsonl"))
    sessions = []
    for _m, n in names:
        try:
            a = actions_of(os.path.join(folder, n))
        except (OSError, ValueError):
            continue
        if a:
            sessions.append(a)
    cut = int(len(sessions) * (1 - HOLDOUT))
    by_kind, by_tool, every = {}, {}, []
    for a in sessions[:cut]:
        for k, t, g in a:
            by_kind.setdefault(k, []).append(g)
            by_tool.setdefault(t, []).append(g)
            every.append(g)
    overall = median(every)
    pk, pt, pa = [], [], []
    for a in sessions[cut:]:
        for k, t, g in a:                       # predicted from what came before it, then learned from
            pa.append((overall, g))
            pt.append((median(by_tool[t]) if by_tool.get(t) else overall, g))
            pk.append((median(by_kind[k]) if len(by_kind.get(k, [])) >= 3 else median(by_tool[t]) if by_tool.get(t) else overall, g))
            by_kind.setdefault(k, []).append(g)
            by_tool.setdefault(t, []).append(g)
    total = sum(sum(v) for v in by_kind.values()) or 1
    top = sorted(by_kind.items(), key=lambda kv: -sum(kv[1]))[:12]
    return {"sessions": len(sessions), "actions": len(every) + len(pa), "held_out": len(pa), "kinds": len(by_kind),
            "by kind": score(pk), "by tool": score(pt), "median of everything": score(pa),
            "top": [[k, len(v), median(v), round(sum(v) / total, 3)] for k, v in top]}


def report(res, path):
    L = ["# by_command: what an action adds to the context", "",
         f"{res['actions']:,} actions of {res['sessions']} sessions in {res['kinds']:,} kinds; the newest sessions' "
         f"{res['held_out']:,} held out.", "",
         "| predicts an action's cost by | off by (median factor) | within a factor of two |", "|---|---|---|"]
    L += [f"| {k} | {res[k][0]} | {res[k][1]:.0%} |" for k in ("by kind", "by tool", "median of everything")]
    L += ["", "| kind | times | median tokens | share of all growth |", "|---|---|---|---|"]
    L += [f"| {k} | {n:,} | {m:,} | {s:.0%} |" for k, n, m, s in res["top"]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--work", default=str(HERE / "out"))
    a = ap.parse_args(argv)
    res = run(a.sessions)
    out = Path(a.work) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "by_command.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(report(res, out / "by_command.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
