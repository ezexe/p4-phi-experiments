#!/usr/bin/env python3
"""test_overrides_suite.py: the overrides suite, offline, so nothing here calls the API.

  1. every grader passes its task's reference and fails each bad answer, the check of the instrument that comes
     before any pathway is measured
  2. the units are found from the plugins' own hooks.json, seed files left out, and with skills each SKILL.md
  3. the pathways are the ones the suite's docstring names; an edits directory adds ~unit, a context directory =name
  4. a --fake run answers every pathway on every task, writes the JSON and resumes from its cache; a context that
     carries FAKE-FIX:<category> measurably helps, and a task's earlier turns reach the request
  5. the jargon scan catches the contracts' vocabulary and leaves ordinary words alone
  6. a context directory's per-task file replaces the pathway's own context for that task alone

The tasks are the p4 checkout's scripts/session/overrides_tasks.jsonl, and the units come from a small marketplace
this file writes, so no git history is read. Run from the root of ezexe/p4-phi, with the p4 checkout at p4/:

    .venv/Scripts/python -I experiments/p4-suite/session/test_overrides_suite.py
"""

import atexit
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P4 = HERE.parents[2] / "p4"
sys.path.insert(0, str(HERE))
import overrides_suite as os_  # noqa: E402

TASKS = P4 / "scripts" / "session" / "overrides_tasks.jsonl"
if not TASKS.exists():
    raise SystemExit(f"test_overrides_suite.py: no tasks at {TASKS}; the p4 checkout belongs at {P4}")

GATE = "# gate\nSay what was not done.\n"
MEMORY = "# memory\nKeep the earlier turns in mind.\n"
CONTRACTS = ["alpha/gate", "beta/memory"]
_MARKET = []


def market():
    """A marketplace of two plugins. alpha's SessionStart hook cats hooks/gate.md, and its Stop hook cats a file that
    is no contract, since only SessionStart counts. beta's SessionStart hook runs hooks/start.sh, which names
    hooks/memory.md and the seed file hooks/memory-seed.md, and beta ships the skill tidy."""
    if _MARKET:
        return _MARKET[0]
    d = Path(tempfile.mkdtemp(prefix="p4-suite-market-"))
    atexit.register(shutil.rmtree, d, ignore_errors=True)

    def put(rel, text):
        f = d / "lbiza" / "plugins" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")

    def hook(command):
        return {"hooks": [{"type": "command", "command": command}]}

    put("alpha/hooks/hooks.json", json.dumps({"hooks": {
        "SessionStart": [hook('cat "${CLAUDE_PLUGIN_ROOT}/hooks/gate.md"')],
        "Stop": [hook('cat "${CLAUDE_PLUGIN_ROOT}/hooks/stop.md"')]}}))
    put("alpha/hooks/gate.md", GATE)
    put("alpha/hooks/stop.md", "# stop\nNo contract.\n")
    put("beta/hooks/hooks.json", json.dumps({"hooks": {
        "SessionStart": [hook('sh "${CLAUDE_PLUGIN_ROOT}/hooks/start.sh"')]}}))
    put("beta/hooks/start.sh", 'cat "$ROOT/hooks/memory.md" "$ROOT/hooks/memory-seed.md"\n')
    put("beta/hooks/memory.md", MEMORY)
    put("beta/hooks/memory-seed.md", "# memory seed\n")
    put("beta/skills/tidy/SKILL.md", "---\nname: tidy\n---\nTidy the folder.\n")
    _MARKET.append(d)
    return d


def tasks():
    return [json.loads(l) for l in open(TASKS, encoding="utf-8") if l.strip()]


def test_graders():
    assert os_.check_graders(tasks()) == []
    ts = tasks()
    assert len({t["id"] for t in ts}) == len(ts)
    assert {t["category"] for t in ts} <= set(os_.GENERAL)


def test_units():
    us = os_.units(market())
    assert list(us) == CONTRACTS, list(us)
    assert all(not u["path"].endswith("-seed.md") for u in us.values())
    assert us["alpha/gate"]["text"] == GATE and us["beta/memory"]["plugin"] == "beta"
    assert list(os_.units(market(), skills=True)) == CONTRACTS + ["skill:beta/tidy"]


def test_pathways():
    us = os_.units(market())
    p = os_.pathways(us, ["builtin", "all", "loo", "solo", "pad"])
    assert set(p) == {"builtin", "all", "pad"} | {f"-{k}" for k in CONTRACTS} | {f"+{k}" for k in CONTRACTS}
    assert p["builtin"] == "" and len(p["pad"]) == len(p["all"])
    assert p["all"] == GATE + "\n\n" + MEMORY and p["-alpha/gate"] == MEMORY and p["+beta/memory"] == MEMORY
    with tempfile.TemporaryDirectory() as d:
        os_.edit_file(d, "beta/memory").write_text("# memory, shorter\n", encoding="utf-8")
        p = os_.pathways(us, ["all"], d)
        assert "# memory, shorter" in p["~beta/memory"] and MEMORY not in p["~beta/memory"]


def test_fake_run():
    with tempfile.TemporaryDirectory() as d:
        Path(f"{d}/ctx").mkdir()
        Path(f"{d}/ctx/fix.md").write_text("FAKE-FIX:honesty FAKE-FIX:judgment FAKE-FIX:memory", encoding="utf-8")
        args = ["--p4", str(P4), "--tasks", str(TASKS), "--model", "claude-haiku-5-5", "--fake",
                "--context-dir", f"{d}/ctx", "--cache", f"{d}/cache", "--out", f"{d}/out.json"]
        assert os_.main(args) == 0
        out = json.loads(Path(f"{d}/out.json").read_text())
        n = len(tasks())
        assert all(p["answered"] == n for p in out["pathways"].values())
        assert out["pathways"]["=fix"]["accuracy"] == 1.0 > out["pathways"]["builtin"]["accuracy"]
        assert out["pathways"]["=fix"]["tasks"]["m1-retraction"]["passed"]
        assert not out["pathways"]["all"]["tasks"]["m1-retraction"]["passed"]
        r = subprocess.run([sys.executable, str(HERE / "overrides_suite.py"), *args], capture_output=True, text=True)
        assert r.returncode == 0 and "0 answers to ask" in r.stdout, r.stdout
    req = os_.request("claude-haiku-5-5", "CTX", "now", "high",
                      [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"}])
    first, last = req["messages"][0]["content"], req["messages"][-1]["content"]
    assert first[0]["text"].endswith("CTX") and first[1]["text"] == "first" and last[-1]["text"] == "now"


def test_jargon():
    assert os_.JARGON.search("I registered the frag and wrote the tombstone.")
    assert os_.JARGON.search("the operator's derivation says FRESH")
    assert not os_.JARGON.search("Here is a fragment of the file; the pour-over coffee is ready.")


def test_context_dir_per_task():
    """--context-dir: `<pathway>.<task id>.md` is that pathway's context for that task alone; a task without one gets
    `<pathway>.md`, or nothing, and nothing is the built-in's request, answered from its cache."""
    with tempfile.TemporaryDirectory() as d:
        ctx = Path(d) / "ctx"
        ctx.mkdir()
        (ctx / "recall.h1-downloads.md").write_text("- FAKE-FIX:honesty\n", encoding="utf-8")
        (ctx / "recall.j3-force-push.md").write_text("- FAKE-FIX:judgment\n", encoding="utf-8")
        (ctx / "plain.md").write_text("FAKE-FIX:memory", encoding="utf-8")
        (ctx / "plain.m2-repeat.md").write_text("nothing here", encoding="utf-8")
        (ctx / "v1.2.md").write_text("a dotted name that is no task's", encoding="utf-8")
        base, per = os_.context_dir(ctx, {t["id"] for t in tasks()})
        assert set(base) == {"=recall", "=plain", "=v1.2"} and base["=recall"] == ""
        assert set(per["=recall"]) == {"h1-downloads", "j3-force-push"} and set(per["=plain"]) == {"m2-repeat"}
        args = ["--p4", str(P4), "--tasks", str(TASKS), "--model", "claude-haiku-5-5", "--fake", "--pathways",
                "builtin", "--context-dir", str(ctx), "--cache", f"{d}/cache", "--out", f"{d}/out.json"]
        assert os_.main(args) == 0
        out = json.loads(Path(f"{d}/out.json").read_text())
        b, r, pl = (out["pathways"][n]["tasks"] for n in ("builtin", "=recall", "=plain"))
        assert r["h1-downloads"]["passed"] and r["j3-force-push"]["passed"]
        assert not b["h1-downloads"]["passed"] and not b["j3-force-push"]["passed"]
        assert [k for k in r if r[k]["passed"] != b[k]["passed"]] == ["h1-downloads", "j3-force-push"]
        assert pl["m1-retraction"]["passed"] and not pl["m2-repeat"]["passed"], "the per-task file replaces the base"
        assert out["pathways"]["=recall"]["per_task_contexts"] == 2
        assert out["pathways"]["=recall"]["silent_tasks"] == len(tasks()) - 2
        # the =recall tasks without a file are the built-in's requests: 2 new answers, the rest from the cache;
        # =plain and =v1.2 differ from the built-in on every task
        n = len(tasks())
        assert len(list(Path(f"{d}/cache").glob("*.json"))) == n + 2 + n + n


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "ok")
    shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
    print("test_overrides_suite.py: all green")
