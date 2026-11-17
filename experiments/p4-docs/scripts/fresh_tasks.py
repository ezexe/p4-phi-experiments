#!/usr/bin/env python3
"""fresh_tasks.py: the task sets of the fresh validation that the suite runs beside its own (fresh-marketplace.md
8.3 and 8.4), in overrides_suite.py's task format.

  short   8.3, short turns held out: from each migrated store's messages/, the newest sessions' short rows (at most
          5 words, the owner's own, completed, with a reading in `addressed:`), about 150 across the stores. Each task
          is the message, with the two rows before it in its session as the conversation it arrived in (the owner's
          words, and what was done about them), and a judge rubric that passes the reply only if it reads the message
          as `addressed:` says it was read. The rows are held out of what recall sees (recall_contexts.py queries a
          copy of the store without the held sessions and without any entry dated from the held cut on).
  newer   8.4, newer than the model: fresh/newer_tasks.jsonl, checked against fresh/newer_facts.md (one fact a task,
          each with since:, checked: and prior:).

  python3 scripts/session/fresh_tasks.py short --out <short.jsonl> --work <dir> [--stores NAME=PATH,...] [--n 150]
  python3 scripts/session/fresh_tasks.py newer --out <newer.jsonl>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recall_contexts as rc  # noqa: E402

N_SHORT = 150
PRIOR_ROWS = 2                # the turns before the message shown as its conversation (8.3)
SESSION_ROWS = 60             # the earlier rows of its session recall may see (the repeat and history lines)
ASK = ("\n\n(One short paragraph: what you take this message to mean, and what you will do now. You have no tools "
       "here.)")
BAD = "I'm not sure what you mean by that. Could you tell me what you would like me to do?"


def rows_of(p4):
    """{s8: [(time, fingerprint, addressed, eligible)]} of the owner's rows, in file order."""
    import store as st
    out = {}
    for f, ents in st.load_store(p4).items():
        if not f.startswith(st.MESSAGES + "/"):
            continue
        s8 = f[len(st.MESSAGES) + 1:-3]
        for e in ents:
            fp = st.unquote(e.fields.get("fingerprint", e.value))
            if not st.owner_message(e, fp):
                continue
            addressed = st.collapse(e.fields.get("addressed", ""))
            done = st.completed(e) and addressed and not addressed.startswith("(")
            tok = st.norm_token(fp)
            out.setdefault(s8, []).append((e.time or "", st.collapse(fp), addressed if done else "",
                                           bool(done and tok and len(tok.split()) <= st.SHORT_WORDS), e.line))
    return out


def short_tasks(stores, work, n_total=N_SHORT, now=None):
    """The 8.3 task set: (tasks, {store: {held sessions, cut, rows}})."""
    now = now or time.strftime("%Y-%m-%d %H:%M")
    prepared = {n: rc.prepare(n, p, Path(work) / "stores", now) for n, p in stores.items()}
    per = {n: rows_of(p) for n, p in prepared.items()}
    avail = {n: sum(r[3] for rows in s.values() for r in rows) for n, s in per.items()}
    total = sum(avail.values()) or 1
    quota = {n: min(avail[n], max(10, round(n_total * avail[n] / total))) for n in per}
    tasks, info = [], {}
    for name, sessions in per.items():
        order = sorted(sessions, key=lambda s: max((r[0] for r in sessions[s]), default=""), reverse=True)
        held, picked = [], []
        for s8 in order:
            if len(picked) >= quota[name]:
                break
            held.append(s8)
            picked += [(s8, k) for k, r in enumerate(sessions[s8]) if r[3]]
        picked = sorted(picked, key=lambda x: sessions[x[0]][x[1]][0], reverse=True)[:quota[name]]
        if not picked:
            continue
        cut = min(sessions[s][k][0] for s, k in picked if sessions[s][k][0])
        info[name] = {"held_sessions": held, "cut": cut, "rows": len(picked), "available": avail[name]}
        for s8, k in sorted(picked):
            t, fp, addressed, _ok, line = sessions[s8][k]
            before = [r for r in sessions[s8][:k] if r[2]]
            history = []
            for bt, bfp, badd, _b, _l in before[-PRIOR_ROWS:]:
                history += [{"role": "user", "content": bfp}, {"role": "assistant", "content": badd}]
            tasks.append({
                "id": f"short-{name}-{s8}-{line}", "category": "short", "split": "test", "from_store": name,
                "session": s8, "time": t, "held_cut": cut, "held_sessions": held, "message": fp,
                "session_rows": [[bt, bfp, badd] for bt, bfp, badd, _b, _l in before[-SESSION_ROWS:]],
                "history": history, "prompt": fp + ASK,
                "grader": {"type": "judge", "rubric": (
                    "In the real session this short message from the owner was read as: \"" + addressed[:400]
                    + "\". Passes if the answer takes the message the same way: the same act or intent (the wording "
                    "may differ, and doing it or saying it will do it both count). Fails if it reads the message as "
                    "something else, or only asks what it means when the earlier turns settle it.")},
                # the key is derived from the rubric's own reading, so a real judge rightly fails it as a copy (two
                # judge checks on 2026-11-01 did); key: derived keeps it out of judge validation, its bad answer in
                "reference": ("You mean this, and I'll act on it now: " + addressed[:400].rstrip(" .;,") + "."),
                "key": "derived", "bad": [BAD]})
    return tasks, info


def newer_tasks():
    """fresh/newer_tasks.jsonl, each task's fact found in fresh/newer_facts.md with since:, checked: and prior:."""
    tasks = [json.loads(l) for l in open(HERE / "fresh" / "newer_tasks.jsonl", encoding="utf-8") if l.strip()]
    facts = {t: (s, dict(p)) for s, t, p in rc.newer_facts()}
    for t in tasks:
        s, f = facts[t["fact"]]
        assert s == t["from_store"], t["id"]
        for k in ("since", "checked", "prior", "source"):
            assert f.get(k), (t["id"], k)
    return tasks


def write(path, tasks):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("which", choices=("short", "newer"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--stores", default="")
    ap.add_argument("--n", type=int, default=N_SHORT)
    ap.add_argument("--now", default="")
    a = ap.parse_args(argv)
    if a.which == "short":
        if not a.work:
            ap.error("short needs --work")
        tasks, info = short_tasks(rc.parse_stores(a.stores), a.work, a.n, a.now or None)
        for n, i in info.items():
            rc.say(f"{n}: {i['rows']} short rows held out of {i['available']}, from {len(i['held_sessions'])} "
                   f"sessions, cut {i['cut']}")
    else:
        tasks = newer_tasks()
    write(a.out, tasks)
    rc.say(f"{a.which}: {len(tasks)} tasks -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
