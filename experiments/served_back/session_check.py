#!/usr/bin/env python3
"""
session_check.py: one running session of a p4-recall store, checked from its rows: what the owner asked, what
recall served and what the reply used, what the close hook judged, what the reply recommended, what the owner had to
correct meanwhile, and what the store holds that bears on each question whether recall printed it or not.

Free and offline; rerun it as the session goes on.

    python experiments/served_back/session_check.py --store <project>/.claude/p4 --session <first 8 of a session id>
    python experiments/served_back/session_check.py --store <p4> --latest        # the newest session file
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import served_back as sb  # noqa: E402

TOP = 5
STAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--session", help="the session's 8-hex id")
    ap.add_argument("--latest", action="store_true")
    a = ap.parse_args(argv)
    r = sb.recall()
    p4 = str(Path(a.store))
    files = r.st.load_store(p4)
    if a.latest:
        newest = max(glob.glob(os.path.join(p4, "store", "messages", "*.md")), key=os.path.getmtime)
        a.session = os.path.basename(newest)[:-3]
    rel = f"{r.st.MESSAGES}/{a.session}.md"
    rows = files.get(rel) or []
    if not rows:
        print(f"no rows for session {a.session} in {rel}")
        return 1
    owner = [e for e in rows if r.st.owner_message(e)]
    tasks = r.lb.read_tasks(p4)
    labels = [x for x in r.lb.read_tsv(r.lb.mpath(p4, r.lb.TRAIN)) if x.get("task", "").startswith((rel, f"turn:{a.session}"))]
    idx, _ = r.st.ensure_index(p4)
    docs = [d for d in sb.docs_of(idx) if d.get("kind") not in ("file", "commit")]
    first, last = owner[0].time, owner[-1].time
    print(f"session {a.session}: {len(rows)} rows, {len(owner)} the owner's, {first} to {last}")
    states = Counter((e.fields.get("state", "").split(" ") or [""])[0] or "(open)" for e in owner)
    print(f"  states: {dict(states)}; kinds in tasks.tsv: {dict(Counter(t['kind'] for t in tasks.values() if t['session'] == a.session))}")
    by_src = Counter(x["source"] for x in labels)
    usage = [x for x in labels if x["source"] == "usage"]
    print(f"  labels: {dict(by_src)}; served lines: {len(usage)}, used {sum(x['label'] == '1' for x in usage)}, "
          f"modes {dict(Counter(x['note'].split('; ')[-1] for x in usage)) or '-'}")
    live = glob.glob(os.path.join(p4, "models", "served", f"turn-{a.session}*.json"))
    live = [l for l in live if not l.endswith(".tmp")]
    if live:
        s = json.load(open(live[0], encoding="utf-8"))
        print(f"  open turn: {s.get('lines')} lines computed, {s.get('shown')} shown, shadow={s.get('shadow')}")
    recs = [(e.time, r.st.collapse(e.fields.get("recommended", ""))) for e in owner if e.fields.get("recommended", "").strip()]
    print(f"  recommendations handed the owner: {len(recs)}")
    for t, s in recs:
        print(f"    {t}: {s[:200]}")
    cases = [c for c in sb.correction_cases(r.st, files) if first <= c["time"] <= (last or "9999")]
    print(f"  corrections and retractions dated inside the session: {len(cases)}")
    for c in cases:
        print(f"    {c['time']} {c['kind']}: {c['head'][:140]}")
    print("  per owner message: what recall prints now, and the store's nearest entries whether printed or not")
    for e in owner:
        text = r.st.unquote(e.fields.get("fingerprint", e.value))
        q = r.cli.query(p4, text, a.session, now=e.time, idx=idx)
        # the hook queries before it opens the row, so the row's own line, a repeat line pointing at it, and an entry
        # the same turn wrote afterwards are artefacts of checking later, never something the session saw
        pairs = [(ln, i) for ln, i in zip(q["lines"], q["ids"]) if i != e.id and not (i == "repeat" and (e.time or "@") in ln)
                 and not ((m := STAMP.search(i)) and e.time and m.group(0) >= e.time)]
        q["lines"] = [ln for ln, _i in pairs]
        pool = sb.before(docs, e.time or "9999", exclude={e.id})
        near = [(s, d) for s, d in sb.bm25_rank(r.rk, text, pool)[:TOP] if s > 0 and not d["id"].startswith(rel)]
        print(f"  - {e.time} [{(e.fields.get('state', '') or '(open)').split(' ')[0]}] {text[:110]}")
        print(f"      recall road {q['road'] or 'silent'}: {len(q['lines'])} line(s)" + (f"; jargon dropped {q['jargon_dropped']}" if q["jargon_dropped"] else ""))
        for ln in q["lines"]:
            print(f"        {ln[:170]}")
        for s, d in near:
            print(f"      near {s:5.1f} [{d.get('kind')} {d.get('time') or ''}] {sb.doc_text(d)[:150]}")
        if e.fields.get("addressed", "").strip():
            print(f"      reply: {r.st.collapse(e.fields['addressed'])[:170]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
