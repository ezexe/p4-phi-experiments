#!/usr/bin/env python3
"""Offline tests of served_back.py and its notebook: a local two-branch store in the p4-recall layout, no project
repository, FAKE client, tiny training.

    python experiments/served_back/test_served_back.py
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import served_back as sb  # noqa: E402
import make_notebook as mk  # noqa: E402

TOPICS = ["ring buffer drain", "midi transport playhead", "toolbar latency readout", "crash dump symbols",
          "plan document decisions", "render thread priority", "cmake floor version", "branch per machine"]


def sh(cwd, *a):
    subprocess.run(a, cwd=cwd, check=True, capture_output=True)


def entry(pairs):
    return "- " + pairs[0][0] + ": " + pairs[0][1] + "\n" + "".join(f"  {k}: {v}\n" for k, v in pairs[1:]) + "\n"


def write_store(root, session, n_tasks, day):
    """A store with rules per topic, one session's message rows, labels that point rules at tasks, tasks.tsv."""
    store = root / "store"
    (store / "messages").mkdir(parents=True, exist_ok=True)
    rules = "The owner's rulings.\nWritten when the owner rules.\nFields: ruling, owner-words, status, time.\n\n"
    rule_ids = []
    for i, t in enumerate(TOPICS):
        time = f"2026-10-{20 + i:02d} 10:00"
        rules += entry([("ruling", f"the {t} rule: keep the {t} as agreed, {t} detail"), ("owner-words", f'"boss said {t}"'),
                        ("status", "LIVE"), ("time", time)])
        import hashlib
        rule_ids.append(f"rules.md:{time}:{hashlib.sha1(f'the {t} rule: keep the {t} as agreed, {t} detail'.encode()).hexdigest()[:8]}")
    (store / "rules.md").write_text(rules, encoding="utf-8")
    (store / "corrections.md").write_text(
        "Readings that were wrong.\nWritten when the owner corrects.\nFields: key, time, match, meant, delta.\n\n"
        + entry([("key", "cmake-floor"), ("time", f"2026-11-{day:02d} 12:00"),
                 ("match", "keep the cmake floor version backward compatible"),
                 ("meant", "the cmake floor version is the latest, never backward compatible"), ("delta", "fixed")])
        + entry([("asked", '"where is the ring buffer drain measured?"'), ("time", f"2026-11-{day:02d} 12:30"),
                 ("omitted", "the ring buffer drain readout in the toolbar")]), encoding="utf-8")
    (store / "retractions.md").write_text(
        "What the owner freed.\nWritten when the owner takes something back.\nFields: freed, time, cause, owner-words, lesson.\n\n"
        + entry([("freed", "the midi transport playhead follows the host clock"), ("time", f"2026-11-{day:02d} 13:00"),
                 ("cause", "superseded"), ("owner-words", '"the boss says the playhead is ours"'),
                 ("lesson", "the midi transport playhead is the plugin's own")]), encoding="utf-8")
    (store / "facts.md").write_text("Claims checked.\nWritten when verified.\nFields: claim, source, time.\n", encoding="utf-8")
    rows = "Every owner message of one session.\nOpened by the prompt hook.\nFields: fingerprint, time, arrival, state, addressed.\n\n"
    tasks = "id\tsession\ttime\ttext\tkind\n"
    labels = "task\tentry\tlabel\tsource\tweight\ttime\tteacher\tedited\tnote\n"
    import hashlib
    for i in range(n_tasks):
        t = TOPICS[i % len(TOPICS)]
        text = f"please check the {t} in the engine and tell me what the boss decided about the {t} last week"
        time = f"2026-11-{day:02d} {11 + i // 6:02d}:{(i * 7) % 60:02d}"
        rid = f"messages/{session}.md:{time}:{hashlib.sha1(f'\"{text}\"'.encode()).hexdigest()[:8]}"   # the row's id: its quoted fingerprint
        rows += entry([("fingerprint", f'"{text}"'), ("time", time), ("arrival", "turn"), ("state", "FRESH"),
                       ("addressed", f"Done: the {t} follows the rule; see the {t} detail.")])
        tasks += f"{rid}\t{session}\t{time}\t{text}\tnew\n"
        labels += f"{rid}\t{rule_ids[i % len(TOPICS)]}\t1\tusage\t0.3\t{time}\t\t\tused\n"
        labels += f"{rid}\t{rule_ids[(i + 1) % len(TOPICS)]}\t0\tusage\t0.3\t{time}\t\t\tunused\n"
        labels += f"{rid}\t{rule_ids[(i + 2) % len(TOPICS)]}\t0\trepo\t0.3\t{time}\t\t\tdropped\n"
    rows += entry([("fingerprint", '"<task-notification>done</task-notification>"'), ("time", f"2026-11-{day:02d} 15:00"),
                   ("arrival", "notification"), ("state", "FRESH"), ("addressed", "(a notification; complete on arrival)")])
    (store / "messages" / f"{session}.md").write_text(rows, encoding="utf-8")
    (root / "models").mkdir(exist_ok=True)
    (root / "models" / "tasks.tsv").write_text(tasks, encoding="utf-8")
    (root / "models" / "labels.tsv").write_text(labels, encoding="utf-8")
    (root / "models" / "eval.tsv").write_text("round\ttime\tr_precision\tprecision\tpromoted\twhy\n0\t2026-11-01 00:00\t0.1\t0.0\tkept old\tlow\n",
                                              encoding="utf-8")
    (root / "memory").mkdir(exist_ok=True)
    (root / "memory" / "MEMORY.md").write_text("# Memory index\n\n- [Branch](branch.md) — each machine on its own branch\n", encoding="utf-8")
    (root / "flights" / "f1").mkdir(parents=True, exist_ok=True)
    (root / "flights" / "f1" / "plan.md").write_text("flight: f1\nplan: X\n", encoding="utf-8")
    (root / "flights" / "f1" / "log").write_text("2026-11-01 10:00\tfiled\n2026-11-01 11:00\tlanded\n", encoding="utf-8")
    (root / ".gitignore").write_text("models/index.json\nmodels/repo.json\nmodels/*.tmp\nmodels/served/\n", encoding="utf-8")


def make_store_repo(root):
    src = root / "src-store"
    src.mkdir()
    sh(src, "git", "init", "-q", "-b", "main")
    sh(src, "git", "config", "user.email", "testy@example.org")
    sh(src, "git", "config", "user.name", "Testy McAuthor")
    (src / "README.md").write_text("empty main\n")
    sh(src, "git", "add", "-A")
    sh(src, "git", "commit", "-qm", "main")
    for b, session, n, day in (("mac", "aaaa1111", 20, 2), ("win", "bbbb2222", 28, 5)):
        sh(src, "git", "checkout", "-q", "--orphan", b)
        sh(src, "git", "rm", "-rfq", "--cached", ".")
        for p in list(src.iterdir()):
            if p.name != ".git":
                (p.unlink() if p.is_file() else __import__("shutil").rmtree(p))
        write_store(src, session, n, day)
        sh(src, "git", "add", "-A")
        sh(src, "git", "commit", "-qm", b)
    sh(src, "git", "checkout", "-q", "main")
    return src


class ServedBack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.src = make_store_repo(cls.tmp)
        sb.configure(FAKE=True, STORE_REPO=str(cls.src), BRANCHES=[], PROJECT_REPO="", WORK=cls.tmp / "work",
                     PRETRAIN_MINUTES=0.02, EPOCHS=3, TEACHER_TASKS=3, USAGE_SAMPLE=4, NEGATIVES_K=3, HOLDOUT_SHARE=0.3)

    def test_1_layout_reads_both_branches_and_phi_reads_the_new_layout(self):
        layout = sb.step_layout()
        self.assertEqual(set(layout), {"mac", "win"})
        x = layout["win"]
        self.assertEqual(x["store"].get("rules.md:LIVE"), len(TOPICS))
        self.assertEqual(x["store"].get("messages:LIVE"), 29)
        # phi_store reads store/ and memory/ since the new-layout reader (says_vs_does target phi-store-new-layout):
        # every owner message with a reply is a pair, and the readers' reach is no longer capped at memory/*.md
        self.assertEqual(x["phi_store"]["error"], "")
        self.assertEqual(x["phi_store"]["pairs"], 28)
        self.assertEqual(x["phi_model"], "accepts the store")
        self.assertGreater(x["phi_rank"]["text_files"], 5)
        self.assertGreater(x["coverage"]["phi_store"], 0.5)
        self.assertEqual(x["flights_landed"], 1)

    def test_2_served_back_counts_turns_and_cases(self):
        served = sb.step_served_back()
        x = served["win"]
        self.assertEqual(x["owner_turns"], 28)            # the notification row is not the owner's
        self.assertEqual(x["served_turns"], 28)
        self.assertEqual(x["served_ids"], 56)
        self.assertEqual(x["used_share"], 0.5)
        self.assertEqual(x["by_kind"]["rules"], {"1": 28, "0": 28})
        self.assertEqual((x["corrections"], x["retractions"]), (2, 1))
        case = next(c for c in x["cases"] if c["kind"] == "correction")
        self.assertTrue(case["trigger"])                  # a row in the 90 minutes before it
        self.assertTrue(case["candidates"])               # the cmake rule predates the correction
        self.assertTrue(case["known_lexical"])
        self.assertTrue(all(d["time"] < case["time"] for d in case["candidates"] if d["time"]))

    def test_3_bridge_trains_on_the_labels(self):
        bridge = sb.step_bridge()
        x = bridge["win"]
        self.assertEqual(x["pools"], 28)
        self.assertEqual(x["train"] + x["held"], 28)
        self.assertTrue(x["cutoff"].startswith("2026-11-05"))
        for k in ("bm25", "pt_fresh", "pt_pretrained"):
            self.assertIn("r-precision", x[k])
        self.assertTrue(Path(x["model"]).exists())
        self.assertGreater(x["pretrain"]["passage_pairs"], 0)

    def test_4_teacher_scores_every_road(self):
        bridge = json.loads((sb.paths()["results"] / "bridge.json").read_text())
        teacher = sb.step_teacher(bridge)
        x = teacher["win"]
        self.assertEqual(x["tasks"], 3)
        row = x["rows"][0]
        self.assertEqual(set(row["roads"]), {"hooks", "bm25", "bridge", "recall"})
        for m in sb.TEACHER_MODELS:
            self.assertEqual(len(row["teacher"][m]), 2)   # the fake picks C1 and C2
            self.assertIn("f1", x["summary"][m]["hooks"])
        self.assertIsNotNone(x["summary"]["teacher_agreement_jaccard"])
        sil = x["summary"]["silence"]                     # the plugin's silent prompts, read from the checkout
        self.assertEqual(sil["tasks"], sb.SILENT_TASKS)
        self.assertGreaterEqual(sil[sb.TEACHER_MODELS[0]]["picks"], 2)   # the fake picks 2 wherever a candidate exists
        self.assertIn("recall", sil["roads_silent"])

    def test_5_adherence_dedupes_cases_across_branches(self):
        served = json.loads((sb.paths()["results"] / "served_back.json").read_text())
        adherence = sb.step_adherence(served)
        self.assertEqual(adherence["cases_judged"], 6)    # 3 per branch, different days so different ids
        self.assertEqual(adherence["usage_checked"], 4)
        self.assertEqual(adherence["usage_agreement"], 1.0)   # the fake judge echoes the hook label
        self.assertIn("KNOWN_NOT_SERVED", adherence["classes"])

    def test_5b_epistemic_rederives_recommendations_and_drafts_facts(self):
        served = json.loads((sb.paths()["results"] / "served_back.json").read_text())
        self.assertEqual(served["win"]["recommendation_turns"], 0)      # "Done: the X follows the rule" recommends nothing
        self.assertEqual(served["win"]["facts_entries"], 0)
        # give one branch's replies a recommendation so the step has something to re-derive
        served["win"]["recommendations"] = [{"id": "messages/bbbb2222.md:2026-11-05 11:00:deadbeef", "session": "bbbb2222",
                                             "time": "2026-11-05 11:00", "source": "ask",
                                             "text": "which cmake floor version should the boss pin for the project?",
                                             "addressed": "You should keep the cmake floor version backward compatible with 3.x."}]
        served["mac"]["recommendations"] = [dict(served["win"]["recommendations"][0], time="2026-11-02 11:00")]
        epistemic = sb.step_epistemic(served)
        self.assertEqual(epistemic["sampled"], 1)                       # the same row id on both branches counts once
        row = epistemic["rows"][0]
        self.assertEqual(row["default"]["basis"], "GUESS")               # the fake default arm sees no URL
        self.assertEqual(row["grounded"]["basis"], "DOCUMENTATION")
        self.assertEqual(row["grounded"]["sources_verified"], ["https://docs.example.org/current"])
        self.assertEqual(epistemic["session_matches"], {"GROUNDED": 1})
        self.assertEqual(epistemic["facts_drafted"], 1)
        self.assertEqual(epistemic["by_source"], {"ask": 1})
        fact = row["fact"]                                 # the mac copy is seen first: its 20 turns, the 11:00 one not later
        self.assertEqual(row["branch"], "mac")
        self.assertEqual(fact["later_turns"], 19)
        self.assertGreaterEqual(fact["later_turns_hit"], 1)   # the cmake turns rank the drafted cmake claim
        self.assertIn("recirculates:", (sb.paths()["results"] / "facts-draft.md").read_text(encoding="utf-8"))
        self.assertTrue(sb.ASK_RE.search("should we pin cmake 4.4 or stay on 3.x?"))
        self.assertIsNone(sb.ASK_RE.search("run the tests again"))
        draft = (sb.paths()["results"] / "facts-draft.md").read_text(encoding="utf-8")
        self.assertIn("- claim: ", draft)
        self.assertIn("  source: https://docs.example.org/current", draft)
        self.assertTrue(sb.RECOMMEND_RE.search("I'd go with ninja instead of make here"))
        self.assertIsNone(sb.RECOMMEND_RE.search("Committed as abc123; nothing pushed."))

    def test_6_report_and_summary(self):
        p = sb.paths()["results"]
        load = lambda n: json.loads((p / n).read_text())  # noqa: E731
        s = sb.step_report(load("layout.json"), load("served_back.json"), load("bridge.json"), load("teacher.json"),
                           load("adherence.json"), load("epistemic.json"))
        self.assertTrue(s["fake"])
        self.assertEqual(s["spent_usd"], 0.0)
        self.assertEqual(s["epistemic"]["sampled"], 1)
        text = (p / "report.md").read_text(encoding="utf-8")
        self.assertIn("## E. Adherence", text)
        self.assertIn("## F. Epistemic", text)
        self.assertIn("FAKE run", text)

    def test_7_scrub_and_prf(self):
        self.assertEqual(sb.scrub("the boss said so"), "the owner said so")
        # the store's committers' names come from its git log, nothing typed in
        self.assertEqual(sb.scrub("Testy told McAuthor's team; testy@example.org"), "owner told owner's team; owner@example.org")
        self.assertNotIn("example", sb.owner_terms())   # the e-mail's domain is not an owner term
        self.assertIn("testy", sb.owner_terms())
        self.assertEqual(sb.prf(["a", "b"], ["b", "c"]), {"precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertEqual(sb.prf([], [])["f1"], 1.0)

    def test_8_budget_guard(self):
        sb.configure(FAKE=False, TOTAL_MAX_USD=0.0)
        try:
            with self.assertRaises(RuntimeError):
                sb.guard("x")
        finally:
            sb.configure(FAKE=True, TOTAL_MAX_USD=10.0)

    def test_9_notebook_cells_run(self):
        nb = mk.notebook()
        ns = {}
        cfg = (f"sb.configure(STORE_REPO={str(self.src)!r}, PROJECT_REPO='', WORK={str(self.tmp / 'work-nb')!r}, "
               "PRETRAIN_MINUTES=0.02, EPOCHS=2, TEACHER_TASKS=2, USAGE_SAMPLE=2, NEGATIVES_K=3, HOLDOUT_SHARE=0.3)")
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                src = "".join(c["source"])
                exec(compile(src, "cell", "exec"), ns)
                if "sb.configure(FAKE=FAKE" in src:
                    exec(cfg, ns)
        self.assertIn("summary", ns)
        json.dumps(nb)


if __name__ == "__main__":
    unittest.main(verbosity=1)
