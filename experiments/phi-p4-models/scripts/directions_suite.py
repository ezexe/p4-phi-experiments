"""
The measured legs of the p4 models' directions (the directions notebook, notebooks/directions_colab.ipynb):

  D1  the JSON ranker: leave-one-pool-out r-precision against term overlap, scored against the store's labels and
      against each teacher's
  D2  pretraining: the single .pt ranker pretrained for each budget in --budgets, with early stopping on held-back
      dev pools (every DEV_EVERY-th pool, kept out of every score). Per budget: steps taken, the best dev step,
      held-out ranking after fine-tuning (scored against every label set), and whether its out-of-fold score raises
      the JSON ranker as its .pt feature
  D4  labels: every model fine-tuned on each label set in turn and scored against each, so a label set's
      learnability (trained and scored on itself) and its transfer (trained on one, scored on another) both show;
      and how far the label sets agree with each other

Label sets: the store's own (pick_model.py pairs: the operator's review and the Haiku children) and one per
--teacher NAME=DIR, a directory of scripts/teacher.py label output.

Resumable through --cache: every pretrained model and every out-of-fold score set is saved as it finishes.

    python scripts/directions_suite.py --p4 <store> --pairs pairs.jsonl --plugin <vlds> \
        --teacher fable=<dir> --teacher opus=<dir> --budgets 0.5 1 2 4 6 --seeds 0 1 --check-s 20 \
        --cache <dir> --out directions.json
"""

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phi_rank as pr  # noqa: E402
from nested_suite import CACHED, RAW, cached, load, load_pools, rp, say, state  # noqa: E402
from nested_vs_one import pairs_of, shards  # noqa: E402

DEV_EVERY = 6


def teacher_labels(pairs, tdir):
    """{pool id: set of picked texts} from a teacher.py label directory, the picks read against the pairs file."""
    rows = {}
    with open(pairs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.setdefault(r["pool_id"], []).append(r["text"])
    out = {}
    for name in os.listdir(tdir):
        if name.endswith(".json"):
            with open(os.path.join(tdir, name), encoding="utf-8") as f:
                g = json.load(f)
            if g.get("picks") is not None and g["pool"] in rows:
                out[g["pool"]] = {rows[g["pool"]][i] for i in g["picks"]}
    return out


def relabel(pool, texts):
    return {**pool, "rows": [{**r, "label": int(r["text"] in texts)} for r in pool["rows"]]}


def label_sets(pools, teachers):
    """{name: {pool id: pool with that set's labels}}, each set over the pools it labels with at least one pick."""
    sets = {"store": {p["id"]: p for p in pools}}
    for name, tl in teachers.items():
        sets[name] = {p["id"]: relabel(p, tl[p["id"]]) for p in pools
                      if p["id"] in tl and any(r["text"] in tl[p["id"]] for r in p["rows"])}
    return sets


def agreement(sets):
    """Mean F1 between every two label sets over the pools both label."""
    out = {}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            f1s = []
            for pid in set(sets[a]) & set(sets[b]):
                x = {r["text"] for r in sets[a][pid]["rows"] if r["label"]}
                y = {r["text"] for r in sets[b][pid]["rows"] if r["label"]}
                if x and y:
                    pr_, rc = len(x & y) / len(y), len(x & y) / len(x)
                    f1s.append(2 * pr_ * rc / (pr_ + rc) if pr_ + rc else 0.0)
            out[f"{a}~{b}"] = round(sum(f1s) / len(f1s), 4) if f1s else None
    return out


# ─── D1: the JSON ranker ───────────────────────────────────────────────────────────────────────────────

def json_ranker(plugin, store, teachers):
    sys.path.insert(0, os.path.join(plugin, "scripts"))
    import phi
    import pick_model as pm
    jpools, data = pm.build_pairs(store)
    feats = [pm.pool_examples(p, data, False) for p in jpools]
    out = {"pools": len(jpools)}
    sets = {"store": jpools}
    for name, tl in teachers.items():
        sets[name] = [{**p, "labels": [int(phi.distilled(data[0][i], 400) in tl.get(p["id"], ())) for i in p["rows"]]}
                      for p in jpools]
    for name, ps in sets.items():
        keep = [k for k, p in enumerate(ps) if sum(p["labels"])]
        sub, fs = [ps[k] for k in keep], [feats[k] for k in keep]
        if len(sub) < 3:
            continue
        model = pm.loo_rprecision(sub, fs, data)
        overlap = sum(pm.metrics(p["labels"], f[1], [])["r-precision"] for p, f in zip(sub, fs)) / len(sub)
        out[name] = {"json_ranker": round(model, 4), "term_overlap": round(overlap, 4), "pools": len(sub)}
        say(f"D1 JSON ranker against the {name} labels: {model:.4f} (term overlap {overlap:.4f}, {len(sub)} pools)")
    return out, (pm, jpools, feats, data)


def deployed_with(pmx, oof, pools):
    """The JSON ranker's leave-one-out r-precision with this out-of-fold .pt score as its feature (store labels)."""
    pm, jpools, feats, data = pmx
    texts_of = {p["id"]: [r["text"] for r in p["rows"]] for p in pools}
    as_oof = {pid: {pm.text_key(t): s for t, s in zip(texts_of[pid], sc)} for pid, sc in oof.items()}
    pts = [pm.pool_pt(p, data, as_oof) for p in jpools]
    with_pt = [(pm.add_pt([dict(x) for x in f[0]], sc), f[1], f[2]) if sc else f for f, sc in zip(feats, pts)]
    return round(pm.loo_rprecision(jpools, with_pt, data), 4)


# ─── D2 and D4: the .pt ranker by pretraining budget, by label set ─────────────────────────────────────

def out_of_fold(st, pools_by_id, ids, folds, epochs, seed):
    got = {}
    for f in range(folds):
        train = [pools_by_id[pid] for j, pid in enumerate(ids) if j % folds != f]
        m = load(st)
        torch.manual_seed(seed * 100 + f)
        pr.finetune(m, train, epochs=epochs, seed=seed * 100 + f)
        for j, pid in enumerate(ids):
            if j % folds == f:
                p = pools_by_id[pid]
                got[pid] = pr.score_pool(m, p["task"], [r["text"] for r in p["rows"]])
    return got


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--teacher", action="append", default=[], help="NAME=DIR, a teacher.py label directory")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1, 2, 4, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--bits", type=int, default=15)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--check-s", type=float, default=20)
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cache:
        os.makedirs(a.cache, exist_ok=True)
    store = next(os.path.join(a.p4, "sessions", d, "vlds") for d in sorted(os.listdir(os.path.join(a.p4, "sessions")))
                 if os.path.isdir(os.path.join(a.p4, "sessions", d, "vlds")))
    pools = load_pools(a.pairs)
    teachers = {}
    for t in a.teacher:
        name, d = t.split("=", 1)
        if os.path.isdir(d):
            teachers[name] = teacher_labels(a.pairs, d)
            say(f"teacher {name}: {len(teachers[name])} pools labelled")
    dev = [p for j, p in enumerate(pools) if j % DEV_EVERY == DEV_EVERY - 1]
    evals = [p for j, p in enumerate(pools) if j % DEV_EVERY != DEV_EVERY - 1]
    if not dev:
        a.check_s = 0
        say(f"only {len(pools)} pools: no dev set, so early stopping is off")
    sets = label_sets(evals, teachers)
    report = {"pools": len(pools), "dev_pools": len(dev), "eval_pools": len(evals), "budgets": a.budgets,
              "label_sets": {n: len(s) for n, s in sets.items()}, "device": str(pr.device()),
              "agreement_f1": agreement(sets)}
    say(f"label sets on the eval pools: {report['label_sets']}; agreement (F1): {report['agreement_f1']}")
    report["json_ranker"], pmx = json_ranker(a.plugin, store, teachers)
    shard_pairs = [pairs_of(a.p4, s) for s in shards(a.p4, 1)]
    all_pairs = shard_pairs[0][0]
    ids = sorted(set.intersection(*(set(s) for s in sets.values()))) if len(sets) > 1 else sorted(sets["store"])
    ids = [p["id"] for p in evals if p["id"] in ids]                       # date order, the folds' order
    say(f"D2/D4: {len(ids)} eval pools labelled by every set; {len(all_pairs):,} passage pairs to pretrain on")
    report["runs"] = []
    pr.featurize = CACHED

    def select(model):
        return sum(rp(p, pr.score_pool(model, p["task"], [r["text"] for r in p["rows"]])) for p in dev) / len(dev)
    for seed in a.seeds:
        for budget in a.budgets:
            tag = a.cache and os.path.join(a.cache, f"dir-b{budget:g}-s{seed}")

            def make():
                torch.manual_seed(seed)
                m = pr.Ranker(1 << a.bits, a.dim)
                info = {}
                say(f"D2 seed {seed}, budget {budget:g} min: pretraining on {pr.device()}")
                steps = pr.pretrain(m, all_pairs, budget, seed=seed, log=lambda *_: None,
                                    select=select if a.check_s > 0 else None, select_every=a.check_s, info=info)
                say(f"D2 seed {seed}, budget {budget:g} min: {steps:,} steps"
                    + (f", best dev {info['best_score']} at step {info['best_steps']:,}" if info else ""))
                return {"state": state(m), "info": info or {"steps": steps}}
            got = cached(tag and tag + "-model.pt", make, "pt")
            run = {"seed": seed, "budget_min": budget, "early_stopping": {k: v for k, v in got["info"].items()
                                                                          if k != "curve"},
                   "curve": got["info"].get("curve"), "quality": {}}
            for train_name, tset in sets.items():
                path = tag and f"{tag}-oof-{train_name}.json"
                oof = cached(path, lambda: out_of_fold(got["state"], tset, ids, a.folds, a.epochs, seed))
                for eval_name, eset in sets.items():
                    run["quality"][f"{train_name}->{eval_name}"] = round(
                        sum(rp(eset[pid], oof[pid]) for pid in ids) / len(ids), 4)
                if train_name == "store":
                    run["deployed"] = deployed_with(pmx, oof, [sets["store"][pid] for pid in ids])
                say(f"D2/D4 seed {seed}, budget {budget:g}: trained on {train_name}: "
                    + ", ".join(f"{k} {v}" for k, v in run["quality"].items() if k.startswith(train_name + "->")))
            report["runs"].append(run)
            if a.out:
                with open(a.out, "w") as f:
                    json.dump(report, f, indent=1)
    pr.featurize = RAW
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
