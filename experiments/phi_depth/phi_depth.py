"""phi_depth: grow phi_rank.tree's Zeckendorf node tree deeper on a p4 store and measure what each level of recursion buys.

phi_rank.tree puts K small rankers at the first K store addresses (p4 = 1, .claude = 0; start and end with 1, no 11):
1, 101, 1001, 10001, 10101, 100001, ... A node's parent is its address with the last run of 0s and its final 1 dropped,
so the tree expands recursively, and every node's feature rows carry a phi_prune mask: one bit per row, 1 kept, 0 pruned,
no two adjacent 1s. The skill ships K = 4. This sweeps K over the Fibonacci counts 1, 2, 3, 5, 8, 13 at a fixed total
pretraining budget and reports, per K:

- held-out r-precision over 4 interleaved folds (the folds tree.out_of_fold uses) at medium (the routed node), at high
  (every node, z-scored and averaged), for each node alone, and for an oracle that picks the best node per pool (the
  ceiling a perfect route would reach), beside BM25 on the same rows
- the end nodes' bits, from a tree fine-tuned on every pool: each mask's kept share and adjacent-1 count, how much of a
  child's 1/0 pattern its parent keeps (Jaccard of the kept rows, against 1/3 for two unrelated density-0.5 masks), how
  many kept rows no task in the store ever lights, and which addresses the medium route sends the pools to

Free, CPU, offline. Nothing is written into the store: every tree lands in out/<name>/k<K>/.

    python -I experiments/phi_depth/phi_depth.py --store <a .claude/p4 store> [--store <another>]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FOLDS = 4
RANDOM_JACCARD = 1 / 3          # two independent density-0.5 masks: p^2 / (2p - p^2)


def bm25(task, texts, words, k1=1.2, b=0.75):
    toks = [[w.lower() for w in words(t)] for t in texts]
    n = len(toks)
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    avgdl = sum(len(t) for t in toks) / max(1, n)
    terms = list(dict.fromkeys(w.lower() for w in words(task)))
    out = []
    for t in toks:
        tf = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in terms:
            f = tf.get(w)
            if f:
                s += math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / (avgdl or 1)))
        out.append(s)
    return out


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def grow(pr, tree, torch, p4, k, minutes, log):
    """The pretrained node states, deepest (oldest shard) first, each node starting from its children's mean as build does."""
    files, spans = tree.shard_files(p4, k)
    nb = 1 << (pr.NB.bit_length() - 1 - int(round(math.log2(k))))
    order = sorted(tree.addresses(k), key=lambda a: (-len(a), a))
    trained, nodes = {}, []
    for i, a in enumerate(order):
        m = pr.Ranker(nb, pr.DIM)
        kids = [trained[c] for c in trained if tree.parent(c) == a]
        if kids:
            m.load_state_dict({key: sum(s[key] for s in kids) / len(kids) for key in kids[0]})
        torch.manual_seed(i)
        pairs = tree.shard_pairs(p4, files[i])
        steps = pr.pretrain(m, pairs, minutes / k, seed=i, log=lambda *_: None)
        trained[a] = {key: v.clone() for key, v in m.state_dict().items()}
        nodes.append({"address": a, "parent": tree.parent(a), "children": [c for c in trained if tree.parent(c) == a],
                      "files": len(files[i]), "pairs": len(pairs), "steps": steps, "span": list(spans[i])})
        log(f"  node {a:>8} ({i + 1}/{k}): {len(files[i])} files, {len(pairs)} pairs, {steps} steps")
    return order, trained, nodes, nb


def fit(pr, tree, torch, states, order, pools, epochs, seed):
    models = []
    for i, a in enumerate(order):
        m = pr.ranker_like(states[a])
        m.load_state_dict(states[a])
        torch.manual_seed(seed + i)
        pr.finetune(m, pools, epochs=epochs, seed=seed + i)
        tree.mask_rows(m)
        models.append(m)
    return models


def rprec(pr, labels, scores):
    return pr.pool_metrics(labels, scores, [])["r-precision"]


def sweep_one(pr, tree, torch, p4, k, minutes, epochs, pools, log):
    t0 = time.time()
    order, states, nodes, nb = grow(pr, tree, torch, p4, k, minutes, log)
    per = {"medium": [], "high": [], "oracle": [], "bm25": []}
    solo = {a: [] for a in order}
    for f in range(FOLDS):
        held = pools[f::FOLDS]
        train = [p for j, p in enumerate(pools) if j % FOLDS != f]
        models = fit(pr, tree, torch, states, order, train, epochs, f * 100)
        rows = tree.node_rows(models)
        for p in held:
            texts = [r["text"] for r in p["rows"]]
            labels = [r["label"] for r in p["rows"]]
            for eff in ("medium", "high"):
                sc, _ = tree.answer(models, rows, p["task"], texts, eff)
                per[eff].append(rprec(pr, labels, sc))
            node_r = [rprec(pr, labels, pr.score_pool(m, p["task"], texts)) for m in models]
            for a, r in zip(order, node_r):
                solo[a].append(r)
            per["oracle"].append(max(node_r))
            per["bm25"].append(rprec(pr, labels, bm25(p["task"], texts, pr.words)))
        log(f"  fold {f + 1}/{FOLDS}: {len(train)} train / {len(held)} held")

    # the end nodes' bits, from a tree fine-tuned on every pool
    models = fit(pr, tree, torch, states, order, pools, epochs, 9000)
    masks = {a: (m.bag.weight.detach().abs().sum(dim=1) != 0).to(torch.int8) for a, m in zip(order, models)}
    lit = torch.zeros(nb, dtype=torch.bool)
    for p in pools:
        lit[torch.tensor(pr.featurize(p["task"], nb))] = True
        for r in p["rows"]:
            lit[torch.tensor(pr.featurize(r["text"], nb))] = True
    rows = tree.node_rows(models)
    routes = {a: 0 for a in order}
    for p in pools:
        routes[order[tree.route(rows, p["task"])]] += 1
    bits = {}
    for a in order:
        m = masks[a]
        kept = m.bool()
        par = tree.parent(a)
        jac = None
        if par in masks:
            pk = masks[par].bool()
            jac = round(float((kept & pk).sum()) / max(1, int((kept | pk).sum())), 4)
        bits[a] = {"kept_share": round(float(kept.float().mean()), 4),
                   "adjacent_ones": int((m[1:] * m[:-1]).sum()),
                   "kept_unlit_share": round(float((kept & ~lit).sum()) / max(1, int(kept.sum())), 4),
                   "jaccard_with_parent": jac,
                   "route_share": round(routes[a] / len(pools), 4),
                   "solo_rprec": mean(solo[a])}
    depth = {}
    for a in order:
        depth.setdefault(len(a), []).append(a)
    return {"k": k, "nb": nb, "addresses": order, "depths": {str(d): v for d, v in sorted(depth.items())},
            "heldout": {e: mean(v) for e, v in per.items()},
            "best_solo": max(((a, mean(v)) for a, v in solo.items()), key=lambda x: x[1]),
            "worst_solo": min(((a, mean(v)) for a, v in solo.items()), key=lambda x: x[1]),
            "nodes": nodes, "bits": bits, "seconds": round(time.time() - t0)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append", required=True, help="a .claude/p4 store (repeatable)")
    ap.add_argument("--ks", default="1,2,3,5,8,13")
    ap.add_argument("--minutes", type=float, default=1.5, help="total pretraining per tree, split over its nodes")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass
    import torch
    import phi_rank as pr
    import phi_rank.tree as tree
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    OUT.mkdir(exist_ok=True)
    ks = [int(x) for x in args.ks.split(",")]
    report = {}
    for store in args.store:
        p4 = str(Path(store).resolve())
        name = Path(p4).name if Path(p4).name not in ("p4",) else Path(p4).parent.parent.name
        pools = pr.read_label_pools(p4)
        print(f"{name}: {len(pools)} labelled pools, {sum(len(p['rows']) for p in pools)} rows")
        if len(pools) < FOLDS:
            print(f"{name}: fewer than {FOLDS} pools, skipped")
            continue
        report[name] = {"store": p4, "pools": len(pools), "git_history": bool(tree.first_added(p4)), "sweep": []}
        for k in ks:
            print(f"{name} K={k}")
            r = sweep_one(pr, tree, torch, p4, k, args.minutes, args.epochs, pools, print)
            h = r["heldout"]
            print(f"{name} K={k}: medium {h['medium']} high {h['high']} oracle {h['oracle']} bm25 {h['bm25']} "
                  f"best node {r['best_solo']} ({r['seconds']} s)")
            report[name]["sweep"].append(r)
            (OUT / "phi_depth.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT / 'phi_depth.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
