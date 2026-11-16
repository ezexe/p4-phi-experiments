"""
The store's directory tree as a network's pathways, tested against the controls that settle what the tree adds:
is it the φ structure, or only splitting one model into small ones? Pathways 1 and 5 of the nested-store note's
next steps, in one run. For ezexe/phi's .docs/p4/nested-store.md; the Colab notebook is notebooks/pathways_colab.ipynb.

Data as nested_suite.py: the store's tracked text cut into K shards by when git first added each file (the oldest
shard is the deepest node), and the store's labelled pools. Every 6th pool (DEV_EVERY) is held back as the dev set
the early stopping reads; every score below is on the other pools only, so no number has seen the dev set.

Arms (each node 2^(B - log2 K) x D, each tree K nodes, the same shards and pretraining budget):
  big         one 2^B x D model on all the text: the single large network
  zeck        the Zeckendorf directory tree: nodes at the first K Zeckendorf addresses (1, 101, 1001, 10001, ...),
              a node's parent its address less its last run of .claude's and the p4 after it (the p4 hooks' rule);
              deepest first on the oldest shard, a node with children starting from their mean
  btree       the binary directory tree: the same K nodes at heap addresses 1..K, node i's parent i // 2, trained
              the same way. The control: the same splitting into small models, a power-of-2 structure
  flat        K independent nodes, no tree: each from scratch on its own shard
  each tree as _ens (every node, z-scored and averaged) and _root (the root alone)
Masks on the Zeckendorf tree's nodes, each keeping half the feature rows, each as _ens and as _routed (the task goes
to the one node whose kept rows carry the most weight on its features), with route_hit (how often that node is the
best node for the pool):
  phi     phi_prune's adjacency mask (no two neighbouring rows kept)
  24      2:4 (two rows of every four)
  rand    a random half of the rows (seeded)
  heavy   the heaviest half of the rows by norm: if this routes as well as phi, the adjacency rule adds nothing
Early stopping (--check-s > 0): every pretraining is scored on the dev pools every check-s seconds, zero-shot, and
keeps its best weights; each model's curve and best step are reported, so the run also shows whether longer
pretraining helps or hurts.

Run:
  python scripts/pathways_suite.py --p4 <store> --pairs <pairs.jsonl> --k 4 --minutes 6 --check-s 30 \
      --seeds 0 --cache <dir> --out <json>
"""

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phi_rank as pr  # noqa: E402
from nested_suite import (CACHED, RAW, cached, load, load_pools, rp, say, state, zeck_addresses,  # noqa: E402
                          zeck_parent)
from nested_vs_one import pairs_of, shards, zmean  # noqa: E402
from phi_prune.masks import phi_mask, two_four_mask  # noqa: E402

DEV_EVERY = 6
MASK_KINDS = ("phi", "24", "rand", "heavy")


# ─── the trees ─────────────────────────────────────────────────────────────────────────────────────────

def btree_parent(a):
    return None if a == 1 else a // 2


def depth(a, parent):
    d = 0
    while parent(a) is not None:
        a, d = parent(a), d + 1
    return d


def grow(addrs, parent, k_bits, dim, shard_pairs, minutes, seed, select, check_s, curves, name):
    """Train a tree: deepest addresses first, on the oldest shards; a node with children starts from their mean.
    Returns the nodes' states in training order (the root last) and their addresses."""
    order = sorted(addrs, key=lambda a: (-depth(a, parent), str(a)))
    trained = {}
    for i, a in enumerate(order):
        m = pr.Ranker(1 << k_bits, dim)
        kids = [trained[c] for c in trained if parent(c) == a]
        if kids:
            m.load_state_dict({key: sum(s[key] for s in kids) / len(kids) for key in kids[0]})
        trained[a] = pretrain(m, shard_pairs[i][0], minutes, seed * 10 + i, select, check_s, curves, f"{name} {a}")
    return [trained[a] for a in order], [str(a) for a in order]


def pretrain(m, pairs, minutes, seed, select, check_s, curves, name):
    info = {}
    say(f"pretrain {name}: {minutes * 60:.0f} s budget, {len(pairs):,} pairs, on {pr.device()}")
    steps = pr.pretrain(m, pairs, minutes, seed=seed, log=lambda *_: None,
                        select=select if check_s > 0 else None, select_every=check_s, info=info)
    curves[name] = info or {"steps": steps}
    say(f"pretrain {name}: {steps:,} steps" + (f"; best dev {info['best_score']} at step {info['best_steps']:,}"
                                              if info else ""))
    return state(m)


def build(args, shard_pairs, all_pairs, dev, seed):
    K, B, D, T = args.k, args.bits, args.dim, args.minutes
    sb = B - int(round(math.log2(K)))
    torch.manual_seed(seed)
    curves = {}
    pr.featurize = CACHED

    def select(model):
        return sum(rp(p, pr.score_pool(model, p["task"], [r["text"] for r in p["rows"]])) for p in dev) / len(dev)
    M = {"big": [pretrain(pr.Ranker(1 << B, D), all_pairs, T, seed, select, args.check_s, curves, "big")]}
    M["zeck"], M["zeck_addresses"] = grow(zeck_addresses(K), zeck_parent, sb, D, shard_pairs, T / K, seed, select,
                                          args.check_s, curves, "zeck")
    M["btree"], M["btree_addresses"] = grow(list(range(1, K + 1)), btree_parent, sb, D, shard_pairs, T / K, seed,
                                            select, args.check_s, curves, "btree")
    M["flat"] = [pretrain(pr.Ranker(1 << sb, D), ps, T / K, seed * 10 + i, select, args.check_s, curves, f"flat {i + 1}")
                 for i, (ps, _b) in enumerate(shard_pairs)]
    pr.featurize = RAW
    return M, curves


# ─── the masks ─────────────────────────────────────────────────────────────────────────────────────────

def row_mask(w, kind, seed):
    """A mask of w's shape keeping about half the feature rows, by kind."""
    rows = w.shape[0]
    if kind == "phi":
        return phi_mask(w, axis=0, density=0.5)
    if kind == "24":
        return two_four_mask(w, axis=0)
    if kind == "rand":
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(rows, generator=g)[: rows // 2]
    else:                                                         # heavy
        keep = torch.topk(w.norm(dim=1), rows // 2).indices
    m = torch.zeros(rows, dtype=w.dtype)
    m[keep] = 1
    return m[:, None].expand_as(w).clone()


def masked_families(M, seed):
    """The Zeckendorf nodes under each mask kind: their states, masks, and the kept rows' norms the route reads."""
    fams = {}
    for kind in MASK_KINDS:
        sts, masks, norms = [], [], []
        for i, st in enumerate(M["zeck"]):
            mask = row_mask(st["bag.weight"], kind, seed * 100 + i)
            st2 = {k: v.clone() for k, v in st.items()}
            st2["bag.weight"] = st["bag.weight"] * mask
            sts.append(st2)
            masks.append(mask)
            norms.append(st2["bag.weight"].norm(dim=1))
        fams[kind] = (sts, masks, norms)
    return fams


def route(norms, task):
    ids = torch.tensor(RAW(task, norms[0].shape[0]))
    return int(torch.stack([n[ids].sum() for n in norms]).argmax())


# ─── quality ───────────────────────────────────────────────────────────────────────────────────────────

def oof_scores(states, masks, pools, args, seed, label, path):
    """Each model's out-of-fold scores over the eval pools: {model index: {pool id: scores}}."""
    def make():
        got = {}
        for i, st in enumerate(states):
            got[i] = {}
            for f in range(args.folds):
                train = [p for j, p in enumerate(pools) if j % args.folds != f]
                m = load(st)
                torch.manual_seed(seed * 100 + f)
                pr.finetune(m, train, epochs=args.epochs, seed=seed * 100 + f)
                if masks:
                    with torch.no_grad():
                        m.bag.weight.mul_(masks[i])
                for j, p in enumerate(pools):
                    if j % args.folds == f:
                        got[i][p["id"]] = pr.score_pool(m, p["task"], [r["text"] for r in p["rows"]])
                say(f"quality {label} node {i + 1}/{len(states)}: fold {f + 1}/{args.folds}")
        return got
    if path and os.path.exists(path):
        say(f"quality {label}: from the cache")
    return {int(k): v for k, v in cached(path, make).items()}


def mean_rp(pools, score_of):
    return round(sum(rp(p, score_of(p)) for p in pools) / len(pools), 4)


def quality(M, fams, pools, args, seed, tag):
    out = {}
    pr.featurize = CACHED
    path = lambda name: tag and f"{tag}-oof-{name}.json"  # noqa: E731
    big = oof_scores(M["big"], None, pools, args, seed, "big", path("big"))
    out["big"] = mean_rp(pools, lambda p: big[0][p["id"]])
    for name in ("zeck", "btree", "flat"):
        s = oof_scores(M[name], None, pools, args, seed, name, path(name))
        idx = sorted(s)
        out[f"{name}_ens"] = mean_rp(pools, lambda p: zmean([s[i][p["id"]] for i in idx]))
        if name != "flat":
            out[f"{name}_root"] = mean_rp(pools, lambda p: s[idx[-1]][p["id"]])
    for kind, (sts, masks, norms) in fams.items():
        s = oof_scores(sts, masks, pools, args, seed, f"mask {kind}", path(f"mask-{kind}"))
        idx = sorted(s)
        out[f"{kind}_ens"] = mean_rp(pools, lambda p: zmean([s[i][p["id"]] for i in idx]))
        out[f"{kind}_routed"] = mean_rp(pools, lambda p: s[route(norms, p["task"])][p["id"]])
        hits = 0
        for p in pools:
            per = [rp(p, s[i][p["id"]]) for i in idx]
            hits += per[route(norms, p["task"])] >= max(per)
        out[f"{kind}_route_hit"] = round(hits / len(pools), 3)
    pr.featurize = RAW
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bits", type=int, default=15)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--check-s", type=float, default=30, help="early stopping: seconds between dev checks; 0 = off")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="")
    args = ap.parse_args(argv)
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)
    pools = load_pools(args.pairs)
    dev = [p for j, p in enumerate(pools) if j % DEV_EVERY == DEV_EVERY - 1]
    evals = [p for j, p in enumerate(pools) if j % DEV_EVERY != DEV_EVERY - 1]
    if not dev and args.check_s > 0:
        say(f"only {len(pools)} pools: none to hold back as a dev set, so early stopping is off")
        args.check_s = 0
    shard_files = shards(args.p4, args.k)
    shard_pairs = [pairs_of(args.p4, s) for s in shard_files]
    all_pairs = [x for ps, _ in shard_pairs for x in ps]
    report = {"k": args.k, "pools": len(pools), "dev_pools": len(dev), "eval_pools": len(evals),
              "minutes": args.minutes, "epochs": args.epochs, "folds": args.folds, "check_s": args.check_s,
              "device": str(pr.device()), "runs": []}
    print(json.dumps(report), flush=True)
    for seed in args.seeds:
        t0 = time.time()
        tag = args.cache and os.path.join(args.cache, f"pw-k{args.k}-s{seed}")
        have = tag and os.path.exists(tag + "-models.pt")
        say(f"seed {seed}: " + ("models from the cache" if have else "pretraining big, zeck, btree and flat"))
        M, curves = cached(tag and tag + "-models.pt", lambda: build(args, shard_pairs, all_pairs, dev, seed), "pt")
        fams = masked_families(M, seed)
        say(f"seed {seed}: quality on {len(evals)} eval pools, {args.folds}-fold")
        run = {"seed": seed, "zeck_addresses": M["zeck_addresses"], "btree_addresses": M["btree_addresses"],
               "quality": quality(M, fams, evals, args, seed, tag),
               "early_stopping": {name: {k: v for k, v in c.items() if k != "curve"} for name, c in curves.items()},
               "curves": {name: c.get("curve") for name, c in curves.items()},
               "seconds": round(time.time() - t0)}
        print(json.dumps({"seed": seed, "quality": run["quality"], "early_stopping": run["early_stopping"]}),
              flush=True)
        report["runs"].append(run)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
