"""phi_open: the questions phi/.docs/p4/nested-store.md leaves open, tested on a p4 store, free and offline.

1. seeds     every arm on several seeds: the note warns that arms move by 0.05 between seeds
2. pretrain  the shipped K=4 tree and a single ranker with their pretraining budget against none at all (the Colab
             replications found the best dev ranking at step 0)
3. routes    the mask as the route: phi_mask (shipped) against the heaviest half of the rows, a random half, 2:4, and
             no mask; route accuracy (the routed node is the best node for the pool) and routed / high / oracle
             r-precision, from one fine-tune per fold shared by every mask
4. ordered   phi_mask over feature rows put in an order that means something (rows sorted along their first
             principal component, so neighbours hold similar weights) against the hashed order: the note's open
             question, whether the adjacency rule carries meaning once the features have an order
5. stream    the single realtime stream of every node's rounds, built and decoded: each routed update as
             [address codeword][Fibonacci-coded header][payload] against the per-node .phid files today's chain writes

Results land in out/phi_open.json; nothing is written into the store.

    python -I experiments/phi_depth/phi_open.py --store .claude/p4 --seeds 0,1,2
"""

import argparse
import copy
import json
import math
import os
import random
import struct
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FOLDS = 4
K = 4
MASKS = ("none", "phi", "phi_ordered", "heavy", "random", "two_four")


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return round(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)), 4)


# ─── masks over the feature rows ───────────────────────────────────────────────────────────────────────

def row_mask(torch, masks_mod, w, kind, seed):
    """A 0/1 mask over w's rows (nb x dim), about half kept."""
    nb = w.shape[0]
    norms = w.norm(dim=1)
    if kind == "none":
        return torch.ones(nb)
    if kind == "phi":
        return masks_mod.phi_mask(w, axis=0, density=0.5)[:, 0]
    if kind == "two_four":
        return masks_mod.two_four_mask(w, axis=0)[:, 0]
    if kind == "heavy":
        m = torch.zeros(nb)
        m[norms.argsort(descending=True)[: nb // 2]] = 1
        return m
    if kind == "random":
        g = torch.Generator().manual_seed(seed)
        m = torch.zeros(nb)
        m[torch.randperm(nb, generator=g)[: nb // 2]] = 1
        return m
    if kind == "phi_ordered":
        # rows sorted along their first principal component: neighbours carry similar weight vectors, so "no two
        # adjacent kept" means "never keep two near-duplicate features"; the mask is computed in that order, mapped back
        x = w - w.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)     # exact, so the order is the same every run
        order = (x @ vh[0]).argsort()
        m_sorted = masks_mod.phi_mask(w[order], axis=0, density=0.5)[:, 0]
        m = torch.zeros(nb)
        m[order] = m_sorted
        return m
    raise ValueError(kind)


def masked(torch, masks_mod, model, kind, seed):
    m = copy.deepcopy(model)
    with torch.no_grad():
        mask = row_mask(torch, masks_mod, m.bag.weight.detach(), kind, seed)
        m.bag.weight.mul_(mask.unsqueeze(1))
    return m, mask


# ─── tests 1-4: one fine-tune per fold, every mask scored on it ────────────────────────────────────────

def states_for(pr, tree, torch, p4, k, minutes, seed):
    files, _ = tree.shard_files(p4, k)
    nb = 1 << (pr.NB.bit_length() - 1 - int(round(math.log2(k))))
    order = sorted(tree.addresses(k), key=lambda a: (-len(a), a))
    trained, steps = {}, {}
    for i, a in enumerate(order):
        torch.manual_seed(seed * 1000 + i)
        m = pr.Ranker(nb, pr.DIM)
        kids = [trained[c] for c in trained if tree.parent(c) == a]
        if kids:
            m.load_state_dict({key: sum(s[key] for s in kids) / len(kids) for key in kids[0]})
        steps[a] = pr.pretrain(m, tree.shard_pairs(p4, files[i]), minutes / k, seed=seed * 1000 + i,
                               log=lambda *_: None) if minutes > 0 else 0
        trained[a] = {key: v.clone() for key, v in m.state_dict().items()}
    return order, trained, steps


def evaluate(pr, tree, torch, masks_mod, order, states, pools, epochs, seed):
    per = {kind: {"medium": [], "high": [], "oracle": [], "route_hit": [], "route_best_share": []} for kind in MASKS}
    for f in range(FOLDS):
        held = pools[f::FOLDS]
        train = [p for j, p in enumerate(pools) if j % FOLDS != f]
        tuned = []
        for i, a in enumerate(order):
            m = pr.ranker_like(states[a])
            m.load_state_dict(states[a])
            torch.manual_seed(seed * 1000 + f * 100 + i)
            pr.finetune(m, train, epochs=epochs, seed=seed * 1000 + f * 100 + i)
            tuned.append(m)
        for kind in MASKS:
            models = [masked(torch, masks_mod, m, kind, seed * 1000 + f * 100 + i)[0] for i, m in enumerate(tuned)]
            rows = tree.node_rows(models)
            for p in held:
                texts = [r["text"] for r in p["rows"]]
                labels = [r["label"] for r in p["rows"]]
                solo = [pr.pool_metrics(labels, pr.score_pool(m, p["task"], texts), [])["r-precision"] for m in models]
                med, i = tree.answer(models, rows, p["task"], texts, "medium")
                high, _ = tree.answer(models, rows, p["task"], texts, "high")
                per[kind]["medium"].append(pr.pool_metrics(labels, med, [])["r-precision"])
                per[kind]["high"].append(pr.pool_metrics(labels, high, [])["r-precision"])
                per[kind]["oracle"].append(max(solo))
                per[kind]["route_hit"].append(float(solo[i] == max(solo)))
                per[kind]["route_best_share"].append(sum(s == max(solo) for s in solo) / len(solo))  # chance of a hit
    return {kind: {k: mean(v) for k, v in d.items()} for kind, d in per.items()}


# ─── test 5: the single stream ─────────────────────────────────────────────────────────────────────────

def stream_item(gaps, address, header, payload, names):
    """[address codeword][header][payload]: the address as Zeckendorf digits least significant first with the comma 1,
    then Fibonacci codes of the tensor count, each tensor's name index, rank, dims and kept count, then its scale as a
    32-bit float, then the payload exactly as the .phid carries it."""
    f = lambda n: gaps.fib_code(n + 1)   # noqa: E731 — codes start at 1
    bits = [address[::-1] + "1", f(len(header["tensors"]))]
    for t in header["tensors"]:
        bits += [f(names.index(t["name"])), f(len(t["shape"]))] + [f(d) for d in t["shape"]] + [f(t["n_kept"])]
        bits.append(format(struct.unpack(">I", struct.pack(">f", t["scale"]))[0], "032b"))
    return "".join(bits) + payload


def read_stream(gaps, delta, stream, names, n_digits):
    """Every item back from the stream: (address, header, payload), the payload's length read by decode_delta's own
    walk, so no length is stored anywhere."""
    out, i = [], 0
    while i < len(stream):
        j = stream.index("11", i)                       # the address ends at its first 11
        address = stream[i:j + 1][::-1]
        i = j + 2
        (nt,), i = gaps.read_codes(stream, i, 1)
        tensors = []
        for _ in range(nt - 1):
            (ni, rank), i = gaps.read_codes(stream, i, 2)
            dims, i = gaps.read_codes(stream, i, rank - 1)
            (kept,), i = gaps.read_codes(stream, i, 1)
            scale = struct.unpack(">f", struct.pack(">I", int(stream[i:i + 32], 2)))[0]
            i += 32
            tensors.append({"name": names[ni - 1], "shape": [d - 1 for d in dims], "n_kept": kept - 1, "scale": scale})
        header = {"n_digits": n_digits, "tensors": tensors}
        start = i
        for t in tensors:                               # walk the payload the way decode_delta does, to find its end
            _, i = gaps.read_codes(stream, i, t["n_kept"])
            for _ in range(t["n_kept"]):
                _, i = gaps.read_codes(stream, i + 1, 1)
        out.append((address, header, stream[start:i]))
    return out


def stream_test(pr, tree, torch, delta, gaps, p4, pools, minutes, epochs, n_new, log):
    order, states, _ = states_for(pr, tree, torch, p4, K, minutes, 0)
    base, new = pools[:-n_new], pools[-n_new:]
    models = []
    for i, a in enumerate(order):
        m = pr.ranker_like(states[a])
        m.load_state_dict(states[a])
        torch.manual_seed(i)
        pr.finetune(m, base, epochs=epochs, seed=i)
        tree.mask_rows(m)
        models.append(m)
    names = list(models[0].state_dict().keys())
    stream, phid_bytes, items, kept, phid_payload_bits = "", 0, [], 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for r, p in enumerate(new):
            rows = tree.node_rows(models)
            i = tree.route(rows, p["task"])
            before = {k: v.clone() for k, v in models[i].state_dict().items()}
            torch.manual_seed(100 + r)
            pr.finetune(models[i], [p], epochs=pr.CONTINUE_EPOCHS, seed=100 + r)
            header, payload = delta.encode_delta(before, models[i].state_dict(), keep=pr.KEEP, n_digits=pr.N_DIGITS,
                                                 changed_only=True)
            path = os.path.join(tmp, f"{order[i]}.r{r:03d}.phid")
            delta.save_delta(path, header, payload, packed=True)
            phid_bytes += os.path.getsize(path)
            phid_payload_bits += len(payload)
            kept += delta.delta_stats(header, payload)["kept"]
            stream += stream_item(gaps, order[i], header, payload, names)
            items.append((order[i], header, payload))
            log(f"  stream round {r + 1}/{n_new}: node {order[i]}, {delta.delta_stats(header, payload)['kept']} kept")
    back = read_stream(gaps, delta, stream, names, pr.N_DIGITS)
    exact = len(back) == len(items) and all(
        a == b[0] and pl == b[2] and [t["n_kept"] for t in h["tensors"]] == [t["n_kept"] for t in b[1]["tensors"]]
        and all(abs(t["scale"] - u["scale"]) <= 1e-6 * abs(t["scale"]) for t, u in zip(h["tensors"], b[1]["tensors"]))
        for (a, h, pl), b in zip(items, back))
    decoded_equal = exact and all(
        all((x[0] == y[0]).all() and (abs(x[1] - y[1]) <= 1e-6 * (abs(x[1]).max() + 1e-12)).all()
            for x, y in zip(delta.decode_delta(h, pl).values(), delta.decode_delta(b[1], b[2]).values()))
        for (a, h, pl), b in zip(items, back))
    overhead = len(stream) - phid_payload_bits
    return {"rounds": n_new, "nodes_hit": sorted({a for a, _, _ in items}), "kept": kept,
            "phid_files_bits": phid_bytes * 8, "phid_bits_per_kept": round(phid_bytes * 8 / max(1, kept), 3),
            "stream_bits": len(stream), "stream_bits_per_kept": round(len(stream) / max(1, kept), 3),
            "payload_bits_per_kept": round(phid_payload_bits / max(1, kept), 3),
            "stream_framing_bits_per_round": round(overhead / n_new, 1),
            "phid_header_bits_per_round": round((phid_bytes * 8 - phid_payload_bits) / n_new, 1),
            "round_trip_exact": bool(exact), "decoded_deltas_equal": bool(decoded_equal)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--minutes", type=float, default=1.0, help="the pretrain arm's total budget per tree")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--stream-rounds", type=int, default=4)
    ap.add_argument("--skip", default="", help="comma list of: tree,single,stream")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass
    import torch
    import phi_rank as pr
    import phi_rank.tree as tree
    import phi_prune.masks as masks_mod
    from phi_prune import delta, gaps
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    OUT.mkdir(exist_ok=True)
    skip = set(filter(None, args.skip.split(",")))
    out_path = OUT / "phi_open.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    for store in args.store:
        p4 = str(Path(store).resolve())
        name = Path(p4).parent.parent.name if Path(p4).name == "p4" else Path(p4).name
        pools = pr.read_label_pools(p4)
        rep = report.setdefault(name, {"store": p4, "pools": len(pools)})
        print(f"{name}: {len(pools)} pools")
        for k_arm, what in ((K, "tree"), (1, "single")):
            if what in skip:
                continue
            for minutes in (0.0, args.minutes):
                arm = f"{what}-k{k_arm}-pre{minutes:g}"
                runs = rep.setdefault(arm, {})
                for seed in [int(s) for s in args.seeds.split(",")]:
                    if str(seed) in runs:
                        continue
                    t0 = time.time()
                    order, states, steps = states_for(pr, tree, torch, p4, k_arm, minutes, seed)
                    res = evaluate(pr, tree, torch, masks_mod, order, states, pools, args.epochs, seed)
                    runs[str(seed)] = {"steps": steps, "masks": res, "seconds": round(time.time() - t0)}
                    print(f"{name} {arm} seed {seed}: " + "; ".join(
                        f"{kd} med {r['medium']} high {r['high']} hit {r['route_hit']}" for kd, r in res.items())
                        + f" ({time.time() - t0:.0f} s)")
                    out_path.write_text(json.dumps(report, indent=1))
        if "stream" not in skip and "stream" not in rep:
            rep["stream"] = stream_test(pr, tree, torch, delta, gaps, p4, pools, args.minutes, args.epochs,
                                        args.stream_rounds, print)
            print(f"{name} stream: {rep['stream']}")
            out_path.write_text(json.dumps(report, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
