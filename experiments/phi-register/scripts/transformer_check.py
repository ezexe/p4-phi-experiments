"""
Pre-registered check of φ-pruning, and of quantized weights written in the
Zeckendorf representation, on a small transformer.

A character-level GPT with 4 layers, width 128 and MLP width 512 learns the
source of the standard library of the Python that runs it, on CPU. Per seed it
trains dense once, then:

Part A prunes MLP hidden neurons, structured, so that a pruned neuron loses its
row in fc1 and its column in fc2, one-shot from the dense weights, and then
fine-tunes with the mask fixed:
    50%:  phi            adjacency_dp on the L1 norm of each neuron's weights,
                         in neuron order
          phi-reorder    the same after reordering neurons so ranks interleave:
                         top, bottom, 2nd, 2nd-to-last and so on; reordering hidden
                         neurons does not change what the model computes, and for
                         distinct scores of even count the mask it keeps is the
                         heaviest half, exactly top's
          2:4            2 of every 4 neighbouring neurons
          top            the heaviest half, no pattern
          random         a random half, as a floor
    25%:  phi-low        adjacency_dp_density(scores, 0.25)
          1:4            1 of every 4 neighbouring neurons
          top            the heaviest quarter, no pattern
--part high instead prunes above the 50% cap, by putting the (1,∞)-RLL
constraint on the pruned neurons, so that no two neighbouring neurons are both
pruned:
    75%:  phi-comp       adjacency_dp_density on max score - score picks the
                         pruned quarter, so pruned neurons never sit side by side
          3:4            3 of every 4 neighbouring neurons
          top            the heaviest three quarters, no pattern
          random         a random three quarters, as a floor

Part B, quantization: every Linear weight is quantized per output channel to L
uniform levels, L in {16, 21, 32}, which take 4 bits, 6 Zeckendorf digits and
5 bits; --part high takes {55, 144, 377, 987}, the level counts of 8, 10, 12
and 14 Zeckendorf digits. For each L it averages the Hamming weight of the
levels in binary, in the Zeckendorf representation and in canonical signed
digit form, CSD, which is the number of partial products a multiplier adds per
weight. FibonacciEncoder's levels are every integer 0 .. F(n+2) - 1, so 6
Zeckendorf digits quantize to 21 uniform levels, and the representations differ
only in how many additions they cost.

Kill criteria. Lower bits per character is better, and σ is the pooled
standard deviation over seeds of the two methods compared:
    K1  φ at 50% survives against 2:4 if mean(phi) - mean(2:4) < σ
    K2  φ-low at 25% survives against 1:4 if mean(1:4) - mean(phi-low) > σ
    K3  reordering prediction holds if |mean(phi-reorder) - mean(top 50%)| < σ;
        phi-reorder keeps top's mask, so K3 compares a method with itself
    K4  the Zeckendorf representation survives if its mean Hamming weight is
        below CSD's at 21 uniform affine levels, the package's (w - min) * scale

Kill criteria for --part high:
    K5  φ-comp at 75% survives against 3:4 if, compared within each seed, it
        is lower in every seed
    K6  the Zeckendorf representation survives at 144 levels if its mean
        Hamming weight over the seeds is below CSD's, on uniform affine levels

--part delta tests the post-training case. Per seed it trains the dense model
on most of the training files, fine-tunes a copy on a held-out tenth of them,
and takes the delta of every Linear layer, the fine-tuned minus the base weights.
Weights and deltas are both quantized symmetrically per output channel, to
round(w / max|w| * m) with m = (L - 1) // 2, and the Hamming weight of their
magnitudes is averaged in binary, in the Zeckendorf representation and in CSD.
Top-k sparsification keeps the largest 10% of each delta's entries per tensor,
and then the largest 1%, and the gaps between kept positions are coded with
Fibonacci coding, Elias gamma, Elias delta and fixed-length codes. Kill criteria:
    K7  the Zeckendorf representation suits deltas if its saving in Hamming
        weight over CSD, (CSD - Zeckendorf) / CSD at 21 levels, is larger on the
        deltas than on the full weights of the same model, averaged over seeds
    K8  Fibonacci-coded gaps survive if, at 10% kept, they take fewer bits per
        kept position than both Elias gamma and fixed-length codes

Usage:
    python experiments/phi-register/scripts/transformer_check.py                 # 3 seeds
    python experiments/phi-register/scripts/transformer_check.py --seeds 1 --steps 300 --finetune 100   # smoke run
"""

import argparse
import copy
import glob
import json
import locale
import math
import platform
import sysconfig
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phi_prune.gaps import gap_bits, pooled_gaps
from phi_prune.masks import adjacency_dp, adjacency_dp_density
from phi_prune.register import BINARY, PHI, from_int

CTX, D, LAYERS, HEADS = 128, 128, 4, 4
BATCH, LR = 32, 3e-3
EVAL_BATCHES = 40


# ── data ──

CHARS = [chr(c) for c in range(32, 127)] + ["\n", "\t"]


def read_files(keep):
    """Encoded text of the standard library files whose sorted index i passes keep(i)."""
    files = sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/*.py"))
    lookup = {c: i for i, c in enumerate(CHARS)}
    text = "".join(open(f, errors="replace").read() for i, f in enumerate(files) if keep(i))
    return torch.tensor([lookup.get(c, lookup["?"]) for c in text], dtype=torch.long)


def load_text():
    return read_files(lambda i: i % 10), read_files(lambda i: i % 10 == 0), len(CHARS)


def run_info(**chars):
    """What a run's numbers depend on that its arguments do not say: the Python, whose standard library is the
    text, the platform and the encoding open() read it with, torch, and the characters of each split."""
    return {"python": platform.python_version(), "platform": platform.platform(),
            "encoding": locale.getpreferredencoding(False), "torch": torch.__version__, "chars": chars}


def batch(data, g):
    ix = torch.randint(len(data) - CTX - 1, (BATCH,), generator=g)
    x = torch.stack([data[i:i + CTX] for i in ix])
    y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
    return x, y


# ── model ──

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc1, self.fc2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, D), nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))


def apply_masks(model, masks):
    """Zero pruned hidden neurons: fc1 rows and bias, fc2 columns."""
    with torch.no_grad():
        for blk, m in zip(model.blocks, masks):
            blk.fc1.weight.mul_(m[:, None])
            blk.fc1.bias.mul_(m)
            blk.fc2.weight.mul_(m[None, :])


def train(model, data, steps, seed, masks=None, lr=LR):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    for _ in range(steps):
        x, y = batch(data, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if masks is not None:
            apply_masks(model, masks)


@torch.no_grad()
def bpc(model, data):
    """Validation bits per character on fixed batches."""
    g = torch.Generator().manual_seed(1234)
    model.eval()
    total = 0.0
    for _ in range(EVAL_BATCHES):
        x, y = batch(data, g)
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return total / EVAL_BATCHES / math.log(2)


# ── Part A: neuron masks ──

def neuron_scores(blk):
    return (blk.fc1.weight.abs().sum(1) + blk.fc2.weight.abs().sum(0)).detach().numpy()


def interleave_order(scores):
    """Neuron order putting ranks 1, n, 2, n-1, ... side by side."""
    ranked = list(np.argsort(-scores))
    order = []
    while ranked:
        order.append(ranked.pop(0))
        if ranked:
            order.append(ranked.pop())
    return np.array(order)


def n_of_four(scores, n):
    m = np.zeros(len(scores))
    for i in range(0, len(scores), 4):
        m[i + np.argsort(-scores[i:i + 4])[:n]] = 1
    return m


def top_k(scores, density):
    m = np.zeros(len(scores))
    m[np.argsort(-scores)[:round(density * len(scores))]] = 1
    return m


def make_mask(method, scores, rng):
    if method == "phi":
        return adjacency_dp(scores).astype(float)
    if method == "phi-reorder":
        order = interleave_order(scores)
        m = np.zeros(len(scores))
        m[order] = adjacency_dp(scores[order])
        return m
    if method == "2:4":
        return n_of_four(scores, 2)
    if method == "top-50":
        return top_k(scores, 0.5)
    if method == "random-50":
        m = np.zeros(len(scores))
        m[rng.permutation(len(scores))[:len(scores) // 2]] = 1
        return m
    if method == "phi-low-25":
        return adjacency_dp_density(scores, 0.25).astype(float)
    if method == "1:4":
        return n_of_four(scores, 1)
    if method == "top-25":
        return top_k(scores, 0.25)
    if method == "phi-comp-75":
        return 1 - adjacency_dp_density(scores.max() - scores + 1e-6, 0.25)
    if method == "3:4":
        return n_of_four(scores, 3)
    if method == "top-75":
        return top_k(scores, 0.75)
    if method == "random-75":
        m = np.zeros(len(scores))
        m[rng.permutation(len(scores))[:round(0.75 * len(scores))]] = 1
        return m
    raise ValueError(method)


PARTS = {
    "low": (["phi", "phi-reorder", "2:4", "top-50", "random-50", "phi-low-25", "1:4", "top-25"], (16, 21, 32)),
    "high": (["phi-comp-75", "3:4", "top-75", "random-75"], (55, 144, 377, 987)),
}


# ── Part B: quantization and Hamming weights ──

def quantize_levels(w, levels):
    """Per-output-channel uniform affine quantization; returns the dequantized weight and the integer levels."""
    lo, hi = w.min(1, keepdim=True).values, w.max(1, keepdim=True).values
    scale = (levels - 1) / (hi - lo).clamp_min(1e-12)
    q = torch.round((w - lo) * scale)
    return q / scale + lo, q.long()


def csd_nonzeros(v):
    """Hamming weight of the canonical signed digit form, CSD, of an integer."""
    n = 0
    while v:
        if v & 1:
            v -= 2 - (v & 3)
            n += 1
        v >>= 1
    return n


def digit_counts(levels, L):
    """
    Mean Hamming weight per weight in binary, Zeckendorf and CSD, of the uniform
    affine levels, under the key offset, and of their sign-magnitude form.
    """
    zeck = np.array([sum(from_int(v, PHI)) for v in range(L)])
    binary = np.array([sum(from_int(v, BINARY)) for v in range(L)])
    csd = np.array([csd_nonzeros(v) for v in range(L)])
    centre = (L - 1) // 2
    mag = np.abs(levels - centre)
    return {
        "offset": {"binary": binary[levels].mean(), "zeckendorf": zeck[levels].mean(), "csd": csd[levels].mean()},
        "sign-magnitude": {"binary": binary[mag].mean(), "zeckendorf": zeck[mag].mean(), "csd": csd[mag].mean()},
    }


def quantized_model(model, L):
    q = copy.deepcopy(model)
    all_levels = []
    with torch.no_grad():
        for mod in q.modules():
            if isinstance(mod, nn.Linear):
                deq, lv = quantize_levels(mod.weight, L)
                mod.weight.copy_(deq)
                all_levels.append(lv.flatten())
    return q, torch.cat(all_levels).numpy()


# ── post-training: deltas ──

def symmetric_magnitudes(w, levels):
    """|round(w / max|w| * m)| per output channel, m = (levels - 1) // 2."""
    m = (levels - 1) // 2
    scale = w.abs().max(1, keepdim=True).values.clamp_min(1e-12) / m
    return torch.round(w / scale).abs().long().flatten().numpy()


def magnitude_digits(mags):
    top = int(mags.max()) + 1
    zeck = np.array([sum(from_int(v, PHI)) for v in range(top)])
    binary = np.array([sum(from_int(v, BINARY)) for v in range(top)])
    csd = np.array([csd_nonzeros(v) for v in range(top)])
    return {"binary": binary[mags].mean(), "zeckendorf": zeck[mags].mean(), "csd": csd[mags].mean()}


def run_delta(args):
    pre = read_files(lambda i: i % 10 not in (0, 1))
    tune = read_files(lambda i: i % 10 == 1)
    split = int(0.9 * len(tune))
    tune_train, tune_val = tune[:split], tune[split:]
    val = read_files(lambda i: i % 10 == 0)
    print(f"pretrain {len(pre):,} chars, fine-tune {len(tune_train):,}, held out {len(tune_val):,}, val {len(val):,}")
    results = {"args": vars(args), "run": run_info(pretrain=len(pre), fine_tune=len(tune_train),
                                                   held_out=len(tune_val), val=len(val)), "seeds": []}
    t0 = time.time()
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        base = GPT(len(CHARS))
        train(base, pre, args.steps, seed)
        tuned = copy.deepcopy(base)
        train(tuned, tune_train, args.finetune, seed + 100, lr=LR / 3)
        r = {"seed": seed, "base": {"val": bpc(base, val), "tune_val": bpc(base, tune_val)},
             "tuned": {"val": bpc(tuned, val), "tune_val": bpc(tuned, tune_val)}, "digits": {}, "gaps": {}}
        pairs = [(a.weight.detach(), b.weight.detach()) for a, b in zip(base.modules(), tuned.modules())
                 if isinstance(a, nn.Linear)]
        deltas = [b - a for a, b in pairs]
        rel = float(torch.cat([d.flatten() for d in deltas]).norm() / torch.cat([a.flatten() for a, _ in pairs]).norm())
        r["update_norm_ratio"] = rel
        print(f"\nseed {seed}: held-out fine-tune bpc {r['base']['tune_val']:.4f} -> {r['tuned']['tune_val']:.4f}, "
              f"val {r['base']['val']:.4f} -> {r['tuned']['val']:.4f}; |delta| / |weights| {rel:.3f} "
              f"({time.time() - t0:.0f} s)")
        for L in (21, 55):
            for name, ws in (("weights", [a for a, _ in pairs]), ("update", deltas)):
                mags = np.concatenate([symmetric_magnitudes(w, L) for w in ws])
                d = magnitude_digits(mags)
                d["saving_vs_csd"] = (d["csd"] - d["zeckendorf"]) / d["csd"]
                r["digits"][f"{name}-{L}"] = d
                print(f"  {L} levels, {name:<7}: mean Hamming weight binary {d['binary']:.3f}, Zeckendorf "
                      f"{d['zeckendorf']:.3f}, CSD {d['csd']:.3f}; Zeckendorf saves {d['saving_vs_csd']:+.1%} over CSD")
        for keep in (0.10, 0.01):
            gaps, n = pooled_gaps(deltas, keep)
            r["gaps"][keep] = gap_bits(gaps, n)
            print(f"  delta, {keep:.0%} kept: bits per position " +
                  ", ".join(f"{k} {v:.2f}" for k, v in r["gaps"][keep].items()))
        results["seeds"].append(r)

    def mean(f):
        return float(np.mean([f(s) for s in results["seeds"]]))

    upd = mean(lambda s: s["digits"]["update-21"]["saving_vs_csd"])
    full = mean(lambda s: s["digits"]["weights-21"]["saving_vs_csd"])
    fib, gamma, fixed = (mean(lambda s, k=k: s["gaps"][0.10][k]) for k in ("fibonacci", "elias-gamma", "fixed-width"))
    verdicts = {
        "K7 Zeckendorf saving vs CSD, update vs weights": (upd - full, upd > full),
        "K8 Fibonacci gap code vs Elias-gamma, fixed": (fib - min(gamma, fixed), fib < gamma and fib < fixed),
    }
    results["verdicts"] = {k: {"difference": float(a), "survives": bool(b)} for k, (a, b) in verdicts.items()}
    print()
    for k, v in results["verdicts"].items():
        print(f"{k:<48} difference {v['difference']:+.4f}  {'SURVIVES' if v['survives'] else 'KILLED'}")
    write_json(results, args.out)
    print(f"\n{time.time() - t0:.0f} s; results in {args.out}")


# ── run ──

def pooled(a, b):
    """Pooled standard deviation of two methods over seeds; None from one seed, where there is none."""
    return math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) if len(a) > 1 else None


def write_json(results, path):
    """Write results as JSON, with every NaN or infinity as null, since JSON has neither."""
    def finite(x):
        if isinstance(x, dict):
            return {k: finite(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [finite(v) for v in x]
        if isinstance(x, (float, np.floating)) and not math.isfinite(x):
            return None
        return x
    text = json.dumps(finite(results), indent=1, default=float, allow_nan=False)
    with open(path, "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500, help="dense training steps")
    ap.add_argument("--finetune", type=int, default=300,
                    help="fine-tune steps: with the mask fixed for low and high, unmasked on held-out files for delta")
    ap.add_argument("--part", choices=sorted(PARTS) + ["delta"], default="low",
                    help="low: masks at 50%% and 25%%, 16-32 levels; high: masks at 75%%, 55-987 levels; "
                         "delta: the delta of a fine-tuned model")
    ap.add_argument("--out", default="transformer_check.json")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.part == "delta":
        return run_delta(args)
    methods, level_counts = PARTS[args.part]

    train_data, val_data, vocab = load_text()
    print(f"train {len(train_data):,} chars, val {len(val_data):,} chars, vocab {vocab}")
    results = {"args": vars(args), "run": run_info(train=len(train_data), val=len(val_data)), "seeds": []}
    t0 = time.time()
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        dense = GPT(vocab)
        train(dense, train_data, args.steps, seed)
        r = {"seed": seed, "dense": bpc(dense, val_data), "prune": {}, "quant": {}}
        print(f"\nseed {seed}: dense {r['dense']:.4f} bpc ({time.time() - t0:.0f} s)")

        rng = np.random.default_rng(seed)
        for method in methods:
            model = copy.deepcopy(dense)
            masks = [torch.tensor(make_mask(method, neuron_scores(b), rng), dtype=torch.float32)
                     for b in model.blocks]
            apply_masks(model, masks)
            oneshot = bpc(model, val_data)
            train(model, train_data, args.finetune, seed + 100, masks=masks, lr=LR / 3)
            density = float(torch.stack(masks).mean())
            r["prune"][method] = {"density": density, "oneshot": oneshot, "finetuned": bpc(model, val_data)}
            print(f"  {method:<12} density {density:.3f}  one-shot {oneshot:.4f}  fine-tuned "
                  f"{r['prune'][method]['finetuned']:.4f}")

        for L in level_counts:
            q, levels = quantized_model(dense, L)
            r["quant"][L] = {"bpc": bpc(q, val_data), "bits": math.log2(L), "digits": digit_counts(levels, L)}
            d = r["quant"][L]["digits"]["offset"]
            print(f"  {L} levels ({math.log2(L):.2f} bits): {r['quant'][L]['bpc']:.4f} bpc; mean Hamming weight "
                  f"binary {d['binary']:.3f}, Zeckendorf {d['zeckendorf']:.3f}, CSD {d['csd']:.3f}")
        results["seeds"].append(r)

    # ── verdicts ──
    def ft(m):
        return [s["prune"][m]["finetuned"] for s in results["seeds"]]

    def digits(L):
        d = [s["quant"][L]["digits"]["offset"] for s in results["seeds"]]
        return np.mean([x["zeckendorf"] for x in d]), np.mean([x["csd"] for x in d])

    if args.part == "high":
        comp, three4 = ft("phi-comp-75"), ft("3:4")
        paired = [a - b for a, b in zip(comp, three4)]
        zeck, csd = digits(144)
        verdicts = {
            "K5 phi-comp vs 3:4 at 75% (paired)": (np.mean(paired), 0.0, all(p < 0 for p in paired)),
            "K6 Zeckendorf vs CSD digits at 144 levels": (zeck - csd, 0.0, zeck < csd),
        }
    else:
        verdicts = low_verdicts(ft, digits)
    results["verdicts"] = {k: {"difference": float(a), "sigma": None if b is None else float(b),
                               "survives": None if c is None else bool(c)} for k, (a, b, c) in verdicts.items()}
    report(results, methods, ft, args, t0)


def low_verdicts(ft, digits):
    phi, two4, reo, top50 = ft("phi"), ft("2:4"), ft("phi-reorder"), ft("top-50")
    low, one4 = ft("phi-low-25"), ft("1:4")
    zeck21, csd21 = digits(21)
    d1, s1 = np.mean(phi) - np.mean(two4), pooled(phi, two4)
    d2, s2 = np.mean(one4) - np.mean(low), pooled(low, one4)
    d3, s3 = np.mean(reo) - np.mean(top50), pooled(reo, top50)
    return {
        "K1 phi vs 2:4 at 50%": (d1, s1, None if s1 is None else d1 < s1),
        "K2 phi-low vs 1:4 at 25%": (d2, s2, None if s2 is None else d2 > s2),
        "K3 phi-reorder vs top-50": (d3, s3, None if s3 is None else abs(d3) < s3),
        "K4 Zeckendorf vs CSD digits at 21 levels": (zeck21 - csd21, 0.0, zeck21 < csd21),
    }


def report(results, methods, ft, args, t0):
    print(f"\n{'method':<12}{'density':>9}{'one-shot':>10}{'fine-tuned':>12}{'± sd':>8}")
    for m in methods:
        v = ft(m)
        dens = np.mean([s["prune"][m]["density"] for s in results["seeds"]])
        one = np.mean([s["prune"][m]["oneshot"] for s in results["seeds"]])
        sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
        print(f"{m:<12}{dens:>9.3f}{one:>10.4f}{np.mean(v):>12.4f}{sd:>8.4f}")
    print(f"dense       {'1.000':>9}{'':>10}{np.mean([s['dense'] for s in results['seeds']]):>12.4f}")
    print()
    for k, v in results["verdicts"].items():
        if v["sigma"] is None:
            print(f"{k:<44} difference {v['difference']:+.4f}  no σ from one seed, UNDECIDED")
        else:
            print(f"{k:<44} difference {v['difference']:+.4f}  σ {v['sigma']:.4f}  "
                  f"{'SURVIVES' if v['survives'] else 'KILLED'}")
    write_json(results, args.out)
    print(f"\n{time.time() - t0:.0f} s; results in {args.out}")


if __name__ == "__main__":
    main()
