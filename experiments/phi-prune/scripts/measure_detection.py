"""
Measure the share of single-digit flips that the minimal-form check catches.

For each setup it quantizes weights with FibonacciEncoder, runs integrity.simulate_corruption, and
sets the result beside the exact expectation: over each kept weight's Zeckendorf digits, the share
of digit positions whose flip from 0 to 1 makes 11. A flip from 1 to 0 never does. "uniform" is
that share averaged over every quantization level, a baseline that does not depend on the weights.
results/measure_detection.txt holds a run with the defaults.

Usage:
    python scripts/measure_detection.py                      # synthetic conv weights
    python scripts/measure_detection.py --models resnet18,resnet50
"""

import argparse
import random

import numpy as np
import torch

from phi_prune import prune
from phi_prune.encoding import FibonacciEncoder
from phi_prune.integrity import simulate_corruption
from phi_prune.masks import phi_mask

SYNTHETIC_SHAPE = (256, 128, 3, 3)
DISTRIBUTIONS = ("gaussian", "laplace", "student-t3", "uniform")


def synthetic_weights(dist, g):
    if dist == "gaussian":
        return torch.randn(SYNTHETIC_SHAPE, generator=g)
    if dist == "laplace":
        u = torch.rand(SYNTHETIC_SHAPE, generator=g) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs())
    if dist == "student-t3":
        z = torch.randn(SYNTHETIC_SHAPE, generator=g)
        chi2 = sum(torch.randn(SYNTHETIC_SHAPE, generator=g) ** 2 for _ in range(3))
        return z / torch.sqrt(chi2 / 3)
    if dist == "uniform":
        return torch.rand(SYNTHETIC_SHAPE, generator=g) * 2 - 1
    raise ValueError(dist)


def per_level_rate(enc):
    """For each quantization level, the exact share of single-digit flips the minimal-form check catches."""
    rates = np.zeros(enc.max_value + 1)
    for v in range(enc.max_value + 1):
        cw = enc.to_codeword(v)
        n = len(cw)
        rates[v] = sum(
            1 for i in range(n)
            if cw[i] == 0 and ((i > 0 and cw[i - 1]) or (i < n - 1 and cw[i + 1]))
        ) / n
    return rates


def measure_layers(layers, enc, n_flips):
    """
    Quantize each weight under its mask and flip digits of its quantization levels.

    Returns detected, flips, exact_rate and rmse, with exact_rate and rmse weighted by each
    layer's share of the flips.
    """
    rates = per_level_rate(enc)
    detected = flips = 0
    exact_sum = sq_err_sum = 0.0
    for w, mask in layers:
        encoded, scale, rmse, offset = enc.encode_tensor(w, mask, return_offset=True)
        active = encoded[mask.bool()].double().cpu().numpy()
        levels = np.clip(np.round((active - offset) * scale), 0, enc.max_value).astype(int)
        d, t = simulate_corruption(encoded, mask, enc, scale, offset, n_flips=n_flips)
        detected += d
        flips += t
        exact_sum += rates[levels].mean() * t
        sq_err_sum += rmse ** 2 * t
    return detected, flips, exact_sum / flips, (sq_err_sum / flips) ** 0.5


def model_layers(name):
    import torchvision

    model = getattr(torchvision.models, name)(weights="DEFAULT").eval()
    pruned, masks = prune(model)
    params = dict(pruned.named_parameters())
    return [(params[k].data, m) for k, m in masks.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--digits", default="8,10", help="comma-separated numbers of Zeckendorf digits")
    ap.add_argument("--models", default="", help="comma-separated torchvision model names, each with its "
                    "DEFAULT weights, which are IMAGENET1K_V2 for ResNet-50, 101 and 152; "
                    "empty measures synthetic weights instead")
    ap.add_argument("--flips", type=int, default=200_000,
                    help="flips per synthetic tensor, or per pruned layer with --models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.models:
        setups = [(m, lambda m=m: model_layers(m)) for m in args.models.split(",")]
    else:
        def synthetic(dist):
            w = synthetic_weights(dist, torch.Generator().manual_seed(args.seed))
            return [(w, phi_mask(w))]
        setups = [(d, lambda d=d: synthetic(d)) for d in DISTRIBUTIONS]

    print(f"{'digits':>6} {'weights':>11} {'rmse':>8} "
          f"{'simulated':>9} {'±2σ':>6} {'exact':>6} {'uniform':>7}")
    for label, get_layers in setups:
        layers = get_layers()
        for n_digits in (int(d) for d in args.digits.split(",")):
            enc = FibonacciEncoder(n_digits)
            random.seed(args.seed)
            det, tot, exact, rmse = measure_layers(layers, enc, args.flips)
            p = det / tot
            print(f"{n_digits:>6} {label:>11} {rmse:8.5f} "
                  f"{p:9.4f} {2 * np.sqrt(p * (1 - p) / tot):6.4f} "
                  f"{exact:6.4f} {per_level_rate(enc).mean():7.4f}")


if __name__ == "__main__":
    main()
