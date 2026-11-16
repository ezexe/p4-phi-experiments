"""
Fine-tune, prune and quantize a ResNet on CIFAR-10, then compare per-tensor and per-channel quantization.

The steps of the variant cells of the Colab notebook notebooks/quickstart_gpu.ipynb, on a CPU: a
dense fine-tune, φ-pruning of the convolutions and a fine-tune with the masks held fixed. The pruned
convolutions are then quantized once per tensor and once per output channel. Each model is scored
in fp32 and, on the first --fp16-test images, under CPU fp16 autocast, with the share of those
images whose logits are not all finite.

With the defaults, 128 px images, 5000 training and 2000 test images and one epoch per fine-tune,
the fine-tunes, pruning, quantization and scoring of ResNet-152 took 50.9 minutes on a GitHub runner
with 2 threads, once the data and weights had loaded, and 101.2 minutes with --ft-epochs 6;
results/finetune-check-36070148204.txt and results/finetune-check-36076634102.txt hold
the two runs, whose printed total is that span.

Usage:
    python scripts/finetune_encode_check.py --arch resnet152 --weights IMAGENET1K_V2
"""

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from phi_prune import finetune, prune
from phi_prune.encoding import FibonacciEncoder

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Batches:
    """CIFAR-10 uint8 images, normalized, optionally flipped and resized per batch, as the Colab notebook does."""

    def __init__(self, x, y, batch, size, train):
        self.x, self.y, self.batch, self.size, self.train = x, y, batch, size, train

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n) if self.train else torch.arange(n)
        for i in range(0, n, self.batch):
            idx = order[i:i + self.batch]
            x = (self.x[idx].float() / 255 - MEAN) / STD
            if self.train:
                flip = torch.rand(len(idx)) < 0.5
                x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            yield F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False), self.y[idx]


def cifar(root, n_train, n_test, fake=False):
    if fake:  # random images in place of CIFAR-10
        g = torch.Generator().manual_seed(0)
        return (torch.randint(0, 256, (n_train, 3, 32, 32), dtype=torch.uint8, generator=g),
                torch.randint(0, 10, (n_train,), generator=g),
                torch.randint(0, 256, (n_test, 3, 32, 32), dtype=torch.uint8, generator=g),
                torch.randint(0, 10, (n_test,), generator=g))
    import torchvision

    tr = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    g = torch.Generator().manual_seed(0)
    pick = torch.randperm(len(tr.targets), generator=g)[:n_train]
    xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2)[pick]
    ytr = torch.as_tensor(tr.targets)[pick]
    # CIFAR-10's test set comes shuffled, so its first n_test images hold roughly equal classes
    xte = torch.from_numpy(te.data).permute(0, 3, 1, 2)[:n_test]
    yte = torch.as_tensor(te.targets)[:n_test]
    return xtr, ytr, xte, yte


def accuracy(model, loader, fp16):
    """Top-1 accuracy in percent, and the percent of images whose logits are not all finite."""
    model.eval()
    correct = bad = total = 0
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16, enabled=fp16):
        for x, y in loader:
            logits = model(x).float()
            bad += (~torch.isfinite(logits).all(1)).sum().item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return 100.0 * correct / total, 100.0 * bad / total


def encoded(model, masks, digits, axis):
    m = copy.deepcopy(model)
    enc = FibonacciEncoder(digits)
    params = dict(m.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            params[name].copy_(enc.encode_tensor(params[name].data, mask=mask, axis=axis)[0])
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", default="resnet152")
    ap.add_argument("--weights", default="IMAGENET1K_V2", help="torchvision weights name, or none")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--train", type=int, default=5000, help="training images")
    ap.add_argument("--test", type=int, default=2000, help="test images")
    ap.add_argument("--fp16-test", type=int, default=256,
                    help="of which also scored under CPU fp16 autocast, which runs far slower than fp32")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dense-epochs", type=int, default=1)
    ap.add_argument("--ft-epochs", type=int, default=1)
    ap.add_argument("--digits", type=int, default=10)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--fake-data", action="store_true",
                    help="random images in place of CIFAR-10; with --weights none the run downloads nothing")
    args = ap.parse_args()

    import torchvision

    torch.manual_seed(0)
    xtr, ytr, xte, yte = cifar(args.data, args.train, args.test, args.fake_data)
    train = Batches(xtr, ytr, args.batch, args.img_size, True)
    test = Batches(xte, yte, 256, args.img_size, False)
    test16 = Batches(xte[:args.fp16_test], yte[:args.fp16_test], 256, args.img_size, False)
    lr_scale = args.batch / 128  # the Colab notebook's learning rates are for batch 128

    weights = None if args.weights.lower() == "none" else args.weights
    net = torchvision.models.get_model(args.arch, weights=weights)
    net.fc = nn.Linear(net.fc.in_features, 10)
    print(f"{args.arch} {args.weights}: {len(ytr)} train / {len(yte)} test images at {args.img_size} px, "
          f"{torch.get_num_threads()} threads", flush=True)

    start = time.perf_counter()
    finetune(net, train, epochs=args.dense_epochs, lr=0.01 * lr_scale, verbose=False)
    print(f"dense fine-tune done, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    pruned, masks = prune(net, inplace=False, layer_types=(nn.Conv2d,))
    finetune(pruned, train, epochs=args.ft_epochs, masks=masks, lr=0.001 * lr_scale, verbose=False)
    print(f"pruned fine-tune done, {(time.perf_counter() - start) / 60:.1f} min\n", flush=True)

    models = [("dense", net), ("pruned", pruned),
              ("per tensor", encoded(pruned, masks, args.digits, None)),
              ("per channel", encoded(pruned, masks, args.digits, 0))]
    print(f"{'model':<17}{'fp32':>8}{'fp16':>8}{'non-finite fp16':>17}  fp16 on the first {len(test16.y)} images")
    for label, m in models:
        a32, _ = accuracy(m, test, False)
        a16, bad16 = accuracy(m, test16, True)
        print(f"{label:<17}{a32:>7.2f}%{a16:>7.2f}%{bad16:>16.1f}%", flush=True)
    print(f"\ntotal {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
