"""Tests for scripts/make_register_notebook.py: the cells that load weights and images, run against a
stand-in torchvision, read from the generator's source without running it."""

import ast
import http.client
import sys
import types
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "make_register_notebook.py"


def code_cells():
    """The source of each code cell the generator writes, in order, from its code(...) calls."""
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "code"]
    return [c.args[0].value for c in sorted(calls, key=lambda c: c.lineno)]


def cell(marker):
    return next(c for c in code_cells() if marker in c)


@pytest.fixture(autouse=True)
def keep_random_state():
    """The weights cell seeds torch's global generator; put it back for the tests that follow."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def stand_in(monkeypatch, resnet18=None, cifar10=None):
    """A torchvision module with only what the cells call, put where `import torchvision` finds it."""
    weights = types.SimpleNamespace(IMAGENET1K_V1="IMAGENET1K_V1 weights", DEFAULT="DEFAULT weights")
    tv = types.ModuleType("torchvision")
    tv.models = types.SimpleNamespace(resnet18=resnet18, ResNet18_Weights=weights)
    tv.datasets = types.SimpleNamespace(CIFAR10=cifar10)
    monkeypatch.setitem(sys.modules, "torchvision", tv)
    return tv


def raising(error):
    def call(*args, **kwargs):
        raise error
    return call


def run(source, namespace=None):
    namespace = {} if namespace is None else namespace
    exec(compile(source, "<notebook cell>", "exec"), namespace)
    return namespace


DOWNLOAD_FAILURES = [urllib.error.URLError("no network"), http.client.IncompleteRead(b"", 8192),
                     RuntimeError("invalid hash value")]


@pytest.mark.parametrize("error", DOWNLOAD_FAILURES, ids=lambda e: type(e).__name__)
def test_weights_cell_falls_back_when_the_download_fails(monkeypatch, error):
    stand_in(monkeypatch, resnet18=raising(error))
    ns = run(cell("resnet18"))
    assert ns["model"] is None and ns["SOURCE"].startswith("synthetic Laplace weights")


def test_weights_cell_falls_back_without_torchvision(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchvision", None)
    ns = run(cell("resnet18"))
    assert ns["model"] is None
    assert ns["SOURCE"] == "synthetic Laplace weights (ModuleNotFoundError: no pretrained weights)"


def test_weights_cell_stops_on_any_other_error(monkeypatch):
    stand_in(monkeypatch, resnet18=raising(TypeError("a fault in the code, not a missing download")))
    with pytest.raises(TypeError):
        run(cell("resnet18"))


def tiny_model():
    """A model whose layer3.0.conv1 is a 3-to-4-channel convolution, enough for the activations cell."""
    block = torch.nn.Module()
    block.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
    model = torch.nn.Module()
    model.layer3 = torch.nn.ModuleList([block])
    model.forward = lambda x: model.layer3[0].conv1(x)
    return model


def activations_namespace(tv):
    return {"np": np, "torch": torch, "F": F, "torchvision": tv, "model": tiny_model(), "LAYER": "layer3.0.conv1",
            "FAN_IN": 27, "SEED": 0}


@pytest.mark.parametrize("error", DOWNLOAD_FAILURES, ids=lambda e: type(e).__name__)
def test_activations_cell_falls_back_when_cifar10_fails(monkeypatch, error):
    tv = stand_in(monkeypatch, cifar10=raising(error))
    ns = run(cell("CIFAR10"), activations_namespace(tv))
    assert ns["ACT_SOURCE"].startswith("random images") and len(ns["acts"]) == ns["N_PATCHES"]


def test_activations_cell_stops_on_any_other_error(monkeypatch):
    tv = stand_in(monkeypatch, cifar10=raising(TypeError("a fault in the code, not a missing download")))
    with pytest.raises(TypeError):
        run(cell("CIFAR10"), activations_namespace(tv))


def test_weights_cell_asks_for_the_weights_it_describes(monkeypatch):
    """The cell names ResNet18_Weights.IMAGENET1K_V1, the ImageNet weights its text describes, where the alias
    DEFAULT is one torchvision's documentation says may change across versions."""
    asked = []

    def record(weights=None):
        asked.append(weights)
        raise urllib.error.URLError("no network")
    tv = stand_in(monkeypatch, resnet18=record)
    run(cell("resnet18"))
    assert asked == [tv.models.ResNet18_Weights.IMAGENET1K_V1]


def test_synthetic_weights_do_not_depend_on_torchvision(monkeypatch):
    """The fallback draws the same synthetic weights whether torchvision is missing or present with its download
    out of reach; building ResNet-18 before the download fails draws from torch's global generator."""
    monkeypatch.setitem(sys.modules, "torchvision", None)
    without = run(cell("resnet18"))["weight"]

    def build_then_fail(weights=None):
        torch.nn.Conv2d(64, 64, 3)
        raise urllib.error.URLError("no network")
    stand_in(monkeypatch, resnet18=build_then_fail)
    present = run(cell("resnet18"))["weight"]
    assert torch.equal(without, present)


def test_synthetic_activations_do_not_depend_on_the_global_generator():
    """With the synthetic weights the activations are drawn the same, whatever drew from torch's generator before."""
    acts = []
    for draws in (0, 1000):
        torch.rand(draws)
        acts.append(run(cell("CIFAR10"), {**activations_namespace(None), "model": None})["acts"])
    assert all(np.array_equal(a, b) for a, b in zip(*acts))
