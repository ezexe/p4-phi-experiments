"""Tests for scripts/transformer_check.py: its verdicts, and whole runs on a stub text with a model of width 8."""

import importlib.util
import json
import locale
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "transformer_check.py"
SIGMA_CRITERIA = ("K1 phi vs 2:4 at 50%", "K2 phi-low vs 1:4 at 25%", "K3 phi-reorder vs top-50")


@pytest.fixture
def check(monkeypatch):
    """The script as a module, reading a 400-character stub text into a one-layer model, on one thread; the thread
    count and torch's global random state, which main seeds, are put back afterwards."""
    spec = importlib.util.spec_from_file_location("transformer_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, value in (("CTX", 8), ("D", 8), ("LAYERS", 1), ("HEADS", 2), ("BATCH", 2), ("EVAL_BATCHES", 1)):
        monkeypatch.setattr(mod, name, value)
    text = torch.randint(0, len(mod.CHARS), (400,), generator=torch.Generator().manual_seed(0))
    monkeypatch.setattr(mod, "read_files", lambda keep: text)
    monkeypatch.setattr(mod, "load_text", lambda: (text, text, len(mod.CHARS)))
    threads, set_threads, state = torch.get_num_threads(), torch.set_num_threads, torch.get_rng_state()
    set_threads(1)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    yield mod
    set_threads(threads)
    torch.set_rng_state(state)


def run(check, monkeypatch, tmp_path, *args):
    out = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv",
                        ["transformer_check.py", "--steps", "1", "--finetune", "1", "--out", str(out), *args])
    check.main()
    return out.read_text()


def strict_json(text):
    """json.loads that refuses NaN and Infinity, which JSON does not have."""
    def refuse(token):
        raise ValueError(f"{token} is not JSON")
    return json.loads(text, parse_constant=refuse)


def seeds(n, mean):
    return [mean + 0.01 * i for i in range(n)]


def verdicts(check, n_seeds):
    """low_verdicts on fine-tuned losses of n_seeds seeds, φ worse than 2:4 by 0.02."""
    losses = {"phi": 2.02, "2:4": 2.00, "phi-reorder": 2.00, "top-50": 2.00, "phi-low-25": 2.10, "1:4": 2.10}
    return check.low_verdicts(lambda m: seeds(n_seeds, losses[m]), lambda L: (1.0, 1.1))


def test_one_seed_decides_no_sigma_criterion(check):
    """With one seed there is no σ, so K1 to K3 are undecided, neither killed nor surviving."""
    v = verdicts(check, 1)
    for name in SIGMA_CRITERIA:
        _, sigma, survives = v[name]
        assert sigma is None and survives is None


def test_two_seeds_decide_every_criterion(check):
    v = verdicts(check, 2)
    assert all(survives in (True, False) for _, _, survives in v.values())
    assert not v["K1 phi vs 2:4 at 50%"][2]  # killed: 0.02 worse, against a σ of 0.0071


def test_one_seed_run_writes_json(check, monkeypatch, tmp_path):
    """A whole run of --part low with one seed writes a file a strict JSON parser reads."""
    out = strict_json(run(check, monkeypatch, tmp_path, "--seeds", "1"))
    for name in SIGMA_CRITERIA:
        assert out["verdicts"][name]["sigma"] is None and out["verdicts"][name]["survives"] is None
    assert out["run"]["chars"] == {"train": 400, "val": 400}


def test_write_json_writes_null_for_nan_and_infinity(check, tmp_path):
    """Every NaN and infinity becomes null, in every container the results hold, and nothing else changes."""
    results = {"verdicts": {"K1": {"difference": 0.25, "sigma": float("nan")}},
               "seeds": [{"gaps": {0.1: {"fibonacci": float("inf"), "elias-gamma": -float("inf")}},
                          "bpc": np.float32("nan"), "dense": np.float64(1.5), "n": 3}]}
    out = tmp_path / "results.json"
    check.write_json(results, out)
    assert strict_json(out.read_text()) == {
        "verdicts": {"K1": {"difference": 0.25, "sigma": None}},
        "seeds": [{"gaps": {"0.1": {"fibonacci": None, "elias-gamma": None}}, "bpc": None, "dense": 1.5, "n": 3}]}


def test_run_records_what_it_trained_on(check, monkeypatch, tmp_path):
    """A run's numbers depend on the Python whose standard library is its text, so its file says which Python, on
    which platform, read in which encoding, with which torch, and how many characters each split held."""
    info = strict_json(run(check, monkeypatch, tmp_path, "--seeds", "1", "--part", "delta"))["run"]
    assert info["python"] == platform.python_version() and info["torch"] == torch.__version__
    assert info["platform"] == platform.platform() and info["encoding"] == locale.getpreferredencoding(False)
    assert info["chars"] == {"pretrain": 400, "fine_tune": 360, "held_out": 40, "val": 400}
