#!/usr/bin/env python3
"""phi_prune's fine-tune and evaluation with amp on a CUDA device: fp16 autocast and GradScaler, which no CPU run reaches.

Run: PHI_REPO=<checkout of ezexe/phi> python -m pytest experiments/phi-suite/test_core_cuda.py
with a Python whose torch is a CUDA build, on a machine with an NVIDIA GPU; without CUDA the test is skipped.
PHI_REPO defaults to the folder `phi` beside `experiments`.
Its Python project, the folder `lbiza` inside the checkout, goes first on sys.path, so its phi_prune is the one tested.
An older checkout keeps the project in its folder `phi` or at the top, and the test reads it there.
The test left ezexe/phi's tests/test_core.py because every machine that runs that suite has a CPU build of torch.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PHI_REPO = Path(os.environ.get("PHI_REPO", Path(__file__).resolve().parents[2] / "phi"))
PHI_PROJECT = next((PHI_REPO / d for d in ("lbiza", "phi") if (PHI_REPO / d / "pyproject.toml").is_file()), PHI_REPO)
sys.path.insert(0, str(PHI_PROJECT))

from phi_prune.api import check, finetune, prune  # noqa: E402


def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast runs on CUDA only")
def test_finetune_amp_on_cuda():
    model = make_model().cuda()
    pruned, masks = prune(model)
    batches = [(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,))) for _ in range(2)]
    result = finetune(pruned, batches, epochs=1, masks=masks, val_loader=batches,
                      verbose=False, amp=True)
    assert result["best_val_acc"] is not None
    assert check(pruned, masks)["_summary"]["all_zeros_enforced"]
