# phi-register

phi's register model, measured: the Zeckendorf register of `phi_prune.register` against a binary register, the dot products of one pruned convolution layer in both, and a pre-registered check of φ-pruning and of quantized weights written in the Zeckendorf representation, on a small transformer.
These files moved here from ezexe/phi, which keeps the package, its fast tests and its pages.

## Run

phi is the package `phi-prune` of ezexe/phi: the project's `.venv` holds it installed from the checkout with `pip install -e phi/lbiza`, and elsewhere `pip install "git+https://github.com/ezexe/phi.git#subdirectory=lbiza"` installs it.
Each command runs from the repository's root, and each file under `results/` is what its script printed or wrote.

```bash
.venv/bin/python -I experiments/phi-register/scripts/growth_rates.py > experiments/phi-register/results/register/growth_rates.txt                  # the standard library alone
.venv/bin/python -I experiments/phi-register/scripts/mask_choice.py > experiments/phi-register/results/register/mask_choice.txt                    # the standard library alone
.venv/bin/python -I experiments/phi-register/scripts/digit_counts_uniform.py > experiments/phi-register/results/register/digit_counts_uniform.txt  # phi
.venv/bin/python -I experiments/phi-register/scripts/register_measure.py > experiments/phi-register/results/register/register_measure.txt          # phi
.venv/bin/python -I experiments/phi-register/scripts/transformer_check.py --part low --out experiments/phi-register/results/transformer-check/low.json   # phi; --part high and --part delta the same way
.venv/bin/python -I experiments/phi-register/scripts/make_register_notebook.py                                                                      # phi and nbformat
.venv/bin/jupyter nbconvert --to notebook --execute --inplace experiments/phi-register/notebooks/register_settling.ipynb                             # phi, nbconvert, ipykernel, and torchvision for the pretrained weights
.venv/bin/python -I -m pytest experiments/phi-register/tests                                                                                         # phi and pytest, offline, a few seconds
```

`transformer_check.py` trains on the standard library of the Python that runs it, so a run on another Python gives other figures.
Each file under `results/transformer-check/` is the record of one run; the three there were made before the script began to write which Python, platform, encoding, torch and characters a run used.
The saved notebook holds one run made without torchvision, and its closing cell, which the generator writes, quotes that run.
