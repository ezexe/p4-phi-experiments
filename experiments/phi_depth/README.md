# phi_depth

`phi_rank.tree` is the p4-model skill's node tree: K small rankers at the first K Zeckendorf store addresses (p4 = 1, .claude = 0; start and end with 1, no `11`): `1`, `101`, `1001`, `10001`, `10101`, `100001`, …
A node's parent is its address with the last run of 0s and the final 1 dropped, so the tree grows recursively; each node pretrains on its shard of the store's text (the oldest shard deepest), starts from its children's mean, fine-tunes on the store's labelled pools, and keeps half its feature rows under `phi_mask`: one bit per row, 1 kept and 0 pruned, never two adjacent 1s.
The skill ships K = 4; it answers at `medium` with the one node the mask routes to and at `high` with every node's scores z-scored and averaged.

```bash
.venv/bin/python -I experiments/phi_depth/phi_depth.py --store <a .claude/p4 store> [--store <another>]   # K = 1,2,3,5,8,13
.venv/bin/python -I experiments/phi_depth/phi_open.py --store <a .claude/p4 store> --seeds 0,1,2
.venv/bin/python -I experiments/phi_depth/test_phi_depth.py                                               # offline, ~5 s
```

Free, CPU; nothing is written into the store.
Results land in `out/phi_depth.json` and `out/phi_open.json` (a rerun resumes per arm and seed).

- **`phi_depth.py`** sweeps K at a fixed total pretraining budget (1.5 min per tree, split over its nodes; 10 fine-tuning epochs) and reports 4-fold held-out R-precision at `medium`, at `high`, for each node alone, for an oracle that picks the best node per pool (the ceiling a perfect route reaches), and for BM25 on the same rows; then the end nodes' bits from a tree fit on every pool: each mask's kept share and adjacent 1s, the Jaccard of a child's kept rows with its parent's (1/3 for two unrelated half masks), the share of kept rows no task in the store ever lights, and where the route sends the pools.
- **`phi_open.py`** tests what `phi/.docs/p4/nested-store.md` leaves open, at K = 4 and K = 1 over several seeds: pretraining against none; the mask as the route (`phi_mask` against the heaviest half of the rows, a random half, 2:4 and no mask, all from one fine-tune per fold); `phi_mask` over rows sorted along their first principal component, so "no two adjacent kept" means "never two near-duplicate features"; and the single realtime stream of every node's rounds, `[address codeword][Fibonacci-coded header][payload]`, built, decoded and compared with the per-node `.phid` files.

## Findings, 2026-11-09

Two stores: a 97-pool store branch with git history (W) and this project's own 9-pool store (S, no git history, so its shards are cut by name rather than age).
S's spreads (±0.02 to ±0.14 between seeds) are as large as most of its differences; W is the one to read.

### Depth (one seed per K)

| W | nodes' rows | medium | high | oracle | best node | BM25 |
|---|---|---|---|---|---|---|
| K = 1 | 32,768 | 0.500 | 0.500 | 0.500 | 0.500 | 0.211 |
| K = 2 | 16,384 | 0.437 | 0.467 | 0.554 | 0.484 | 0.211 |
| K = 3 | 8,192 | 0.523 | 0.595 | 0.657 | 0.528 | 0.211 |
| K = 5 | 8,192 | 0.467 | 0.588 | 0.716 | 0.503 | 0.211 |
| K = 8 | 4,096 | 0.429 | **0.602** | 0.821 | 0.481 | 0.211 |
| K = 13 | 2,048 | 0.440 | 0.596 | **0.850** | 0.478 | 0.211 |

1. **Every arm beats BM25 by more than twice** on these pools (which always hold their labelled positives; p4-recall's own candidate stage often misses them, see `../recall_levers`).
2. **Averaging the nodes (`high`) gains about 0.10 over one ranker and plateaus from K = 3** (0.595 to 0.602).
3. **The ceiling keeps climbing with depth, the route does not follow it.** The oracle rises from 0.50 to 0.85, so deeper nodes do specialise; but `medium`, the one routed node, stays at or below the single ranker at every K but K = 3 (0.523), and the best node alone beats it by at most 0.03 (0.528 at K = 3). On S the same shape: oracle 0.65 to 0.89, `high` 0.65 to 0.79.
4. **The bits.** No mask ever keeps two adjacent rows (0 at every node of every tree). A child keeps about half the rows its parent keeps (Jaccard 0.47 to 0.53 from K = 3, against 1/3 for unrelated masks; 0.90 at K = 2, where the root starts as its one child). At K = 1, half the rows the mask keeps are never lit by any task in W (0.498); at K ≥ 8 none are (0.002, 0.000): the small tables use every row they keep. The route spreads the pools over the nodes (no node takes more than a third at K = 13), though one of three nodes gets none at K = 3.

Note the row counts: K = 3 and K = 5 both get 8,192 rows a node (`2^(15 − round(log2 K))`), so K = 5 holds 1.25× the parameters of one ranker and K = 13 0.81×.

### The open questions (K = 4, three seeds on W, five on S)

| W, held-out R-precision | high (all 4 nodes) | medium (routed) | one ranker (K = 1) |
|---|---|---|---|
| no mask | **0.633 ± 0.029** | 0.557 ± 0.019 | **0.567 ± 0.026** |
| `phi_mask` (as shipped) | 0.586 ± 0.022 | 0.485 ± 0.008 | 0.507 ± 0.039 |
| `phi_mask`, rows in PC order | 0.585 ± 0.011 | 0.478 ± 0.012 | 0.502 ± 0.026 |
| heaviest half | 0.595 ± 0.042 | 0.479 ± 0.022 | 0.514 ± 0.044 |
| random half | 0.588 ± 0.023 | 0.470 ± 0.035 | 0.483 ± 0.015 |
| 2:4 | 0.586 ± 0.035 | 0.500 ± 0.032 | 0.505 ± 0.051 |

5. **The mask costs 0.05 to 0.06, and its pattern does not matter.** Every half mask lands within its spread of every other (φ, φ in PC order, heaviest, random, 2:4) and all sit below no mask, on the tree and on one ranker. What costs is halving the rows, not which half.
6. **φ over ordered features means nothing more than over hashed ones.** With rows sorted so neighbours hold similar weights, `phi_mask` scores what it scores on the hashed order (0.585 against 0.586; on S, 0.663 against 0.731 for one ranker). The note's open question closes negative for this ranker.
7. **The route is at chance.** The routed node is the best node for 47% of W's held-out pools under `phi_mask`, where a random pick would hit 50% (ties counted); 52% against 54% unmasked. On S, 40 to 44% against 37 to 44%.
8. **Pretraining on the store's text lowers ranking.** One ranker: 0.567 without, 0.519 with a minute of pretraining; the tree at `high`: 0.633 and 0.621 (on S: 0.749 and 0.686). The Colab replications found the best dev ranking at step 0; so do these.
9. **The single stream works and is a little smaller.** Four routed rounds as one stream decode back exactly (every address, header and delta). It costs 12.38 bits per kept weight against 12.99 for the per-node `.phid` files on W (4.7% less; 6.1% on S): the stream's framing is 616 bits a round against about 7,650 bits of JSON header. The payload itself (12.32 bits per kept weight) is the same code in both.

### What the tree would ship on this evidence

Four unmasked nodes answering at `high`, with no pretraining: 0.633 on W against 0.586 for the tree as shipped (masked) at `high` and 0.567 for one unmasked ranker, at the same total parameters.
The `medium` route, the mask and the pretraining each cost ranking here; the Fibonacci stream and the gap codes earn their place on the wire, not in the model.
These are proposals for `phi_rank.tree` in [ezexe/phi](https://github.com/ezexe/phi), measured on two stores of one kind; a third store would show whether they carry.

### Checked along the way

- `nested-store.md`'s structural claims hold when re-run: address counts by length are 1, 0, 1, 1, 2, 3, 5, 8, 13, … to 144 at 14 digits; the 144 addresses of up to 12 digits are 144 distinct integers starting 1, 4, 6, 9, 12, 14, 17, 19, 22; a stream of them with one `1` appended each parses with no header; the two child subtrees step towards φ (1.615 at 10 digits, 1.618 at 14); `stream_logic.py`'s growth result reproduces on three seeds.
- `phi/.docs/phi/Q.md`'s "Fibonacci coding outperforms Elias delta for integers up to approximately 514,228" does not hold as stated: the last integer the Fibonacci code is strictly shorter on is 75,024, the last tie 317,810, and averaged over 1..N Fibonacci stops winning near N = 100,000.
- Fibonacci gap codes against a fitted Golomb-Rice (untested in `phi/.docs/register.md`), on positions of a real ranker update: 3.99 against 8.56 bits a position at 10% kept, 8.21 against 11.87 at 1%; Elias-gamma 3.91 and 9.81. The changed entries come in whole 64-wide rows, so most gaps are 1, which a geometric code fits badly.
