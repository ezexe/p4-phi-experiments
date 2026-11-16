# Neural pathways on Colab: pathways-1

Pinned commits: `{"phi": "be6e94fdb41a7dabc82013136ed8d5182dd9606a", "p4": "04eee31281f5636b2abdd6d84809d80ecafb22fb", "store": "e5df5faaba85f0fafff1dcab6120a52d8bef429c"}`

Held-out r-precision, mean over seeds (dev pools excluded):

| k | seeds | one large | Zeckendorf tree | binary tree | flat | Zeck root | binary root |
|---|---|---|---|---|---|---|---|
| 4 | 3 | 0.1837 | 0.1848 | 0.1931 | 0.1903 | 0.1764 | 0.193 |

Routing by each mask (routed r-precision / how often the routed node is the best node):

| k | φ | 2:4 | random half | heaviest half |
|---|---|---|---|---|
| 4 | 0.1622 / 0.5 | 0.1541 / 0.45 | 0.1588 / 0.5 | 0.1841 / 0.6333 |

Early stopping, the one large model (best dev score at step, of steps run):

- k = 4, seed 0: best 0.1375 at step 0 of 1,201
- k = 4, seed 1: best 0.04167 at step 68 of 1,206
- k = 4, seed 2: best 0.0375 at step 224 of 1,217

## Verdicts

- the Zeckendorf directory tree beats the binary directory tree (ensemble, every seed): **does not hold**
- the Zeckendorf tree's root beats the binary tree's root (every seed): **does not hold**
- the Zeckendorf tree beats flat independent nodes (ensemble, every seed): **does not hold**
- the Zeckendorf tree beats one large model (ensemble, every seed): **does not hold**
- the φ mask routes to the best node more often than the heaviest-rows mask (every seed): **does not hold**
- φ routing ranks better than routing by a random half of the rows (every seed): **does not hold**
- φ routing ranks better than 2:4 routing (every seed): **holds**
- the one large model's best dev ranking comes before its pretraining budget ends (every seed): **holds**
