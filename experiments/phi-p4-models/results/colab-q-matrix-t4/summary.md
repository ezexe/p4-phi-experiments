# p4 Q matrix on Colab: q-matrix-t4

Pinned commits: `{"phi": "d039f00225c5749220045e14ea4bb8f0fa32880b", "p4": "0157bbda3f6349494953605e4e995c61e942f08a", "store": "bdd69831bad870d663d0b39fae48f36c4f140385"}`
Trained on: Tesla T4, cpu

| k | seeds | one large | Zeckendorf ensemble | beats it every seed | φ-routed | 2:4-routed | route hit (φ / 2:4) | chance | pretrained on | one large's steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 0.185 | 0.172 | no | 0.161 | 0.152 | 0.77 / 0.75 | 0.50 | cuda Tesla T4 | 35870, 36072 |
| 3 | 2 | 0.191 | 0.219 | no | 0.179 | 0.160 | 0.67 / 0.54 | 0.33 | cuda Tesla T4 | 36120, 34976 |
| 4 | 2 | 0.196 | 0.186 | no | 0.152 | 0.196 | 0.40 / 0.48 | 0.25 | cuda Tesla T4 | 35331, 36677 |

Tree (k = 4) held out: medium 0.158, high 0.158.

| .pt source | with | without | kept |
|---|---|---|---|
| ranker | 0.3751 | 0.3788 | no |
| medium | 0.3757 | 0.3788 | no |
| high | 0.3652 | 0.3788 | no |

Teacher claude-fable-5-1: 24 pools compared, agreement with the store's labels precision 0.5075, recall 0.6048, f1 0.5081.

| trained on | scored against the store's labels | scored against the teacher's |
|---|---|---|
| the store's labels | 0.165 | 0.1652 |
| the teacher's labels | 0.1606 | 0.2019 |

## Verdicts

- k = 4 is the only node count where the Zeckendorf ensemble beats one large model on every seed: **does not hold**
- the matrix route finds the best node above chance at every k: **holds**
- φ routing ranks better than 2:4 routing at every k: **does not hold**
- the tree at high ranks better than at medium (held out): **does not hold**
- the JSON ranker keeps the tree's .pt weight at high: **does not hold**
- training on the teacher's labels predicts the store's own labels at least as well as training on them: **does not hold**
