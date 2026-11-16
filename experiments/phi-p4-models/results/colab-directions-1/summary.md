# The p4 models' directions: directions-1

Pinned commits: `{"phi": "fd61ea629bfc5a12c3b217e40829869c7898b0de", "p4": "04eee31281f5636b2abdd6d84809d80ecafb22fb", "store": "e5df5faaba85f0fafff1dcab6120a52d8bef429c"}`

## Verdicts

- D1 the JSON ranker ranks better than plain word overlap (store's labels): **holds**
- D2 a shorter pretraining budget ranks better than the longest: **holds**
- D2 some pretraining budget's .pt raises the JSON ranker on every seed: **does not hold**
- D4 the fable labels are more learnable than the store's (trained and scored on themselves, best budget): **holds**
- D4 the opus labels are more learnable than the store's (trained and scored on themselves, best budget): **holds**
- D4 training on the fable labels predicts the store's labels at least as well as training on them: **holds**
- D4 training on the opus labels predicts the store's labels at least as well as training on them: **does not hold**
- D4 the two teachers agree with each other more than either agrees with the store's labels: **holds**
- D5 the φ structure matters: **not measured**

## Implied settings

- **pool-effort:** low or unnamed (the JSON ranker alone)
- **pretraining budget:** 1 min (best held-out ranking)
- **labels:** train on the teachers' labels: they are more learnable than the store's

## D2 by pretraining budget (mean over seeds)

| budget (min) | steps | best dev step | store->store | store->fable | store->opus | fable->store | fable->fable | fable->opus | opus->store | opus->fable | opus->opus | JSON + .pt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 94 | 40 | 0.1825 | 0.1961 | 0.1893 | 0.1992 | 0.2172 | 0.2189 | 0.1649 | 0.2109 | 0.222 | 0.3728 |
| 1 | 176 | 38 | 0.186 | 0.1925 | 0.1868 | 0.1962 | 0.2172 | 0.2204 | 0.1701 | 0.2091 | 0.2205 | 0.3728 |
| 2 | 456 | 44 | 0.1777 | 0.1915 | 0.1893 | 0.1978 | 0.2206 | 0.2293 | 0.1652 | 0.2148 | 0.2222 | 0.374 |
| 4 | 991 | 47 | 0.1784 | 0.1931 | 0.186 | 0.1965 | 0.2187 | 0.2293 | 0.1652 | 0.2109 | 0.2192 | 0.374 |
| 6 | 1,640 | 47 | 0.1784 | 0.1931 | 0.186 | 0.1965 | 0.2187 | 0.2293 | 0.1652 | 0.2109 | 0.2192 | 0.374 |

JSON ranker alone: 0.3788 (word overlap 0.3129); label-set agreement (F1): {'store~fable': 0.5322, 'store~opus': 0.5024, 'fable~opus': 0.7538}
