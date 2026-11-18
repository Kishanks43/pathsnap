# Experiment log

The brief requires you to "retain experiment logs, random seeds, configurations and
results" and to support every claim in the report with evidence. This file is the
human-readable half of that record; `experiments/results/*.json` is the machine half.

**Rule:** if it is not in here, it did not happen, and it does not go in the report.

---

## How to use this file

- Append a block **before** you run something, fill in the numbers after.
- One block per meaningful run. Not per keystroke, not per notebook cell.
- Record what you expected as well as what happened. "Expected X, got Y" is the
  most useful sentence in an experiment log, and it is exactly what the viva
  questions probe.
- Do not delete failed runs. A documented failure that led to a better decision is
  worth marks; a suspiciously clean history invites questions.

---

## Template — copy this block

```markdown
### [MM-DD HH:MM] <owner> — <short title>

**Question:** what am I trying to find out?

**Setup**
- Data partitions: train (80) / student_test (10) — never evaluator-test
- CV: <n> folds, GroupKFold by trip_id, seed <seed>
- Model / config: <name>, <key hyperparameters>
- Features: schema v<n>, <n_features> columns
- Command: `python ...`

**Expected:** <what you thought would happen and why>

**Result**
| metric | value |
|---|---|
| CV macro F1 (mean ± std) | |
| student-test top-1 accuracy | |
| student-test macro F1 | |
| log loss | |
| route continuity | |
| train time | |
| inference ms/obs | |

**Artifacts:** `experiments/results/<file>`

**What this changes:** decision taken, or what to try next.

**Failure mode / caveat:** anything that weakens confidence in the result.
```

---

## Run log

### [P1] Dataset validation — `scripts/check_data.py`

**Question:** is the local copy complete and internally consistent?

**Setup:** train + student_test, all four file types per trip.

**Result**
```
train        : 80 trips, 208,417 observations, 1,933,485 nodes, 3,876,904 arcs
student_test : 10 trips,  19,160 observations,    90,689 nodes,   185,041 arcs
observations/trip: min 361, median 1,529, max 19,313
median dt 1.00 s, worst gap 153,064 s (train), 210 s (student_test)
lon range -100.4 .. 137.2, lat -34.6 .. 61.3
no validation issues
```

**What this changes:** trips are in different cities → projection must be per-trip;
worst gap is ~42 h → decoder must not assume dt≈1 s.

---

### [P1] Reference-route coverage diagnostic

**Question:** do the supplied `.route` files actually describe the whole trip?

**Setup:** for all 80 train and 10 test trips, compare `sum(route arc lengths)`
against the path length measured from the GPS track.

**Expected:** ratio near 1 (route slightly shorter than GPS path, because GPS
noise inflates measured travel distance).

**Result:** median 0.84 (train). Ten train trips below 0.50; two extreme:
`00000033` ratio 0.13, `00000054` ratio 0.19. Consecutive arcs are 99–100%
node-connected, so the routes are chains, just truncated.

**What this changes:** per-observation labels are unreliable on those trips →
introduced the label-trust filter (residual ≤ 25 m). Recorded as Fact 6.

---

### [P1] Candidate radius sweep

**Question:** which radius/candidate budget minimises miss rate per unit of
compute?

**Setup:** all 80 train trips, `max_candidates` 10, miss rate computed on trusted
labels only.

**Result**
```
radius 50  m, k=10 : miss 1.86%   rows 1,083,514
radius 75  m, k=10 : miss 1.44%   rows 1,313,622
radius 100 m, k=10 : miss 1.22%   rows 1,453,943
radius 150 m, k=10 : miss 0.85%   rows 1,708,157
radius 100 m, k= 6 : miss 1.75%   rows 1,010,267
```

**What this changes:** chose 100 m / k=10 — the knee. Recorded as Fact 7.

---

### [P1] Prepared dataset build

**Question:** does the full pipeline run end to end at usable cost?

**Setup:** `python scripts/prepare_dataset.py --force`, 90 trips.

**Result**
```
train        : 80 trips, 1,453,943 candidate rows, 6.98 cand/obs,
               miss mean 3.41% (median 0.73%), trusted 93.0%
student_test : 10 trips,   148,752 candidate rows, 7.76 cand/obs,
               miss mean 1.17%, trusted 98.5%
X: float64 (1,453,943 x 19), positive rate 12.7%
load_partition("train") takes 3.6 s, 221 MB resident
```

**Artifacts:** `experiments/results/candidate_quality.json`,
`artifacts/feature_schema.json`, `data/processed/*/*.npz`

**Failure mode / caveat:** the 3.41% headline miss rate is inflated by the two
route-truncated trips. On trusted labels it is 1.22%. Both must be quoted.

---

## Decisions log

Short, dated entries for choices that are hard to reverse. This is the section the
report's methodology quietly draws from.

| Date | Decision | Alternatives considered | Why |
|---|---|---|---|
| | Project to a local metric frame instead of using degrees | Euclidean in degrees, full geodesic (pyproj) | Degrees are meaningless for distance; local projection is sub-metre at city scale and much faster in inner loops |
| | Anchor the projection per trip | One global projection for the dataset | Trips are in different countries (lon -100..137); a global frame distorts most trips |
| | Grid index over arc bounding boxes + separate long-arc list | Brute force obs×arcs, KD-tree on midpoints only, bbox rasterisation | Brute force is ~4e9 segment tests per trip; midpoint KD-trees miss long arcs; bbox binning is exact and long arcs are few |
| | Candidate generation by point-to-segment distance | Endpoint distance | Endpoint distance unfairly penalises long arcs |
| | Radius 100 m, max 10 candidates | 50/75/150 m, k=6 | Measured knee: 1.22% miss on trusted labels for 1.45M rows |
| | Binary per-candidate labels | Multiclass over arc ids | Multiclass over a global per-trip arc namespace does not transfer across trips; binary scores feed the Viterbi emission directly |
| | Label trust = residual ≤ 25 m | Drop whole trips; no filter | Filters the specific observations whose reference route does not cover them, keeps 93% of rows, and applies identically to train and test |
| | Primary selection metric: CV macro F1 | Top-1 accuracy | Accuracy hides rare-arc performance; brief asks for macro F1 |
| | | | |

---

## Things we tried that did not work

Keep this section. It is the difference between "we experimented" and "we ran one
model and wrote a report". Each entry: what, why it seemed reasonable, what
actually broke.

- _e.g. Unrestricted Random Forest (max_depth=None): CV macro F1 0.xx but train
  accuracy 0.99 — memorised per-trip geometry. Fixed with min_samples_leaf=5._
- _e.g. Raw degree coordinates as features: distances are meaningless, model
  learned nothing. Superseded by metre-based features in schema v1._
