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

### [P1] Model 1 — Logistic Regression, feature ablation

**Question:** how much of the candidate-scoring signal is geometry alone, and what
does a fitted linear model add over the nearest-candidate rule?

**Setup**
- Partitions: train (80 trips, 1,352,108 trusted rows), student_test (10 trips, 146,499 trusted rows)
- CV: 5-fold GroupKFold by trip, seed 42. Temporary splitter until P2's `cv.py` lands.
- Model: StandardScaler + LogisticRegression(C=1.0, max_iter=2000), fitted per fold
- Command: `python scripts/run_model1.py`
- Full log: `experiments/logs/model1_20261118_122042.log`

**Expected:** the sequential block (previous observation, connectivity, speed) should
add signal beyond geometry. A geometry-only linear model cannot do better than the
nearest-candidate rule on its own, since its score is monotone in distance.

**Result** (observation level, trusted labels)

| config | CV macro F1 (mean ± std) | CV top-1 | student-test top-1 | student-test macro F1 | student-test log loss |
|---|---|---|---|---|---|
| nearest candidate (no model) | — | — | 0.5581 | 0.4156 | N/A (raw distances, not probabilities) |
| Logistic, geometry only (10 cols) | 0.3863 ± 0.0420 | 0.5756 | 0.5581 | 0.4156 | 0.8721 |
| **Logistic, all 19 cols** | **0.6444 ± 0.0290** | **0.7855** | **0.7938** | **0.6316** | **0.6351** |

Train time 3.2 s, inference 0.003 ms/obs, 0 convergence warnings.

**What this changes:** the sequential block is worth ~26 points of CV macro F1. Geometry
alone reproduces nearest-candidate exactly on student_test, which confirms the linear
model is not learning anything beyond distance from those columns. Model 1 is the
baseline every other model must beat.

**Failure mode / caveat:**
- Route continuity is N/A here; it needs the HMM-decoded route (P4).
- Miss rate on the student-test trusted rows is 1.53%. The full, unfiltered student-test
  figure in `candidate_quality.json` is 1.17%. They use different denominators; quote both
  with their definitions.
- Metrics are on trusted labels only. Untrusted observations (residual > 25 m) are not scored.

---

### [P2] KNN (Model 2) — k sweep, cost scaling, student-test

**Question:** which k gives the best observation-level CV macro F1, and what does
KNN inference actually cost as the reference set grows?

**Setup**
- Data partitions: train (80 trips, 1,352,108 trusted candidate rows) /
  student_test (10 trips, 146,499 trusted rows) — never evaluator-test
- CV: 5 folds, `GroupKFold` on trip index, seed 42 (`evaluation/cv.py`)
- Model: `KNNModel`, weights=distance, metric=euclidean, algorithm=brute,
  StandardScaler inside the pipeline, fitted on training folds only
- Training cap: 100,000 rows per fold (label-stratified). Validation folds are
  scored in full. Rationale: one full fold is a ~270k x 1.16M distance matrix;
  the measured cost curve below is the evidence.
- k sweep: {3, 5, 7, 11}
- Features: schema v1, 19 columns
- Command: `python scripts/run_p2.py --skip-svm`

**Expected:** larger k smooths the vote and improves macro F1 at the cost of
coarser, less confident probabilities. Inference cost should grow roughly
linearly with the reference-set size.

**Result**

| metric | value |
|---|---|
| best k | 11 (of the pre-registered {3, 5, 7, 11}) |
| CV macro F1 (mean ± std) | 0.6064 ± 0.0260 |
| student-test top-1 accuracy | 0.7750 |
| student-test macro F1 | 0.5818 |
| log loss | N/A (infinite): 0.5898% of test candidate rows received exactly zero probability for their true label |
| route continuity | 0.9004 |
| train time | 0.03 s (final fit on 100k rows; KNN is lazy, all cost is inference) |
| inference ms/obs | 0.532 (at 100k reference rows) |

Cost scaling on a shared 10,000-row query set: 25k train -> 0.0201 ms/row,
50k -> 0.0421 ms/row, 100k -> 0.0840 ms/row. Doubling the reference set roughly
doubles inference time, which is the expected O(n_train) behaviour.

The zero-probability problem is systematic, not a test-set fluke: across the five
CV folds at k=11 it affects 0.42–1.53% of candidate rows (at k=3, 2.15–4.72%).
Larger k softens it, which is consistent with the vote becoming less coarse.

**Artifacts:** `experiments/results/model_knn.json`,
`experiments/results/cv_knn.json`, `experiments/results/knn_inference_cost.json`

**What this changes:** k=11 is selected. KNN beats a simple geometry prior on
top-1 accuracy (0.7750) but its test log loss is undefined because it assigns
zero probability to the truth on 0.59% of rows; the Viterbi emission term would
need explicit smoothing. Its inference cost is the only cost that grows with the
training set, so it is not the deployment candidate.

**Failure mode / caveat:** the recorded CV score is the best of four k values on
the same folds, so model selection biases it upwards; the fold std is not a
corrected estimate of that selection. The sweep had **not** plateaued at k=11
(0.5601 -> 0.5841 -> 0.5949 -> 0.6064), so k=11 is a boundary choice and larger
k was not tested; KNN's CV ceiling may be higher. KNN probabilities are
k-neighbour votes (resolution ~1/k), which is why the log-loss cell reads N/A.

---

### [P2] RBF SVM (Model 5) — Platt-scaling cost and student-test

**Question:** what accuracy does an RBF SVM reach on a defensible subsample, and
what does mandatory probability calibration cost?

**Setup**
- CV: 5 folds, `GroupKFold` on trip index, seed 42
- Model: `SVC(kernel=rbf, C=1.0, gamma=scale, probability=True)` with
  StandardScaler fitted inside each training fold
- Training cap: 30,000 rows per fold (label-stratified). Validation folds are
  scored in full. In this run a 30k fit took ~23–27 s and a whole CV fold
  (fit + scoring 270k rows) 78–104 s. 50k was not attempted; SVC cost grows
  superlinearly, so it would exceed the run budget.
- Platt cost measured by fitting identical SVMs with `probability=False` and
  `probability=True` on the same slice (`measure_platt_scaling_cost`)
- Features: schema v1, 19 columns
- Command: `python scripts/run_p2.py --skip-knn`

**Expected:** high accuracy per parameter, but the worst training and inference
cost of the five models; Platt scaling should add a multiple of the raw fit time.

**Result**

| metric | value |
|---|---|
| CV macro F1 (mean ± std) | 0.6431 ± 0.0271 |
| student-test top-1 accuracy | 0.8062 |
| student-test macro F1 | 0.6297 |
| log loss | 0.1357 |
| route continuity | 0.8997 |
| train time (Platt) | 23.46 s final fit on 30k; CV folds 78–104 s each |
| Platt overhead | 6.69× (3.99 s raw -> 26.69 s calibrated on the same 30k slice) |
| inference ms/obs | 1.730 (4,533 support vectors) |

CV fold detail: macro F1 0.6288, 0.6822, 0.6578, 0.6451, 0.6015 across five
trip-disjoint folds; log loss 0.1243–0.2022. No test row received exactly zero
probability, so the log loss is finite and directly comparable with the tree
models once they report.

**Artifacts:** `experiments/results/model_svm_rbf.json`,
`experiments/results/cv_svm_rbf.json`

**What this changes:** the SVM is the strongest of P2's two models on every
accuracy metric and the only one whose probabilities are usable as Viterbi
emissions without smoothing. It is also 3.3× slower per observation than KNN
(1.730 vs 0.532 ms/obs) and needed a 2.2% training subsample, so it is evidence
for "accuracy per parameter is good, cost is bad" rather than a deployment
choice.

**Failure mode / caveat:** the model is trained on 30,000 of 1.35M trusted
candidate rows (2.2%). The comparison is therefore not a full-data comparison
for this row; the cap and the actual row count are in `describe()` and the
record notes. This is reported, not hidden. `C=1.0`, `gamma="scale"` are sklearn
defaults and were not tuned (tuning belongs to P3 and only for the winner).

---

### [P2] Models 2 & 5 — calibration and cost comparison (report §3.1/§3.2 B–C)

**Question:** KNN and SVM have similar accuracy on paper. Which one produces
probabilities the Viterbi emission term can actually use, and what does each
cost at inference?

**Setup:** same 10 student_test trips and the same 146,499 trusted candidate rows
as the two model runs above. Metrics from `model_knn.json` / `model_svm_rbf.json`;
cost curve from `knn_inference_cost.json`.

**Result**

| | KNN (k=11) | RBF SVM |
|---|---|---|
| Top-1 accuracy | 0.7750 | 0.8062 |
| Macro F1 | 0.5818 | 0.6297 |
| Log loss | N/A — infinite (0.59% of rows given p=0 for the truth) | 0.1357 |
| Inference ms/obs | 0.532 | 1.730 |
| Training rows used | 100,000 / 1,352,108 | 30,000 / 1,352,108 |
| Probability mechanism | distance-weighted 11-neighbour vote | Platt-scaled (internal 5-fold) |
| Training cost | lazy fit (0.03 s); all cost at inference | 23.46 s final; Platt 6.69× raw fit |

KNN inference against a shared 10k-row query set: 25k reference rows -> 0.0201
ms/row, 50k -> 0.0421, 100k -> 0.0840. Four times the reference set gives 4.2×
the inference time, i.e. the expected O(n_train) behaviour.

**What this changes:** KNN's accuracy is only 3.1 points behind the SVM, but its
log loss is undefined because it assigns exactly zero probability to the true
class on 0.59% of rows. In a Viterbi decoder a zero emission kills the correct
path no matter how good the transition score is; using KNN would require an
arbitrary probability floor. The SVM's probabilities are finite and usable.
Cost runs the other way: the SVM is 3.3× slower per observation and needed a
2.2% training subsample. So P2's honest conclusion is "SVM for calibrated
probabilities if the whole table says so; KNN demonstrates the calibration
failure, not a deployment option".

**Failure mode / caveat:** one seed, and the fold std is ~0.027 for both models.
The 0.037 macro-F1 gap is only ~1.4× the fold std, so it is suggestive rather
than decisive. Neither model has been tuned; that is P3's stage and only for the
model that wins the five-way comparison.

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
| 2026-11-18 | KNN/SVM training capped with label-stratified subsampling (100k / 30k), validation folds scored in full | Train on all 1.35M rows; reduce folds; reduce candidate count | Full-data KNN is a ~270k x 1.16M distance matrix per fold and full-data Platt SVM is O(n^3); the cap is the largest size whose 5-fold run fits the budget, and it is disclosed in every record |
| 2026-11-18 | Candidate-miss observations keep true label `-1` in observation-level metrics | Drop them; treat them as wrong only for accuracy | A miss is a real failure of the pipeline; dropping it would inflate every model's score. Macro F1 includes `-1` as a class that can never be predicted, which is deliberately conservative |
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
