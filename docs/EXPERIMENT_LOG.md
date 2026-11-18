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
- CV: 5-fold GroupKFold by trip, seed 42, through P2's shared `evaluation/cv.py`.
  Re-scored on 2026-11-18 after the temporary splitter was retired; see the note
  below. Student-test uses `evaluation/holdout.evaluate_on_test`.
- Model: StandardScaler + LogisticRegression(C=1.0, max_iter=2000), fitted per fold
- Command: `python scripts/run_model1.py`
- Full log: `experiments/logs/model1_20261118_185506.log`

**Expected:** the sequential block (previous observation, connectivity, speed) should
add signal beyond geometry. A geometry-only linear model cannot do better than the
nearest-candidate rule on its own, since its score is monotone in distance.

**Result** (observation level, trusted labels; arc ids namespaced per trip,
candidate-row log loss via `safe_log_loss`, as in every other model record)

| config | CV macro F1 (mean ± std) | CV top-1 | student-test top-1 | student-test macro F1 | student-test log loss |
|---|---|---|---|---|---|
| nearest candidate (no model) | — | — | 0.5581 | 0.4156 | N/A (raw distances, not probabilities) |
| Logistic, geometry only (10 cols) | 0.3784 ± 0.0368 | 0.5756 | 0.5581 | 0.4020 | 0.2033 |
| **Logistic, all 19 cols** | **0.6269 ± 0.0252** | **0.7855** | **0.7938** | **0.6105** | **0.1488** |

Train time 3.7 s, inference 0.0016 ms/obs, 0 convergence warnings.

**What this changes:** the sequential block is worth ~25 points of CV macro F1. Geometry
alone reproduces nearest-candidate on student-test top-1, which confirms the linear
model is not learning anything beyond distance from those columns. Model 1 is the
baseline every other model must beat.

**2026-11-18 re-score:** the first run used P1's temporary splitter, global arc ids
and observation-level renormalised log loss (CV 0.6444 ± 0.0290, test macro F1 0.6316,
log loss 0.6351). It was re-run through the shared harness so that all four records in
`comparison.csv` use the same metric definitions. Top-1 is unchanged; the macro F1 drop
is the correction for arc-id collisions between trips. Quote the new numbers only.

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

### [P3] Random Forest (Model 4 candidate) - untuned baseline, 25-trial Bayesian search, full-data decision

**Question:** does a depth-limited random forest beat the linear and kernel models on
the same folds, and does 25 trials of Bayesian tuning improve it once the chosen
configuration is re-scored on all 1.35M training rows?

**Setup**
- Data partitions: train (80 trips, 1,352,108 trusted candidate rows, 3,650 after
  candidate filtering per trip on average) / student_test (10 trips, 146,499 trusted
  rows) - evaluator-test never read
- CV: 5 folds, `GroupKFold` on trip index, seed 42
- Model: `RandomForestModel`, n_estimators=300, max_depth=20, max_features=sqrt,
  class_weight=balanced_subsample, gini impurity, n_jobs=-1
- Search: Optuna TPE, 25 trials, objective = 5-fold group CV macro F1 on a
  150,201-row observation subsample (whole observations, seed 42); the search space
  is n_estimators {100..400}, max_depth {5, 10, 15, 20, 25, None}, min_samples_split
  and min_samples_leaf {2..20}, max_features {sqrt, log2, 0.5}, class_weight
  {None, balanced, balanced_subsample}
- Decision rule, fixed before looking: re-fit the best configuration on the full
  training rows, compare its full-data CV macro F1 with the untuned forest, keep the
  higher one, then score that one on student_test exactly once
- Features: schema v1, 19 columns
- Commands: `python scripts/run_p3.py untuned` then
  `python scripts/run_p3.py tune --model random_forest --n-trials 25` then
  `python scripts/run_p3.py final --model random_forest`

**Expected:** the forest should beat logistic regression (0.6269 after the shared
harness re-score) because candidate
geometry is non-linear, and tuning should add a small amount, mostly through deeper
trees and a lower learning bias. The subsample objective was expected to overstate
the full-data score.

**Result**

| metric | untuned (kept) | tuned |
|---|---|---|
| CV macro F1 (mean +- std), full data | **0.6627 +- 0.0260** | 0.6596 +- 0.0285 |
| CV macro F1, tuning subsample | not run | 0.6927 (best trial 17) |
| student-test top-1 accuracy | **0.8148** | 0.8058 |
| student-test macro F1 | **0.6443** | 0.6344 |
| student-test log loss | **0.1307** | 0.1775 |
| route continuity | 0.9031 | - |
| train time (full 1.35M rows) | 234.8 s | 224.4 s |
| inference ms/obs | 0.0342 | - |

Best searched configuration: n_estimators=200, max_depth=10, min_samples_split=5,
min_samples_leaf=16, max_features=0.5, class_weight=balanced_subsample. The search
took 43.6 min (2,618 s) and all 25 trials completed; trials.csv records every trial.

Top feature importances (mean decrease in impurity, untuned model):
dist_rank 0.239, dist_ratio 0.210, dist_m 0.096, heading_change_deg 0.091,
log1p_dist 0.085, t_along 0.076, prev_dist_m 0.064. Rank and ratio features carry
more signal than the raw distance, which is the expected signature of per-candidate
normalisation working.

**Artifacts:** `experiments/results/rf_untuned.json`,
`experiments/results/model_random_forest.json`,
`experiments/results/tuned_vs_untuned_random_forest.json`,
`experiments/results/trials.csv`, `experiments/results/importances_rf.csv`,
`artifacts/model.joblib` (the kept untuned forest)

**What this changes:** the random forest is the strongest single model
(CV 0.6627, student-test top-1 0.8148, macro F1 0.6443) and becomes the API's
`model.joblib`. Tuning is rejected: the best subsample configuration is more
regularised (depth 10, leaf 16), scores 0.0031 lower on full-data CV and 0.009 lower
top-1 on student_test. The 0.6927 subsample score was optimistic by about 0.03,
which is the expected selection effect of choosing the best of 25 trials on 150k
rows.

**Failure mode / caveat:** the kept model is chosen by comparing two configurations
on the same five folds, so its CV score is a selection-biased estimate; the fold
spread is large (0.629-0.699) because whole trips differ in difficulty. The search
only saw 11% of the training rows, so a configuration that needs more data to pay
off (for example deeper trees) could have been missed. Log loss is reported on
student_test only: the full-data CV helper did not record it for these folds
(`null` in the comparison file). The forest is 665 MB on disk, which is why
`artifacts/` stays out of git and the manifest carries its checksum instead.

---

### [P4] HistGradientBoosting (Model 4) - early stopping and the speed/accuracy trade-off

**Question:** does sequential boosting beat the random forest, and how much training
and inference time does it save?

**Setup**
- Data partitions: train (80 trips, 1,352,108 trusted rows) / student_test
  (10 trips, 146,499 trusted rows) - evaluator-test never read
- CV: 5 folds, `GroupKFold` by trip, seed 42, through the shared
  `evaluation/cv.py` and `evaluation/holdout.py`
- Model: `HistGradientBoostingModel`, learning_rate=0.1, max_iter=300,
  max_leaf_nodes=31, l2_regularization=0.0, early_stopping=True, random_state=42
- Early stopping held out a random 10% of rows; 85 of the 300 iterations were used
- Features: schema v1, 19 columns
- Command: `python scripts/run_model4.py`

**Expected:** boosting should match or beat the forest's CV macro F1 while training
in seconds rather than minutes, because histogram binning and early stopping cut the
cost of each tree. With only 80 trips, the fold spread should stay large.

**Result**

| metric | value |
|---|---|
| CV macro F1 (mean +- std) | 0.6565 +- 0.0297 |
| student-test top-1 accuracy | 0.8005 |
| student-test macro F1 | 0.6368 |
| log loss | N/A (zero probability for the true label on 6.8e-06 of candidate rows) |
| route continuity | N/A (classifier only; HMM stage is separate) |
| train time (full 1.35M rows) | 5.5 s |
| inference ms/obs | 0.0088 |
| iterations used | 85 of 300 |

Against the random forest: CV 0.6565 vs 0.6627, test top-1 0.8005 vs 0.8148,
macro F1 0.6368 vs 0.6443. Boosting is about **43x faster to train** (5.5 s vs
234.8 s) and **3.9x faster at inference** (0.0088 vs 0.0342 ms/obs), but it is
second on every accuracy metric. Log loss is N/A for HGB because a tiny number of
rows receive exactly zero probability, while the forest records 0.1307.

**Artifacts:** `experiments/results/model_hist_gradient_boosting.json`,
`experiments/results/model4_detail.json`,
`experiments/results/model4_student_test_predictions.csv`,
`artifacts/hist_gradient_boosting.joblib`

**What this changes:** the fifth model record exists and `assert_five_models` now
passes. The random forest remains the deployed model: it wins CV, top-1, macro F1,
continuity and log loss. HistGradientBoosting is the speed option and the model to
use if evaluation-day latency or retraining time becomes the binding constraint.

**Failure mode / caveat:** scikit-learn's early stopping holds out a random 10% of
**rows**, not whole trips, so rows from the same trip appear on both sides of its
internal validation split. The 85-iteration stopping point may therefore be later
than a trip-aware split would choose. The accuracy gap to the forest (0.0062 CV
macro F1) is smaller than the fold std (0.026-0.030), so "forest wins" is supported
by the test numbers and log loss, not by a decisive CV margin.

---

### [P4] Viterbi decoder - settings chosen on held-out TRAIN trips, then student_test HMM vs classifier-only

**Question:** does the HMM improve on the per-observation classifier, which settings
should it use, and where does it make the route worse?

**Setup**
- Decoder tuning: 10 training trips drawn by `RandomState(42).permutation(80)[:10]`
  (`00000001, 00000005, 00000013, 00000016, 00000024, 00000028, 00000035,
  00000037, 00000038, 00000088`). The scorer is a random forest trained on the
  other 70 trips only (1,231,364 trusted rows). student_test is not read in this stage.
- Sweep: beta_m in {10, 25, 50, 100, 200} x max_hops in {1, 2}; emission weight 1.0,
  probability floor 1e-6, connectivity bonus 5.0, same-arc bonus 1.0, gap reset at
  30 s. Selection: pooled observation-level top-1 on the held-out trips, ties broken
  by fewer forced emission-only steps.
- Final comparison: the exact API pipeline (`Predictor`), once on each of the 10
  student_test trips, against `classifier_only` on the same probabilities.
- Commands: `python scripts/run_p4.py decoder-tune`,
  `python scripts/run_p4.py student-test`

**Expected:** decoding should smooth isolated classifier errors and improve top-1
on trips with long, coherent routes, while a too-large beta_m should over-smooth
and lose accuracy. max_hops=2 should reduce forced steps at lane changes.

**Result**

| metric | value |
|---|---|
| chosen settings | beta_m=10.0, max_hops=2 |
| held-out train top-1, HMM | 0.8796 |
| held-out train top-1, classifier-only | 0.8743 |
| held-out train forced steps | 3,693 (hops=2) vs 4,196 (hops=1) |
| student-test pooled top-1, HMM | **0.8249** |
| student-test pooled top-1, classifier-only | 0.8148 |
| student-test mean route continuity | 0.9638 |
| student-test mean road-set Jaccard | 0.7398 |
| trips HMM better / worse / equal | 5 / 5 / 0 |

Per-trip deltas ranged from -7.09 pp (`00000009`, 0.7541 vs 0.8251) to +7.25 pp
(`00000077`, 0.7145 vs 0.6421). The worst trip has the lowest Jaccard (0.612) and
the second-highest forced-step count (157): the decoder locks onto a nearby parallel
road and stays there. The best trip is long (3,911 observations) and the classifier
is locally noisy, which is exactly the case the transition term is meant to fix.
Prediction maps were written for both cases:
`reports/maps/pred_00000009.html` and `reports/maps/pred_00000077.html`.

**Artifacts:** `experiments/results/decoder_sweep.csv`,
`experiments/results/decoder_tuning.json`,
`experiments/results/hmm_vs_classifier.csv`,
`experiments/results/hmm_vs_classifier.json`,
`artifacts/decoder_config.json`, `reports/maps/pred_*.html`

**What this changes:** Viterbi is the API's default decode mode. It gains 1.01 pp
pooled top-1 on student_test and improves continuity, but it is not uniformly
better: 5 of 10 trips get worse. The report must show the worst case, not only the
pooled average. `decoder_config.json` is now written by the tuning stage rather than
by hand.

**Failure mode / caveat:** beta_m=10 is the smallest value in the pre-registered
grid, so the choice sits on the boundary; smaller values (2.5, 5) were not tested
and could be better or could collapse towards classifier-only. The setting was
selected on 10 training trips, so its held-out score is still a selection estimate.
The student-test comparison uses one model and one seed; the 5/10 split of wins and
losses is the honest summary of a small effect. Forced emission-only steps remain
in the thousands on the held-out trips; no trip in student_test lost all candidate
coverage (0 observations without candidates), but that path is handled and tested.

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
| 2026-11-18 | Bayesian tuning runs on a 150,201-row observation subsample, but the tuned-vs-untuned decision is made on full-data CV | Tune on all 1.35M rows; tune on a fixed trip subset | 25 trials on the full data would cost roughly 10x the 44 min spent. The decision rule was pre-registered: the subsample only proposes configurations, the full data decides |
| 2026-11-18 | Keep the untuned forest as `model.joblib` | Ship the tuned configuration because it won the search | The tuned configuration scores 0.6596 vs 0.6627 on full-data CV and 0.8058 vs 0.8148 top-1 on student_test. Shipping the search winner would contradict the pre-registered decision rule and the held-out evidence |
| 2026-11-18 | Five-model table completed: HistGradientBoosting was run on all 80 training trips and scored with the shared harness | Insert the `standin_model4_loto_summary.json` numbers as the fifth row | The stand-in uses 9-trip training folds, not the official 80-trip protocol. Once the training partition was available, `scripts/run_model4.py` produced a real record (`model_hist_gradient_boosting.json`); `assert_five_models` passes and the stand-in file stays out of `comparison.csv` |
| 2026-11-18 | Viterbi settings chosen on 10 held-out training trips, not on student_test | Tune on student_test for the best demo numbers | Student_test is the only held-out data this build has; tuning on it would leak the evaluation. The tuning stage fits a separate forest on the other 70 training trips |
| 2026-11-18 | max_hops=2 wins the tie-break at beta_m=10 by needing 503 fewer forced steps | Keep the simpler hops=1 | Top-1 was identical to four decimal places; fewer forced emission-only steps means more of the route is explained by the model, which is the safer failure mode |
| 2026-11-18 | Report the 5/10 trips where the HMM makes top-1 worse, with the worst map | Report only the pooled gain | The pooled +1.01 pp hides a -7.09 pp trip. The brief asks for a case where the decoder got worse, and the parallel-road failure is the honest limitation |
| 2026-11-18 | Re-ran logistic regression through P2's shared CV and held-out harness | Keep P1's original numbers and just note the difference | One authoritative CV implementation is a team rule. The old splitter also used global arc ids and observation-level log loss, so the logreg log-loss cell was not comparable with KNN/SVM/RF. The re-score changed CV macro F1 0.6444 -> 0.6269 and log loss 0.6351 -> 0.1488; only the new numbers are quoted |
| 2026-11-18 | Build HistGradientBoosting as the fifth model, reversing the earlier exclusion | Leave four records and keep the guard failing | The complete project needs five real models. The owner's runner was pointed at the shared CV harness, its official record was produced on all 80 training trips, and `assert_five_models` now passes |
| 2026-11-18 | Keep the random forest as the deployed model even though HGB is 43x faster to train | Deploy HGB because the plan called it the natural deployment model | The forest wins on CV (0.6627 vs 0.6565), test top-1 (0.8148 vs 0.8005), macro F1 (0.6443 vs 0.6368) and log loss (0.1307 vs N/A). HGB is documented as the speed option, with a 5.5 s fit and 0.0088 ms/obs |
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
