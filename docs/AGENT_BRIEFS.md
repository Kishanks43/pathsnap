# Agent briefs — copy-paste prompts for your coding assistant

Each person opens their own agent (Claude Code, Cursor, Copilot, Windsurf,
whatever you use) **in this repository root** and pastes two things:

1. the **shared preamble** below, then
2. their own workstream prompt.

The prompts are written to be self-contained because the agent starts with no
memory of this conversation. Keep the agent scoped to the files you own — an
agent that "helpfully" edits someone else's module is how four people lose an
afternoon.

---

## Shared preamble — paste this first, every session

```text
You are helping on a university ML hackathon project called PathSnap: distributed
map-matching and route reconstruction. Read these files before doing anything:

  README.md
  docs/TEAM_PLAN.md
  docs/CONTRACTS.md
  docs/DATA_FACTS.md
  src/pathsnap/io/schema.py
  src/pathsnap/config.py

Problem: given noisy GPS observations (.track) and a road network (.nodes/.arcs)
for one trip, predict which road arc the vehicle was on at each observation, then
reconstruct a continuous route. Ground truth is .route (a chronological list of
arc ids) and exists only for train + student_test trips.

Hard rules you must never violate:
1. The 10 evaluator-testing trips are not in this repo and must never be used.
2. Splits are always by trip_id, never by row. Pass groups=trip_id.
3. Reference routes (.route) must never be read inside a prediction path, only
   for scoring after the prediction is made.
4. Fit every learned preprocessing step on training trips only.
5. No fabricated metrics, no hardcoded predictions, no synthetic curves.
6. Do not modify these frozen files: src/pathsnap/io/schema.py,
   src/pathsnap/io/loaders.py, src/pathsnap/geo/distance.py,
   src/pathsnap/evaluation/metrics.py, src/pathsnap/models/base.py.
7. Stay inside the files I list as mine. If you believe a change is needed in
   another person's module, stop and tell me instead of editing it.

Working style: implement, then run the tests you wrote, then show me the real
output. Do not claim something works without running it. If something fails,
show the traceback rather than guessing.
```

---

## P1 — Data, candidates, features, EDA

```text
My workstream: candidate generation, feature engineering, EDA and the EDA Folium map.

Files I own (implement these, do not touch others):
  src/pathsnap/candidates/generator.py     -> generate_candidates()
  src/pathsnap/features/build.py           -> build_feature_matrix()
  src/pathsnap/viz/folium_maps.py          -> only build_eda_map()
  notebooks/01_eda.ipynb
  scripts/check_data.py                    (already written; extend if useful)

Task 1 - verify the data.
Run: python scripts/check_data.py
Report back the actual numbers (trip counts, observations per trip, node/arc counts,
timestamp deltas, worst GPS gaps). Flag anything surprising.

Task 2 - candidate generation.
Read docs/DATA_FACTS.md first: networks have up to ~200k arcs and one trip has
19,313 observations, so brute-force obs x arcs will not fit (3.9e9 segment
evaluations for one trip). Build a per-trip spatial index (uniform grid over arc
bounding boxes, or a KD-tree) and only evaluate arcs within radius_m. Cache the
result per trip so candidate generation runs once per trip, not once per
experiment. For each GPS observation, find nearby arcs. Requirements:
- Project observations and nodes into metres with
  pathsnap.geo.distance.LocalProjection, anchored at the trip centroid.
- Use point_to_segment_distance, not endpoint distance, so long arcs are not
  unfairly penalised.
- Keep arcs within radius_m, take the nearest max_candidates, but always keep at
  least one candidate per observation.
- Return a CandidateSet whose arrays are sorted by obs_idx (see docs/CONTRACTS.md C2).
- Add a function that reports candidate miss rate over a set of trips
  (using pathsnap.evaluation.metrics.candidate_miss_rate). This number is the
  quality gate for everything downstream - report it on all 80 train trips.

Task 3 - features.
Implement build_feature_matrix(trip, candidates) -> (X, y, meta) per CONTRACTS C3.
Start with the geometry block only (distance to arc, log1p distance, distance rank
among candidates, arc length, junction degree). Get it working end to end, report
the shape, then add the sequential block (previous-observation distance,
connectivity flag, heading change, observed speed, time gap) as a second pass.
Encode "no previous observation" with a flag column, not NaN.

Task 4 - freeze the schema.
Write artifacts/feature_schema.json with {"version": 1, "columns": [...]} and print
the exact column order. Tell me the order so I can announce it to the team.

Task 5 - EDA notebook.
notebooks/01_eda.ipynb: dataset inventory, observation-count distribution, dt
distribution, GPS trajectory scatter, network size/connectivity summary. Execute
it top to bottom so outputs are real.

Task 6 - EDA map.
build_eda_map(trip, out_path): Folium map with raw GPS observations and the road
network as separate layers, legend, layer control, sensible zoom. The network can
have tens of thousands of arcs - subsample or use a single MultiPolyline so it
opens fast. Save to reports/maps/eda_<trip_id>.html.

Definition of done: check_data.py clean, candidate miss rate reported, feature
schema frozen and printed, EDA notebook executed, one HTML map committed, and
tests for the geometry helpers in tests/.
```

---

## P2 — Models 1–2 and the evaluation harness

```text
My workstream: Model 1 (Logistic Regression), Model 2 (distance-weighted KNN), the
group-aware CV harness, and the model comparison table for the report.

Files I own (implement these, do not touch others):
  src/pathsnap/models/linear.py            (class skeleton exists; refine it)
  src/pathsnap/models/instance_based.py    (class skeleton exists; refine it)
  src/pathsnap/evaluation/cv.py            -> group_kfold(), cross_validate_model()
  src/pathsnap/evaluation/compare.py       -> save_record/load_records/table/guard
  tests/test_cv.py, tests/test_compare.py

Start now, do not wait for the feature engineer: build a synthetic
np.random.rand(500, 6) matrix with a deterministic label rule and use it to
develop everything below. Swap in the real feature matrix when it lands.

Task 1 - group-aware CV.
Implement group_kfold(X, y, groups, n_splits=5) using sklearn GroupKFold. Whole
trips stay on one side of a split. If a group is too small for n_splits, fall
back to GroupShuffleSplit and print a warning saying so.
Implement cross_validate_model(...) -> CVResult: fit on each training fold, score
the held-out fold, record per-fold macro F1, top-1 accuracy and log loss, plus
mean and std of macro F1. Stability across folds is an explicit selection
criterion in the brief, so the std matters as much as the mean.

Task 2 - finish the two models.
Make both satisfy pathsnap.models.base.check_classifier. Both must standardise
features inside a sklearn Pipeline (fit on train folds only - never on validation
or test). Record describe() properly: every hyperparameter, the seed, and the
scaling choice.

Task 3 - persist results.
Implement save_record / load_records / build_comparison_table / assert_five_models
per CONTRACTS C5. The markdown table must match the Deliverables 3.1 columns:
Model | Family | Key configuration | CV macro F1 | Top-1 accuracy | Route continuity
| Student-test macro F1 | Log loss | Inference ms/obs.
Write experiments/results/comparison.csv AND comparison.md from the records -
never hand-typed numbers.

Task 4 - train and evaluate for real.
Once the real feature matrix exists: train both models on the 80 train trips,
cross-validate, then evaluate on the 10 student-test trips. Report:
  - CV macro F1 mean and std per model
  - student-test top-1 accuracy, macro F1, log loss
  - training time and inference ms/obs
  - for KNN specifically: how coarse are the probabilities, and what does that do
    to log loss? This is a report point, not a footnote.

Task 5 - analysis I need for the report.
One short table per model of "what we tried -> what happened" (e.g. KNN k=3/5/7/11).
These become the 3.2 B/C answers. Do not invent any of it; run it.

Definition of done: both models pass check_classifier, CV + student-test numbers
exist in experiments/results/, comparison table generates from records, guard
passes with 5 models present, tests green.
```

---

## P3 — Models 3–5 and Bayesian optimisation

```text
My workstream: Model 3 (Random Forest), Model 4 (HistGradientBoosting), Model 5
(RBF SVM), then Bayesian hyperparameter optimisation of the winning model.

Files I own (implement these, do not touch others):
  src/pathsnap/models/tree.py              (skeleton exists)
  src/pathsnap/models/boosting.py          (skeleton exists)
  src/pathsnap/models/kernel.py            (skeleton exists)
  src/pathsnap/tuning/bayes.py             -> suggest_params(), tune(), export_trials()
  tests/test_models.py, tests/test_tuning.py

Start now, do not wait for the feature engineer: develop against a synthetic
np.random.rand(500, 6) matrix with a deterministic label rule, then swap in the
real feature matrix when it is announced.

Task 1 - three models, contract-compliant.
All three must satisfy pathsnap.models.base.check_classifier. Seeds set and
recorded. describe() must contain every hyperparameter actually used.
Also: for RandomForest export feature importances to
experiments/results/importances_rf.csv; for HistGradientBoosting record the number
of iterations early stopping actually used.

Task 2 - measure honestly.
Read docs/DATA_FACTS.md first: the training matrix is roughly 2M candidate rows.
An RBF SVM with probability=True on 2M rows is not feasible, so subsample for that
model only, state the sample size and how you chose it, and report the cost - do
not silently train on a small slice and present it as if it used everything.
Train each model on the 80 trips using Person 2's cross_validate_model (import it,
do not write your own splitter), then evaluate on the 10 student-test trips.
Record: CV macro F1 mean/std, top-1 accuracy, macro F1, log loss, training time,
inference ms/obs. Note that the RBF SVM with probability=True runs internal
5-fold Platt scaling - measure how slow it actually is and report that as a real
finding, not a caveat.

Task 3 - compare 3 vs 4 vs 5.
Produce a short evidence table: RF vs HistGB training/inference time, RF feature
importances, and any depth/leaf settings you changed and why. This is the
"sequential error correction and computational efficiency" comparison the brief
asks for.

Task 4 - Bayesian optimisation.
Implement tuning/bayes.py with Optuna's TPESampler (must be Bayesian, not random
search). Objective = group-aware CV macro F1 from Person 2's harness, with
groups=trip_id. Persist the study to experiments/results/trials.db and export
every trial to experiments/results/trials.csv - the brief requires the search
record to be retained and committed.
Search space: justify each range in a comment. Reasonable starting points:
  logreg: C log-uniform 1e-3..1e2
  random_forest: n_estimators 100..600, max_depth 3..30, min_samples_leaf 1..20
  hist_gradient_boosting: learning_rate 1e-3..3e-1 log, max_leaf_nodes 15..127,
                          l2_regularization 1e-8..1e1 log
  svm_rbf: C 1e-2..1e2 log, gamma 1e-4..1e0 log
  knn: n_neighbors 1..25, weights categorical
Only tune the model that actually won on the comparison - do not tune all five.

Task 5 - tuned vs untuned.
Same data partitions, same metrics, same seed: CV macro F1, student-test top-1
accuracy, macro F1, log loss, inference time. State plainly whether tuning helped
and by how much. If it did not help, say that - it is a legitimate result and
faking an improvement is explicitly prohibited.

Definition of done: three models pass check_classifier, all numbers in
experiments/results/, importances CSV committed, trials.csv committed, tuned-vs-
untuned comparison produced, tests green.
```

---

## P4 — HMM/Viterbi, FastAPI, UI, deployment

```text
My workstream: the HMM + Viterbi route reconstruction, the FastAPI backend, the
evaluator UI on laptop 2, and the two-laptop deployment.

Files I own (implement these, do not touch others):
  src/pathsnap/hmm/viterbi.py              -> transition_logprob, viterbi_decode,
                                              classifier_only
  src/pathsnap/api/inference.py            -> Predictor.load / predict_trip /
                                              score_against_reference
  src/pathsnap/api/main.py                 -> /health, /trips, /predict/{trip_id}
  src/pathsnap/viz/folium_maps.py          -> only build_prediction_map and
                                              arc_sequence_to_coords
  ui/app.py, ui/requirements.txt
  artifacts/decoder_config.json, artifacts/eval_trips.txt

Start now, do not wait for the models: write and test the decoder against a
random probability matrix over a synthetic 3-arc chain until a real classifier is
available.

Task 1 - Viterbi.
Hidden state = the candidate arc at each observation. Emission from the
classifier probability (state your mapping). Transition from network connectivity
plus the match between network distance and the GPS-implied distance, e.g.
exp(-|d_network - d_observed| / beta).
Implement in LOG space - multiplying hundreds of probabilities underflows to zero
and yields a garbage route with no error.
Handle explicitly:
  - observations with zero candidates (skip, output -1, count them)
  - steps with no valid transition (fall back to emission-only, count them)
  - large GPS gaps (reset or widen tolerance; document the choice)
Return DecodeResult with arc_per_obs aligned to the GPS timeline.

Task 2 - the required comparison.
classifier_only() baseline (implement independently, ~5 lines) vs viterbi_decode,
on the same trips, same model. Report observation-level top-1 accuracy, route
continuity and sequence-level performance. Then find at least one trip where
decoding made things WORSE and explain why - the brief explicitly asks about
cases where HMM did not help.

Task 3 - inference pipeline.
Predictor must load model.joblib, preprocessor.joblib, feature_schema.json,
decoder_config.json once at startup and fail loudly if any is missing. predict_trip
reads .track/.nodes/.arcs ONLY - never .route. Scoring against the reference is a
separate method called by the API layer after prediction.
Critical: the feature matrix at inference must use exactly the training column
order from feature_schema.json, or predictions are silently wrong.

Task 4 - API.
/health, /trips, /predict/{trip_id} with pydantic response models per CONTRACTS C7.
Bind 0.0.0.0 so laptop 2 can reach it. Log every request with trip id and elapsed
ms - you will be asked to show these logs during evaluation. Never return raw
reference arc ids; metrics only.

Task 5 - UI + deployment.
Streamlit + streamlit-folium is the fastest route: fetch /trips, let the evaluator
pick a trip, POST /predict/{id}, draw the returned route_geometry over the raw GPS
points, show metrics. API base URL comes from a PATHSNAP_API env var, not
localhost.
Then do a real two-machine test (or two processes + firewall rules) and capture:
  - screenshot of the UI showing a route
  - screenshot of the API terminal logs on the receiving laptop
Both screenshots are required in the report.

Task 6 - prediction map.
build_prediction_map: raw GPS, road network, reference route, classifier-only route
and Viterbi route as distinguishable layers with legend + layer control. Also
implement arc_sequence_to_coords so consecutive arcs sharing a node produce one
continuous polyline, not a zig-zag of arc midpoints.

Definition of done: decoder handles the three edge cases, comparison table exists
including a negative case, API serves all ten reserved trips, UI works from a
second laptop, logs and screenshots captured, prediction map committed.
```

---

## Tips for working with the agent

- **Give it the real numbers.** Paste the actual error, the actual shape, the
  actual test output. Agents guess badly when given a summary.
- **One task at a time.** "Implement candidate generation and features and EDA and
  the map" produces four half-finished things.
- **Make it show evidence.** End prompts with "show me the command you ran and its
  real output", not "does it work?".
- **It will try to write your report for you.** Don't let it produce prose you
  cannot defend — the viva is where that gets caught, and the integrity policy
  treats fabricated analysis as a serious violation.
- **Commit after each green test run**, under your own git identity. See
  `docs/SETUP.md`.
