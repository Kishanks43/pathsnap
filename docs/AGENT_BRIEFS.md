# Agent briefs — copy-paste prompts for your coding assistant

Each person opens their own agent (Claude Code, Cursor, Copilot, Windsurf,
whatever you use) **in this repository root** and pastes two things:

1. the **shared preamble** below, then
2. their own workstream prompt.

The prompts are self-contained because the agent starts with no memory of this
conversation. Keep the agent scoped to the files you own — an agent that
"helpfully" edits someone else's module is how four people lose an afternoon.

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

The data pipeline is ALREADY BUILT and frozen. Do not rebuild it. Load it:

  from pathsnap.features.dataset import load_partition, TRUST_RESIDUAL_M
  d = load_partition("train")      # X (1,453,943 x 19), y, groups, label_residual_m
  d = load_partition("student_test")
  X, y, groups = d.X, d.y, d.groups
  trusted = d.trusted()            # drop untrusted labels from fitting/scoring

`y` is binary per candidate row (1 = this arc is the observation's reference arc).
`groups` is the trip index: always pass it to GroupKFold.

Hard rules you must never violate:
1. The 10 evaluator-testing trips are not in this repo and must never be used.
2. Splits are always by trip (pass groups=), never by row.
3. Reference routes must never be read inside a prediction path, only for scoring.
4. Fit every learned preprocessing step on training folds only.
5. Per-observation Top-1 accuracy = argmax over an observation's candidate scores.
   Macro F1 must be computed at the OBSERVATION level after that argmax, never
   over candidate rows (rows are ~87% negative; macro F1 there is meaningless).
6. No fabricated metrics, no hardcoded predictions, no synthetic curves.
7. Do not modify these frozen files: src/pathsnap/io/schema.py,
   src/pathsnap/io/loaders.py, src/pathsnap/geo/distance.py,
   src/pathsnap/evaluation/metrics.py, src/pathsnap/models/base.py,
   src/pathsnap/labels.py, src/pathsnap/features/build.py,
   src/pathsnap/features/dataset.py.
8. Stay inside the files I list as mine. If you believe a change is needed in
   another person's module, stop and tell me instead of editing it.

Working style: implement, then run the tests you wrote, then show me the real
output. Do not claim something works without running it. If something fails, show
the traceback rather than guessing.
```

---

## P1 — Data & features (done) + Model 1

Your data pipeline is already implemented, so this brief is mostly about the EDA
artefacts, the ablation evidence, and your model.

```text
My workstream: I own the data pipeline (already built) plus Model 1, Logistic
Regression. I also own the EDA notebook and the EDA Folium map.

Files I own:
  src/pathsnap/candidates/generator.py     DONE
  src/pathsnap/features/build.py           DONE (schema v1 frozen)
  src/pathsnap/features/dataset.py         DONE
  src/pathsnap/labels.py                   DONE
  src/pathsnap/models/linear.py            -> Model 1
  src/pathsnap/viz/folium_maps.py          -> only build_eda_map()
  notebooks/01_eda.ipynb
  scripts/prepare_dataset.py               DONE

Task 1 - confirm the pipeline on my machine.
Run: python scripts/check_data.py
Run: python scripts/prepare_dataset.py --force
Report the actual summary line (trips, candidate rows, candidate miss rate,
trusted fraction) and explain in one sentence what "trusted fraction" filters.

Task 2 - the feature ablation (report evidence).
Train the same model twice on the same folds: once with the 10 geometry columns,
once with all 19. Report CV macro F1 for each and say which column group earns its
keep. This is the "what features and why" answer for section 2.2. Do not invent
the numbers; run both.

Task 3 - Model 1, Logistic Regression.
Implement fit/predict_proba/predict/describe per pathsnap/models/base.py. Use a
Pipeline with StandardScaler fitted inside each fold. Because features mix metres,
degrees and counts, unscaled logistic regression is meaningless - say so in the
report. Hyperparameters to record: C, max_iter, class_weight, and whether it
converged (surface ConvergenceWarning rather than swallowing it).

Task 4 - EDA notebook.
notebooks/01_eda.ipynb, executed top to bottom so outputs are real:
  - dataset inventory (train 80 / student_test 10, observation counts)
  - observations-per-trip and dt distributions
  - arc-length distribution (why candidate generation needs a spatial index)
  - residual (obs -> reference arc) distribution, and the two anomalous trips
  - candidate miss rate before/after the label-trust filter
  - GPS trajectory scatter + network for one trip
Save figures to reports/figures/.

Task 5 - EDA map.
build_eda_map(trip, out_path): Folium with raw GPS observations and the road
network as separate layers, legend, layer control, sensible zoom, and a note in
the title if the network is subsampled for display. Networks reach 300k arcs, so
filter to the track's bounding box and/or use one MultiPolyline.
Save to reports/maps/eda_<trip_id>.html and take one screenshot for the report.

Definition of done: EDA notebook executed and committed, one HTML map committed,
ablation table exists, Model 1 CV + student-test numbers recorded in
experiments/results/, tests green.
```

---

## P2 — Models 2 & 5, evaluation harness

```text
My workstream: Model 2 (distance-weighted KNN), Model 5 (RBF SVM), the group-aware
CV harness, and the model comparison table for the report.

Files I own:
  src/pathsnap/models/instance_based.py    -> Model 2
  src/pathsnap/models/kernel.py            -> Model 5
  src/pathsnap/evaluation/cv.py            -> group_kfold, cross_validate_model
  src/pathsnap/evaluation/compare.py       -> save_record/load_records/table/guard
  tests/test_cv.py, tests/test_compare.py

Task 1 - group-aware CV (highest priority, three people import this).
Implement group_kfold using sklearn GroupKFold over `groups` (the trip index from
load_partition). Whole trips stay on one side. If a group is too small for
n_splits, fall back to GroupShuffleSplit and print a warning.
Implement cross_validate_model -> CVResult with per-fold macro F1, top-1 accuracy,
log loss, and mean/std of macro F1. The brief names "stability across validation
folds" as a selection criterion, so the std is not decoration.

IMPORTANT: per-observation metrics, not per-row. For each observation take argmax
over its candidate scores, then compute accuracy and macro F1 over observations.
Use the frozen helpers in pathsnap.evaluation.metrics.

Task 2 - the two models.
Both must satisfy pathsnap.models.base.check_classifier. Both standardise inside a
Pipeline fitted on training folds only.
KNN: choose k by CV within your workstream (k in {3,5,7,11}) and justify it.
SVM: probability=True (Platt scaling) is mandatory because the HMM needs
probabilities. At 1.45M candidate rows it is not feasible directly: subsample for
this model only, state the sample size and the rule used to choose it, and report
the cost. Do not silently train on a small slice and present it as if it used all
data - that is an integrity violation, not a shortcut.

Task 3 - persist results.
Implement save_record/load_records/build_comparison_table/assert_five_models per
docs/CONTRACTS.md C5. The markdown table must match the Deliverables 3.1 columns:
Model | Family | Key configuration | CV macro F1 | Top-1 accuracy | Route continuity
| Student-test macro F1 | Log loss | Inference ms/obs.
Generate comparison.csv AND comparison.md from the records. Never hand-type a
number into the report.

Task 4 - the two analyses only you can do.
  (a) Calibration: KNN's predict_proba is a k-neighbour vote (jumps of 1/k) and
      SVM's is Platt-scaled. Both degrade the Viterbi emission term. Show it with
      log loss next to accuracy for all five models.
  (b) Cost: inference ms/obs for KNN and SVM versus the trees. KNN cost grows with
      training size; SVM scores everything through support vectors.
These two paragraphs are sections 3.2 B/C for Models 2 and 5, and feed 3.1.

Definition of done: cv.py and compare.py implemented and importable by P3, both
models pass check_classifier, five-row comparison table generates from records,
guard passes, tests green.
```

---

## P3 — Model 3, Bayesian optimisation

```text
My workstream: Model 3 (Random Forest) and the Bayesian hyperparameter
optimisation of whichever model wins the comparison.

Files I own:
  src/pathsnap/models/tree.py              -> Model 3
  src/pathsnap/tuning/bayes.py             -> suggest_params, tune, export_trials
  tests/test_models.py, tests/test_tuning.py

Task 1 - Random Forest.
Implement to the contract in pathsnap/models/base.py. Set and record the seed.
Record n_estimators, max_depth, min_samples_leaf and whether depth was capped to
control overfitting - on 80 trips an unrestricted forest memorises trip geometry,
and you should be able to show that claim with numbers, not assert it.
Export feature importances to experiments/results/importances_rf.csv.

Task 2 - report your model's numbers.
Use P2's cross_validate_model (import it; do not write a second splitter). Record
CV macro F1 mean/std, student-test top-1 accuracy, student-test macro F1, log
loss, training time and inference ms/obs for Model 3. Compare training time
against HistGradientBoosting (P4's model) - "sequential error correction versus
bagging, at what cost" is a strong comparison paragraph.

Task 3 - Bayesian optimisation.
1. Your objective must be cross-validated macro F1 from P2's harness, with
   groups=trip_id. Use optuna.samplers.TPESampler - random search fails the
   requirement.
3. Tune ONLY the model that actually won the comparison, not all five.
4. Persist the study to experiments/results/trials.db and export every trial to
   experiments/results/trials.csv. The brief requires the search record to be
   retained and committed.
5. Justify each range in a comment. Starting points:
     random_forest        : n_estimators 100..600, max_depth 3..30,
                            min_samples_leaf 1..20
     hist_gradient_boosting: learning_rate 1e-3..3e-1 log, max_leaf_nodes 15..127,
                            l2_regularization 1e-8..1e1 log
     logreg               : C log-uniform 1e-3..1e2
     knn                  : n_neighbors 1..25, weights categorical
   (Tuning code must handle whichever model wins, even if it is not mine. If the
   winner is P4's HistGB, coordinate with P4 for the model class - do not edit
   their file.)

Task 4 - tuned vs untuned.
Same partitions, same metrics, same seed: CV macro F1, student-test top-1
accuracy, macro F1, log loss, inference time. State plainly whether tuning helped
and by how much. If it did not help, report that - it is a legitimate result, and
fabricating an improvement is explicitly prohibited by the integrity policy.

Definition of done: Model 3 passes check_classifier, importances committed,
trials.csv committed, tuned-vs-untuned comparison produced, tests green.
```

---

## P4 — Model 4, decoder, API

```text
My workstream: Model 4 (HistGradientBoosting), the HMM + Viterbi decoder, and the
FastAPI backend. The UI is a later stage - do not start it until the five models
report real numbers.

Files I own:
  src/pathsnap/models/boosting.py          -> Model 4
  src/pathsnap/hmm/viterbi.py              -> transition_logprob, viterbi_decode,
                                              classifier_only
  src/pathsnap/api/inference.py            -> Predictor
  src/pathsnap/api/main.py                 -> /health, /trips, /predict/{trip_id}
  src/pathsnap/viz/folium_maps.py          -> only build_prediction_map and
                                              arc_sequence_to_coords
  ui/                                       (later stage)
  artifacts/decoder_config.json, artifacts/eval_trips.txt

Task 1 - Model 4.
Implement HistGradientBoosting to the contract. Enable early stopping and record
how many iterations were actually used, plus learning_rate, max_leaf_nodes and
l2_regularization. Use P2's cross_validate_model (import it, do not write a second
splitter) and report the same metric set as everyone else. Note that this learner
bins features internally, so it is far less scale-sensitive than LogReg/KNN/SVM -
a concrete cross-model insight for the report, not a textbook sentence.

Task 2 - Viterbi.
Hidden state = the candidate arc at each observation. Emission from the classifier
probability (state your mapping). Transition from network connectivity plus the
match between network distance and GPS-implied distance, e.g.
exp(-|d_network - d_observed| / beta).
Implement in LOG space - multiplying hundreds of probabilities underflows to zero
and produces a confident, wrong route with no error.
Handle explicitly: observations with zero candidates (skip, output -1, count
them); steps with no valid transition (fall back to emission-only, count them);
large GPS gaps (docs/DATA_FACTS.md Fact 4 - the worst gap is 153,064 s). Return
arc_per_obs aligned to the GPS timeline.

Task 3 - the required comparison.
classifier_only() baseline (implement independently, ~5 lines) vs viterbi_decode on
the same trips and the same model. Report observation-level top-1 accuracy, route
continuity and sequence-level performance. Then find at least one trip where
decoding made things WORSE and explain why - the brief explicitly asks about cases
where the HMM did not help, and an honest negative case scores better than a
suspiciously perfect story.

Task 4 - inference pipeline + API.
Predictor loads model.joblib, preprocessor.joblib, feature_schema.json,
decoder_config.json once at startup and fails loudly if any is missing.
predict_trip reads .track/.nodes/.arcs ONLY - never .route. Scoring against the
reference is a separate method called by the API layer after prediction.
Critical: build the inference feature matrix with exactly the 19 columns in
artifacts/feature_schema.json, in that order, or predictions are silently wrong.
API: /health, /trips, /predict/{trip_id} with pydantic response models
(docs/CONTRACTS.md C7). Bind 0.0.0.0. Log trip id and elapsed ms per request -
you will be asked to show those logs during evaluation. Never return raw reference
arc ids; aggregated metrics only.

Task 5 - prediction map.
build_prediction_map: raw GPS, road network, reference route, classifier-only route
and Viterbi route as distinguishable layers with legend + layer control. Implement
arc_sequence_to_coords so consecutive arcs sharing a node produce one continuous
polyline rather than a zig-zag of arc midpoints.

Definition of done: Model 4 passes check_classifier, decoder handles the three edge
cases, comparison table exists including a negative case, API serves all ten
reserved trips, prediction map committed.
```

---

## Tips for working with the agent

- **Give it the real numbers.** Paste the actual error, shape and test output.
  Agents guess badly when given a summary.
- **One task at a time.** "Implement the model and the decoder and the API" produces
  three half-finished things.
- **Make it show evidence.** End with "show me the command you ran and its real
  output", not "does it work?".
- **It will try to write your report for you.** Do not let it produce prose you
  cannot defend — the viva catches that, and the integrity policy treats fabricated
  analysis as a serious violation.
- **Commit after each green test run**, under your own git identity (`docs/SETUP.md`).
