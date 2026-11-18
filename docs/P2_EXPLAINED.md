# PathSnap, explained from scratch (Part 2 focus)

This document explains the whole project and, in more detail, **Person 2's part**:
the two models KNN and SVM, the way we test them fairly, the results, and the code
behind them. It assumes basic school maths (averages, percentages, distances,
squares, square roots) and **no machine learning background**. Every technical
word is explained the first time it appears, and there is a presentation guide and
a set of likely questions with answers at the end.

Everything in this document was produced by the code in this repository. Numbers
quoted here come from the files in `experiments/results/`.

---

## Contents

1. [The problem in one picture](#1-the-problem-in-one-picture)
2. [Words you need](#2-words-you-need)
3. [The data](#3-the-data)
4. [The whole pipeline, step by step](#4-the-whole-pipeline-step-by-step)
5. [Who does what in the team](#5-who-does-what-in-the-team)
6. [Person 2's part: what P2 builds](#6-person-2s-part-what-p2-builds)
7. [How we judge a model fairly](#7-how-we-judge-a-model-fairly)
8. [The two models, explained simply](#8-the-two-models-explained-simply)
9. [Results](#9-results)
10. [What the results mean](#10-what-the-results-mean)
11. [Honest limitations (say these before someone asks)](#11-honest-limitations)
12. [Code tour](#12-code-tour)
13. [Presentation guide](#13-presentation-guide)
14. [Likely questions and answers](#14-likely-questions-and-answers)
15. [Status and next steps](#15-status-and-next-steps)

---

## 1. The problem in one picture

A car drives through a city with a phone that records its position about once per
second. The phone's position is **noisy**: it can be 5 to 50 metres off, and the
road network is a map of the city's roads.

The question is:

> For every GPS point, which road was the car on?

Then, using those answers in order, we rebuild the car's **route**: a chain of roads
that a car could actually drive.

```
GPS points  +  road map
      |
      v
"Which road was I on?"  for every point   --> this is what we predict
      |
      v
A continuous, drivable route               --> the final output
```

This task is called **map matching**. Navigation apps do it every second.

---

## 2. Words you need

| Word | Plain meaning |
|---|---|
| **Observation** | One GPS point at one moment (one row of a `.track` file). |
| **Trip** | One whole drive: many observations plus the road map of that area. |
| **Arc** | One road segment (a piece of road between two junctions). Each arc has a number (its *arc id*). |
| **Node** | A junction point. Arcs connect nodes. |
| **Candidate** | A road that *could* be the right one for a given GPS point, because it lies within 100 m of it. We keep at most 10 per point. |
| **Feature** | A number that describes a candidate (for example, its distance from the GPS point). Each candidate has 19 features. |
| **Label** | The known answer for a candidate: `1` if it is the road the car really used, `0` otherwise. The answers come from the `.route` file. |
| **Model** | A rule, learned from examples, that turns the 19 features of a candidate into a score: "how likely is this candidate the right road?" |
| **Training** | Letting a model look at examples with known answers so it can learn the rule. |
| **Validation / test** | Data the model did *not* train on. Used to check how good it really is. |
| **Probability** | A number between 0 and 1. 0.9 means "90% sure". |
| **Accuracy** | Fraction of answers that are right. 80 right out of 100 = 0.80 = 80%. |
| **Precision** | Of the times the model said "yes", how often it was right. |
| **Recall** | Of the times the true answer was "yes", how often the model found it. |
| **F1 score** | One number that balances precision and recall: `F1 = 2 × precision × recall / (precision + recall)`. |
| **Macro F1** | F1 calculated for each road separately, then averaged equally over roads. Fair to rare roads. |
| **Log loss** | A penalty for being confidently wrong. Lower is better. Explained in section 7. |
| **Cross-validation (CV)** | Splitting the training data into 5 parts, training on 4 and testing on the 5th, five times, then averaging. |
| **Group split** | A split where whole trips go into one part. A trip is never split between train and test. |
| **Subsample** | Using a random part of the data (for example 100,000 rows out of 1.35 million) because the full amount is too slow. |
| **Viterbi / HMM** | The step after the classifier (Person 4): it chooses the most sensible *sequence* of roads, not just the best road for each point alone. |
| **Route continuity** | The fraction of neighbouring predicted roads that actually connect to each other in the map. |
| **API** | A small web service that answers requests such as "give me the route for trip 7". Person 4. |

---

## 3. The data

The data was given to us in folders, one folder per trip:

| Set | Trips | GPS observations | Used for |
|---|---|---|---|
| **train** | 80 | 208,417 | Training and cross-validation |
| **student_test** | 10 | 19,160 | Final honest check (never used to tune anything) |
| **evaluator_test** | 10 | not in our repo | Used by the teaching staff on evaluation day |

Each trip folder contains four text files:

- `.track`: the GPS points (longitude, latitude, time).
- `.nodes`: the junctions (their coordinates).
- `.arcs`: the roads (which two junctions each road connects).
- `.route`: the **known correct** sequence of roads (the answer key). It is used only for
  scoring, never as an input to predictions.

Two facts about the data shaped every decision:

1. **Trips come from all over the world.** Longitudes range from -100 to 137. So we never
   compare raw coordinates across trips.
2. **Some answer keys are incomplete.** In a few trips, the `.route` file covers only
   13 to 19% of the actual drive. For those GPS points the "correct road" is not
   really known. We therefore **trust** a label only when the GPS point is within
   **25 metres** of the road in the answer key. This keeps 93% of the training rows and
   98.5% of the student-test rows. The rule is applied the same way to both sets.

---

## 4. The whole pipeline, step by step

```
 Step 1            Step 2                 Step 3              Step 4           Step 5
 Candidates   -->  Features (19 numbers) -->  Classifier     -->  Viterbi/HMM -->  API + UI
 (Person 1)        per candidate          (Persons 1-4)       (Person 4)       (Person 4)
 "which roads      "how far, how          "how likely is      "pick one road   "serve the
  could it be?"    aligned, how           this candidate?"    per point that   route to the
                   connected?"                                 forms a path"    evaluator"
```

### Step 1: candidates (done, Person 1)
For each GPS point, find every road within 100 m. Keep the 10 closest. Measured
result: the correct road is among the candidates for about 98.8% of trusted points
(so 1.2% are lost before any model runs). This is why we chose 100 m and 10:
a bigger radius found a few more correct roads but made the data 17% bigger.

### Step 2: features (done, Person 1)
Each candidate gets 19 numbers, in a fixed order (the order is frozen in
`artifacts/feature_schema.json`). They fall into two groups:

**Geometry: 10 numbers, about the road and the point alone**
`dist_m` (distance in metres), `log1p_dist` (a squashed version of the distance),
`dist_rank` (is it the 1st, 2nd, ... closest?), `dist_ratio` (distance divided by the
closest distance), `arc_len_m` (road length), `t_along` (where along the road the
closest point is, from 0 to 1), `tail_out_degree` and `head_in_degree` (how many roads
leave and enter the junctions), `has_reverse` (is there a road going the other way),
`arc_bearing_deg` (the compass direction of the road).

**Sequence: 9 numbers, about the drive so far**
`has_prev` (is there a previous point), `prev_dist_m`, `delta_dist` (how the distance
changed from the previous point), `connected_prev` (does this road connect to the road
chosen for the previous point), `step_m` (distance moved since the last point),
`dt_s` (seconds since the last point), `speed_mps` (speed), `heading_change_deg`
(how much the direction turned), `seq_pos` (position in the trip).

Why it matters: Person 1 measured that a model using **only** the geometry group scores
0.386 macro F1 in cross-validation, and the full 19 numbers score 0.644. The sequence
numbers are worth about **26 points**.

### Step 3: the classifier (Persons 1 to 4)
A classifier looks at the 19 numbers of one candidate and gives it a probability. Each
GPS point has about 7 candidates, so the model produces about 7 probabilities per
point. For each point we then pick the candidate with the highest probability. That
pick is the model's answer for the point.

The team compares five classifiers under the same rules:

| Model | Owner | Family (the "kind" of model) |
|---|---|---|
| 1. Logistic regression | Person 1 | Straight-line rule with probabilities |
| 2. KNN (nearest neighbours) | **Person 2** | Compare with stored examples |
| 3. Random forest | Person 3 | Many decision trees voting |
| 4. HistGradientBoosting | Person 4 | Trees built one after another, each fixing the last |
| 5. RBF SVM | **Person 2** | Curved boundary around the examples |

### Step 4: Viterbi / HMM (Person 4, built)
The classifier judges each point alone. Viterbi looks at the whole sequence and asks:
"which chain of roads is most likely, given the probabilities *and* the fact that
consecutive roads must connect?" This is why the classifier's probabilities must be
**usable**, and why a probability of exactly 0 is dangerous (section 10).

### Step 5: API and user interface (Person 4, built)
A web service that returns the predicted route for a trip, and a screen that draws it.
It never returns the answer key.

---

## 5. Who does what in the team

| Person | Part | Status (as of this document) |
|---|---|---|
| P1 | Data, candidates, features, labels, EDA, Model 1 (logistic regression) | Data pipeline done; Model 1 pushed |
| **P2 (this document)** | **Evaluation harness, Model 2 (KNN), Model 5 (RBF SVM), comparison table** | **Done and pushed** |
| P3 | Model 3 (random forest) and the Bayesian tuning of the winner | Pushed; tuning run, untuned forest kept |
| P4 | Model 4 (HistGradientBoosting), Viterbi, API, UI | Decoder, API and UI built; Model 4 excluded from this build by request |

The rule from the team plan: **every file has exactly one owner**. Nobody edits another
person's files without asking.

---

## 6. Person 2's part: what P2 builds

P2 has four jobs. Think of them as four layers.

1. **The fair test rig (`evaluation/cv.py`).** A single, shared way to test any model:
   split by trip, train on four fifths, test on one fifth, repeat five times, and
   report the average and the spread. Everybody's model is tested this way, so the
   comparison is fair. Two other people (P3 and P4) depend on it.
2. **Model 2: KNN (`models/instance_based.py`).**
3. **Model 5: RBF SVM (`models/kernel.py`).**
4. **The comparison table (`evaluation/compare.py`).** Each model writes a small
   record file (`model_<name>.json`). The table is built *automatically* from those
   files, so no number is ever typed in by hand.

The runner `scripts/run_p2.py` does everything in order, and `tests/` checks that the
pieces behave as they should (25 tests for P2; 93 tests in total across the project).

---

## 7. How we judge a model fairly

### 7.1 Rule 1: split by trip, never by point
If points from the same drive were in both the training and the test part, the model
would recognise the roads of that drive and look better than it really is. This is
called **leakage**. So we split by trip: a trip is entirely in one part or the other.

Example: 80 trips, 5 folds. Each fold has about 16 trips for checking and 64 for
training. The folds never share a trip.

### 7.2 Rule 2: score per GPS point, not per candidate
A candidate table has about 1.35 million rows, but ~87% of them are "no". A model that
says "no" to everything would look good by row-level counting. So:

- **Accuracy (Top-1)** asks: for each GPS point, is the highest-scored candidate the
  correct road? The score is the **percentage of points answered correctly**.
  Example: 1,000 points, 806 correct → accuracy 0.806 (80.6%).
- **Macro F1** works per point too. For each road, count:
  - *true positives*: the point was on this road and we said so,
  - *false positives*: we said this road but it was not,
  - *false negatives*: the point was on this road and we missed it.

  Precision and recall come from those counts, then F1 combines them. Macro F1 is the
  **average over roads**, so a rarely used road counts as much as a common one.

### 7.3 Log loss: rewarding honest confidence
Log loss punishes being *confidently wrong*. For the true answer, the penalty for a
probability `p` is `-ln(p)`:

| Probability given to the true answer | Penalty `-ln(p)` |
|---|---|
| 0.99 (very sure, right) | 0.01 (tiny) |
| 0.90 | 0.11 |
| 0.50 (coin toss) | 0.69 |
| 0.01 (almost sure it was wrong) | 4.61 (big) |
| 0 (impossible) | **infinity** |

The score is the average penalty over all candidate rows. **Lower is better.** A
model that outputs exactly 0 for the true answer even once has an infinite log loss,
so we report it as N/A and also count how often it happens (the "zero-probability
fraction").

### 7.4 Route continuity
Take the predicted road for each point in order. Count the neighbouring pairs that
connect (the first road ends where the second starts, or they are the same road, or
the reverse). The fraction of connected pairs is the continuity. 1.0 means a fully
drivable route. Both KNN and SVM score about 0.90 here, which means about 1 in 10
neighbouring pairs is a jump that a real car could not make. That is what the HMM
(Person 4) is for.

### 7.5 Cross-validation: mean and spread
The five fold scores are averaged (**mean**) and their spread is measured
(**standard deviation**, "std"). A small std means the result is stable from one part
of the data to another. We report both, for example `0.6431 ± 0.0271`.

**Simple rule of thumb:** a difference between two models that is smaller than the std
is not clearly a real difference.

---

## 8. The two models, explained simply

### 8.1 KNN: "find the most similar examples and copy their vote"

**Analogy.** A new candidate arrives. Look at the 100,000 training candidates that
look most like it and see whether they were mostly correct roads or not.

**The steps.**

1. **Store** the training candidates. KNN does not really learn anything in advance.
2. **Standardise** the 19 numbers (see the box below) so that metres and counts count
   equally.
3. For a new candidate, **measure the distance** to every stored example:

   ```
   distance = square root of ( (a1 - b1)^2 + (a2 - b2)^2 + ... + (a19 - b19)^2 )
   ```

   That is Pythagoras extended to 19 dimensions.
4. Keep the **k = 11 closest** examples.
5. **Vote, weighted by closeness.** Closer neighbours count more: weight = 1 / distance.
   The probability of "correct road" is the total weight of the correct-road neighbours
   divided by the total weight of all 11.

**Why probabilities are "coarse".** Only 11 neighbours vote, so the probability can only
take a small number of values. It cannot say 0.8734 in a meaningful way. This is the
main weakness of KNN for our purpose.

**Why it can output exactly 0.** If all 11 nearest examples are wrong roads, the
probability of "correct road" is exactly 0. Then the true answer, if it happens to be
that candidate, gets an infinite log loss. This happened for 0.59% of test candidates.

**Why it is slow at prediction.** Every prediction compares the new point with *all*
stored examples. Double the stored examples, double the time. We measured this:

| Stored examples | Time per candidate |
|---|---|
| 25,000 | 0.020 ms |
| 50,000 | 0.042 ms |
| 100,000 | 0.084 ms |

**Standardising** (what `StandardScaler` does): for each of the 19 columns, subtract the
average and divide by the spread (standard deviation):

```
new value = (old value - average) / standard deviation
```

After this, a column measured in metres and a column that is a count are on the same
scale. Without it, the metres would dominate the distance.

**Our choices.**
- `k` was chosen by trying 3, 5, 7 and 11. Score rose each time: 0.5601, 0.5841, 0.5949,
  0.6064. We kept 11, the best of those tried. The score had not stopped rising, so a
  larger k might be better; we did not try it. This is stated in the log.
- We used 100,000 of the 1.35 million training candidates per fold (7.4%) to keep the
  run fast. The test model was trained on the same 100,000.

### 8.2 RBF SVM: "draw a curved fence between correct and wrong"

**Analogy.** Imagine plotting every candidate as a dot in 19-dimensional space, with
correct roads in one colour and wrong roads in another. The SVM tries to draw the
fence that separates the colours with the widest possible margin. Only a few dots sit
right next to the fence: those are the **support vectors**. They are the only dots that
matter.

**The "RBF" part (the curved fence).** Instead of a straight line, the fence bends
around clusters of examples. The similarity between two candidates is:

```
similarity = e^( -gamma × distance² )
```

where `e` is about 2.718. Two identical candidates have similarity 1. Very different
ones get a similarity close to 0. The distance here is the same Pythagoras distance as
before. `gamma = "scale"` sets how quickly similarity fades. `C = 1.0` controls how
strictly the fence must separate the colours (higher means stricter).

**The probability part (Platt scaling).** The SVM's raw output is a score, not a
probability. Platt scaling converts it with a sigmoid curve:

```
probability = 1 / (1 + e^( A × score + B ))
```

`A` and `B` are fitted on the training data (using an internal 5-fold check). This is
the costly step: it made training **6.69 times slower** (3.99 s without it, 26.69 s with
it, on the same 30,000 rows).

**Why it is slow.** The fence depends on pairs of examples, so training cost grows very
fast with the number of rows. Prediction also compares each candidate with every
**support vector**: we ended with 4,533 of them. So prediction is slower than a tree.

**Why we used only 30,000 rows.** Training on all 1.35 million rows would take hours and
would not fit the team's time budget. So 30,000 rows were drawn at random, keeping the
share of correct roads at about 12.7%. That is 2.2% of the data. **The results say so
explicitly**, in the log and in the saved record. Validation folds were always scored in
full, so the test is honest; only the training was smaller.

---

## 9. Results

All numbers below come from `experiments/results/` (generated by the code, not typed).

### 9.1 The comparison table so far (4 of 5 models)

| Model | Owner | CV macro F1 | Test top-1 accuracy | Test macro F1 | Test log loss | Route continuity | Time per point (ms) |
|---|---|---|---|---|---|---|---|
| Logistic regression | P1 | 0.6444 | 0.7938 | 0.6316 | 0.6351 | N/A | 0.0031 |
| **KNN (k = 11)** | **P2** | 0.6064 | 0.7750 | 0.5818 | **N/A** (0.59% zero probability) | 0.9004 | 0.532 |
| **RBF SVM** | **P2** | 0.6431 | **0.8062** | **0.6297** | **0.1357** | 0.8997 | 1.730 |
| Random forest | P3 | **0.6627** | **0.8148** | **0.6443** | **0.1307** | **0.9031** | 0.0342 |

The random forest is the strongest single model in this build and is the one saved
as `artifacts/model.joblib`. HistGradientBoosting (P4) is deliberately excluded from
this build, so the table has four rows and the "five models" check reports
incomplete rather than inventing a fifth row.

### 9.2 P2's extra evidence

| Question | Answer | Where it is |
|---|---|---|
| How does KNN cost grow? | Linearly: 25k → 0.020, 50k → 0.042, 100k → 0.084 ms per candidate | `knn_inference_cost.json` |
| How often does KNN give exactly 0? | 0.42–1.53% of rows across the 5 CV folds at k = 11; 2.15–4.72% at k = 3 | `cv_knn.json` |
| What does Platt scaling cost? | 6.69× training time (3.99 s → 26.69 s) | `cv_svm_rbf.json` |
| How many support vectors? | 4,533 | `model_svm_rbf.json` |
| How stable is the SVM across folds? | Macro F1 per fold: 0.629, 0.682, 0.658, 0.645, 0.602 (std 0.027) | `cv_svm_rbf.json` |

### 9.3 KNN's k sweep

| k | CV macro F1 (mean ± std) | CV top-1 |
|---|---|---|
| 3 | 0.5601 ± 0.0281 | 0.7211 |
| 5 | 0.5841 ± 0.0270 | 0.7323 |
| 7 | 0.5949 ± 0.0266 | 0.7383 |
| **11** | **0.6064 ± 0.0260** | **0.7447** |

### 9.4 What the decoder adds

The decoder was tuned on 10 held-out **training** trips, never on the student test.
With `beta_m = 10` and `max_hops = 2` it scores:

| Data | Decoder (Viterbi) top-1 | Classifier-only top-1 | Trips better / worse |
|---|---|---|---|
| 10 held-out training trips | 0.8796 | 0.8743 | - |
| 10 student-test trips | **0.8249** | 0.8148 | 5 better, 5 worse |

The average gain is +1.01 points, but it hides a loss of 7.09 points on one trip
(`00000009`) and a gain of 7.25 points on another (`00000077`). The decoder is not
uniformly better: on the bad trip it settles on a parallel road and stays there.
Both maps are in `reports/maps/`.

### 9.5 Did Bayesian tuning help the forest?

No. The 25-trial search found a configuration that scored 0.6927 on its 150k-row
subsample, but when both models were re-scored on all 1.35M training rows the tuned
version scored **0.6596** against **0.6627** for the untuned forest, and its
student-test top-1 was 0.8058 against 0.8148. The untuned forest was kept. This is a
useful, honest result: a search can win on a subsample and lose on the full data,
which is why the decision rule was fixed before looking.

---

## 10. What the results mean

1. **The SVM is the best of P2's two models** on every accuracy-type number: test
   top-1 0.806 vs 0.775, test macro F1 0.630 vs 0.582.
2. **Only the SVM gives usable probabilities.** KNN assigns exactly 0 to the right
   road for 0.59% of test candidates, so its log loss is infinite. In Viterbi, a zero
   probability for the right road kills the correct path, however good the other
   evidence is. KNN would need an artificial "minimum probability" to be used. This is
   the single most important finding for the HMM.
3. **The SVM is expensive.** It takes 1.730 ms per point against 0.532 for KNN, and it
   needed a 2.2% training subsample to be trainable at all.
4. **Logistic regression is a serious competitor.** It scores almost the same macro F1
   as the SVM (0.6316 vs 0.6297 on test; 0.6444 vs 0.6431 in CV), and it runs about
   **550 times faster** (0.0031 ms per point). Its log loss is worse (0.635 vs 0.136).
   This is an honest result, and the report should say it. Choosing the final model is
   the comparison table's job, not any one person's.
5. **Differences near the fold spread are not conclusive.** The SVM and logistic
   regression differ by less than the fold std on macro F1. One seed was used.
6. **The random forest wins overall.** It has the best CV macro F1 (0.6627), the best
   student-test top-1 (0.8148), the best macro F1 (0.6443) and the best route
   continuity (0.9031). It is also much faster at prediction than the SVM
   (0.0342 ms per point against 1.730), because it does not have to compare the point
   with thousands of stored examples.
7. **The decoder is a small average gain with large trip-to-trip variation.** It adds
   1.01 points pooled but makes 5 of 10 trips worse. Reporting only the average would
   be misleading; the report shows both.

---

## 11. Honest limitations

Say these yourself during the presentation. Stating them first makes the rest credible.

- **KNN's k was chosen on the same folds that report its score.** The reported CV score
  is therefore a little optimistic. This is written in the log.
- **k = 11 was the largest value tried**, and the score was still rising. Larger k was
  not tested.
- **Both models were trained on subsamples** (KNN 7.4%, SVM 2.2%). The comparison is
  fair in the sense that every model is scored on the same validation and test rows,
  but the SVM and KNN saw less data than the tree and linear models may see.
- **Neither P2 model was tuned.** Hyperparameter tuning was Person 3's job, and only
  for the model that won the comparison. The forest was tuned (25 trials); the tuning
  did not beat the untuned forest on full-data CV, so the untuned one shipped.
- **One random seed (42).** Repeating with other seeds would show how stable the ranking is.
- **Logistic regression's CV in the table used a temporary splitter** (P1's own code,
  noted in its record). For a strictly fair comparison it should be rerun with
  `evaluation/cv.py`. Raise this with Person 1.
- **Only 4 of 5 models exist** in the table. Person 4's HistGradientBoosting is
  excluded from this build by request, so no comparable record exists for it.
- **The decoder's `beta_m = 10` was the smallest value in its grid**, so the setting
  sits on the boundary. Smaller values were not tested.
- **The forest file is about 665 MB.** It is not stored in git; the manifest records
  its checksum and the code to rebuild it is in the repository.
- **Route continuity is not an accuracy measure.** It only checks that neighbouring
  roads connect. A route can be connected and still be wrong.
- **The reference route is not perfect ground truth.** A predicted road next to the
  true one may be physically reasonable and still count as wrong.

---

## 12. Code tour

### 12.1 Where things are

```
src/pathsnap/
  evaluation/
    cv.py          <- the fair test rig (P2)
    compare.py     <- saves records, builds comparison table, five-model check (P2)
  models/
    instance_based.py   <- Model 2, KNN (P2) + cost measurement
    kernel.py           <- Model 5, RBF SVM (P2) + Platt cost measurement
    base.py             <- the rules every model must follow (frozen, shared)
    registry.py         <- the list of five models in report order
    linear.py, tree.py, boosting.py   <- other people's models
  features/dataset.py   <- loads the prepared data (P1)
scripts/
  run_p2.py        <- runs everything for KNN and SVM (P2)
tests/
  test_cv.py       <- tests for cv.py and the P2 models
  test_compare.py  <- tests for compare.py
experiments/
  results/         <- model_*.json, cv_*.json, comparison.csv/.md (generated)
  configs/p2_run.json   <- what settings the last run used
artifacts/
  p2_knn.joblib, p2_svm_rbf.joblib  <- saved trained models (not pushed to GitHub)
docs/
  EXPERIMENT_LOG.md  <- diary of every run, with results and what they changed
  P2_EXPLAINED.md    <- this document
```

### 12.2 The key functions, in plain words

**`cv.py`**

- `group_kfold(...)` splits the data into 5 parts so that whole trips stay together.
  It uses scikit-learn's `GroupKFold`, a standard library tool.
- `cross_validate_model(...)` for each fold: copies the model, trains it on four
  fifths, scores the other fifth, and records macro F1, top-1 accuracy, log loss and
  the zero-probability fraction. Returns the mean and standard deviation.
- `observation_level_predictions(...)` does the "one pick per GPS point" step: for
  each point, the candidate with the highest probability wins. It returns the true
  road and the predicted road for every point.
- `zero_true_probability_fraction(...)` counts how often the true road got probability
  exactly 0.

**`instance_based.py` (KNN)**

- `KNNModel.fit(...)`: standardise the features and store the (sub-sampled) training
  rows. Uses scikit-learn's `KNeighborsClassifier` inside a `Pipeline`, which chains
  "standardise, then KNN" so the scaler only ever sees training data.
- `KNNModel.predict_proba(...)`: returns, for each candidate, the probability of being
  the correct road.
- `measure_knn_inference_cost(...)`: times predictions for 25k, 50k and 100k stored
  rows to show how cost grows.

**`kernel.py` (SVM)**

- `SVMModel.fit(...)`: subsamples the rows (stratified, so the share of correct roads
  is kept), standardises, and trains `SVC(kernel="rbf", probability=True)`.
- `SVMModel.predict_proba(...)`: returns probabilities after Platt scaling.
- `measure_platt_scaling_cost(...)`: trains the same SVM with and without Platt
  scaling and reports the time ratio.

**`compare.py`**

- `save_record(...)`: writes one model's results to `model_<name>.json`.
- `load_records(...)` and `build_comparison_table(...)`: read all records and write
  `comparison.csv` and `comparison.md` in the order the report needs.
- `assert_five_models(...)`: fails loudly if fewer than five models have a record.

**`scripts/run_p2.py`**, in order:

1. Load train and student_test, keep trusted rows only.
2. For k = 3, 5, 7, 11: run the 5-fold CV for KNN and keep the best k.
3. Measure KNN cost at three sizes.
4. Train the final KNN, score it on student_test.
5. Measure the Platt cost, run the 5-fold CV for SVM, train the final SVM, score it.
6. Save the two records and the two joblib files, rebuild the table, check the five-model rule.

### 12.3 Commands to know

```powershell
python -m pytest -q                        # all tests (should say 62 passed)
python scripts/run_p2.py                   # rerun KNN + SVM (about 20 minutes)
python scripts/run_p2.py --skip-svm        # KNN only
python scripts/run_p2.py --skip-knn        # SVM only
python scripts/run_p2.py --table-only      # rebuild the comparison table (no training, seconds)
```

### 12.4 Two details worth knowing

- **Metrics are computed on trusted rows only.** The 1.5% (student test) to 7% (train)
  of rows whose answer key is unreliable are not scored, but they are still available
  for predictions.
- **Candidate-miss points get the label -1.** If the correct road was never among the
  candidates, the point is counted as wrong. This is deliberate: it stops a pipeline
  problem from being hidden.

---

## 13. Presentation guide

### 13.1 A 60-second version

> "Our system reads a car's noisy GPS trail and works out which road it drove on, so
> we can rebuild the route. My part is the test rig and two of the five models, KNN
> and an RBF support vector machine. Every model is tested by splitting data by trip,
> so nothing leaks. The SVM is the best of mine: 81% of GPS points correctly matched on
> the student test, with well-behaved probabilities. KNN is close on accuracy, but it
> sometimes gives the true road a probability of exactly zero, which breaks the route
> decoder. The SVM is slow, so it was trained on 2% of the data, and we say so."

### 13.2 A 5-minute outline

1. **The problem** (section 1): one picture, GPS trail to road route. 30 seconds.
2. **The data and the pipeline** (sections 3 and 4): candidates, 19 features, classifier,
   Viterbi. Name the five-step chain. 60 seconds.
3. **Fair testing** (section 7): trip-level splits, per-point accuracy, macro F1, log
   loss. One example of log loss from the table. 60 seconds.
4. **KNN** (section 8.1): "copy the vote of the 11 most similar examples". Show the
   zero-probability issue. 60 seconds.
5. **SVM** (section 8.2): "a curved fence with support vectors", Platt scaling and its
   6.7× cost. 60 seconds.
6. **Results** (section 9.1): the table. Point to the SVM's 0.806 and the infinite log
   loss of KNN. 45 seconds.
7. **Honest limits** (section 11): k selection, subsampling, logistic regression's
   strength. 45 seconds.
8. **What's next**: random forest and boosting, the HMM, the API. 15 seconds.

### 13.3 Things to show live if asked (no ML needed)

- `experiments/results/comparison.md`: the table, generated, not typed.
- `experiments/results/model_svm_rbf.json`: one record, with the subsample size stated.
- `python -m pytest -q`: 62 passed.

### 13.4 Phrases that impress and are true

- "Every number in the report is generated by code from saved results; none is typed by hand."
- "We split by trip, so the model cannot see a drive it was trained on."
- "Accuracy alone would hide the zero-probability problem; log loss shows it."
- "We disclose every subsample and every choice that biases a score."

---

## 14. Likely questions and answers

**Q: What is machine learning here, in one sentence?**
A: We show the computer many candidate roads whose right or wrong answers we know, and
it learns a rule that scores new candidates.

**Q: Why not just pick the nearest road?**
A: We tested it. Using only distance (geometry) gives the same answers as the nearest
road. The sequence numbers (how the car moved from the previous point) add about 26
points of macro F1.

**Q: Why KNN and SVM?**
A: They come from different families. KNN compares with stored examples, the SVM draws a
curved boundary. The brief asks for several families, and these two cover two of them.

**Q: Why macro F1 and not accuracy?**
A: Accuracy can look high while rare roads are always missed. Macro F1 gives every road
the same weight, so rare roads count.

**Q: What does "log loss N/A" mean?**
A: The model gave the true answer probability 0 at least once, so the penalty would be
infinite. We report N/A and also say how often it happens: 0.59% of test candidates.

**Q: Why is KNN's log loss worse than its accuracy suggests?**
A: Accuracy only asks whether the top candidate was right. Log loss also asks how
confident the model was. KNN is often right, but when it is wrong it is sometimes
certain, and it gives the true road exactly zero. The SVM's probabilities are smoother
and never reach zero on test.

**Q: Why is the SVM not the final model?**
A: It is the best of our two, but it is the slowest per point, it was trained on only
2.2% of the data, and logistic regression matches its macro F1 at a fraction of the
cost. The final choice belongs to the five-model table and to the decoder's needs.

**Q: What is a support vector?**
A: A training example that sits right next to the SVM's boundary. The other examples do
not change the boundary. We ended with 4,533 of them.

**Q: What is Platt scaling?**
A: A way to turn the SVM's raw score into a probability using a sigmoid curve. It needs
extra training work, which is why it made training 6.7 times slower.

**Q: Why did you subsample?**
A: Full data would take hours, and the team has a time budget. KNN uses 7.4% and the SVM
2.2%. We state this in every record and in the table's configuration column. We did not
pretend it was full data.

**Q: Did you tune the models?**
A: No. KNN's k was chosen from four values, which is the only tuning. Tuning the winner
is Person 3's job.

**Q: How do you know there is no cheating?**
A: The trip split means no drive is in both training and testing. The answer key is used
only for scoring, never as an input. The student test set was not used to choose k or
any other setting; it is only used to report results.

**Q: What is route continuity?**
A: The fraction of neighbouring predicted roads that connect on the map. About 0.90 for
both P2 models: roughly one jump in ten is not drivable. The Viterbi decoder (Person 4)
uses the road network to enforce connections and reaches 0.96 on the student-test trips,
but it does not fix every trip (it helps 5 of 10).

**Q: How long does prediction take?**
A: Per GPS point: logistic regression 0.003 ms, random forest 0.034 ms, KNN 0.53 ms,
SVM 1.73 ms. All are far below the time a car needs to travel between points.

**Q: Why 100 metres and 10 candidates?**
A: Person 1 measured it. 100 m and 10 roads contain the right road for 98.8% of trusted
points. A larger radius adds 17% more data for 0.4 more percentage points.

---

## 15. Status and next steps

**Done (P2):**
- Fair CV harness, shared by everyone.
- KNN (k = 11): trained, scored, cost measured.
- RBF SVM: trained, Platt cost measured, scored.
- Comparison table generator with the five-model check.
- 25 P2-specific tests; the full suite is now 93 tests, all passing.

**Done by the rest of the team in this build:**
- P3: random forest (CV 0.6627, student-test top-1 0.8148), 25-trial Optuna search.
  The tuned configuration lost on full-data CV, so the untuned forest is the saved
  `artifacts/model.joblib`.
- P4: Viterbi decoder tuned on held-out training trips (student test 0.8249 vs
  0.8148 classifier-only, 5 trips better and 5 worse), FastAPI service, Streamlit UI,
  prediction maps and an artifact manifest.
- MLOps: Dockerfile, GitHub Actions workflow template (`ci/github-actions.yml`),
  PowerShell run scripts and JSONL request logging.

**Open before submission:**
- The table has 4 of 5 rows. Person 4's HistGradientBoosting is excluded from this
  build; do **not** fill the fifth row with the stand-in numbers, which use a
  different protocol.
- Run the UI from a second laptop against the API (only headless and same-laptop
  tests were run here).
- Build the Docker image on a machine with Docker installed (this laptop does not
  have it); the Dockerfile and CI workflow are written but not executed locally.
- Person 1 should rerun logistic regression with `evaluation/cv.py` (the temporary
  splitter is recorded in its record).
- Person 4's model 4 remains the owner's deliverable.
- The 665 MB forest is not in git. Ship it separately with `artifacts/manifest.json`
  so the checksum and git commit can be verified.

Answer the viva question "Can you explain why KNN probabilities are coarse and what
that costs the HMM?" using sections 8.1 and 10.
