# Measured dataset facts

Numbers produced by `python scripts/check_data.py` on the provided data. Re-run it
if anything looks off. These matter because two of them change the design.

```
train         : 80 trips, 208,417 observations, 1,933,485 nodes, 3,876,904 arcs
student_test  : 10 trips,  19,160 observations,    90,689 nodes,   185,041 arcs
observations/trip : min 361, median 1,529, max 19,313
nodes/trip        : up to ~100k      arcs/trip : up to ~200k
median dt         : 1.00 s           worst gap : 153,064 s (train), 210 s (student_test)
```

## Fact 1 — trips are from different cities around the world

The combined longitude range is `-100.4 .. 137.2` and latitude `-34.6 .. 61.3`.
A single trip is locally coherent, but **different trips are in different
countries**. Consequences:

- Never compute features or normalise anything across trips in a way that assumes
  a shared geography. No global bounding box, no "distance from city centre".
- A local projection must be anchored **per trip** (its own centroid). A global
  projection would distort distances for most trips.
- Any spatial index (grid, KD-tree) is rebuilt per trip, not once for the dataset.
- Don't compare raw lon/lat values between trips; they are not comparable.

This is also a leakage trap: normalising coordinates globally means a validation
trip's location influences a training trip's features. Normalise within trip only.

## Fact 2 — the networks are large; brute-force candidate generation will not fit

`~200k arcs` for the biggest trips and up to `19,313 observations` in one trip.
Naively comparing every observation against every arc is `19k x 200k = 3.9e9`
point-to-segment evaluations for a single trip. In vectorised numpy that is tens
of GB of intermediates, and it will not finish inside the hackathon.

Required approach:
- Build a **uniform grid or KD-tree over arc bounding boxes** per trip, query only
  arcs within `radius_m` of each observation.
- Alternatively, vectorise in chunks of observations (e.g. 2,000 at a time) and
  discard candidates beyond `radius_m` as you go.
- Cache the result per trip; candidate generation should run once per trip, not
  once per model, not once per experiment.

Measure the time. If candidate generation takes minutes per trip, it will take
hours across 90 trips, and you will re-run it many times.

## Fact 3 — the candidate count drives every downstream cost

`208k observations x ~10 candidates = ~2M candidate rows` for training. That is
the real size of the learning problem, and it has direct consequences:

| Model | Practical note at ~2M rows |
|---|---|
| Logistic Regression | Fine with `saga` / `lbfgs`; a few minutes at most |
| KNN | Memory-heavy, and inference cost scales with training size — measure it |
| Random Forest | Fine with `n_jobs=-1`, but training time grows with rows |
| HistGradientBoosting | Fine; it bins features, so it is the fastest strong learner |
| RBF SVM | **Not feasible on the full matrix.** `probability=True` adds internal 5-fold Platt scaling. Subsample deliberately, state the sample size and how it was chosen, and report the cost. Do not silently train on 5% and present it as if it used all data. |

If every model is too slow, reduce `max_candidates` (e.g. 10 -> 5) rather than
reducing the number of trips: fewer trips means fewer CV folds and a weaker
evaluation, which is a bigger loss.

## Fact 4 — real GPS gaps exist, so the decoder must handle them

A 153,064 s gap (about 42 hours) appears in the training data. That is not a
mis-parse; timestamps are seconds and some trips have long discontinuities.
The decoder must not assume `dt` is always ~1 s:

- Do not use `dt` as a hard divisor without guarding against zeros and large values.
- Decide whether a gap resets the HMM chain, widens the transition tolerance, or
  allows a multi-hop path between candidates. Report how many observations were
  affected by whichever rule you choose.
- When computing observed speed, clip or flag implausible values instead of
  letting an outlier dominate a feature's scale.

## Fact 5 — every trip has a reference route, but it is not flawless

`80/80` train and `10/10` student-test trips have `.route` files. The brief warns
that reference routes are "reference information for the task", not verified
physical ground truth. So:

- Use `.route` for labels and scoring.
- Do not describe accuracy against it as "correctness"; describe it as agreement
  with the supplied reference.
- Note in the report that a predicted arc adjacent to the reference arc may be
  geographically reasonable but still counted wrong by an exact-match metric.

## Fact 6 — some reference routes cover only part of their trip (measured)

Comparing total route length against the distance the GPS actually travelled:

```
route_len / observed_travel_distance
  train         p5 = 0.38   p50 = 0.84   p75 = 0.92   max = 0.99
  student_test  p5 = 0.60   p50 = 0.87   max = 0.95
```

Ten training trips fall below 0.50, two of them badly:

| Trip | ratio | observed travel | route length | trusted obs |
|---|---|---|---|---|
| 00000033 | 0.13 | 47.8 km | 6.4 km | 36.7% |
| 00000002 | 0.14 | 55.9 km | 8.0 km | 89.1% |
| 00000054 | 0.19 | 169.3 km | 31.4 km | 32.0% |

These routes are connected chains (99–100% of consecutive arcs share a node), so
they are not corrupted — they simply describe a fraction of the trajectory. A
reference route that covers 13% of a drive cannot label the other 87%.

**Consequence:** per-observation labels derived from these routes are wrong for
large stretches, and any model trained on them learns noise. Measured residual
(distance from each observation to the reference arc it was assigned):

```
train        p50 = 5.6 m   p90 = 74.9 m   p95 = 140.2 m
             72.6% <= 10 m    87.1% <= 25 m    89.2% <= 50 m
student_test p50 = 5.2 m   p90 = 13.3 m
             79.9% <= 10 m    98.6% <= 25 m
```

**Decision taken (before any model was trained):** label trust = residual
`<= 25 m`. That keeps 93% of training candidate rows and 98.5% of student-test
rows. The criterion is a label-quality filter, is applied identically to both
partitions, and observations are still available at inference time.

Report this as a dataset limitation, not a modelling trick. It also explains why
trip 00000054's residual median is 138 m: the GPS is ~84 m from the nearest arc
in the whole network, so that trip is displaced from the network as well.

## Fact 7 — candidate radius: 100 m is the measured trade-off

Candidate miss rate (reference arc absent from the candidate set) over the 80
training trips, using trusted labels, versus the size of the resulting matrix:

| radius | max candidates | miss rate (trusted) | candidate rows |
|---|---|---|---|
| 50 m | 10 | 1.86% | 1,083,514 |
| 75 m | 10 | 1.44% | 1,313,622 |
| **100 m** | **10** | **1.22%** | **1,453,943** |
| 150 m | 10 | 0.85% | 1,708,157 |
| 100 m | 6 | 1.75% | 1,010,267 |

Chosen: **radius 100 m, max 10 candidates**. Going to 150 m buys 0.37 percentage
points for 17% more rows; dropping to 6 candidates costs 0.53 points to save 31%
of rows. The 100 m / 10 configuration is the knee.

Final prepared dataset (`experiments/results/candidate_quality.json`):

```
train        80 trips   208,417 obs   1,453,943 candidate rows   6.98 cand/obs
             miss mean 3.41% (median 0.73%)   trusted 93.0%
student_test 10 trips    19,160 obs     148,752 candidate rows   7.76 cand/obs
             miss mean 1.17%                 trusted 98.5%
```

The gap between the 3.41% headline miss rate and the 1.22% trusted-label rate is
entirely the two route-truncated trips. Quote both numbers, and say which is
which — quoting only the better one is the kind of selective reporting the
integrity policy is aimed at.
