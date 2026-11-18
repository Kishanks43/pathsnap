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

### [--] P1 — scaffold verification (template entry — replace or delete)

**Question:** does the loader/validator pipeline work on the real data?

**Setup**
- Partitions: train only
- Command: `python scripts/check_data.py`

**Expected:** 80 trips, no validation issues.

**Result:** _fill in the actual printed numbers_

**What this changes:** gates everyone else's work.

---

## Decisions log

Short, dated entries for choices that are hard to reverse. This is the section the
report's methodology quietly draws from.

| Date | Decision | Alternatives considered | Why |
|---|---|---|---|
| | Project to a local metric frame instead of using degrees | Euclidean in degrees, full geodesic (pyproj) | Degrees are meaningless for distance; local projection is sub-metre at city scale and much faster in inner loops |
| | Candidate generation by point-to-segment distance | Endpoint distance | Endpoint distance unfairly penalises long arcs |
| | Group-aware CV by `trip_id` | Random KFold | Rows from one trip leak that trip's geometry; random KFold inflates scores |
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
