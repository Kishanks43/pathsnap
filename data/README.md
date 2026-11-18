# Data directory

Raw data is **not committed** (see `.gitignore`). Every teammate places it locally
in exactly this layout:

```
data/raw/
├── train/                     80 trips
│   └── <trip_id>/
│       ├── <trip_id>.track    lon  lat  timestamp
│       ├── <trip_id>.nodes    lon  lat              (row index = node_id)
│       ├── <trip_id>.arcs     from_node  to_node    (row index = arc_id)
│       └── <trip_id>.route    arc_id ...            (chronological ground truth)
└── student_test/              10 trips, same structure
```

The evaluator-testing partition (10 trips) is **not** in this repository and must
never be used during development. Do not request it from another team, do not
try to reconstruct it.

## Copying the data in (Windows, from the provided materials folder)

```powershell
# from the directory that contains the provided "Student folder"
Copy-Item -Recurse "..\Student folder\train"         data\raw\train
Copy-Item -Recurse "..\Student folder\student_test"  data\raw\student_test
```

Then verify:

```powershell
python scripts/check_data.py
```

Expected: `train: 80 trips`, `student_test: 10 trips`, no validation issues.

## Rules that are easy to break by accident

- **Never** copy `.route` files into `data/processed/` and then load them as
  features. The reference route is the answer.
- Preprocessing that learns parameters (scalers, feature rankings, candidate
  radius chosen from outcomes) must be fitted on **training trips only**.
- Do not move rows between trips. Splits are always by `trip_id`.
