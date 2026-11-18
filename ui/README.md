# Evaluator UI

A Streamlit page that calls the FastAPI service and draws the predicted route.
It never receives ground truth: it shows the predicted route, the predicted road
per observation and the aggregate metrics the API returns.

## Run it

Laptop 1 (API, holds the models and the trip data):

```powershell
.\scripts\run_api.ps1
```

Laptop 2 (UI), replacing the address with Laptop 1's LAN address:

```powershell
.\scripts\run_ui.ps1 -Api http://192.168.1.20:8000
```

Then open <http://localhost:8501>, pick a trip and press **Predict route**.

## What the page shows

- Model name and API address (top of the left column).
- Four numbers: observations, route continuity, top-1 accuracy of the decoded route
  and top-1 of the classifier alone. Accuracy is only shown when a reference route
  exists for the trip; otherwise N/A.
- The predicted route on an interactive map (start and end markers).
- Decoder diagnostics (gap resets, forced steps, observations without candidates).
- The first 50 predicted road ids.
