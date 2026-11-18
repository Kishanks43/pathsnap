# UI (Laptop 2)

OWNER: Person 4

The evaluator-facing interface. It runs on a **different laptop** from the API,
talks to it over HTTP, and must:

1. List the ten evaluator trips (fetched from `GET /trips`).
2. Send the selected trip to `POST /predict/{trip_id}`.
3. Draw the returned `route_geometry` on a map, with the raw GPS observations.
4. Show the server-returned metrics.

Suggested implementation: Streamlit + `streamlit-folium`. It is the fastest way
to a working UI and it is easy to screenshot for the report.

```powershell
# API base URL is the other laptop's LAN address, not localhost
$env:PATHSNAP_API="http://192.168.1.42:8000"
streamlit run ui/app.py
```

Rubric reminders for this component:

- Screenshots of the working UI go in the report (Deliverables section 7).
- Screenshots of the API responses **on the receiving laptop's terminal** also
  go in the report. Set the API logging up so those lines are readable.
- The submitted code must match what is demonstrated. Do not demo a notebook
  while the repo contains something else.
