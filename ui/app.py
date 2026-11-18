"""PathSnap evaluator UI (Streamlit). Runs on Laptop 2 and talks to the API on Laptop 1.

OWNER: Person 4

    streamlit run ui/app.py -- --api http://<laptop-1-ip>:8000

or set the environment variable PATHSNAP_API (default http://127.0.0.1:8000).

The UI never sees ground truth: it only displays what the API returns (predicted
route geometry, predicted road per observation and aggregate metrics).
"""

from __future__ import annotations

import os
import sys

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_API = os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")


def _api_base() -> str:
    args = sys.argv[1:]
    if "--api" in args:
        idx = args.index("--api")
        if idx + 1 < len(args):
            return args[idx + 1].rstrip("/")
    return DEFAULT_API.rstrip("/")


def _get(base: str, path: str) -> dict:
    response = requests.get(f"{base}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def _post(base: str, path: str) -> dict:
    response = requests.post(f"{base}{path}", timeout=300)
    response.raise_for_status()
    return response.json()


def _route_map(route_geometry: list[list[float]]) -> str:
    """Folium map of the predicted route only (no reference data is available here)."""
    if not route_geometry:
        return "<p>No route could be decoded for this trip.</p>"
    lats = [p[1] for p in route_geometry]
    lons = [p[0] for p in route_geometry]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=15)
    folium.PolyLine(
        [[lat, lon] for lon, lat in route_geometry], color="#8e44ad", weight=5, opacity=0.9,
        tooltip="Predicted route (classifier + Viterbi)",
    ).add_to(m)
    folium.Marker([lats[0], lons[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lats[-1], lons[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m.get_root().render()


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def main() -> None:
    base = _api_base()
    st.set_page_config(page_title="PathSnap", layout="wide")
    st.title("PathSnap: GPS map matching")
    st.caption(f"API: {base}")

    try:
        health = _get(base, "/health")
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {base}: {exc}")
        st.info("Start it with: uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000")
        return

    if not health.get("artifacts_loaded"):
        st.error(f"API is up but artifacts are not loaded (status: {health.get('status')}).")
        return

    left, right = st.columns([1, 3])
    with left:
        st.subheader("Trip")
        st.write(f"Model: **{health['model']}**")
        trips = _get(base, "/trips")["trips"]
        trip_id = st.selectbox("Choose a trip", trips)
        run = st.button("Predict route", type="primary", use_container_width=True)

    with right:
        if not run:
            st.info("Pick a trip and press **Predict route**.")
            return
        with st.spinner("Matching GPS points to roads..."):
            try:
                result = _post(base, f"/predict/{trip_id}")
            except requests.HTTPError as exc:
                st.error(f"Prediction failed: {exc.response.status_code} {exc.response.text}")
                return
            except requests.RequestException as exc:
                st.error(f"Prediction failed: {exc}")
                return

        metrics = result["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", f"{result['n_observations']:,}")
        c2.metric("Route continuity", _pct(metrics.get("route_continuity")))
        c3.metric("Top-1 (HMM)", _pct(metrics.get("top1_accuracy")))
        c4.metric("Top-1 (classifier only)", _pct(metrics.get("classifier_only_top1_accuracy")))

        if metrics.get("reference_available"):
            st.caption(
                "Accuracy is measured against the reference route (available for this trip). "
                "Evaluator trips have no reference, so these fields show N/A."
            )
        else:
            st.caption("No reference route for this trip: only the prediction is shown.")

        st.iframe(_route_map(result["route_geometry"]), height=520)

        with st.expander("Decoder diagnostics"):
            st.json(result.get("diagnostics", {}))
        with st.expander("Predicted road per observation (first 50)"):
            st.dataframe({"observation": range(min(50, len(result["arc_per_obs"]))),
                          "road_id": result["arc_per_obs"][:50]})


main()
