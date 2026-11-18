"""Folium maps - mandatory deliverable, and the most under-rated one.

OWNER: Person 1 (EDA maps) and Person 4 (predicted-route maps)
STATUS: STUB

WHY IT IS WORTH DOING PROPERLY
------------------------------
Deliverables section 6 makes Folium mandatory and asks for specific things:
distinguishable colours per layer, a legend, sensible zoom/extent, and layer
controls. It also requires at least one map screenshot in the PDF plus the map
code committed. This is a checklist you can score almost for free.

The map that actually earns marks is the comparison one: raw GPS points, road
network, reference route, and predicted route on the same map, with layer
toggles. That single artefact demonstrates the whole pipeline visually.

LAYERS TO BUILD
---------------
    raw GPS observations      - small marker circles, one colour
    road network              - thin grey polylines (subsample if the network is huge)
    reference route           - solid line, distinct colour (only where .route exists)
    predicted route (classifier only) - dashed line
    predicted route (+ Viterbi)        - solid line, strongest colour

REQUIRED OUTPUTS
----------------
    reports/maps/eda_<trip_id>.html            Person 1 (network + raw GPS)
    reports/maps/pred_<trip_id>.html           Person 4 (all layers)
    reports/maps/static_<trip_id>.png          screenshot for the PDF

Commit the HTML files (they are small) so the repository itself satisfies
"commit the interactive Folium HTML map or the code required to generate it".

PERFORMANCE NOTE
----------------
Some networks have tens of thousands of arcs. Drawing every arc as a separate
PolyLine makes a map that takes 30+ seconds to open on the evaluator's laptop.
Subsample the network for display, or use a single MultiPolyline. State what you
did; a map that fails to load is worse than no map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pathsnap.io.schema import RoadNetwork, Trip


def build_eda_map(trip: Trip, out_path: Path, max_network_arcs: int = 5000) -> Path:
    """Raw GPS observations + road network. Person 1's EDA artefact."""
    raise NotImplementedError("Person 1: implement the EDA map.")


def build_prediction_map(
    trip: Trip,
    pred_arc_per_obs: np.ndarray,
    out_path: Path,
    classifier_only_arc_per_obs: np.ndarray | None = None,
) -> Path:
    """Overlay observations, network, reference route and predicted route(s)."""
    raise NotImplementedError("Person 4: implement the prediction comparison map.")


def arc_sequence_to_coords(arc_ids: np.ndarray, network: RoadNetwork) -> list[list[float]]:
    """Convert a predicted arc sequence to a [[lon, lat], ...] polyline.

    Consecutive arcs that share a node should produce a continuous line; the
    polyline must not zig-zag back to each arc's midpoint. Deduplicate shared
    endpoints or the drawn route will look broken.

    Shared with the API, which returns route_geometry for the UI to draw.
    """
    raise NotImplementedError("Person 4: implement arc-sequence-to-polyline.")
