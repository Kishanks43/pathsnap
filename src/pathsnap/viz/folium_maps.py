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


def build_eda_map(
    trip: Trip,
    out_path: Path,
    max_network_arcs: int = 5000,
    max_track_points: int = 3000,
) -> Path:
    """Raw GPS observations + road network, as two toggleable layers.

    OWNER: Person 1 (implemented).

    Display decisions, all of which the report should mention:
    - Networks reach 307k arcs. Drawing them all makes a map that takes tens of
      seconds to open on the evaluator's laptop, so arcs are clipped to the
      track's bounding box (plus a small margin) and then uniformly subsampled to
      `max_network_arcs` for display only. Feature computation always uses the
      full network - this cap is cosmetic.
    - Every arc is drawn as one MultiPolyline collection rather than 5000 separate
      PolyLine objects; same picture, far faster to render.
    - The reference route is drawn when the trip has one, so the reader can see
      where the GPS actually sat relative to the labelled route.
    """
    import math

    import folium

    track, net = trip.track, trip.network
    lon0, lon1 = float(track.lon.min()), float(track.lon.max())
    lat0, lat1 = float(track.lat.min()), float(track.lat.max())
    pad = 0.002

    node_lon, node_lat = net.nodes[:, 0], net.nodes[:, 1]
    a_lon, a_lat = node_lon[net.arcs[:, 0]], node_lat[net.arcs[:, 0]]
    b_lon, b_lat = node_lon[net.arcs[:, 1]], node_lat[net.arcs[:, 1]]

    inside = (
        (a_lon >= lon0 - pad) & (a_lon <= lon1 + pad) & (a_lat >= lat0 - pad) & (a_lat <= lat1 + pad)
    ) | ((b_lon >= lon0 - pad) & (b_lon <= lon1 + pad) & (b_lat >= lat0 - pad) & (b_lat <= lat1 + pad))
    visible = np.flatnonzero(inside)
    n_visible = int(visible.size)
    if n_visible > max_network_arcs:
        step = math.ceil(n_visible / max_network_arcs)
        visible = visible[::step]

    m = folium.Map(
        location=[(lat0 + lat1) / 2.0, (lon0 + lon1) / 2.0],
        zoom_start=15,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    # --- road network ---------------------------------------------------------
    net_lines = [
        [[float(a_lat[i]), float(a_lon[i])], [float(b_lat[i]), float(b_lon[i])]] for i in visible
    ]
    fg_net = folium.FeatureGroup(
        name=f"Road network ({len(visible):,} of {n_visible:,} visible arcs drawn)"
    )
    folium.PolyLine(net_lines, color="#7f8c8d", weight=1, opacity=0.45).add_to(fg_net)
    fg_net.add_to(m)

    # --- reference route (if supplied) ---------------------------------------
    if trip.has_reference_route:
        route = trip.route
        r_lon, r_lat = node_lon[net.arcs[route, 0]], node_lat[net.arcs[route, 0]]
        r_lon2, r_lat2 = node_lon[net.arcs[route, 1]], node_lat[net.arcs[route, 1]]
        route_lines = [
            [[float(y0), float(x0)], [float(y1), float(x1)]]
            for x0, y0, x1, y1 in zip(r_lon, r_lat, r_lon2, r_lat2)
        ]
        fg_route = folium.FeatureGroup(name=f"Reference route ({len(route)} arcs)")
        folium.PolyLine(route_lines, color="#27ae60", weight=3, opacity=0.8).add_to(fg_route)
        fg_route.add_to(m)

    # --- raw GPS -------------------------------------------------------------
    stride = max(1, len(track) // max_track_points)
    idx = range(0, len(track), stride)
    pts = [[float(track.lat[i]), float(track.lon[i])] for i in idx]
    fg_gps = folium.FeatureGroup(name=f"Raw GPS observations (n={len(track):,})")
    folium.PolyLine(pts, color="#2980b9", weight=2, opacity=0.7).add_to(fg_gps)
    for p in pts:
        folium.CircleMarker(
            p, radius=2, color="#1b4f72", weight=0, fill=True, fill_color="#2980b9", fill_opacity=0.9
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    _add_legend(m, title=f"PathSnap EDA — trip {trip.trip_id}", stride=stride, n_drawn=len(visible))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat0, lon0], [lat1, lon1]])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return out_path


def _add_legend(map_obj, title: str, stride: int, n_drawn: int) -> None:
    """Small fixed-position HTML legend; the brief explicitly asks for one."""
    note = (
        f'<div style="font-size:11px;color:#555">GPS drawn every {stride} obs</div>'
        if stride > 1
        else ""
    )
    html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 12px; border: 1px solid #bbb;
                border-radius: 4px; font-family: sans-serif; font-size: 12px;">
      <b>{title}</b>
      <div style="margin-top:6px">
        <span style="display:inline-block;width:22px;height:3px;background:#2980b9;
                     vertical-align:middle;margin-right:6px"></span>raw GPS observations
      </div>
      <div>
        <span style="display:inline-block;width:22px;height:3px;background:#7f8c8d;
                     vertical-align:middle;margin-right:6px"></span>road network ({n_drawn:,} arcs)
      </div>
      <div>
        <span style="display:inline-block;width:22px;height:3px;background:#27ae60;
                     vertical-align:middle;margin-right:6px"></span>reference route
      </div>
      {note}
    </div>
    """
    import folium

    map_obj.get_root().html.add_child(folium.Element(html))


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
