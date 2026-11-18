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
    max_network_arcs: int = 5000,
    max_track_points: int = 3000,
) -> Path:
    """Raw GPS, road network, reference route, classifier-only route and Viterbi route.

    OWNER: Person 4. Layers are toggleable; the Viterbi route is drawn solid and
    strongest, the classifier-only route dashed, the reference route green.
    Route lines are built with arc_sequence_to_coords so consecutive roads join
    into one polyline. Unplaced observations (-1) are skipped.
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
        visible = visible[:: math.ceil(n_visible / max_network_arcs)]

    m = folium.Map(
        location=[(lat0 + lat1) / 2.0, (lon0 + lon1) / 2.0],
        zoom_start=15,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    fg_net = folium.FeatureGroup(name=f"Road network ({len(visible):,} of {n_visible:,} arcs drawn)")
    net_lines = [
        [[float(a_lat[i]), float(a_lon[i])], [float(b_lat[i]), float(b_lon[i])]] for i in visible
    ]
    folium.PolyLine(net_lines, color="#7f8c8d", weight=1, opacity=0.45).add_to(fg_net)
    fg_net.add_to(m)

    if trip.has_reference_route:
        ref_coords = arc_sequence_to_coords(trip.route, net)
        fg_ref = folium.FeatureGroup(name=f"Reference route ({len(trip.route)} arcs)")
        folium.PolyLine(
            [[lat, lon] for lon, lat in ref_coords], color="#27ae60", weight=4, opacity=0.8
        ).add_to(fg_ref)
        fg_ref.add_to(m)

    if classifier_only_arc_per_obs is not None:
        clf_arcs = _drop_unplaced(classifier_only_arc_per_obs)
        clf_coords = arc_sequence_to_coords(clf_arcs, net)
        fg_clf = folium.FeatureGroup(name=f"Classifier-only route ({len(clf_arcs)} obs)")
        folium.PolyLine(
            [[lat, lon] for lon, lat in clf_coords], color="#e67e22", weight=3, opacity=0.8, dash_array="6 6"
        ).add_to(fg_clf)
        fg_clf.add_to(m)

    hmm_arcs = _drop_unplaced(pred_arc_per_obs)
    hmm_coords = arc_sequence_to_coords(hmm_arcs, net)
    fg_hmm = folium.FeatureGroup(name=f"Predicted route: classifier + Viterbi ({len(hmm_arcs)} obs)")
    folium.PolyLine(
        [[lat, lon] for lon, lat in hmm_coords], color="#8e44ad", weight=5, opacity=0.9
    ).add_to(fg_hmm)
    fg_hmm.add_to(m)

    stride = max(1, len(track) // max_track_points)
    fg_gps = folium.FeatureGroup(name=f"Raw GPS observations (n={len(track):,})")
    for i in range(0, len(track), stride):
        folium.CircleMarker(
            [float(track.lat[i]), float(track.lon[i])],
            radius=2, color="#1b4f72", weight=0, fill=True, fill_color="#2980b9", fill_opacity=0.9,
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    _add_prediction_legend(m, trip.trip_id, stride)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat0, lon0], [lat1, lon1]])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return out_path


def _drop_unplaced(arc_per_obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(arc_per_obs, dtype=np.int64)
    return arr[arr >= 0]


def _add_prediction_legend(map_obj, trip_id: str, stride: int) -> None:
    import folium

    note = (
        f'<div style="font-size:11px;color:#555">GPS drawn every {stride} obs</div>' if stride > 1 else ""
    )
    html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 12px; border: 1px solid #bbb;
                border-radius: 4px; font-family: sans-serif; font-size: 12px;">
      <b>PathSnap prediction, trip {trip_id}</b>
      <div style="margin-top:6px"><span style="display:inline-block;width:22px;height:4px;background:#8e44ad;vertical-align:middle;margin-right:6px"></span>classifier + Viterbi (solid)</div>
      <div><span style="display:inline-block;width:22px;height:3px;background:#e67e22;vertical-align:middle;margin-right:6px"></span>classifier only (dashed)</div>
      <div><span style="display:inline-block;width:22px;height:4px;background:#27ae60;vertical-align:middle;margin-right:6px"></span>reference route (if available)</div>
      <div><span style="display:inline-block;width:22px;height:3px;background:#2980b9;vertical-align:middle;margin-right:6px"></span>raw GPS observations</div>
      <div><span style="display:inline-block;width:22px;height:3px;background:#7f8c8d;vertical-align:middle;margin-right:6px"></span>road network</div>
      {note}
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(html))


def arc_sequence_to_coords(arc_ids: np.ndarray, network: RoadNetwork) -> list[list[float]]:
    """Convert a predicted arc sequence to one continuous [[lon, lat], ...] polyline.

    For each arc we add its tail then its head node. When the next arc starts where
    the previous one ends, the shared node is recorded once, so the line does not
    zig-zag through midpoints. A break in connectivity shows as a straight jump.

    Shared with the API, which returns route_geometry for the UI to draw.
    """
    arcs = np.asarray(arc_ids, dtype=np.int64)
    arcs = arcs[arcs >= 0]
    if arcs.size == 0:
        return []
    coords: list[list[float]] = []
    prev_arc = None
    for arc in arcs.tolist():
        if arc == prev_arc:
            continue
        tail, head = network.arcs[arc]
        for node in (int(tail), int(head)):
            lon, lat = float(network.nodes[node, 0]), float(network.nodes[node, 1])
            if not coords or coords[-1] != [lon, lat]:
                coords.append([lon, lat])
        prev_arc = arc
    return coords
