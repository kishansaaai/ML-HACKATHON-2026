"""Folium map for the report: GPS points, road network, reference route and predicted route (development trips only).

    python scripts/make_prediction_map.py --trip 00000052 --part student_test

The reference route (.route) is read here only AFTER the prediction has been made, to draw it for the
report; the prediction itself never sees it. Do not run this on evaluator data.
Output: reports/prediction_map_<trip>.html
"""
import argparse
from pathlib import Path

import folium
import numpy as np

from pathsnap.io import load_trip
from pathsnap.paths import DATA_ROOT
from pathsnap.pipeline import predict

ROOT = Path(__file__).resolve().parents[1]


def _arc_segments(trip, arc_ids):
    """lon/lat -> folium (lat, lon) polylines; consecutive arcs that share a node are joined, others start a new line."""
    lines, current, prev_to = [], [], None
    for a in arc_ids:
        u, v = (int(x) for x in trip.arcs[int(a)])
        if prev_to is not None and u != prev_to and current:
            lines.append(current)
            current = []
        if not current:
            current.append((float(trip.node_lonlat[u, 1]), float(trip.node_lonlat[u, 0])))
        current.append((float(trip.node_lonlat[v, 1]), float(trip.node_lonlat[v, 0])))
        prev_to = v
    if current:
        lines.append(current)
    return lines


def build_map(trip_id: str, part: str) -> folium.Map:
    trip_dir = DATA_ROOT / part / trip_id
    pred = predict(trip_dir)                                   # no ground truth is read here
    trip = load_trip(trip_dir, with_labels=True)               # reference route loaded only after predicting
    center = [float(np.median(trip.lat)), float(np.median(trip.lon))]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    # road network: only arcs inside the GPS bounding box (+ margin) to keep the file small
    pad = 0.004
    lo, hi = (trip.lon.min() - pad, trip.lat.min() - pad), (trip.lon.max() + pad, trip.lat.max() + pad)
    net = folium.FeatureGroup(name="Road network (this trip)", show=True)
    for u, v in trip.arcs:
        (x1, y1), (x2, y2) = trip.node_lonlat[int(u)], trip.node_lonlat[int(v)]
        if max(x1, x2) < lo[0] or min(x1, x2) > hi[0] or max(y1, y2) < lo[1] or min(y1, y2) > hi[1]:
            continue
        folium.PolyLine([(y1, x1), (y2, x2)], color="#888888", weight=1, opacity=0.5).add_to(net)
    net.add_to(fmap)

    ref = folium.FeatureGroup(name="Reference route (ground truth, drawn after prediction)", show=True)
    for line in _arc_segments(trip, trip.route.tolist()):
        folium.PolyLine(line, color="#ff7f0e", weight=6, opacity=0.8).add_to(ref)
    ref.add_to(fmap)

    hmm = folium.FeatureGroup(name="Predicted route (classifier + HMM)", show=True)
    for seg in pred["route_polyline"]:                          # decoder gives [lon, lat]
        folium.PolyLine([(p[1], p[0]) for p in seg], color="#1f77b4", weight=3, opacity=0.95).add_to(hmm)
    hmm.add_to(fmap)

    gps = folium.FeatureGroup(name="Raw GPS points", show=True)
    step = max(1, trip.n_obs // 1500)
    for i in range(0, trip.n_obs, step):
        folium.CircleMarker((float(trip.lat[i]), float(trip.lon[i])), radius=2, color="#d62728",
                            fill=True, fill_opacity=0.8, weight=0).add_to(gps)
    gps.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;'
              'border:1px solid #999;font-size:12px;">'
              '<b>Trip %s</b><br><span style="color:#d62728">&#9679;</span> raw GPS<br>'
              '<span style="color:#ff7f0e">&#9644;</span> reference route<br>'
              '<span style="color:#1f77b4">&#9644;</span> predicted route<br>'
              '<span style="color:#888">&#9644;</span> road network</div>' % trip_id)
    fmap.get_root().html.add_child(folium.Element(legend))
    return fmap


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="00000052")
    ap.add_argument("--part", default="student_test", choices=["train", "student_test"])
    a = ap.parse_args()
    out = ROOT / "reports" / f"prediction_map_{a.trip}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_map(a.trip, a.part).save(str(out))
    print("saved", out, f"({out.stat().st_size / 1e6:.2f} MB)")
