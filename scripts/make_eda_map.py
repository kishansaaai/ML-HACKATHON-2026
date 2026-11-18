"""Create an EDA Folium map for one supplied trip (not a prediction result)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.io import load_trip  # noqa: E402


def make_map(trip_dir: Path, output: Path, max_road_arcs: int = 2500):
    import folium

    trip = load_trip(trip_dir, with_labels=True)
    latlon = list(zip(trip.lat.tolist(), trip.lon.tolist()))
    center = [sum(p[0] for p in latlon) / len(latlon), sum(p[1] for p in latlon) / len(latlon)]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")

    roads = folium.FeatureGroup(name="Road network (sampled)", show=False)
    stride = max(1, (len(trip.arcs) + max_road_arcs - 1) // max_road_arcs)
    for arc_id in range(0, len(trip.arcs), stride):
        source, target = trip.arcs[arc_id]
        a, b = trip.node_lonlat[source], trip.node_lonlat[target]
        folium.PolyLine(
            [(a[1], a[0]), (b[1], b[0])],
            color="#8794a5", weight=1, opacity=0.45,
            tooltip=f"road arc {arc_id}",
        ).add_to(roads)
    roads.add_to(fmap)

    folium.PolyLine(latlon, color="#2166f3", weight=3, opacity=0.9, tooltip="Raw GPS track").add_to(
        folium.FeatureGroup(name="GPS trajectory", show=True).add_to(fmap)
    )
    for label, point, color in (("Start", latlon[0], "green"), ("End", latlon[-1], "red")):
        folium.CircleMarker(
            point, radius=5, color=color, fill=True, fill_opacity=1,
            tooltip=f"{label} observation",
        ).add_to(fmap)

    route = folium.FeatureGroup(name="Reference route (.route)", show=True)
    route_points = []
    for arc_id in trip.route if trip.route is not None else ():
        source, target = trip.arcs[arc_id]
        for node_id in (source, target):
            lon, lat = trip.node_lonlat[node_id]
            point = (lat, lon)
            if not route_points or point != route_points[-1]:
                route_points.append(point)
    if route_points:
        folium.PolyLine(
            route_points, color="#e53935", weight=4, opacity=0.9,
            tooltip="Supplied reference route",
        ).add_to(route)
    route.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(latlon)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", default="00000001", help="trip folder name (default: 00000001)")
    parser.add_argument("--partition", choices=("train", "student_test"), default="train")
    parser.add_argument("--data", type=Path, default=ROOT / "Student folder")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "eda_map.html")
    args = parser.parse_args()
    print(make_map(args.data / args.partition / args.trip, args.out))


if __name__ == "__main__":
    main()
