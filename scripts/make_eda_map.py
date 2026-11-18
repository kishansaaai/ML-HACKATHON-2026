"""Create an EDA Folium map for one supplied trip (not a prediction result)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.io import load_trip  # noqa: E402
from pathsnap.pipeline import predict  # noqa: E402


def make_map(trip_dir: Path, output: Path, max_road_arcs: int = 2500):
    import folium
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prediction = predict(trip_dir)  # prediction reads track/nodes/arcs; labels are loaded below for the reference layer only
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

    gps_layer = folium.FeatureGroup(name="Raw GPS observations", show=True)
    for index, point in enumerate(latlon):
        folium.CircleMarker(point, radius=2, color="#2166f3", weight=1, fill=True,
                            fill_color="#2166f3", fill_opacity=0.65,
                            tooltip=f"GPS observation {index}").add_to(gps_layer)
    gps_layer.add_to(fmap)
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

    predicted = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)", show=True)
    for segment in prediction["route_polyline"]:
        if len(segment) > 1:
            folium.PolyLine(
                [(lat, lon) for lon, lat in segment],
                color="#00a65a", weight=5, opacity=0.9,
                tooltip="Predicted route (HMM/Viterbi)",
            ).add_to(predicted)
    predicted.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(latlon)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output)

    # A static export of the same real trip layers for the consolidated PDF.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    stride = max(1, (len(trip.arcs) + max_road_arcs - 1) // max_road_arcs)
    for arc_id in range(0, len(trip.arcs), stride):
        source, target = trip.arcs[arc_id]
        a, b = trip.node_lonlat[source], trip.node_lonlat[target]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#aab3bf", linewidth=0.45, alpha=0.35)
    ax.scatter(trip.lon, trip.lat, color="#2166f3", s=5, alpha=0.65, label="Raw GPS observations", zorder=3)
    if route_points:
        ax.plot([p[1] for p in route_points], [p[0] for p in route_points],
                color="#e53935", linewidth=2.0, linestyle="--", label="Reference route", zorder=4)
    for index, segment in enumerate(prediction["route_polyline"]):
        if len(segment) > 1:
            ax.plot([p[0] for p in segment], [p[1] for p in segment], color="#00a65a",
                    linewidth=2.5, label="Predicted HMM route" if index == 0 else None, zorder=5)
    ax.scatter([trip.lon[0], trip.lon[-1]], [trip.lat[0], trip.lat[-1]],
               c=["green", "black"], s=28, label="Start / end", zorder=4)
    ax.set(title=f"{trip_dir.parent.name.title()} trip {trip.trip_id}: GPS, road network and route layers",
           xlabel="Longitude", ylabel="Latitude")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    static_path = Path(__file__).resolve().parents[1] / "reports" / "figures" / "report_map_view.png"
    static_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(static_path, dpi=180)
    plt.close(fig)
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
