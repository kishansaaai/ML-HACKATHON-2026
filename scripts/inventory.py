"""Generate a reproducible per-trip data quality inventory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from statistics import median
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.io import load_trip  # noqa: E402


def graph_summary(node_count, arcs):
    adjacency = [[] for _ in range(node_count)]
    degree = [0] * node_count
    invalid = 0
    self_loops = 0
    for source, target in arcs:
        if source >= node_count or target >= node_count or source < 0 or target < 0:
            invalid += 1
            continue
        adjacency[source].append(target)
        adjacency[target].append(source)
        degree[source] += 1
        degree[target] += 1
        self_loops += source == target
    seen = bytearray(node_count)
    components = 0
    for start in range(node_count):
        if seen[start] or not adjacency[start]:
            continue
        components += 1
        seen[start] = 1
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if not seen[neighbor]:
                    seen[neighbor] = 1
                    queue.append(neighbor)
    return {
        "connected_components_nonisolated_undirected": components,
        "isolated_nodes": sum(value == 0 for value in degree),
        "invalid_arc_endpoints": invalid,
        "self_loop_arcs": self_loops,
        "max_undirected_degree": max(degree, default=0),
    }


def summarize(directory: Path, partition: str):
    trip = load_trip(directory, with_labels=True)
    track = list(zip(trip.lon.tolist(), trip.lat.tolist(), trip.t.tolist()))
    nodes = [tuple(row) for row in trip.node_lonlat.tolist()]
    lons = [row[0] for row in track]
    lats = [row[1] for row in track]
    times = [row[2] for row in track]
    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    invalid_track = sum(
        not all(math.isfinite(v) for v in row)
        or not (-180 <= row[0] <= 180 and -90 <= row[1] <= 90)
        for row in track
    )
    invalid_nodes = sum(
        not all(math.isfinite(v) for v in row)
        or not (-180 <= row[0] <= 180 and -90 <= row[1] <= 90)
        for row in nodes
    )
    gap_counts = [sum(g > threshold for g in gaps) for threshold in (1, 2, 5, 10)]
    result = {
        "partition": partition,
        "trip_id": trip.trip_id,
        "track_rows": len(track),
        "node_rows": len(nodes),
        "arc_rows": len(trip.arcs),
        "route_rows": len(trip.route) if trip.route is not None else 0,
        "invalid_track_points": invalid_track,
        "invalid_node_coordinates": invalid_nodes,
        "duplicate_track_rows": len(track) - len(set(track)),
        "duplicate_node_coordinates": len(nodes) - len(set(nodes)),
        "nonincreasing_timestamps": sum(g <= 0 for g in gaps),
        "timestamp_gap_gt_1s": gap_counts[0],
        "timestamp_gap_gt_2s": gap_counts[1],
        "timestamp_gap_gt_5s": gap_counts[2],
        "timestamp_gap_gt_10s": gap_counts[3],
        "median_timestamp_gap_s": median(gaps) if gaps else "",
        "track_lon_min": min(lons, default=""),
        "track_lon_max": max(lons, default=""),
        "track_lat_min": min(lats, default=""),
        "track_lat_max": max(lats, default=""),
    }
    result.update(graph_summary(len(nodes), trip.arcs))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "Student folder")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "data_inventory.csv")
    args = parser.parse_args()
    rows = []
    for partition in ("train", "student_test"):
        partition_dir = args.data / partition
        for directory in sorted(p for p in partition_dir.iterdir() if p.is_dir()):
            rows.append(summarize(directory, partition))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = {
        part: {
            "trips": sum(row["partition"] == part for row in rows),
            "track_rows": sum(row["track_rows"] for row in rows if row["partition"] == part),
            "nodes": sum(row["node_rows"] for row in rows if row["partition"] == part),
            "arcs": sum(row["arc_rows"] for row in rows if row["partition"] == part),
        }
        for part in ("train", "student_test")
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} trips to {args.out} and {summary_path}")


if __name__ == "__main__":
    main()
