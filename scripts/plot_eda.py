"""Plot dataset inventory summaries; all plots are generated from the CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def histogram_svg(title, xlabel, by_partition, output):
    width, height = 900, 500
    left, right, top, bottom = 85, 25, 65, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max((max(values, default=0) for values in by_partition.values()), default=1)
    max_value = max(1, max_value)
    bins = 16
    max_x = max((max(values, default=0) for values in by_partition.values()), default=1)
    step = max(1, math.ceil((max_x + 1) / bins))
    bucket_count = max(1, math.ceil((max_x + 1) / step))
    colors = {"train": "#2563eb", "student_test": "#f97316"}
    counts = {
        name: [0] * bucket_count
        for name in by_partition
    }
    for name, values in by_partition.items():
        for value in values:
            counts[name][min(bucket_count - 1, value // step)] += 1
    max_y = max(1, max(max(row, default=0) for row in counts.values()))
    bar_slot = plot_w / bucket_count
    bar_w = max(2, bar_slot / (len(counts) + 0.5))

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>',
    ]
    for tick in range(5):
        value = round(max_y * tick / 4)
        y = height - bottom - plot_h * value / max_y
        pieces.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        pieces.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#334155">{value}</text>')
    for index in range(bucket_count):
        x = left + index * bar_slot
        for series_index, (name, frequencies) in enumerate(counts.items()):
            value = frequencies[index]
            bar_h = plot_h * value / max_y
            bx = x + 3 + series_index * bar_w
            by = height - bottom - bar_h
            pieces.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w-2:.1f}" height="{bar_h:.1f}" fill="{colors.get(name, "#64748b")}" opacity="0.8"/>')
        if index % max(1, bucket_count // 8) == 0:
            pieces.append(f'<text x="{x+bar_slot/2:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#334155">{index*step}</text>')
    pieces.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="15">{escape(xlabel)}</text>')
    pieces.append(f'<text transform="translate(24 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15">Trips</text>')
    for index, name in enumerate(counts):
        x = width - right - 150
        y = 55 + index * 24
        pieces.append(f'<rect x="{x}" y="{y-12}" width="14" height="14" fill="{colors.get(name, "#64748b")}"/>')
        pieces.append(f'<text x="{x+21}" y="{y}" font-family="Arial" font-size="14">{escape(name)}</text>')
    pieces.append('</svg>')
    output.write_text("\n".join(pieces), encoding="utf-8")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=ROOT / "results" / "data_inventory.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "figures")
    args = parser.parse_args()
    with args.inventory.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    partitions = ("train", "student_test")

    charts = (
        ("GPS observations per trip", "Observation rows", "track_rows", "track_rows_by_trip.svg"),
        ("Weakly connected road components per trip", "Non-isolated components", "connected_components_nonisolated_undirected", "network_components_by_trip.svg"),
        ("Timestamp gaps longer than 2 seconds", "Gap count per trip", "timestamp_gap_gt_2s", "time_gaps_by_trip.svg"),
    )
    for title, xlabel, field, filename in charts:
        data = {
            partition: [int(r[field]) for r in rows if r["partition"] == partition]
            for partition in partitions
        }
        histogram_svg(title, xlabel, data, args.out_dir / filename)
    print(f"Wrote EDA figures to {args.out_dir}")


if __name__ == "__main__":
    main()
