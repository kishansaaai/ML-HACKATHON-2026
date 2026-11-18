"""Report figures generated from real data/results (written to docs/report/fig/).

    python scripts/make_report_figures.py

  eda_gps_network.png        GPS track, road network and reference route of one TRAINING trip
  eda_data_quality.png       per-trip points / long time gaps / road arcs, by partition (data_inventory.csv)
  all_models_comparison.png  CV macro-F1, student-test Top-1 and macro-F1, student-test log loss for all learned models
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from pathsnap.io import load_trip
from pathsnap.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report" / "fig"
TRIP = "00000002"          # training partition


def eda_gps_network():
    trip = load_trip(DATA_ROOT / "train" / TRIP, with_labels=True)
    pad = 0.002
    x0, x1 = trip.lon.min() - pad, trip.lon.max() + pad
    y0, y1 = trip.lat.min() - pad, trip.lat.max() + pad
    a = trip.node_lonlat[trip.arcs[:, 0]]
    b = trip.node_lonlat[trip.arcs[:, 1]]
    keep = ~((np.maximum(a[:, 0], b[:, 0]) < x0) | (np.minimum(a[:, 0], b[:, 0]) > x1) |
             (np.maximum(a[:, 1], b[:, 1]) < y0) | (np.minimum(a[:, 1], b[:, 1]) > y1))
    segs = np.stack([a[keep], b[keep]], axis=1)
    ref = trip.arcs[trip.route.astype(int)]
    rsegs = np.stack([trip.node_lonlat[ref[:, 0]], trip.node_lonlat[ref[:, 1]]], axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.add_collection(LineCollection(segs, colors="#b0b0b0", linewidths=0.4, label=f"Road network ({int(keep.sum()):,} arcs in view)"))
    ax.add_collection(LineCollection(rsegs, colors="#ff7f0e", linewidths=4.0, label="Reference route (.route)", zorder=2))
    ax.scatter(trip.lon, trip.lat, s=1.6, c="#d62728", label=f"Raw GPS points ({trip.n_obs:,})", zorder=3)
    ax.scatter([trip.lon[0]], [trip.lat[0]], s=45, c="green", zorder=4, label="Start")
    ax.scatter([trip.lon[-1]], [trip.lat[-1]], s=45, c="black", zorder=4, label="End")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(1 / np.cos(np.radians(trip.lat.mean())))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Training trip {TRIP}: GPS track, road network and reference route")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(OUT / "eda_gps_network.png", dpi=170); plt.close(fig)


def eda_quality():
    inv = pd.read_csv(ROOT / "results" / "data_inventory.csv", dtype={"trip_id": str})
    parts = [("train", "#2563eb"), ("student_test", "#f97316")]
    cols = [("track_rows", "GPS points per trip", "points"), ("timestamp_gap_gt_5s", "Time steps longer than 5 s, per trip", "count"),
            ("arc_rows", "Directed road arcs per trip", "arcs")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (c, title, xl) in zip(axes, cols):
        hi = float(inv[c].max())
        bins = np.linspace(0, hi, 16)
        for p, col in parts:
            ax.hist(inv.loc[inv.partition == p, c], bins=bins, alpha=0.65, color=col, label=f"{p} ({int((inv.partition == p).sum())} trips)")
        ax.set_title(title, fontsize=9); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("trips", fontsize=8); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "eda_data_quality.png", dpi=170); plt.close(fig)


def all_models():
    e = pd.read_csv(ROOT / "results" / "experiments.csv").drop_duplicates("model", keep="last").set_index("model")
    order = ["lightgbm", "hist_gb", "xgboost", "extra_trees", "random_forest", "logistic_regression", "gaussian_nb", "knn"]
    names = ["LightGBM", "HistGB", "XGBoost", "ExtraTrees", "RandomForest", "LogReg", "NaiveBayes", "KNN"]
    e = e.loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(len(order))
    axes[0].bar(x, e.cv_macro_f1_mean, yerr=e.cv_macro_f1_std, color="#2563eb", capsize=2)
    axes[0].set_title("Cross-validation macro-F1 (5 folds, mean ± std)", fontsize=9)
    axes[0].set_ylim(0.6, 1.0)
    w = 0.38
    axes[1].bar(x - w / 2, e.test_top1, w, color="#2563eb", label="Top-1 accuracy")
    axes[1].bar(x + w / 2, e.test_macro_f1, w, color="#f97316", label="macro-F1")
    axes[1].set_title("Student-test Top-1 accuracy and macro-F1", fontsize=9); axes[1].set_ylim(0.6, 1.0); axes[1].legend(fontsize=7)
    axes[2].bar(x, e.test_logloss, color="#059669")
    axes[2].set_title("Student-test log loss (lower is better)", fontsize=9)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7); ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "all_models_comparison.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    eda_gps_network(); eda_quality(); all_models()
    print("wrote", [p.name for p in sorted(OUT.glob("*.png"))])
