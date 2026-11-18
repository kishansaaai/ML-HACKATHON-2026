"""Plots and tables for the report, all generated from real results.

* candidate_sweep.csv / plot : candidate miss rate vs radius R (from reference-arc distances)
* model_comparison.png       : CV Top-1, student_test Top-1/macro-F1, log loss for every model
* calibration_lightgbm.png   : reliability diagram of the exported model on student_test
* confusion_lightgbm.png     : candidate-level confusion matrix of the exported model on student_test
* confusion_all_models.png    : the same confusion matrix for every model (from results/experiments.csv)
* metrics_table.csv          : the consolidated comparison table (official metrics) for the report
* label_quality.csv          : label-alignment quality summary (from results/label_stats.csv)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from pathsnap.clf import score_candidates
from pathsnap.data import load_part
from pathsnap.paths import RESULTS_DIR

PLOTS = RESULTS_DIR / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def candidate_sweep() -> None:
    _, obs = load_part("train")
    lab = obs[obs.ref_arc >= 0]
    radii = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    rows = [dict(radius_m=r, miss_rate=float((lab.ref_dist_m > r).mean())) for r in radii]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "candidate_sweep.csv", index=False)
    stats = pd.read_csv(RESULTS_DIR / "label_stats.csv")
    print("candidates per point at R=50, k_min=6, cap=24 (train):",
          round(stats[stats.part == "train"].cand_per_obs.mean(), 1))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(df.radius_m, df.miss_rate * 100, marker="o")
    ax.axvline(50, ls="--", c="grey", lw=1)
    ax.set_xlabel("candidate radius R (m)")
    ax.set_ylabel("reference arc outside radius (% of labelled points)")
    ax.set_title("Candidate miss rate vs radius (80 training trips)")
    fig.tight_layout()
    fig.savefig(PLOTS / "candidate_sweep.png", dpi=150)
    plt.close(fig)
    print(df.to_string(index=False))


def label_quality() -> None:
    s = pd.read_csv(RESULTS_DIR / "label_stats.csv")
    out = s.groupby("part").agg(
        trips=("trip_id", "count"), points=("n_obs", "sum"),
        frac_labelled_mean=("frac_labelled", "mean"), frac_labelled_min=("frac_labelled", "min"),
        dist_median_m=("dist_median_m", "median"), dist_p90_m=("dist_p90_m", "median"),
        frac_gt_30m_mean=("frac_dist_gt_30m", "mean"), frac_ambiguous_mean=("frac_ambiguous", "mean"),
        miss_rate_mean=("miss_rate", "mean"), route_consecutive_min=("route_consecutive_frac", "min"))
    out.to_csv(RESULTS_DIR / "label_quality.csv")
    print(out.T.to_string())
    worst = s.sort_values("frac_labelled").head(5)[["part", "trip_id", "n_obs", "frac_labelled", "frac_dist_gt_30m"]]
    print("trips with the most unlabelled points:\n", worst.to_string(index=False))


def model_comparison() -> None:
    d = pd.read_csv(RESULTS_DIR / "experiments.csv").drop_duplicates("model", keep="last")
    d = d[d.model != "lightgbm_tuned"].set_index("model")
    order = ["nearest_arc", "nearest_heading_aware", "extra_trees", "random_forest", "xgboost", "hist_gb", "lightgbm"]
    d = d.loc[[m for m in order if m in d.index]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(d))
    axes[0].bar(x - 0.2, d.cv_top1_mean * 100, 0.4, yerr=d.cv_top1_std * 100, label="CV (80 trips)", capsize=3)
    axes[0].bar(x + 0.2, d.test_top1 * 100, 0.4, label="student_test")
    axes[0].set_ylabel("observation Top-1 (%)")
    axes[0].set_ylim(40, 95)
    axes[0].legend()
    m = d.dropna(subset=["test_macro_f1"])
    xm = np.arange(len(m))
    axes[1].bar(xm - 0.2, m.cv_macro_f1_mean, 0.4, label="CV")
    axes[1].bar(xm + 0.2, m.test_macro_f1, 0.4, label="student_test")
    axes[1].set_ylabel("candidate macro F1")
    axes[1].set_ylim(0.85, 0.96)
    axes[1].set_xticks(xm)
    axes[1].set_xticklabels(m.index, rotation=30, ha="right")
    axes[1].legend()
    axes[2].bar(xm, m.test_logloss)
    axes[2].set_ylabel("log loss, student_test")
    axes[2].set_xticks(xm)
    axes[2].set_xticklabels(m.index, rotation=30, ha="right")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(d.index, rotation=30, ha="right")
    fig.suptitle("Model comparison (error bars: std over 5 trip-grouped folds)")
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison.png", dpi=150)
    plt.close(fig)


def calibration_and_confusion() -> None:
    test_c, _ = load_part("student_test")
    p = score_candidates(test_c)
    y = test_c["label"].to_numpy()
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(p, bins) - 1, 0, 9)
    conf = np.array([p[which == b].mean() if (which == b).any() else np.nan for b in range(10)])
    freq = np.array([y[which == b].mean() if (which == b).any() else np.nan for b in range(10)])
    cnt = np.array([(which == b).sum() for b in range(10)])
    ece = float(np.nansum(np.abs(conf - freq) * cnt) / cnt.sum())
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.plot(conf, freq, marker="o")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"LightGBM calibration, student_test (ECE={ece:.4f})")
    fig.tight_layout()
    fig.savefig(PLOTS / "calibration_lightgbm.png", dpi=150)
    plt.close(fig)
    print("ECE (10 bins) on student_test:", round(ece, 4))
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    pd.DataFrame(cm, index=["true 0", "true 1"], columns=["pred 0", "pred 1"]).to_csv(RESULTS_DIR / "confusion_lightgbm.csv")
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title("Candidate-level confusion, student_test\n(threshold 0.5)")
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_lightgbm.png", dpi=150)
    plt.close(fig)
    print(cm)


def confusion_all_models() -> None:
    d = pd.read_csv(RESULTS_DIR / "experiments.csv")
    d = d.dropna(subset=["test_tn"]).drop_duplicates("model", keep="last").set_index("model")
    order = [m for m in ["random_forest", "extra_trees", "xgboost", "hist_gb", "lightgbm", "lightgbm_tuned"] if m in d.index]
    fig, axes = plt.subplots(1, len(order), figsize=(3.1 * len(order), 3.2))
    for ax, name in zip(np.atleast_1d(axes), order):
        r = d.loc[name]
        cm = np.array([[r.test_tn, r.test_fp], [r.test_fn, r.test_tp]], dtype=float)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=8)
        ax.set_title(name, fontsize=9)
    fig.suptitle("Candidate-level confusion matrices, student_test (threshold 0.5; colour = row-normalised)", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_all_models.png", dpi=150)
    plt.close(fig)


def metrics_table() -> None:
    d = pd.read_csv(RESULTS_DIR / "experiments.csv").dropna(subset=["test_tn"]).drop_duplicates("model", keep="last")
    cols = {"model": "model", "family": "family", "cv_macro_f1_mean": "cv_macro_f1", "cv_top1_mean": "cv_top1",
            "cv_route_continuity_mean": "cv_route_continuity", "test_macro_f1": "test_macro_f1",
            "test_top1": "test_top1", "test_accuracy": "test_candidate_accuracy", "test_precision": "test_precision",
            "test_recall": "test_recall", "test_logloss": "test_logloss", "test_route_continuity": "test_route_continuity",
            "test_candidate_miss_rate": "test_candidate_miss_rate", "test_seq_arc_f1": "test_seq_arc_f1",
            "test_seq_arc_jaccard": "test_seq_arc_jaccard", "test_seq_edit_dist_norm": "test_seq_edit_dist_norm",
            "train_time_s": "train_time_s", "infer_time_s_per_trip": "infer_time_s_per_trip"}
    out = d[[c for c in cols if c in d.columns]].rename(columns=cols).round(4)
    out.to_csv(RESULTS_DIR / "metrics_table.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    candidate_sweep()
    label_quality()
    model_comparison()
    calibration_and_confusion()
    confusion_all_models()
    metrics_table()
