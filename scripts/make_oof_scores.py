"""Build out-of-fold classifier scores for the 80 training trips (for decoder tuning only).

    python scripts/make_oof_scores.py

Prints the pooled and per-fold OOF Top-1 so it can be checked against the grouped-CV Top-1 in
results/experiments.csv (same models, same folds, so they must agree).
"""
import numpy as np
import pandas as pd

from pathsnap.cand_metrics import observation_metrics
from pathsnap.data import load_folds, load_part
from pathsnap.oof import OOF_PATH, build_oof

if __name__ == "__main__":
    oof = build_oof()
    print("saved", OOF_PATH, len(oof), "candidate rows,", oof.trip_id.nunique(), "trips")
    cand, obs = load_part("train")
    cand = cand.copy()
    cand["trip_id"] = cand["trip_id"].astype(str)
    key = ["trip_id", "obs_idx", "arc_id"]
    merged = cand.merge(oof[key + ["p_oof"]], on=key, how="left")
    assert merged["p_oof"].notna().all(), "some candidates have no OOF score"
    obs = obs.copy()
    obs["trip_id"] = obs["trip_id"].astype(str)
    tops = []
    for k, val_trips in enumerate(load_folds()):
        vt = set(val_trips)
        m = merged[merged.trip_id.isin(vt)]
        tops.append(observation_metrics(m, m["p_oof"].to_numpy(), obs[obs.trip_id.isin(vt)])["top1"])
        print(f"fold {k}: OOF Top-1 {tops[-1]:.4f}")
    print(f"mean over folds {np.mean(tops):.4f}  (std {np.std(tops, ddof=1):.4f})")
    ref = pd.read_csv("results/experiments.csv").query("model == 'lightgbm'").iloc[-1]
    print(f"grouped-CV Top-1 recorded in results/experiments.csv: {ref.cv_top1_mean:.4f} (std {ref.cv_top1_std:.4f})")
