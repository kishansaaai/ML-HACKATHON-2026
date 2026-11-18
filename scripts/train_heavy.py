"""Train + compare the heavy models with trip-grouped CV, then evaluate on student_test.

Selection protocol: 5-fold grouped CV on the 80 training trips (configs/cv_folds.json, seed 42);
student_test (10 trips) is the development-stage hold-out. Evaluator data is never used.
Every model appends one row to results/experiments.csv.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from pathsnap.cand_metrics import candidate_metrics, observation_metrics, route_continuity, top1_predictions
from pathsnap.data import load_folds, load_part, stride_subsample, trip_arcs, xy
from pathsnap.models.registry import MODEL_SPECS, SEED, make_model, proba
from pathsnap.paths import RESULTS_DIR


def evaluate(df: pd.DataFrame, obs: pd.DataFrame, score: np.ndarray, part: str) -> dict:
    m = observation_metrics(df, score, obs)
    m.update(candidate_metrics(df["label"].to_numpy(), score))
    pred = top1_predictions(df, score)
    conts, breaks = [], 0
    for tid, g in pred.groupby("trip_id"):
        c, b = route_continuity(g.sort_values("obs_idx")["arc_id"].to_numpy(), trip_arcs(part, tid))
        conts.append(c)
        breaks += b
    m["route_continuity"] = float(np.mean(conts))
    m["route_breaks"] = breaks
    return m


def append_row(row: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "experiments.csv"
    new = pd.DataFrame([row])
    (pd.concat([pd.read_csv(path), new], ignore_index=True) if path.exists() else new).to_csv(path, index=False)


def run(name: str, stride: int, params: dict | None = None, tag: str = "") -> dict:
    train_c, train_o = load_part("train")
    test_c, test_o = load_part("student_test")
    fold_rows = []
    for k, val_trips in enumerate(load_folds()):
        vt = set(val_trips)
        tr = train_c[~train_c.trip_id.isin(vt)]
        va = train_c[train_c.trip_id.isin(vt)]
        Xtr, ytr = xy(stride_subsample(tr, stride))
        model = make_model(name, params)
        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t0
        Xva, _ = xy(va)
        m = evaluate(va, train_o[train_o.trip_id.isin(vt)], proba(model, Xva), "train")
        m["fit_s"] = fit_s
        fold_rows.append(m)
        print(f"  [{name}] fold {k}: top1={m['top1']:.4f} macroF1={m['macro_f1']:.4f} "
              f"logloss={m['logloss']:.4f} fit={fit_s:.0f}s", flush=True)
    cv = pd.DataFrame(fold_rows)

    Xtr, ytr = xy(stride_subsample(train_c, stride))
    model = make_model(name, params)
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_s = time.time() - t0
    Xte, _ = xy(test_c)
    t0 = time.time()
    score = proba(model, Xte)
    infer_s = time.time() - t0
    te = evaluate(test_c, test_o, score, "student_test")
    row = dict(model=name + tag, family=MODEL_SPECS[name]["family"],
               config_json=json.dumps({**MODEL_SPECS[name]["params"], **(params or {})}), seed=SEED,
               train_stride=stride, n_train_rows=len(ytr),
               cv_macro_f1_mean=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(),
               cv_top1_mean=cv.top1.mean(), cv_top1_std=cv.top1.std(), cv_logloss_mean=cv.logloss.mean(),
               cv_route_continuity_mean=cv.route_continuity.mean(),
               test_macro_f1=te["macro_f1"], test_precision=te["precision"], test_recall=te["recall"],
               test_top1=te["top1"], test_logloss=te["logloss"], test_route_continuity=te["route_continuity"],
               test_route_breaks=te["route_breaks"], test_candidate_miss_rate=te["candidate_miss_rate"],
               train_time_s=train_s, infer_time_s_per_trip=infer_s / test_o.trip_id.nunique(),
               timestamp=datetime.now().isoformat(timespec="seconds"))
    append_row(row)
    print(f"[{name}] CV top1 {row['cv_top1_mean']:.4f}+-{row['cv_top1_std']:.4f} | test top1 {row['test_top1']:.4f} "
          f"macroF1 {row['test_macro_f1']:.4f} logloss {row['test_logloss']:.4f} "
          f"cont {row['test_route_continuity']:.4f}", flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    ap.add_argument("--stride", type=int, default=3, help="train on every n-th GPS point per trip")
    ap.add_argument("--n-estimators", type=int, default=None, help="override tree count (ET/RF time-box)")
    args = ap.parse_args()
    for n in args.models:
        run(n, args.stride, {"n_estimators": args.n_estimators} if args.n_estimators else None)

