"""Train Person C's Logistic Regression, Gaussian NB and distance-weighted KNN.

Uses A's candidates, labels, feature order and fixed trip-group CV splits. KNN is
fit on a reproducible 3,000-row subset because exact neighbor scoring over every
candidate row is otherwise impractical; validation and student-test rows remain
complete. Results append to the shared experiments.csv ledger.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from pathsnap.cand_metrics import (
    candidate_metrics,
    observation_metrics,
    route_continuity,
    top1_predictions,
)
from pathsnap.data import load_folds, load_part, stride_subsample, trip_arcs, xy
from pathsnap.paths import RESULTS_DIR

SEED = 42
STRIDE = 3
KNN_FIT_ROWS = 3_000


def _model(name: str):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=SEED),
        )
    if name == "gaussian_nb":
        return GaussianNB(var_smoothing=1e-9)
    if name == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15, weights="distance", p=2, n_jobs=-1),
        )
    raise ValueError(name)


def _fit_data(frame: pd.DataFrame, name: str):
    sampled = stride_subsample(frame, STRIDE)
    if name == "knn" and len(sampled) > KNN_FIT_ROWS:
        sampled = sampled.sample(n=KNN_FIT_ROWS, random_state=SEED)
    return xy(sampled)


def _evaluate(cand: pd.DataFrame, obs: pd.DataFrame, score: np.ndarray, part: str):
    metrics = observation_metrics(cand, score, obs)
    metrics.update(candidate_metrics(cand["label"].to_numpy(), score))
    predictions = top1_predictions(cand, score)
    continuity_values, breaks = [], 0
    for trip_id, group in predictions.groupby("trip_id"):
        predicted = group.sort_values("obs_idx")["arc_id"].to_numpy()
        continuity, count = route_continuity(predicted, trip_arcs(part, trip_id))
        continuity_values.append(continuity)
        breaks += count
    metrics["route_continuity"] = float(np.mean(continuity_values)) if continuity_values else 1.0
    metrics["route_breaks"] = int(breaks)
    return metrics


def _fit_model(name: str, frame: pd.DataFrame):
    x, y = _fit_data(frame, name)
    model = _model(name)
    model.fit(x, y)
    return model, x, y


def run(name: str, train_c: pd.DataFrame, train_o: pd.DataFrame,
        test_c: pd.DataFrame, test_o: pd.DataFrame):
    fold_metrics = []
    fit_seconds = []
    for fold_index, val_ids in enumerate(load_folds()):
        val_set = set(val_ids)
        train_fold = train_c[~train_c.trip_id.isin(val_set)]
        val_fold = train_c[train_c.trip_id.isin(val_set)]
        model = _model(name)
        x_train, y_train = _fit_data(train_fold, name)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds.append(time.perf_counter() - started)
        x_val, _ = xy(val_fold)
        started = time.perf_counter()
        score = model.predict_proba(x_val)[:, 1]
        infer_seconds = time.perf_counter() - started
        metrics = _evaluate(
            val_fold, train_o[train_o.trip_id.isin(val_set)], score, "train"
        )
        metrics["fit_s"] = fit_seconds[-1]
        metrics["infer_s"] = infer_seconds
        fold_metrics.append(metrics)
        print(
            f"[{name}] fold {fold_index + 1}/5: top1={metrics['top1']:.4f} "
            f"macroF1={metrics['macro_f1']:.4f} logloss={metrics['logloss']:.4f} "
            f"fit={fit_seconds[-1]:.2f}s infer={infer_seconds:.2f}s",
            flush=True,
        )

    model, x_train, y_train = _fit_model(name, train_c)
    started = time.perf_counter()
    test_score = model.predict_proba(xy(test_c)[0])[:, 1]
    test_infer_s = time.perf_counter() - started
    test_metrics = _evaluate(test_c, test_o, test_score, "student_test")
    fold = pd.DataFrame(fold_metrics)
    if name == "logistic_regression":
        family = "Linear / probabilistic"
        config = {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "standardized": True}
    elif name == "gaussian_nb":
        family = "Probabilistic / generative"
        config = {"var_smoothing": 1e-9}
    else:
        family = "Instance-based (distance-weighted)"
        config = {
            "n_neighbors": 15, "weights": "distance", "p": 2,
            "standardized": True, "fit_rows_per_fold": KNN_FIT_ROWS,
        }
    row = {
        "model": name,
        "family": family,
        "config_json": json.dumps(config, sort_keys=True),
        "seed": SEED,
        "train_stride": STRIDE,
        "n_train_rows": len(y_train),
        "cv_macro_f1_mean": fold.macro_f1.mean(),
        "cv_macro_f1_std": fold.macro_f1.std(),
        "cv_top1_mean": fold.top1.mean(),
        "cv_top1_std": fold.top1.std(),
        "cv_logloss_mean": fold.logloss.mean(),
        "cv_route_continuity_mean": fold.route_continuity.mean(),
        "test_macro_f1": test_metrics["macro_f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_top1": test_metrics["top1"],
        "test_logloss": test_metrics["logloss"],
        "test_route_continuity": test_metrics["route_continuity"],
        "test_route_breaks": test_metrics["route_breaks"],
        "test_candidate_miss_rate": test_metrics["candidate_miss_rate"],
        "train_time_s": sum(fit_seconds),
        "infer_time_s_per_trip": test_infer_s / max(1, test_o.trip_id.nunique()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "experiments.csv"
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    combined.to_csv(path, index=False)
    print(
        f"[{name}] CV Top-1={row['cv_top1_mean']:.4f} "
        f"student_test Top-1={row['test_top1']:.4f} "
        f"macroF1={row['test_macro_f1']:.4f} logloss={row['test_logloss']:.4f}",
        flush=True,
    )
    return row


def main():
    train_c, train_o = load_part("train")
    test_c, test_o = load_part("student_test")
    for name in ("logistic_regression", "gaussian_nb", "knn"):
        run(name, train_c, train_o, test_c, test_o)


if __name__ == "__main__":
    main()
