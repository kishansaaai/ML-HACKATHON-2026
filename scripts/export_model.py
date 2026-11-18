"""Refit the CV-selected untuned LightGBM on all training trips and save its artifact."""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from pathsnap.data import load_part, stride_subsample, xy
from pathsnap.features import FEATURE_ORDER
from pathsnap.models.registry import MODEL_SPECS, SEED, make_model
from pathsnap.paths import ARTIFACT_DIR, RESULTS_DIR


def main():
    experiments = pd.read_csv(RESULTS_DIR / "experiments.csv")
    selected = experiments.loc[experiments["model"] == "lightgbm"].iloc[-1]
    tuned = experiments.loc[experiments["model"] == "lightgbm_tuned"].iloc[-1]
    if float(selected["cv_top1_mean"]) < float(tuned["cv_top1_mean"]):
        raise RuntimeError("the untuned LightGBM is no longer the CV-selected artifact")
    candidates, _ = load_part("train")
    x, y = xy(stride_subsample(candidates, int(selected["train_stride"])))
    model = make_model("lightgbm")
    started = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - started
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib", compress=3)
    (ARTIFACT_DIR / "feature_order.json").write_text(json.dumps(FEATURE_ORDER, indent=2) + "\n", encoding="utf-8")
    card = {
        "model": "lightgbm",
        "family": MODEL_SPECS["lightgbm"]["family"],
        "seed": SEED,
        "params": MODEL_SPECS["lightgbm"]["params"],
        "train_stride": int(selected["train_stride"]),
        "n_train_rows": int(len(y)),
        "trained_on": "all 80 training trips only; labels from monotone route alignment",
        "selected_by": "higher trip-grouped CV Top-1 of untuned vs tuned LightGBM; student_test not used for selection",
        "cv_top1_mean": float(selected["cv_top1_mean"]),
        "tuned_cv_top1_mean": float(tuned["cv_top1_mean"]),
        "student_test_top1": float(selected["test_top1"]),
        "student_test_logloss": float(selected["test_logloss"]),
        "feature_count": len(FEATURE_ORDER),
        "refit_seconds": fit_seconds,
    }
    (ARTIFACT_DIR / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {ARTIFACT_DIR / 'model.joblib'} using {len(y):,} training candidate rows in {fit_seconds:.1f}s")


if __name__ == "__main__":
    main()
