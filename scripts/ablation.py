"""Feature-group ablation for the selected model (LightGBM, untuned config).

For each feature group, retrain WITHOUT it using the same trip-grouped CV folds and training
protocol (stride 3), and report the change in CV Top-1 / log loss and in student_test Top-1.
Also reports 'only <group>' to show what each group can do alone. Output: results/ablation.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pathsnap.cand_metrics import candidate_metrics, observation_metrics
from pathsnap.data import load_folds, load_part, stride_subsample
from pathsnap.features import FEATURE_GROUPS, FEATURE_ORDER
from pathsnap.models.registry import make_model, proba
from pathsnap.paths import RESULTS_DIR

STRIDE = 3


def cv_and_test(cols, train_c, train_o, test_c, test_o):
    tops, lls = [], []
    for val_trips in load_folds():
        vt = set(val_trips)
        m = train_c.trip_id.isin(vt)
        tr = stride_subsample(train_c[~m], STRIDE)
        va = train_c[m]
        model = make_model("lightgbm")
        model.fit(tr[cols].to_numpy(np.float32), tr["label"].to_numpy())
        p = proba(model, va[cols].to_numpy(np.float32))
        tops.append(observation_metrics(va, p, train_o[train_o.trip_id.isin(vt)])["top1"])
        lls.append(candidate_metrics(va["label"].to_numpy(), p)["logloss"])
    tr = stride_subsample(train_c, STRIDE)
    model = make_model("lightgbm")
    model.fit(tr[cols].to_numpy(np.float32), tr["label"].to_numpy())
    p = proba(model, test_c[cols].to_numpy(np.float32))
    return (float(np.mean(tops)), float(np.std(tops, ddof=1)), float(np.mean(lls)),
            observation_metrics(test_c, p, test_o)["top1"])


if __name__ == "__main__":
    train_c, train_o = load_part("train")
    test_c, test_o = load_part("student_test")
    configs = [("all features", FEATURE_ORDER)]
    for g, cols in FEATURE_GROUPS.items():
        configs.append((f"without {g}", [c for c in FEATURE_ORDER if c not in cols]))
    for g, cols in FEATURE_GROUPS.items():
        configs.append((f"only {g}", cols))
    rows = []
    for name, cols in configs:
        t0 = time.time()
        top1, std, ll, test_top1 = cv_and_test(cols, train_c, train_o, test_c, test_o)
        rows.append(dict(setting=name, n_features=len(cols), cv_top1=top1, cv_top1_std=std,
                         cv_logloss=ll, test_top1=test_top1, seconds=round(time.time() - t0)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "ablation.csv", index=False)
