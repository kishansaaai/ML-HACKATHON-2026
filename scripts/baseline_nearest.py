"""Non-learned baselines the models must beat (no training, same metrics, same partitions).

  nearest_arc          : highest score = smallest perpendicular distance (ties by arc order)
  nearest_heading_aware: smallest perp distance + 0.25 m per degree of heading mismatch
They use only features, never labels. Rows are appended to results/experiments.csv.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from pathsnap.data import load_folds, load_part
from pathsnap.paths import RESULTS_DIR
from train_heavy import append_row, evaluate  # noqa: E402  (scripts dir is on sys.path when run directly)

BASELINES = {
    "nearest_arc": lambda c: -c["perp"].to_numpy(),
    "nearest_heading_aware": lambda c: -(c["perp"].to_numpy() + 0.25 * c["head_absdiff"].to_numpy()),
}

if __name__ == "__main__":
    train_c, train_o = load_part("train")
    test_c, test_o = load_part("student_test")
    for name, fn in BASELINES.items():
        cvs = []
        for val_trips in load_folds():
            vt = set(val_trips)
            va = train_c[train_c.trip_id.isin(vt)]
            cvs.append(evaluate(va, train_o[train_o.trip_id.isin(vt)], fn(va), "train"))
        te = evaluate(test_c, test_o, fn(test_c), "student_test")
        row = dict(model=name, family="Non-learned baseline", config_json=json.dumps({"rule": name}), seed=42,
                   cv_top1_mean=np.mean([m["top1"] for m in cvs]), cv_top1_std=np.std([m["top1"] for m in cvs], ddof=1),
                   cv_route_continuity_mean=np.mean([m["route_continuity"] for m in cvs]),
                   cv_route_continuity_distinct_mean=np.mean([m["route_continuity_distinct"] for m in cvs]),
                   test_top1=te["top1"], test_route_continuity=te["route_continuity"],
                   test_route_continuity_distinct=te["route_continuity_distinct"],
                   test_route_breaks=te["route_breaks"], test_candidate_miss_rate=te["candidate_miss_rate"],
                   timestamp=datetime.now().isoformat(timespec="seconds"))
        append_row(row)
        print(name, "CV top1 %.4f | test top1 %.4f continuity %.4f" % (row["cv_top1_mean"], row["test_top1"], row["test_route_continuity"]))
