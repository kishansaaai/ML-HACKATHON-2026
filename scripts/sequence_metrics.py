"""Sequence-level route comparison, classifier-only vs classifier + HMM, on the 10 student_test trips.

    python scripts/sequence_metrics.py

The reference route (.route) is loaded only after each prediction, to score it (development data only).
Writes results/sequence_metrics_student_test.csv and prints pooled means.
"""
from pathlib import Path

import pandas as pd

from pathsnap.decoder.metrics import route_sequence_metrics
from pathsnap.io import load_trip
from pathsnap.paths import DATA_ROOT
from pathsnap.pipeline import predict

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    rows = []
    for d in sorted(p for p in (DATA_ROOT / "student_test").iterdir() if p.is_dir()):
        pred = predict(d)                                          # no ground truth is read here
        ref = load_trip(d, with_labels=True).route.tolist()        # reference loaded only after predicting
        clf_route = [int(a) for a in pred["classifier_arc_ids"] if a is not None]
        hmm_route = [int(a) for a in pred["route_arc_ids"]]
        c, h = route_sequence_metrics(clf_route, ref), route_sequence_metrics(hmm_route, ref)
        row = {"trip_id": d.name}
        for k, v in c.items():
            row[f"clf_{k}"] = v
        for k, v in h.items():
            row[f"hmm_{k}"] = v
        rows.append(row)
        print(d.name, {k: round(v, 4) for k, v in row.items() if k.endswith("arc_set_f1") or "edit" in k}, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "sequence_metrics_student_test.csv"
    df.to_csv(out, index=False)
    num = df.drop(columns="trip_id").select_dtypes("number")
    print("mean over 10 trips:\n", num.mean().round(4).to_string())
    print("saved", out)
