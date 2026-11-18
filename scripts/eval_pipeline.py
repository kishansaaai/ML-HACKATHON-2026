"""Evaluate the full pipeline (classifier alone vs classifier + HMM) on DEVELOPMENT trips.

    python scripts/eval_pipeline.py                       # all student_test trips, default decoder config
    python scripts/eval_pipeline.py --beta 20 --tag b20   # one decoder setting
    python scripts/eval_pipeline.py --trips 00000052 00000065

Prediction never sees labels. The reference arc per GPS point is derived afterwards, from `.route`,
by the monotone alignment in `pathsnap.labels`, and scored with `pathsnap.decoder.metrics`.
Only the `train` and `student_test` partitions are accepted; never point this at evaluator data.
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import pandas as pd

from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.metrics import observation_top1, route_continuity
from pathsnap.geo import TripGraph
from pathsnap.io import load_trip
from pathsnap.labels import derive_point_labels
from pathsnap.paths import DATA_ROOT, RESULTS_DIR, list_trips
from pathsnap.pipeline import make_components
from pathsnap.decoder.inference import predict_trip


def evaluate_trip(part: str, trip_id: str, cfg: DecoderConfig) -> dict:
    trip_dir = DATA_ROOT / part / trip_id
    t0 = time.perf_counter()
    pred = predict_trip(trip_dir, make_components(trip_dir), cfg)          # no labels used here
    seconds = time.perf_counter() - t0
    labelled = load_trip(trip_dir, with_labels=True)                        # labels only now
    ref = derive_point_labels(labelled, TripGraph(labelled))
    ref_l = [int(r) if r >= 0 else None for r in ref]
    clf, hmm = pred["classifier_arc_ids"], pred["hmm_arc_ids"]
    # continuity needs the decoder graph; rebuild it cheaply from the cached copy
    from pathsnap.decoder.inference import _GRAPH_CACHE
    graph = next(reversed(_GRAPH_CACHE.values()))
    cc, hc = route_continuity(clf, graph), route_continuity(hmm, graph)
    return dict(part=part, trip_id=trip_id, n_obs=len(ref), n_labelled=int((ref >= 0).sum()),
                clf_top1=observation_top1(clf, ref_l), hmm_top1=observation_top1(hmm, ref_l),
                clf_continuity=cc["route_continuity"], clf_breaks=cc["route_break_count"],
                hmm_continuity=hc["route_continuity"], hmm_breaks=hc["route_break_count"],
                stitched_breaks=len(pred["route_breaks"]), seconds=round(seconds, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="student_test", choices=["train", "student_test"])
    ap.add_argument("--trips", nargs="*")
    ap.add_argument("--beta", type=float, default=DecoderConfig().beta_m)
    ap.add_argument("--gap", type=float, default=DecoderConfig().gap_threshold_s)
    ap.add_argument("--sigma", type=float, default=None, help="optional Gaussian emission distance penalty (m)")
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    cfg = DecoderConfig(beta_m=a.beta, gap_threshold_s=a.gap, emission_sigma_m=a.sigma)
    trips = a.trips or list_trips(a.part)
    rows = []
    for tid in trips:
        rows.append(evaluate_trip(a.part, tid, cfg))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    w = df.n_labelled
    summary = dict(tag=a.tag, part=a.part, trips=len(df), points=int(w.sum()), config=dataclasses.asdict(cfg),
                   clf_top1=float(np.average(df.clf_top1, weights=w)), hmm_top1=float(np.average(df.hmm_top1, weights=w)),
                   clf_continuity=float(df.clf_continuity.mean()), hmm_continuity=float(df.hmm_continuity.mean()),
                   hmm_better_trips=int((df.hmm_top1 > df.clf_top1).sum()), hmm_worse_trips=int((df.hmm_top1 < df.clf_top1).sum()))
    print(summary)
    df.to_csv(RESULTS_DIR / f"pipeline_{a.part}_{a.tag}.csv", index=False)
