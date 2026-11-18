#!/usr/bin/env python3
"""Run reproducible HMM comparisons on development trips with OOF scores for training."""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.decoder.config import DecoderConfig
from pathsnap.paths import RESULTS_DIR, list_trips
from eval_pipeline import evaluate_trip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["train", "student_test"], default="train")
    parser.add_argument("--trips", nargs="*", help="defaults to every trip in the selected development partition")
    parser.add_argument("--beta", type=float, default=200.0)
    parser.add_argument("--gap", type=float, default=30.0)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--unreachable-penalty", type=float, default=-20.0)
    parser.add_argument("--tag", required=True, help="unique result name; existing files are never overwritten")
    parser.add_argument("--oof", action="store_true", help="use OOF classifier scores (training only)")
    args = parser.parse_args()
    if args.oof and args.part != "train":
        parser.error("--oof only applies to training trips")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("tag may contain only letters, numbers, dot, underscore, and hyphen")

    output_dir = RESULTS_DIR / "hmm_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.tag}.csv"
    json_path = output_dir / f"{args.tag}.json"
    if csv_path.exists() or json_path.exists():
        parser.error(f"refusing to overwrite existing result: {csv_path if csv_path.exists() else json_path}")

    cfg = DecoderConfig(beta_m=args.beta, gap_threshold_s=args.gap,
                        emission_sigma_m=args.sigma if args.sigma > 0 else None,
                        unreachable_log_penalty=args.unreachable_penalty)
    trip_ids = args.trips if args.trips else list_trips(args.part)
    rows = []
    for trip_id in trip_ids:
        row = evaluate_trip(args.part, trip_id, cfg, oof=args.oof)
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    weights = df.n_labelled
    summary = {
        "tag": args.tag, "part": args.part, "score_source": "OOF LightGBM" if args.oof else "exported LightGBM",
        "trips": len(df), "trip_ids": list(map(str, df.trip_id)), "labelled_observations": int(weights.sum()),
        "config": dataclasses.asdict(cfg),
        "classifier_top1_weighted": float(np.average(df.clf_top1, weights=weights)),
        "hmm_top1_weighted": float(np.average(df.hmm_top1, weights=weights)),
        "top1_delta": float(np.average(df.hmm_top1, weights=weights) - np.average(df.clf_top1, weights=weights)),
        "classifier_continuity_mean_trip": float(df.clf_continuity.mean()),
        "hmm_continuity_mean_trip": float(df.hmm_continuity.mean()),
        "classifier_breaks_total": int(df.clf_breaks.sum()), "hmm_breaks_total": int(df.hmm_breaks.sum()),
        "hmm_better_trips": int((df.hmm_top1 > df.clf_top1).sum()),
        "hmm_tied_trips": int((df.hmm_top1 == df.clf_top1).sum()),
        "hmm_worse_trips": int((df.hmm_top1 < df.clf_top1).sum()),
        "runtime_seconds_sum": float(df.seconds.sum()),
    }
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"saved {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
