"""Run the prediction/scoring API handler over a development partition and save metrics."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.api import PredictRequest, predict, trips  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("train", "student_test"), default="student_test")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "api_student_test.csv")
    args = parser.parse_args()
    os.environ["PATHSNAP_SERVE_PARTITION"] = args.partition
    # The API reads this setting at import time; CLI's partition is applied directly here.
    from pathsnap import api
    api._DEV_PARTITION = args.partition
    rows = []
    for trip_id in trips()["trip_ids"]:
        body = predict(PredictRequest(trip_id=trip_id))
        metrics = body["metrics"] or {}
        classifier = metrics.get("classifier_observation_metrics", {})
        hmm = metrics.get("hmm_observation_metrics", {})
        rows.append({
            "trip_id": trip_id,
            "partition": args.partition,
            "observations": len(body["observations"]),
            "classifier_top1": metrics.get("classifier_top1_accuracy"),
            "hmm_top1": metrics.get("hmm_top1_accuracy"),
            "classifier_macro_f1": classifier.get("macro_f1"),
            "hmm_macro_f1": hmm.get("macro_f1"),
            "classifier_route_continuity": metrics.get("classifier_route_continuity", {}).get("route_continuity"),
            "hmm_route_continuity": metrics.get("hmm_route_continuity", {}).get("route_continuity"),
            "route_breaks": len(body["route_breaks"]),
            "inference_seconds": body["inference_seconds"],
        })
        print(f'{trip_id}: classifier={rows[-1]["classifier_top1"]:.4f} '
              f'hmm={rows[-1]["hmm_top1"]:.4f} inference={rows[-1]["inference_seconds"]:.2f}s')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["trip_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} trips to {args.out}")


if __name__ == "__main__":
    main()
