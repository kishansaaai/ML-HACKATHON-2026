"""Run the HTTP API contract against student_test and compare with official pipeline output."""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import api  # noqa: E402
from pathsnap.paths import DATA_ROOT  # noqa: E402


async def evaluate(out: Path) -> None:
    api._validate_evaluator_partition()
    api._warm_up()
    transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=True)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health = await client.get("/health")
        health.raise_for_status()
        if health.json()["partition"] != "student_test":
            raise SystemExit("Refusing to evaluate: API is not serving student_test.")
        trip_items = (await client.get("/trips")).raise_for_status().json()["trips"]
        if len(trip_items) != 10:
            raise SystemExit(f"Expected 10 student_test trips; found {len(trip_items)}")
        rows = []
        for item in trip_items:
            trip_id = item["trip_id"]
            response = await client.post("/predict", json={"trip_id": trip_id})
            response.raise_for_status()
            body = response.json()
            metrics = body["metrics"]
            if metrics is None:
                raise SystemExit(f"No server-side development metrics for trip {trip_id}")
            classifier, hmm = metrics["classifier_only"], metrics["hmm"]
            rows.append({
                "part": "student_test", "trip_id": trip_id,
                "n_obs": body["num_observations"],
                "n_labelled": classifier["n_labelled"],
                "clf_top1": classifier["top1_accuracy"], "hmm_top1": hmm["top1_accuracy"],
                "clf_continuity": classifier["route_continuity"],
                "clf_breaks": classifier["continuity_breaks"],
                "hmm_continuity": hmm["route_continuity"],
                "hmm_breaks": hmm["continuity_breaks"],
                "stitched_breaks": len(body["hmm_route_segments"]) - 1 if body["hmm_route_segments"] else 0,
                "seconds": body["timings_sec"]["total"],
            })
            print(f'{trip_id}: clf={classifier["top1_accuracy"]:.4f} '
                  f'hmm={hmm["top1_accuracy"]:.4f} {body["timings_sec"]["total"]:.2f}s', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame(rows).sort_values("trip_id")
    current.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    reference_path = ROOT / "results" / "hmm_tuning" / "final_student_test_check.csv"   # official pipeline output with the frozen (tuned) decoder
    reference = pd.read_csv(reference_path, dtype={"trip_id": str}).sort_values("trip_id")
    current["trip_id"] = current.trip_id.astype(str).str.zfill(8)
    if set(current.trip_id) != set(reference.trip_id):
        raise SystemExit("API and official pipeline trip IDs do not match")
    for col in ("clf_top1", "hmm_top1"):
        pooled_current = float((current[col] * current.n_labelled).sum() / current.n_labelled.sum())
        pooled_reference = float((reference[col] * reference.n_labelled).sum() / reference.n_labelled.sum())
        delta = abs(pooled_current - pooled_reference)
        print(f"pooled {col}: API={pooled_current:.4%}, official={pooled_reference:.4%}, delta={delta:.4%}")
        if delta > 0.002:
            raise SystemExit(f"API {col} differs from official pipeline by more than 0.2 percentage points")
    slowest = current.loc[current.seconds.idxmax()]
    print(f"Slowest API trip: {slowest.trip_id}, {slowest.seconds:.2f}s")
    print(f"Saved API metrics: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "api_student_test.csv")
    args = parser.parse_args()
    if os.getenv("PATHSNAP_EVALUATOR_DIR"):
        raise SystemExit("Unset PATHSNAP_EVALUATOR_DIR; this script only evaluates student_test.")
    asyncio.run(evaluate(args.out))


if __name__ == "__main__":
    main()
