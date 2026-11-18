"""Build the reproducible PathSnap technical report PDF from saved results."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "output" / "pdf" / "PathSnap_technical_report.pdf"


def load_api_log() -> pd.DataFrame:
    saved = ROOT / "results" / "api_student_test.csv"
    if saved.exists():
        existing = pd.read_csv(saved, dtype={"trip_id": str})
        if "classifier_top1" in existing.columns:
            return existing.sort_values("trip_id")
    log_path = ROOT / "logs" / "api.log"
    latest: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            match = re.search(r'INFO (\{.*\})$', line)
            if not match:
                continue
            try:
                obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "request_complete" and obj.get("partition") == "student_test":
                latest[obj["trip_id"]] = obj
    columns = ["trip_id", "observations", "inference_seconds", "hmm_top1", "route_breaks"]
    df = pd.DataFrame(latest.values(), columns=columns).sort_values("trip_id") if latest else pd.DataFrame(columns=columns)
    if len(df):
        (ROOT / "results").mkdir(exist_ok=True)
        df.to_csv(ROOT / "results" / "api_student_test.csv", index=False)
    elif (ROOT / "results" / "api_student_test.csv").exists():
        df = pd.read_csv(ROOT / "results" / "api_student_test.csv")
    return df


def map_figure() -> Path:
    from pathsnap.io import load_trip
    trip = load_trip(ROOT / "Student folder" / "train" / "00000001", with_labels=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    stride = max(1, len(trip.arcs) // 3000)
    for arc_id in range(0, len(trip.arcs), stride):
        a, b = trip.arcs[arc_id]
        pts = trip.node_lonlat[[a, b]]
        ax.plot(pts[:, 0], pts[:, 1], color="#a8b0bd", lw=0.35, alpha=0.42)
    if trip.route is not None:
        route_xy = []
        for arc_id in trip.route:
            a, b = trip.arcs[int(arc_id)]
            for node in (a, b):
                xy = trip.node_lonlat[node]
                if not route_xy or (xy != route_xy[-1]).any():
                    route_xy.append(xy)
        if route_xy:
            arr = pd.DataFrame(route_xy, columns=["lon", "lat"])
            ax.plot(arr.lon, arr.lat, color="#e53935", lw=1.1, label="Reference route")
    ax.plot(trip.lon, trip.lat, color="#2166f3", lw=0.7, alpha=0.85, label="GPS track")
    ax.scatter([trip.lon[0], trip.lon[-1]], [trip.lat[0], trip.lat[-1]], c=["green", "black"], s=14, zorder=5, label="Start / end")
    ax.set_title("Train trip 00000001 — sampled road network and reference layers")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path = ROOT / "reports" / "figures" / "report_map_view.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 12 * mm, "PathSnap | Development report | evaluator data excluded")
    canvas.drawRightString(192 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ex = pd.read_csv(ROOT / "results" / "experiments.csv").drop_duplicates("model", keep="last")
    api = load_api_log()
    inv = json.loads((ROOT / "results" / "data_inventory.summary.json").read_text())
    tuning = json.loads((ROOT / "results" / "tuning_summary_lightgbm.json").read_text())
    map_path = map_figure()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=26, leading=32))
    story = []
    story += [Spacer(1, 24 * mm), Paragraph("PathSnap", styles["TitleCenter"]),
              Paragraph("Technical report: data, models, decoder and API integration", styles["Heading2"]) ,
              Spacer(1, 8 * mm), Paragraph("Versioned from the supplied project checkout on 18 November 2026", styles["BodyText"]),
              Spacer(1, 12 * mm), Paragraph("Status: Person C implementation and development-set in-process integration are documented. Person D UI/screenshots and a two-laptop network rehearsal remain team handoffs. See Section 7.", styles["BodyText"]), PageBreak()]

    story += [Paragraph("1. Problem and solution", styles["Heading1"]),
              Paragraph("The task is to recover the road-arc sequence driven by a vehicle from noisy, approximately 1 Hz GPS observations. Nearby parallel roads and GPS drift make independent nearest-road snapping unreliable. PathSnap generates candidate road arcs for each observation, computes geometric and contextual features, scores candidates with a trained LightGBM classifier, decodes a coherent sequence with an HMM/Viterbi stage, and stitches a route through the trip-local road graph.", styles["BodyText"]),
              Spacer(1, 3 * mm), Paragraph("The supplied data has 80 training trips and 10 student_test development trips. Every trip carries its own nodes/arcs graph. Ground-truth `.route` files are used to derive labels/evaluate only after prediction; the withheld evaluator set is absent. The prediction endpoint takes a trip ID and does not return reference labels.", styles["BodyText"]),
              Paragraph("2. Dataset and quality", styles["Heading1"])]
    data = [["Partition", "Trips", "GPS rows", "Nodes", "Arcs"]]
    for part, vals in inv.items():
        data.append([part, vals["trips"], f'{vals["track_rows"]:,}', f'{vals["nodes"]:,}', f'{vals["arcs"]:,}'])
    story.append(Table(data, style=TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce6f1")), ("GRID", (0,0), (-1,-1), .35, colors.grey), ("PADDING", (0,0), (-1,-1), 5)])))
    story += [Spacer(1, 3 * mm), Paragraph("The inventory found valid coordinates, arc endpoints, and increasing timestamp sequences, with no isolated nodes. However, 79/80 training road graphs have multiple weakly connected components (median 20; maximum 421), so disconnected transitions need explicit handling. Training has 2,937 timestamp gaps longer than two seconds, including 1,260 longer than five seconds. Label alignment coverage averages about 98.0% on training; trip 00000054 is a severe outlier (14.7% labelled; 90.2% of observations more than 30 m from the aligned arc), followed by trip 00000033 (49.0% labelled). These trips limit label reliability.", styles["Small"]),
              Paragraph("3. Model comparison", styles["Heading1"]),
              Paragraph("All candidates use the shared five-fold trip-grouped CV split, seed 42 and training-only fits. Candidate rows are not independent trip samples, so grouped folds keep each trip intact. KNN uses a deterministic 3,000-row fit subsample per fold; evaluation covers the full feature rows. The table reports saved output values; `student_test` is development-only.", styles["Small"])]
    heavy = ex[ex.model.isin(["lightgbm", "xgboost", "hist_gb", "extra_trees", "random_forest", "logistic_regression", "gaussian_nb", "knn", "nearest_arc"])].copy()
    rows = [["Model", "CV Top-1", "Test Top-1", "Test macro-F1", "Test log loss"]]
    for _, r in heavy.iterrows():
        rows.append([str(r.model), f'{r.cv_top1_mean:.3f}' if pd.notna(r.cv_top1_mean) else "—", f'{r.test_top1:.3f}' if pd.notna(r.test_top1) else "—", f'{r.test_macro_f1:.3f}' if pd.notna(r.test_macro_f1) else "—", f'{r.test_logloss:.3f}' if pd.notna(r.test_logloss) else "—"])
    tab = Table(rows, repeatRows=1, colWidths=[43*mm, 28*mm, 28*mm, 34*mm, 30*mm])
    tab.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce6f1")), ("GRID", (0,0), (-1,-1), .3, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 7), ("PADDING", (0,0), (-1,-1), 4)]))
    story.append(tab)
    story += [Spacer(1, 3*mm), Image(str(ROOT / "results" / "plots" / "model_comparison.png"), width=165*mm, height=43*mm),
              Paragraph("Selected LightGBM: CV Top-1 0.8748 vs tuned 0.8746; student_test observation Top-1 0.9021 vs nearest-arc 0.5360. Candidate-level macro-F1 is a separate metric.", styles["Small"]), PageBreak()]

    story += [Paragraph("4. Bayesian optimization and exported model", styles["Heading1"]),
              Paragraph(f"Optuna TPE retained {tuning.get('n_trials', 34)} trials (see `results/tuning_trials_lightgbm.csv`). The objective was mean trip-grouped CV Top-1 minus 0.05 times log loss. Tuned LightGBM achieved CV Top-1 {tuning.get('best_trial_cv_top1', 0):.4f}; the selected untuned configuration achieved 0.8748, so the untuned configuration was refit. The exported model uses seed 42, 300 estimators, learning rate 0.1, 63 leaves, 40 minimum child samples and 27 ordered features. It was refit from 678,859 sampled candidate rows from all 80 training trips; the student-test score was not used to select it. Full configuration and provenance are in `artifacts/model_card.json`.", styles["BodyText"]),
              Paragraph("5. Decoder, API and development evaluation", styles["Heading1"]),
              Paragraph("Person B's decoder constructs a per-trip projected road graph, spatially generates candidates via the adapter, runs HMM/Viterbi and stitches the selected arcs. Person C's FastAPI service exposes `/health`, `/trips`, and `/predict`; scoring reads `.route` after `predict_trip` returns and is retained server-side. The all-ten integration test invoked the API handler in-process, verified response lengths and metrics, and passed a separate prediction with the route file omitted. Local HTTP binding was blocked by the sandbox, so this does not verify TCP/remote-client behavior.", styles["BodyText"])]
    if len(api):
        rows = [["Trip", "Obs.", "Classifier", "HMM", "Breaks", "Seconds"]]
        for _, r in api.iterrows():
            rows.append([str(r.trip_id).zfill(8), int(r.observations), f'{r.classifier_top1:.3f}' if pd.notna(r.get("classifier_top1")) else "—", f'{r.hmm_top1:.3f}', int(r.route_breaks), f'{r.inference_seconds:.1f}'])
        tab = Table(rows, repeatRows=1, colWidths=[24*mm, 26*mm, 29*mm, 24*mm, 24*mm, 25*mm])
        tab.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce6f1")), ("GRID", (0,0), (-1,-1), .3, colors.grey), ("FONTSIZE", (0,0), (-1,-1), 7), ("PADDING", (0,0), (-1,-1), 4)]))
        story.append(tab)
        weighted_hmm = (api.hmm_top1 * api.observations).sum() / api.observations.sum()
        if "classifier_top1" in api.columns:
            weighted_classifier = (api.classifier_top1 * api.observations).sum() / api.observations.sum()
            continuity_weight = api.observations - 1
            classifier_continuity = (api.classifier_route_continuity * continuity_weight).sum() / continuity_weight.sum()
            hmm_continuity = (api.hmm_route_continuity * continuity_weight).sum() / continuity_weight.sum()
            comparison = (f"Observation-weighted Top-1: classifier {weighted_classifier:.3f}; HMM {weighted_hmm:.3f}. "
                          f"Weighted continuity: classifier {classifier_continuity:.3f}; HMM {hmm_continuity:.3f}.")
        else:
            comparison = f"Observation-weighted HMM Top-1: {weighted_hmm:.3f}; classifier scores were not saved in the request log."
        story.append(Paragraph(f"{comparison} Total stitched-route breaks: {int(api.route_breaks.sum())}. Development-set results only. Slowest trip took {api.inference_seconds.max():.1f} s inside inference, above the plan's 30 s target; B should profile it.", styles["Small"]))
    else:
        story.append(Paragraph("No all-trip API evaluation log was available when this report was generated.", styles["BodyText"]))
    story += [PageBreak(), Paragraph("6. Maps and visual diagnostics", styles["Heading1"]),
              Paragraph("The repository contains an interactive Folium map with layer controls for sampled road arcs, raw GPS and the development reference route (`reports/eda_map.html`). The figure below renders the same train trip layers for inclusion in this PDF. It is an EDA illustration, not a prediction-quality claim.", styles["Small"]),
              Image(str(map_path), width=165*mm, height=99*mm),
              Paragraph("Candidate sweep, calibration, confusion matrix and label-quality outputs are in `results/plots/` and `results/`.", styles["Small"]), PageBreak()]

    story += [Paragraph("7. API/UI demo, limitations and handoffs", styles["Heading1"]),
              Paragraph("Person C deliverables in this checkout include the data loader/inventory, EDA tooling, simple-model comparison, model-analysis plots, the A/B inference adapter, post-prediction scoring, FastAPI routes/logging, route-free check, README and this report. A/B contributions are integrated locally but need their owners' review. Person D's UI, UI screenshots, console screenshots, demo checklist and section 7 narrative have not been supplied in this checkout. A two-laptop test against a reachable API has not been performed.", styles["BodyText"]),
              Paragraph("Open handoffs", styles["Heading2"]),
              Paragraph("• A: validate the reconstructed `artifacts/model.joblib` against A's export and review feature/candidate adapter parity.\n• B: review projection conversion and diagnose the 43.6-second trip 00000056 inference.\n• D: provide UI, screenshots, and demo checklist; confirm required response fields.\n• Team: merge/reconcile branches, run API over TCP on Laptop 1 and UI on Laptop 2, then preserve genuine screenshots/logs and rerun the report. The evaluator trips must remain withheld until the final session.", styles["BodyText"]),
              Paragraph("Reproduction", styles["Heading2"]),
              Paragraph("Install `requirements.txt`, set `PYTHONPATH=src`, run `scripts/inventory.py`, `scripts/train_simple.py`, `scripts/analysis_plots.py`, `scripts/export_model.py`, the API integration unittest, then `scripts/build_report.py`. Saved numeric sources are `results/experiments.csv`, `results/label_stats.csv`, `results/data_inventory.summary.json`, and the API request log. AI assistance disclosure and run commands are in the README.", styles["Small"])]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=17*mm, bottomMargin=19*mm, title="PathSnap Technical Report")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
