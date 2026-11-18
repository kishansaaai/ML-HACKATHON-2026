"""Build a report draft from saved official results. Team-authored analysis remains marked."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "output" / "pdf" / "PathSnap_technical_report.pdf"
BLUE = colors.HexColor("#dce6f1")


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(14 * mm, 9 * mm, "PathSnap | student_test is development data; evaluator data excluded")
    canvas.drawRightString(283 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def make_eda_plot():
    inv = pd.read_csv(ROOT / "results" / "data_inventory.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for part, group in inv.groupby("partition"):
        axes[0].plot(np.arange(len(group)), group.track_rows.sort_values().to_numpy(), marker=".", label=part)
    axes[0].set(title="Track observations per trip", xlabel="Trip rank", ylabel="GPS rows")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    if "timestamp_gap_gt_5s" in inv:
        axes[1].bar(np.arange(len(inv)), inv.timestamp_gap_gt_5s)
        axes[1].set(title="Timestamp gaps > 5 seconds", xlabel="Trip row", ylabel="Gap count")
    else:
        axes[1].bar([0, 1], [0, 0])
        axes[1].set(title="Timestamp-gap counts", ylabel="Gap count")
    fig.tight_layout()
    path = ROOT / "reports" / "figures" / "report_inventory.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def rows_for_model_table():
    official = pd.read_csv(ROOT / "results" / "metrics_table.csv")
    exp = pd.read_csv(ROOT / "results" / "experiments.csv").drop_duplicates("model", keep="last")
    exp = exp.set_index("model")
    rows = []
    simple = {"logistic_regression", "gaussian_nb", "knn"}
    for _, r in official.iterrows():
        name = r["model"]
        if name in simple or name == "lightgbm_tuned":
            continue
        cfg = exp.loc[name, "config_json"] if name in exp.index else "—"
        rows.append({"model": name, "family": r["family"], "config": str(cfg),
                     "cv_f1": r["cv_macro_f1"], "cv_top1": r["cv_top1"],
                     "cv_cont": r["cv_route_continuity"], "test_f1": r["test_macro_f1"],
                     "test_top1": r["test_top1"], "test_cont": r["test_route_continuity"],
                     "logloss": r["test_logloss"], "infer": r["infer_time_s_per_trip"]})
    for name in simple:
        if name not in exp.index:
            continue
        r = exp.loc[name]
        rows.append({"model": name, "family": r["family"], "config": str(r["config_json"]),
                     "cv_f1": r["cv_macro_f1_mean"], "cv_top1": r["cv_top1_mean"],
                     "cv_cont": r.get("cv_route_continuity_mean", np.nan), "test_f1": r["test_macro_f1"],
                     "test_top1": r["test_top1"], "test_cont": r.get("test_route_continuity", np.nan),
                     "logloss": r["test_logloss"], "infer": r["infer_time_s_per_trip"]})
    return rows


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.5, leading=9))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=6.5, leading=7.5))
    styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], alignment=TA_CENTER, fontSize=30))
    story = [Spacer(1, 35*mm), Paragraph("PathSnap", styles["Cover"]),
             Paragraph("Technical report draft · data, models, decoder, API and maps", styles["Heading2"]),
             Spacer(1, 8*mm), Paragraph("PES University ML Hackathon · 18 November 2026", styles["BodyText"]),
             Spacer(1, 15*mm), Paragraph("This draft uses saved repository results. Team members must add their own model analyses and demo screenshots before submission; those sections are deliberately not written on their behalf.", styles["BodyText"]), PageBreak()]

    inv = json.loads((ROOT / "results" / "data_inventory.summary.json").read_text())
    quality = pd.read_csv(ROOT / "results" / "label_quality.csv", index_col=0)
    eda = make_eda_plot()
    story += [Paragraph("2. Dataset understanding and exploratory analysis", styles["Heading1"]),
              Paragraph("Each trip contains GPS observations (`.track`), a trip-local road graph (`.nodes`, `.arcs`) and a reference arc sequence (`.route`). The prediction loader reads track, nodes and arcs only; route labels are opt-in and are loaded by the scoring layer after prediction. Candidate features are generated from local projected geometry and graph context. The 80 training trips support fitting and grouped CV; the 10 `student_test` trips are development diagnostics only.", styles["BodyText"])]
    table = [["Partition", "Trips", "GPS rows", "Nodes", "Arcs"]]
    for part, row in inv.items():
        table.append([part, row["trips"], f'{row["track_rows"]:,}', f'{row["nodes"]:,}', f'{row["arcs"]:,}'])
    story.append(Table(table, style=TableStyle([("BACKGROUND", (0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.3,colors.grey),("PADDING",(0,0),(-1,-1),4)])))
    train = quality.loc["train"]
    story += [Spacer(1,2*mm), Paragraph(f"Inventory checks found valid coordinates, arc endpoints and increasing timestamps. Training label coverage averages {train.frac_labelled_mean:.1%}; the weakest trip, 00000054, has {train.frac_labelled_min:.1%} aligned observations, with substantial alignment uncertainty. There are 79/80 training graphs with multiple weak components. The feature/candidate process is documented in the A pipeline notes; candidate radius 50 m with nearest-K fallback has measured candidate misses in `results/label_stats.csv`.", styles["Small"]),
              Image(str(eda), width=175*mm, height=64*mm),
              Paragraph("The interactive Folium map is generated at `reports/eda_map.html`. The map and figures show development/training data, not evaluator data.", styles["Small"]), PageBreak()]

    rows = rows_for_model_table()
    story += [Paragraph("3. Model experimentation and comparison", styles["Heading1"]),
              Paragraph("Metrics use the official `results/metrics_table.csv` for integrated A/B models; C's simple-model rows were appended to `results/experiments.csv` after rerunning on the shared candidate pipeline. Route continuity uses the official adjacency definition. Inference time is saved per trip. KNN uses a deterministic 3,000-row fit subsample per fold; validation and student-test rows are complete.", styles["Small"])]
    data = [["Model / family", "Key config", "CV F1", "CV Top-1", "CV cont.", "Test F1", "Test Top-1", "Test cont.", "Log loss", "Infer s"]]
    for r in rows:
        data.append([f'{r["model"]}\n{r["family"]}', r["config"][:70],
                     *(f'{r[k]:.3f}' if pd.notna(r[k]) else "N/A" for k in ("cv_f1","cv_top1","cv_cont","test_f1","test_top1","test_cont","logloss","infer"))])
    tbl = Table(data, repeatRows=1, colWidths=[40*mm, 51*mm, 15*mm, 17*mm, 17*mm, 17*mm, 18*mm, 18*mm, 16*mm, 15*mm])
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.2),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("PADDING",(0,0),(-1,-1),3)]))
    story += [tbl, Spacer(1,2*mm), Image(str(ROOT / "results" / "plots" / "model_comparison.png"), width=132*mm, height=39*mm),
              Image(str(ROOT / "results" / "plots" / "simple_model_comparison.png"), width=132*mm, height=44*mm), PageBreak()]

    story += [Paragraph("3.2 Required per-model discussion — team completion needed", styles["Heading1"]),
              Paragraph("The submission guide requires the team to answer why each algorithm was considered, task-specific strengths and limits, configuration/objective, measured performance, and over/underfitting evidence in the team's own words. These narrative answers are intentionally left for the team. The measured table and plots above are generated from repository files.", styles["BodyText"])]
    for r in rows:
        story.append(Paragraph(f'{r["model"]}: Team members — add your algorithm rationale, observed strengths/limitations, objective/configuration explanation, and over/underfit conclusion from the saved evidence.', styles["Small"]))
        story.append(Spacer(1,1.5*mm))
    story += [Paragraph("4. Bayesian hyperparameter optimisation", styles["Heading1"])]
    tune = json.loads((ROOT / "results" / "tuning_summary_lightgbm.json").read_text())
    story += [Paragraph(f"Optuna TPE completed {tune['n_trials']} trials using 5 trip-grouped CV folds on training data. The maximised objective was mean CV Top-1 − 0.05 × candidate log loss; training folds used every sixth GPS point for the search, with full validation folds. Search ranges are implemented in `pathsnap.tuning.suggest`: leaves 15–127; learning rate 0.03–0.2 (log); estimators 100–500 by 50; min child samples 10–200 (log); row/column subsampling 0.5–1.0; L1/L2 regularisation 0.001–10/30 (log). Best values: {json.dumps(tune['best_params'], sort_keys=True)}. The tuned and untuned rows are compared in `results/metrics_table.csv`; selection/export used grouped CV, never student_test. Trial values are in `results/tuning_trials_lightgbm.csv`.", styles["Small"]), PageBreak()]

    pipeline = pd.read_csv(ROOT / "results" / "pipeline_student_test_default.csv", dtype={"trip_id": str})
    n = int(pipeline.n_labelled.sum())
    clf = float((pipeline.clf_top1 * pipeline.n_labelled).sum() / n)
    hmm = float((pipeline.hmm_top1 * pipeline.n_labelled).sum() / n)
    clf_cont = float(pipeline.clf_continuity.mean())
    hmm_cont = float(pipeline.hmm_continuity.mean())
    story += [Paragraph("5. HMM and Viterbi route reconstruction", styles["Heading1"]),
              Paragraph("The decoder uses candidate road arcs as hidden states. Emissions come from the classifier's candidate probabilities; transitions use directed graph reachability and GPS displacement/time, computed in log space by Viterbi. Gaps restart decoding segments, and route stitching keeps disconnected segments separate. Parameters shown are the default configuration; the integration notes state decoder tuning is not yet complete.", styles["BodyText"])]
    htab = [["Output", "Pooled Top-1", "Trip-mean official continuity", "Route breaks"],
            ["Classifier only", f"{clf:.2%}", f"{clf_cont:.2%}", int(pipeline.clf_breaks.sum())],
            ["Classifier + HMM", f"{hmm:.2%}", f"{hmm_cont:.2%}", int(pipeline.hmm_breaks.sum())]]
    story += [Table(htab, style=TableStyle([("BACKGROUND",(0,0),(-1,0),BLUE),("GRID",(0,0),(-1,-1),.3,colors.grey),("PADDING",(0,0),(-1,-1),5)])),
              Paragraph("On the one-time development comparison, HMM continuity improves while pooled Top-1 is slightly lower; it is better on five trips and worse on five. Do not claim an overall accuracy gain. This is not a final evaluator estimate. The per-trip source is `results/pipeline_student_test_default.csv`.", styles["Small"]),
              Paragraph("6. Geographical visualisation using Folium", styles["Heading1"]),
              Paragraph("The actual interactive map is `reports/eda_map.html`, generated by `scripts/make_eda_map.py`, with GPS, sampled road geometry, reference-route layers and controls. This PDF includes an exported static view derived from the same trip data. A real browser screenshot may be added after the team checks the interactive map.", styles["Small"]),
              Image(str(ROOT / "reports" / "figures" / "report_map_view.png"), width=160*mm, height=96*mm), PageBreak()]

    story += [Paragraph("7. API, UI and deployment", styles["Heading1"]),
              Paragraph("The FastAPI service implements `/health`, `/trips`, and `/predict` to match the latest Person D contract. Startup loads the exported model, warms the pipeline, and rejects an evaluator directory unless exactly ten valid trip folders are present. Without that setting, the API reports `student_test`. The prediction handler uses `pathsnap.pipeline`, then performs server-side scoring; responses include `[lat, lon]` Folium coordinates and omit reference labels/routes. A readable Laptop 1 console line is emitted per prediction.", styles["BodyText"]),
              Paragraph("The in-process TestClient checks and API scoring run cover the three HTTP routes and ten development trips. They do not constitute a two-laptop network rehearsal. This checkout does not include Person D's UI screenshots or a genuine remote CMD screenshot; add them after running on the demo laptops. Do not substitute fabricated screenshots.", styles["BodyText"]),
              Paragraph("Team-owned completion", styles["Heading2"]),
              Paragraph("A: review the integrated model artifact and experiments. B: review decoder integration and timings. D: add UI screenshots and demo checklist. Team: add own per-model narrative answers, perform the real laptop-to-laptop run, and regenerate/verify this PDF before submission.", styles["BodyText"]),
              Paragraph("Reproduction: `pip install -r requirements.txt`; `pip install -e .`; run `scripts/inventory.py`, `scripts/make_candidates.py`, `scripts/train_simple.py`, `scripts/analysis_plots.py`, `scripts/plot_simple_models.py`, `scripts/evaluate_api.py`, then `scripts/build_report.py`. AI assistance is disclosed in README. All reported metrics are from saved repository CSV/JSON outputs.", styles["Small"])]
    doc = SimpleDocTemplate(str(OUT), pagesize=landscape(A4), leftMargin=14*mm, rightMargin=14*mm,
                            topMargin=13*mm, bottomMargin=15*mm, title="PathSnap Technical Report Draft")
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUT)


if __name__ == "__main__":
    build()
