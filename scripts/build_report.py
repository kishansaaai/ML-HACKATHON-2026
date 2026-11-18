"""Build the consolidated report in the exact section order required by Deliverables.pdf.

The script only writes repository-backed measurements. Team-authored interpretations and
live-demo screenshots are called out as required review items rather than fabricated.
"""
from __future__ import annotations

import json
import html
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "PathSnap_technical_report.pdf"
BLUE = colors.HexColor("#dce6f1")
RED = colors.HexColor("#fce4d6")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(12 * mm, 8 * mm, "PathSnap | development partitions only; evaluator data excluded")
    canvas.drawRightString(285 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def table(data, widths, font=6.5, repeat=1):
    result = Table(data, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("GRID", (0, 0), (-1, -1), .3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font + 1.2), ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return result


def number(value, pct=False, digits=3):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}%}" if pct else f"{float(value):.{digits}f}"


def models_data():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv").drop_duplicates("model", keep="last").set_index("model")
    official = pd.read_csv(ROOT / "results" / "metrics_table.csv").drop_duplicates("model", keep="last").set_index("model")
    names = ["nearest_arc", "nearest_heading_aware", "logistic_regression", "gaussian_nb", "knn",
             "extra_trees", "random_forest", "xgboost", "hist_gb", "lightgbm", "lightgbm_tuned"]
    out = []
    for name in names:
        if name not in exp.index:
            continue
        row = exp.loc[name].to_dict()
        row["model"] = name
        if name in official.index:
            row.update(official.loc[name].to_dict())
        out.append(row)
    return out


def get_styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="Cover", parent=s["Title"], alignment=TA_CENTER, fontSize=26, leading=31))
    s.add(ParagraphStyle(name="Small", parent=s["BodyText"], fontSize=8, leading=10))
    s.add(ParagraphStyle(name="Tiny", parent=s["BodyText"], fontSize=6.2, leading=7.3))
    s.add(ParagraphStyle(name="Callout", parent=s["BodyText"], backColor=RED, borderPadding=6, fontSize=9, leading=12))
    return s


def model_analysis(model, styles):
    n = model["model"]
    family = str(model.get("family", ""))
    configs = json.loads(model.get("config_json", "{}") or "{}")
    top1 = number(model.get("test_top1"), True, 2)
    f1 = number(model.get("test_macro_f1"), False, 4)
    loss = number(model.get("test_logloss"), False, 4)
    cvf1 = number(model.get("cv_macro_f1_mean"), False, 4)
    c = number(model.get("cv_top1_mean"), True, 2)
    infer = number(model.get("infer_time_s_per_trip"), False, 4)

    rationales = {
        "logistic_regression": "A simple probabilistic linear baseline for candidate features; it is fast and produces probabilities, but a linear boundary may miss nonlinear interactions.",
        "gaussian_nb": "A fast generative probability baseline; its conditional-independence and Gaussian feature assumptions are strong and can misrepresent correlated geometric features.",
        "knn": "A local, nonlinear baseline that can use similarity in scaled feature space; inference cost and sensitivity to feature scaling are limitations, and fitting used a fixed 3,000-row sample per fold.",
        "lightgbm": "A fast gradient-boosted tree model that captures nonlinear interactions in tabular candidate features and supplies emission probabilities; it has more capacity to overfit than a linear baseline.",
        "lightgbm_tuned": "A Bayesian-tuned LightGBM variant tests whether the default tree structure and regularisation can be improved; the measured held-out results show no practical gain over the untuned model.",
        "hist_gb": "Histogram gradient boosting is a nonlinear tree ensemble suited to large tabular data; its measured results are close to the other boosters, so small rank differences should not be over-interpreted.",
        "xgboost": "Regularised depth-wise gradient boosting is a strong nonlinear tabular comparator; it is accurate here, though training was slower than the selected LightGBM run.",
        "extra_trees": "Randomized tree splits provide a nonlinear ensemble comparator with variance reduction; this run used a time-boxed 50-tree, stride-6 configuration.",
        "random_forest": "Bagged trees capture nonlinear feature interactions and reduce individual-tree variance; this run used a time-boxed 50-tree, stride-6 configuration.",
        "nearest_arc": "A transparent geometric reference rule that selects the candidate with the smallest perpendicular distance; it needs no learned parameters but ignores motion and graph context.",
        "nearest_heading_aware": "A simple geometry-plus-heading reference rule; heading helps resolve parallel-road ambiguity, but it has no learned probabilities or graph-wide sequence reasoning.",
    }
    if n in ("nearest_arc", "nearest_heading_aware"):
        loss_obj = "No loss is optimised: this is a deterministic ranking rule. Macro F1 and log loss are not applicable because no calibrated candidate probability is produced."
        overfit = "Not applicable: this baseline has no fitted parameters."
    elif n == "logistic_regression":
        loss_obj = "L2-regularised logistic regression, fitted by the LBFGS solver; candidate binary labels are the targets."
        overfit = f"The saved evidence gives CV Top-1 {c} and student_test Top-1 {top1}; it does not save training-set scores for this model, so this gap alone is not proof of overfitting."
    elif n == "gaussian_nb":
        loss_obj = "Gaussian Naive Bayes estimates class-conditional feature distributions; no separate iterative loss-minimisation objective is configured."
        overfit = f"CV Top-1 is {c} versus student_test {top1}. No training-score diagnostic is saved, so this comparison cannot establish overfitting."
    elif n == "knn":
        loss_obj = "No parametric loss is minimised; predictions use distance-weighted neighbour votes (Euclidean distance, p=2)."
        overfit = f"CV Top-1 is {c} versus student_test {top1}. The 3,000-row fit sample is documented; no training-score diagnostic is saved to establish overfitting."
    elif n in ("extra_trees", "random_forest"):
        loss_obj = "Trees use the configured Gini impurity criterion; the ensemble is fitted from sampled training rows (seed 42)."
        overfit = f"CV Top-1 is {c} versus student_test {top1}. No train-vs-validation learning history is saved; the reduced stride-6/50-tree run also limits conclusions about an unrestricted model."
    else:
        loss_obj = "Binary classification uses the estimator's binary log-loss objective; evaluation includes candidate log loss and trip-grouped observation Top-1."
        overfit = f"CV Top-1 is {c} versus student_test {top1}. For untuned LightGBM, the saved notes additionally report 94.5% training Top-1 vs 87.5% grouped-CV Top-1, evidence of a generalisation gap. For other boosters, no direct training-score comparison is recorded."
    config = json.dumps(configs, sort_keys=True, ensure_ascii=True)
    config_intro = "Tuned configuration (best values; initial registry configuration is in src/pathsnap/models/registry.py and is compared in Section 4): " if n == "lightgbm_tuned" else "Run configuration (the initial configured values for this experiment): "
    text = (f"<b>Why considered / task-specific strengths and limitations.</b> {rationales[n]}<br/>"
            f"<b>Configuration and objective.</b> {config_intro}<font name='Courier'>{config}</font>. "
            f"Seed {model.get('seed', 42)}; training stride {number(model.get('train_stride'), False, 0)}; {loss_obj}<br/>"
            f"<b>Observed student_test results.</b> Observation Top-1 {top1}; candidate Macro F1 {f1}; candidate Log Loss {loss}; "
            f"CV Macro F1 {cvf1}; mean inference time {infer} seconds per trip. {overfit}")
    return Paragraph(f"<b>{n} ({family})</b><br/>{text}", styles["Small"])


def make_eda_plot():
    data = pd.read_csv(ROOT / "results" / "data_inventory.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2))
    for part, group in data.groupby("partition"):
        ordered = group.sort_values("track_rows").reset_index(drop=True)
        axes[0, 0].plot(range(len(ordered)), ordered.track_rows, marker=".", ms=2, label=part)
    axes[0, 0].set(title="GPS observations per trip", xlabel="Trip rank within partition", ylabel="Rows")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=.2)
    for part, group in data.groupby("partition"):
        ordered = group.sort_values("timestamp_gap_gt_5s", ascending=False).reset_index(drop=True)
        axes[0, 1].plot(range(len(ordered)), ordered.timestamp_gap_gt_5s, marker=".", ms=2, label=part)
    axes[0, 1].set(title="Timestamp gaps > 5 seconds", xlabel="Trip rank", ylabel="Gap count")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=.2)
    axes[1, 0].bar(["train", "student_test"],
                   [data.loc[data.partition.eq(p), "connected_components_nonisolated_undirected"].mean()
                    for p in ("train", "student_test")], color=["#5277b5", "#4b9b77"])
    axes[1, 0].set(title="Mean weak components per trip", ylabel="Components (undirected diagnostic)")
    invalid_cols = ["invalid_track_points", "invalid_node_coordinates", "invalid_arc_endpoints", "nonincreasing_timestamps"]
    totals = data.groupby("partition")[invalid_cols].sum()
    axes[1, 1].bar(["train", "student_test"], [totals.loc[p].sum() for p in ("train", "student_test")],
                   color=["#5277b5", "#4b9b77"])
    axes[1, 1].set(title="Invalid/missing structural records", ylabel="Count")
    axes[1, 1].text(.5, .9, "0 in both partitions", transform=axes[1, 1].transAxes,
                    ha="center", va="top", fontsize=8)
    fig.suptitle("Dataset quality and structure (repository inventory)")
    fig.tight_layout()
    destination = ROOT / "reports" / "figures" / "report_inventory.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def build():
    styles = get_styles()
    story = [Spacer(1, 28 * mm), Paragraph("PathSnap", styles["Cover"]),
             Paragraph("Technical Report | Consolidated team draft", styles["Heading2"]),
             Spacer(1, 4 * mm), Paragraph("PES University ML Hackathon", styles["BodyText"]),
             Paragraph("Report format: Deliverables.pdf, required sections 2.2 through 7", styles["BodyText"]),
             Spacer(1, 8 * mm), Paragraph("Submission status: report structure and repository evidence are assembled. Real two-laptop UI/API screenshots are not present yet; this PDF is not submission-ready until section 7 is completed and the team reviews its analysis.", styles["Callout"]), PageBreak()]

    # Section 2.2: data, preprocessing, patterns, features and partition provenance.
    inv = json.loads((ROOT / "results" / "data_inventory.summary.json").read_text(encoding="utf-8"))
    quality = pd.read_csv(ROOT / "results" / "label_quality.csv", index_col=0)
    eda_img = make_eda_plot()
    story += [Paragraph("2.2 Dataset Understanding and Exploratory Data Analysis", styles["Heading1"]),
              Paragraph("The supplied development dataset contains 80 training trips and 10 student_test trips. Each trip has its own road graph. Raw files are whitespace-separated without headers: .track stores longitude, latitude and timestamp; .nodes stores longitude/latitude; .arcs stores directed node-index pairs whose row number is the arc ID; .route stores a reference arc sequence. The final evaluator partition is separate and was not used for these analyses.", styles["BodyText"]),
              Paragraph("Preprocessing loads each trip independently, validates coordinates and arc endpoints, and projects GPS fixes and road nodes into a trip-local metric coordinate frame before computing geometric features. The prediction pipeline loads only .track, .nodes and .arcs. The route is enabled only in post-prediction scoring or explicitly labelled EDA maps. Candidate generation uses a 50 m radius with nearest-K fallback; route-derived observation labels use monotone alignment.", styles["BodyText"])]
    inv_table = [["Partition", "Trips", "GPS observations", "Road nodes", "Directed arcs"]]
    for part, row in inv.items():
        inv_table.append([part, row["trips"], f'{row["track_rows"]:,}', f'{row["nodes"]:,}', f'{row["arcs"]:,}'])
    story += [table(inv_table, [35*mm, 20*mm, 40*mm, 38*mm, 38*mm], font=8), Spacer(1, 3*mm)]
    train_q, test_q = quality.loc["train"], quality.loc["student_test"]
    story.append(Paragraph(
        f"<b>Missing-value and quality analysis.</b> The inventory checks found no invalid GPS coordinates, invalid arc endpoints, non-increasing timestamp transitions or isolated nodes in the supplied 90-trip snapshot. "
        f"Mean route-alignment coverage is {train_q.frac_labelled_mean:.2%} on train and {test_q.frac_labelled_mean:.2%} on student_test. Train trip 00000054 is a documented alignment outlier ({train_q.frac_labelled_min:.2%} aligned points). "
        f"Train graphs are often disconnected (79/80 have multiple weak components), so route reachability and segment breaks matter. About {train_q.miss_rate_mean:.3%} of labelled train observations and {test_q.miss_rate_mean:.3%} of student_test labels are missed by the saved candidate policy. These are descriptive checks on the supplied development partitions.", styles["Small"]))
    story.append(Paragraph("<b>Observed patterns and feature rationale.</b> GPS sampling is mostly near 1 Hz, with longer gaps in both partitions; networks vary in size and are trip-local. The 27 candidate features combine perpendicular distance and projection fraction, heading/speed/time-gap motion, candidate rank/count/gap, endpoint topology, and neighbouring-observation support. Geometry narrows spatially plausible arcs; motion and neighbour support resolve parallel-road ambiguity; topology gives the decoder connectivity context. Features are computed per observation-candidate pair and the model is trained with trip-grouped folds.", styles["Small"]))
    if eda_img:
        story += [Spacer(1, 2*mm), Image(str(eda_img), width=170*mm, height=62*mm),
                  Paragraph("Figure 2.1. GPS row counts and timestamp-gap counts by trip. Train and student_test are labelled separately. Generated from results/data_inventory.csv by scripts/build_report.py.", styles["Tiny"])]
    story += [Paragraph("The repository also contains per-trip GPS trajectory and network plots under reports/figures/ and an interactive Folium map at reports/eda_map.html. That map uses train trip 00000001 and contains GPS, sampled road geometry, reference route and predicted HMM route layers. The final evaluator partition is not included in this EDA.", styles["Small"]), PageBreak()]

    # Section 3.1: one table includes all learned variants; baseline-only metrics are N/A.
    models = models_data()
    story += [Paragraph("3. Model Experimentation and Comparison", styles["Heading1"]),
              Paragraph("The experiment ledger includes nine learned model runs (five main A models, one tuned LightGBM variant, and three C simple models) plus two non-learned reference baselines. The tuned variant is reported separately because it was actually trained and evaluated. student_test is a development comparison partition, not final evaluator data.", styles["Small"]),
              Paragraph("3.1 Overall Model Comparison", styles["Heading2"])]
    cmp = [["Model", "Algorithm family", "Key configuration (ledger)", "CV Macro F1", "student_test Top-1", "Route continuity", "student_test Macro F1", "Log Loss", "Inference s/trip"]]
    for m in models:
        macro = m.get("cv_macro_f1", m.get("cv_macro_f1_mean"))
        test_macro = m.get("test_macro_f1")
        cont = m.get("test_route_continuity")
        cfg_values = json.loads(m.get("config_json", "{}") or "{}")
        cfg = Paragraph("<br/>".join(html.escape(f"{key}: {value}") for key, value in cfg_values.items()), styles["Tiny"])
        cmp.append([m["model"], str(m.get("family", "")), cfg,
                    number(macro, False, 3), number(m.get("test_top1"), True, 2),
                    number(cont, True, 2), number(test_macro, False, 3),
                    number(m.get("test_logloss"), False, 4), number(m.get("infer_time_s_per_trip"), False, 4)])
    story += [table(cmp, [24*mm, 31*mm, 68*mm, 18*mm, 25*mm, 23*mm, 25*mm, 20*mm, 20*mm], font=6.2),
              PageBreak(),
              Paragraph("3.2 Compulsory Questions for Every Model", styles["Heading2"]),
              Paragraph("The answers below connect algorithm choice, configuration and measured results to this task. The team must verify and be able to explain this analysis before submission. No training curves are shown because the recorded estimators do not provide epoch-wise training histories.", styles["Small"])]
    for model in models:
        story += [Spacer(1, 2*mm), model_analysis(model, styles)]
    story += [PageBreak(), Paragraph("3.3 Model Visualisations (all learned models)", styles["Heading2"]),
              Paragraph("The comparison charts are generated from the experiment ledger: trip-grouped CV Macro F1, student_test Top-1 and Macro F1, and student_test Log Loss. Baseline-only scores appear in Top-1 comparisons; Macro F1 and Log Loss are omitted for baselines because they do not emit probabilities. Confusion matrices are candidate-level diagnostics for the learned models. No artificial training or validation curves are included.", styles["Small"])]
    for img_path, caption, width, height in [
        ("model_comparison.png", "Figure 3.1. Actual model comparison; CV error bars show fold spread.", 260, 75),
        ("confusion_all_models.png", "Figure 3.2. Candidate-level confusion matrices from actual student_test predictions.", 260, 62),
        ("calibration_lightgbm.png", "Figure 3.3. LightGBM probability calibration on student_test.", 105, 90),
    ]:
        p = ROOT / "results" / "plots" / img_path
        if p.is_file():
            story += [Image(str(p), width=width*mm, height=height*mm), Paragraph(caption, styles["Tiny"])]
    story.append(PageBreak())

    # Section 4: compare same chosen model, same folds and same student_test partition.
    metrics = pd.read_csv(ROOT / "results" / "metrics_table.csv").set_index("model")
    untuned, tuned = metrics.loc["lightgbm"], metrics.loc["lightgbm_tuned"]
    tune = json.loads((ROOT / "results" / "tuning_summary_lightgbm.json").read_text(encoding="utf-8"))
    tuning_rows = [["Version", "Search / configuration", "CV Macro F1", "student_test Top-1", "student_test Macro F1", "student_test Log Loss", "Inference s/trip"],
                   ["Untuned LightGBM", "Initial registry values", number(untuned.cv_macro_f1, False, 4), number(untuned.test_top1, True, 2), number(untuned.test_macro_f1, False, 4), number(untuned.test_logloss, False, 4), number(untuned.infer_time_s_per_trip, False, 4)],
                   ["Bayesian-tuned LightGBM", "Optuna TPE; 34 trials", number(tuned.cv_macro_f1, False, 4), number(tuned.test_top1, True, 2), number(tuned.test_macro_f1, False, 4), number(tuned.test_logloss, False, 4), number(tuned.infer_time_s_per_trip, False, 4)]]
    story += [Paragraph("4. Bayesian Hyperparameter Optimisation", styles["Heading1"]),
              Paragraph(f"The selected model family was LightGBM. Optuna TPE ran {tune['n_trials']} trials with seed 42 and eight random-start trials. The maximised objective was mean 5-fold trip-grouped CV observation Top-1 minus 0.05 times candidate log loss. Only the 80 training trips were used; training rows were sampled at stride 6 during search, and validation folds were scored on all points. Search ranges: num_leaves 15–127 (log); learning_rate 0.03–0.2 (log); n_estimators 100–500 by 50; min_child_samples 10–200 (log); subsample and colsample_bytree 0.5–1.0; reg_lambda 0.001–30 (log); reg_alpha 0.001–10 (log).", styles["BodyText"]),
              Paragraph("<b>Best parameters:</b><br/>" + "<br/>".join(
                  html.escape(f"{key}: {value}") for key, value in sorted(tune["best_params"].items())), styles["Small"]),
              table(tuning_rows, [38*mm, 61*mm, 31*mm, 35*mm, 35*mm, 34*mm, 30*mm], font=7),
              Paragraph("The tuned model did not outperform the untuned model overall. The comparison above uses the same trip-grouped CV and student_test partitions; the untuned model remains exported based on the CV selection rule. Saved trial records and the tuning summary are in results/tuning_trials_lightgbm.csv and results/tuning_summary_lightgbm.json.", styles["Small"]), PageBreak()]

    # Section 5: frozen decoder results, not obsolete default pipeline values.
    hmm = json.loads((ROOT / "results" / "hmm_tuning" / "final_student_test_check.json").read_text(encoding="utf-8"))
    per_trip = pd.read_csv(ROOT / "results" / "hmm_tuning" / "final_student_test_check.csv", dtype={"trip_id": str})
    worse = ", ".join(per_trip.loc[per_trip.hmm_top1 < per_trip.clf_top1, "trip_id"].astype(str).tolist())
    story += [Paragraph("5. HMM and Viterbi Route Reconstruction", styles["Heading1"]),
              Paragraph("The hidden states at each GPS time are the generated candidate road arcs with a fractional position on each directed arc. The emission log score is log(candidate probability) minus perpendicular-distance-squared divided by 2 sigma-squared. The transition log score is -|directed network distance - GPS displacement| / beta. Directed network distance includes the remainder of the source arc, a bounded shortest path, and the prefix of the destination arc; unreachable pairs receive the configured finite penalty. Viterbi maximises the summed scores in log space. Missing candidate sets, invalid/non-increasing timestamps and gaps above the configured threshold split decoding into separate segments. The stitcher uses legal directed paths where reachable and preserves route breaks rather than drawing fabricated links.", styles["BodyText"]),
              Paragraph("The frozen settings were selected with out-of-fold training scores on the 80 train trips: beta 200 m, emission sigma 25 m, gap threshold 30 s, transition cutoff 10,000 m, unreachable log penalty -5. student_test was used once as a post-freeze development check, not to choose the settings. The separate evaluator partition was not accessed.", styles["Small"])]
    hdata = [["Prediction", "Weighted Top-1", "Mean trip continuity", "Adjacent-pair breaks"],
             ["Classifier only", f"{hmm['classifier_top1_weighted']:.2%}", f"{hmm['classifier_continuity_mean_trip']:.2%}", hmm["classifier_breaks_total"]],
             ["Classifier + HMM/Viterbi", f"{hmm['hmm_top1_weighted']:.2%}", f"{hmm['hmm_continuity_mean_trip']:.2%}", hmm["hmm_breaks_total"]]]
    story += [table(hdata, [58*mm, 45*mm, 55*mm, 48*mm], font=8),
              Paragraph(f"The frozen check covers {hmm['trips']} student_test trips and {hmm['labelled_observations']:,} labeled observations. HMM Top-1 improved by {hmm['top1_delta']:.2%} overall, and mean trip continuity improved. HMM Top-1 was lower on trips {worse}; these are concrete cases where decoding did not help. Results are from results/hmm_tuning/final_student_test_check.csv/.json and do not estimate final evaluator performance.", styles["Small"]), PageBreak(),
              Paragraph("6. Geographical Visualisation Using Folium", styles["Heading1"]),
              Paragraph("The interactive Folium map was generated from the actual pipeline on train trip 00000001. It contains the raw GPS trajectory, sampled road-network geometry, the reference route (available only for this labelled development-trip map), and predicted HMM route as distinct layers with labels and layer controls. The map is committed as reports/eda_map.html; reproducible code is scripts/make_eda_map.py. The static figure below is exported from those same trip inputs and prediction for the PDF.", styles["Small"])]
    map_image = ROOT / "reports" / "figures" / "report_map_view.png"
    if map_image.is_file():
        story.append(Image(str(map_image), width=210*mm, height=124*mm))
        story.append(Paragraph("Figure 6.1. Train trip 00000001, with GPS, road network, reference route and predicted HMM route. Layer colors are labelled in the map and in the interactive Folium version.", styles["Tiny"]))
    else:
        story.append(Paragraph("REQUIRED BEFORE SUBMISSION: regenerate reports/figures/report_map_view.png and reports/eda_map.html with scripts/make_eda_map.py.", styles["Callout"]))
    story.append(PageBreak())

    # Section 7 must contain real captures; do not use mock UI or invented images.
    story += [Paragraph("7. API, UI and Deployment", styles["Heading1"]),
              Paragraph("The submitted code implements the FastAPI backend in src/pathsnap/api.py and the Streamlit/Folium evaluator in ui/. The API exposes GET /health, GET /trips and POST /predict. On startup it validates the configured evaluator folder contains exactly ten trips, loads the exported model and frozen decoder configuration, and warms up prediction. The API accepts only trip_id; it predicts from track/nodes/arcs and scores route labels server-side after prediction. The UI calls the API and renders its output; it performs no model inference.", styles["BodyText"]),
              Paragraph("Deployment commands and the API schema are in README.md, docs/API_CONTRACT.md and docs/DEMO_CHECKLIST.md. The expected deployment is Laptop 1 hosting the API on 0.0.0.0:8000 and Laptop 2 connecting over the same LAN/hotspot to run the UI. This environment has not completed or verified a real two-laptop session.", styles["Small"])]
    required_screenshots = [
        ("docs/screenshots/ui_trip_selector.png", "Figure 7.1. Working UI on Laptop 2, showing model, partition and dynamically loaded trips."),
        ("docs/screenshots/ui_route_map_and_metrics.png", "Figure 7.2. Real prediction map and metrics from the working UI."),
        ("docs/screenshots/laptop1_api_cmd_logs.png", "Figure 7.3. Successful API responses and prediction logs on the receiving Laptop 1 terminal."),
    ]
    for rel, caption in required_screenshots:
        image_path = ROOT / rel
        if image_path.is_file():
            story += [Image(str(image_path), width=220*mm, height=125*mm), Paragraph(caption, styles["Tiny"]), Spacer(1, 2*mm)]
        else:
            story.append(Paragraph(f"REQUIRED REAL CAPTURE MISSING: {rel}. Capture from the real implementation during the two-laptop rehearsal; mock or fabricated screenshots do not meet Deliverables.pdf.", styles["Callout"]))
            story.append(Spacer(1, 2*mm))
    story += [Paragraph("Before final submission, run the real API/UI on both laptops, verify all ten student_test trips during rehearsal, capture the required screenshots, confirm the screenshots match the submitted source, and have the team review the technical interpretations and AI-use disclosure.", styles["Small"])]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=landscape(A4), leftMargin=12*mm, rightMargin=12*mm,
                            topMargin=12*mm, bottomMargin=14*mm, title="PathSnap Technical Report")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
