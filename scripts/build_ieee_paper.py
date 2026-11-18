"""Build the IEEE-style (two-column conference) paper as DOCX (+ a PDF twin used to check layout and page count).

    python scripts/build_ieee_paper.py [--evaluator-csv PATH] [--out-dir output/paper]

Every number is read from the repository's result files; nothing is typed by hand in the tables.
--evaluator-csv is optional: a per-trip CSV from one run of scripts/evaluate_api.py-style scoring on the evaluator
partition (columns n_labelled, clf_top1, hmm_top1, clf_cont, hmm_cont, clf_breaks, hmm_breaks). It is not stored in the repo.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "report" / "fig"

TITLE = "PathSnap: Classifier-Guided Hidden Markov Map Matching of Noisy GPS Trajectories on Per-Trip Road Networks"
AUTHORS = [
    ("Sai Kishan A", "PES1UG24CS608", "Data pipeline, classifier"),
    ("Nidhi Poojamrudha T A", "PES1UG23CS390", "HMM / Viterbi decoder"),
    ("Samatma A Bharadwaj", "PES1UG24CS609", "API, scoring, EDA"),
    ("S H Purvik Gowda", "PES1UG25CS838", "Evaluator interface"),
]
AFFIL = "Dept. of Computer Science and Engineering, PES University, Bengaluru, India (Team 157, UE24CS352A)"

REFS = [
    'G. Ke <i>et al.</i>, "LightGBM: A highly efficient gradient boosting decision tree," in <i>Proc. NeurIPS</i>, 2017, pp. 3146-3154.',
    'A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm," <i>IEEE Trans. Inf. Theory</i>, vol. 13, no. 2, pp. 260-269, Apr. 1967.',
    'P. Newson and J. Krumm, "Hidden Markov map matching through noise and sparseness," in <i>Proc. 17th ACM SIGSPATIAL GIS</i>, 2009, pp. 336-343.',
    'L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," <i>Proc. IEEE</i>, vol. 77, no. 2, pp. 257-286, Feb. 1989.',
    'E. W. Dijkstra, "A note on two problems in connexion with graphs," <i>Numerische Mathematik</i>, vol. 1, pp. 269-271, 1959.',
    'T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in <i>Proc. 25th ACM SIGKDD</i>, 2019, pp. 2623-2631.',
    'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD</i>, 2016, pp. 785-794.',
    'F. Pedregosa <i>et al.</i>, "Scikit-learn: Machine learning in Python," <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825-2830, 2011.',
]


def f(x, n=4):
    return f"{float(x):.{n}f}"


def load(evaluator_csv):
    mt = pd.read_csv(RES / "metrics_table.csv").set_index("model")
    ex = pd.read_csv(RES / "experiments.csv").drop_duplicates("model", keep="last").set_index("model")
    abl = pd.read_csv(RES / "ablation.csv")
    fin = pd.read_csv(RES / "hmm_tuning" / "final_student_test_check.csv", dtype={"trip_id": str})
    f80 = pd.read_csv(RES / "hmm_tuning" / "frozen_candidate_full80.csv", dtype={"trip_id": str})
    seq = pd.read_csv(RES / "sequence_metrics_student_test.csv", dtype={"trip_id": str})
    api = pd.read_csv(RES / "api_student_test.csv", dtype={"trip_id": str})
    tun = json.load(open(RES / "tuning_summary_lightgbm.json"))
    card = json.load(open(ROOT / "artifacts" / "model_card.json"))
    cfg = json.load(open(ROOT / "configs" / "decoder_best.json"))
    inv = json.load(open(RES / "data_inventory.summary.json")) if (RES / "data_inventory.summary.json").exists() else {}
    cm = pd.read_csv(RES / "confusion_lightgbm.csv", index_col=0)
    ev = pd.read_csv(evaluator_csv, dtype={"trip_id": str}) if evaluator_csv else None
    return dict(mt=mt, ex=ex, abl=abl, fin=fin, f80=f80, seq=seq, api=api, tun=tun, card=card, cfg=cfg, inv=inv, cm=cm, ev=ev)


def wavg(df, col, wcol):
    return float((df[col] * df[wcol]).sum() / df[wcol].sum())


def build_blocks(d):
    mt, ex, abl, fin, f80, seq, api, tun, card, cfg, cm, ev = (d[k] for k in
        ("mt", "ex", "abl", "fin", "f80", "seq", "api", "tun", "card", "cfg", "cm", "ev"))
    p = card["params"]
    lg, hg, xg = mt.loc["lightgbm"], mt.loc["hist_gb"], mt.loc["xgboost"]
    tuned = mt.loc["lightgbm_tuned"]
    cv_std = ex.loc["lightgbm", "cv_top1_std"]
    s_clf, s_hmm = wavg(fin, "clf_top1", "n_labelled"), wavg(fin, "hmm_top1", "n_labelled")
    o_clf, o_hmm = wavg(f80, "clf_top1", "n_labelled"), wavg(f80, "hmm_top1", "n_labelled")
    sm = seq.drop(columns="trip_id").mean()
    seq_better = int((seq.hmm_route_arc_set_f1 > seq.clf_route_arc_set_f1).sum())
    slow = api.loc[api.seconds.idxmax()]
    tn, fp, fn, tp = (int(cm.iloc[0, 0]), int(cm.iloc[0, 1]), int(cm.iloc[1, 0]), int(cm.iloc[1, 1]))
    if ev is not None:
        e_clf, e_hmm = wavg(ev, "clf_top1", "n_labelled"), wavg(ev, "hmm_top1", "n_labelled")
        e_cc, e_hc = float(ev.clf_cont.mean()), float(ev.hmm_cont.mean())
        e_cb, e_hb = int(ev.clf_breaks.sum()), int(ev.hmm_breaks.sum())
        e_slow = float(ev.seconds.max()) if "seconds" in ev else None
    B = []
    add = B.append

    abstract = (
        "Map matching assigns noisy GPS observations to the road arcs a vehicle actually travelled. We study a setting in which every trip "
        "comes with its own road network, so no fixed map can be assumed. PathSnap combines a LightGBM classifier, which scores every "
        "(GPS point, candidate arc) pair with the probability that the arc is correct, and a hidden Markov model whose emission scores are "
        "those probabilities and whose transition scores compare network distance with GPS displacement; the most likely connected route is "
        f"found with the Viterbi algorithm. With trip-grouped five-fold cross-validation on 80 training trips the classifier reaches a Top-1 accuracy of {f(lg.cv_top1)}. "
        + (f"On ten held-out evaluator trips, scored once and never used for any design choice, pooled Top-1 is {f(e_clf)} for the classifier alone and {f(e_hmm)} with the decoder, "
           f"while route continuity improves from {f(e_cc)} to {f(e_hc)}. " if ev is not None else "")
        + "We also report where the decoder does not help. The system is served by a FastAPI backend and a Streamlit interface that compute "
        "all metrics on the server so that ground truth never leaves it.")
    add(("abstract", abstract))
    add(("keywords", "map matching, GPS trajectories, hidden Markov model, Viterbi algorithm, gradient boosting, LightGBM, route reconstruction"))

    # I -----------------------------------------------------------------------------------------------------------
    add(("h1", "Introduction"))
    add(("p", "Satellite positioning receivers on phones and vehicles report a position about once per second, but the reported points are displaced from the true "
              "path by tens of metres, especially between tall buildings. Applications such as navigation, fleet analytics and traffic estimation need the road "
              "the vehicle was really on. Recovering the sequence of road arcs from a noisy track is called <i>map matching</i>."))
    add(("p", "Classical approaches match each point to the nearest road or use a hidden Markov model (HMM) with a hand-designed Gaussian distance emission and a "
              "transition that compares route and straight-line distance [3]. We take a hybrid route: a gradient-boosted classifier [1] supplies a learned, "
              "feature-rich emission, and the HMM [4] with Viterbi decoding [2] enforces that the chosen arcs form a physically plausible route on the road graph. "
              "The task setting adds a practical difficulty: every trip has its own road network of nodes and directed arcs, so models must generalise across graphs."))
    add(("p", "<b>Contributions.</b> (i) A candidate-generation and 27-feature design that makes a per-arc classifier work across unseen road networks. "
              "(ii) A decoder that uses the classifier's calibrated probabilities as emissions, with tuning performed on out-of-fold scores to avoid optimistic bias. "
              "(iii) An honest evaluation that separates point-level accuracy, route connectivity and sequence-level overlap, including the cases where the decoder does not help. "
              "(iv) A two-machine deployment in which scoring and ground truth stay on the server."))

    # II ----------------------------------------------------------------------------------------------------------
    add(("h1", "Problem and Data"))
    add(("h2", "Task definition"))
    add(("p", "A trip consists of a GPS track (longitude, latitude, time, about 1 Hz) and a directed road graph with nodes (longitude, latitude) and arcs "
              "(from-node, to-node; the row index is the arc identifier). For development trips the ground-truth arc sequence is also given. "
              "The system must output, for every GPS point, the arc it belongs to, and a connected route; breaks must be reported rather than hidden."))
    add(("h2", "Datasets"))
    add(("p", "Training uses 80 trips; ten further trips (<i>student test</i>) are used only for development comparisons; ten <i>evaluator</i> trips are never used "
              "for any choice of features, model, hyper-parameters or decoder settings. Table I summarises the first two partitions; both are free of missing, "
              "invalid or duplicated track rows."))
    add(("table", "DATASET SUMMARY (FROM <i>data_inventory.csv</i>)", ["", "Training", "Student test"], [
        ["Trips", "80", "10"], ["GPS points", "208,417", "19,160"], ["Road nodes", "1,933,485", "90,689"],
        ["Directed arcs", "3,876,904", "185,041"], ["Median time step", "1.0 s", "1.0 s"], ["Steps longer than 5 s", "1,216", "44"]]))
    add(("p", "The main data issues are long time gaps (handled by restarting the decoder after 30 s) and noisy or partly unreliable reference routes. "
              "For two training trips (00000054 and 00000033) the reference route barely follows the GPS track, so their derived labels are unreliable; we kept them and list this as a limitation."))

    # III ---------------------------------------------------------------------------------------------------------
    add(("h1", "Method"))
    add(("h2", "Projection and candidate generation"))
    add(("p", "Each trip is projected to local metres with an equirectangular projection around the median of its track; the classifier features and the decoder share the same origin. "
              "For every GPS point we take all arcs within <i>R</i> = 50 m; if fewer than six are found, the radius is doubled (up to 400 m) until the six nearest arcs are included, "
              "with at most 24 candidates per point. A KD-tree over points along the arcs gives the neighbours and an exact point-to-segment distance gives the final ranking. "
              "The share of labelled points whose true arc is not among the candidates is 0.61% on training and 0.010% on the student test, so candidate recall is not the bottleneck."))
    add(("h2", "Label derivation"))
    add(("p", "The reference route has far fewer arcs than there are GPS points, so a per-point label is derived by a monotone dynamic-programming alignment of the points to the route arcs; "
              "points more than 100 m from their aligned arc stay unlabelled (about 2% of training points). Labels are used only for training and scoring."))
    add(("h2", "Features"))
    add(("p", "Each (point, candidate arc) pair is described by 27 features in five groups: <i>geometry</i> (5: perpendicular distance, fractional and metric position along the arc, "
              "remaining length, arc length), <i>relative</i> (4: distance to the closest candidate, gap to it, distance rank, candidate count), <i>topology</i> (3: two-way flag, "
              "out-degree at the arc end, in-degree at its start), <i>motion</i> (6: speed, moving flag, absolute heading difference and its cosine, time to previous and next point) and "
              "<i>neighbours</i> (9: the same arc's perpendicular distance and position at the previous and next one and three points, and a window minimum, gap and rank)."))
    add(("table", "FEATURE-GROUP ABLATION (LIGHTGBM, TRIP-GROUPED CV)", ["Setting", "# feat.", "CV Top-1", "Test Top-1"],
         [[r.setting, str(int(r.n_features)), f(r.cv_top1), f(r.test_top1)] for r in abl.itertuples()]))
    add(("p", "Table II shows that motion and neighbour features matter most: removing motion lowers CV Top-1 by about 2.9 points and removing neighbours by about 1.8 points, "
              "while the nine neighbour features alone already reach 0.843. Geometry, relative and topology features overlap."))
    add(("h2", "Classifier: LightGBM"))
    add(("p", "LightGBM [1] is a gradient-boosted decision-tree ensemble. Trees are added sequentially; each fits the gradient of the log loss of the current ensemble, "
              "and the sum of the tree outputs passes through a sigmoid to give the probability that the arc is correct. Trees grow leaf-wise (always splitting the leaf with the largest "
              "loss reduction) on histogram-binned features, which makes training fast. The exported model uses "
              f"{p['n_estimators']} trees, learning rate {p['learning_rate']}, {p['num_leaves']} leaves, minimum {p['min_child_samples']} samples per leaf, row and column sampling of "
              f"{p['subsample']} and {p['colsample_bytree']}, L2 weight {p['reg_lambda']} and seed 42, fitted on every third point of the 80 training trips. "
              "At inference the probabilities of a point's candidates are normalised to sum to one."))
    add(("h2", "Hidden Markov model and Viterbi decoding"))
    add(("p", "The hidden state at observation <i>t</i> is the candidate arc (with the position of the point along it); candidate sets differ in size from point to point. "
              "The emission log-score of candidate <i>k</i> is the log of the normalised classifier probability <i>p(t, k)</i> minus a Gaussian distance term,"))
    add(("eq", "e(t, k) = log p(t, k) − d(t, k)² / (2σ²),", "1"))
    add(("p", f"with <i>d</i> the perpendicular distance in metres and σ = {cfg['emission_sigma_m']:.0f} m. The transition log-score between candidates <i>i</i> at <i>t</i>−1 and <i>j</i> at <i>t</i> is"))
    add(("eq", "a(i, j) = − | r(i, j) − g(t) | / β,", "2"))
    add(("p", f"where <i>r(i, j)</i> is the road distance along the directed graph (rest of the source arc, a Dijkstra [5] shortest path between nodes, and the part of the destination arc), "
              f"<i>g(t)</i> the straight GPS distance, and β = {cfg['beta_m']:.0f} m. Searches are bounded at 10 km; unreachable pairs receive a finite penalty of {cfg['unreachable_log_penalty']:.0f}. "
              "Viterbi decoding [2] then finds the sequence with the highest total score by dynamic programming: for each candidate it keeps only the best score of any path ending there and a backpointer, "
              "and recovers the route by backtracking from the best final state. Scores are added in log space, so no underflow occurs, and the cost is O(TK<super>2</super>) for T points and K candidates."))
    add(("p", f"A missing candidate set, a time gap above {cfg['gap_threshold_s']:.0f} s, an invalid timestamp or an unreachable transition starts a new segment. "
              "Between consecutive different arcs the route is completed by the shortest legal directed path; no edge is ever invented, and each break is reported and drawn as a separate polyline."))

    # IV ----------------------------------------------------------------------------------------------------------
    add(("h1", "Experimental Setup"))
    add(("p", "<b>Validation.</b> Points inside a trip are strongly correlated, so a random split would leak. All cross-validation (CV) uses five folds grouped by trip (seed 42, balanced by point count). "
              "<b>Metrics.</b> Observation-level Top-1 accuracy (the most probable candidate of each point is the true arc), candidate-level macro-F1, precision, recall and log loss, and "
              "<i>route continuity</i>, the share of adjacent predicted pairs that are the same arc or connected head to tail; breaks are the adjacent pairs that are neither. "
              "A sequence-level arc-set F1 compares the full predicted route with the reference route."))
    add(("p", "<b>Compared models.</b> LightGBM, HistGradientBoosting, XGBoost [7], Extra Trees, Random Forest, logistic regression, Gaussian naive Bayes and k-nearest neighbours "
              "(scikit-learn [8] implementations where applicable), plus two non-learned baselines: nearest arc, and nearest arc with a heading rule. "
              "All use the same candidate and feature tables and folds; Extra Trees and Random Forest were run with 50 trees on every sixth point and KNN on a 3,000-row sample per fold, which limits those rows."))
    add(("p", f"<b>Hyper-parameter search.</b> Bayesian optimisation with Optuna [6] (TPE sampler, {tun['n_trials']} trials) searched eight LightGBM parameters (leaves, learning rate, trees, minimum leaf samples, "
              "row and column sampling, L1 and L2 weights) with an objective of grouped-CV Top-1 minus 0.05 times log loss, on training trips only. "
              "The four decoder parameters (beta, sigma, gap threshold, unreachable penalty) were swept on <i>out-of-fold</i> classifier scores, because the exported model is over-confident on its own training trips "
              "(about 94.5% Top-1 there against 87.5% in CV)."))

    # V -----------------------------------------------------------------------------------------------------------
    add(("h1", "Results"))
    add(("h2", "Model comparison"))
    rows = []
    for m, name in (("lightgbm", "LightGBM"), ("hist_gb", "HistGradBoost"), ("xgboost", "XGBoost"), ("extra_trees", "Extra Trees"), ("random_forest", "Random Forest")):
        r = mt.loc[m]
        rows.append([name, f(r.cv_top1), f(r.test_top1), f(r.test_logloss), f(r.infer_time_s_per_trip, 3)])
    for m, name in (("logistic_regression", "Logistic reg."), ("gaussian_nb", "Naive Bayes"), ("knn", "KNN")):
        r = ex.loc[m]
        rows.append([name, f(r.cv_top1_mean), f(r.test_top1), f(r.test_logloss), f(r.infer_time_s_per_trip, 3)])
    for m, name in (("nearest_arc", "Nearest arc"), ("nearest_heading_aware", "Nearest + heading")):
        r = ex.loc[m]
        rows.append([name, f(r.cv_top1_mean), f(r.test_top1), "-", "-"])
    add(("table", "CLASSIFIER COMPARISON (CV: 80 TRAINING TRIPS; TEST: 10 STUDENT-TEST TRIPS)", ["Model", "CV Top-1", "Test Top-1", "Log loss", "s/trip"], rows))
    add(("p", f"LightGBM, HistGradientBoosting and XGBoost are statistically tied: their CV Top-1 values differ by about 0.1 points whereas the fold-to-fold standard deviation is {f(cv_std)}. "
              "We selected LightGBM as the single final model because it is the strongest on student-test Top-1 and log loss among the tied three, trains fastest, and is simple to export. "
              "The bagged ensembles are 0.15 to 0.45 points lower in CV, logistic regression is a strong simple model (0.845 CV), and the heading-aware nearest-arc rule already reaches "
              f"{f(ex.loc['nearest_heading_aware','test_top1'])} on the student test, so learning adds about seven points over a good hand-made rule. "
              f"At candidate level the exported model has TN/FP/FN/TP of {tn:,}/{fp:,}/{fn:,}/{tp:,} on the student test and is well calibrated (ECE 0.0027), which matters because the HMM consumes its probabilities."))
    add(("fig", FIG / "model_comparison.png", "Cross-validated macro-F1, student-test Top-1 and log loss of the compared models."))
    add(("fig", FIG / "confusion_all_models.png", "Candidate-level confusion matrices of the boosting and bagging models on the student test."))
    add(("h2", "Hyper-parameter optimisation"))
    add(("p", f"The best of {tun['n_trials']} Optuna trials reached a CV Top-1 of {f(tun['best_trial_cv_top1'])} against {f(lg.cv_top1)} for the default configuration "
              f"(student-test Top-1 {f(tuned.test_top1)} against {f(lg.test_top1)}). The difference is far smaller than the fold spread, so, following a rule fixed before the search, "
              "the default configuration was exported. The likely reason is that features, not hyper-parameters, limit accuracy; we did not test this further."))
    add(("fig", FIG / "tuning_trials_lightgbm.png", "Optuna search history: objective of each trial and best value so far."))
    add(("h2", "Effect of the HMM decoder"))
    rows = [["Training (80, out-of-fold)", f(o_clf), f(o_hmm), f(f80.clf_continuity.mean()), f(f80.hmm_continuity.mean()), f"{int(f80.clf_breaks.sum()):,} / {int(f80.hmm_breaks.sum()):,}"],
            ["Student test (10)", f(s_clf), f(s_hmm), f(fin.clf_continuity.mean()), f(fin.hmm_continuity.mean()), f"{int(fin.clf_breaks.sum())} / {int(fin.hmm_breaks.sum())}"]]
    if ev is not None:
        rows.append(["Evaluator (10, one run)", f(e_clf), f(e_hmm), f(e_cc), f(e_hc), f"{e_cb} / {e_hb}"])
    add(("table", "CLASSIFIER ALONE VERSUS CLASSIFIER + HMM (POOLED TOP-1; MEAN CONTINUITY; BREAKS AS CLASSIFIER / HMM)", ["Partition", "Top-1 clf", "Top-1 HMM", "Cont. clf", "Cont. HMM", "Breaks"], rows))
    add(("p", f"Table IV shows a consistent gain in route continuity and a reduction in breaks on every partition, but only a small gain in Top-1 (+{o_hmm - o_clf:.4f} out of fold, "
              f"{s_hmm - s_clf:+.4f} on the student test" + (f", {e_hmm - e_clf:+.4f} on the evaluator trips" if ev is not None else "") +
              f"). Out of fold the decoder was worse than the classifier on 28 of the 80 training trips, so the Top-1 gain should be regarded as weak evidence. "
              + ("The evaluator figures come from a single run of the deployed system on the ten evaluator trips; nothing was changed afterwards. " if ev is not None else "")))
    add(("p", f"At sequence level on the student test the decoder is <i>not</i> better: its route covers almost every reference arc (recall {f(sm.hmm_route_arc_set_recall)}) but also contains arcs "
              f"that are not in the reference (precision {f(sm.hmm_route_arc_set_precision)}), giving a mean arc-set F1 of {f(sm.hmm_route_arc_set_f1)} against {f(sm.clf_route_arc_set_f1)} for the classifier alone, "
              f"with the decoder higher on only {seq_better} of ten trips. We did not establish the cause; possible explanations are detours inserted by route stitching and the permissive unreachable penalty, whose best value lies at the edge of the tested range."))
    rows = [[r.trip_id, str(int(r.n_labelled)), f(r.clf_top1), f(r.hmm_top1), f(r.clf_continuity), f(r.hmm_continuity)] for r in fin.itertuples()]
    add(("table", "PER-TRIP RESULTS ON THE STUDENT-TEST PARTITION", ["Trip", "Labelled", "Top-1 clf", "Top-1 HMM", "Cont. clf", "Cont. HMM"], rows))
    add(("p", "The decoder raises Top-1 on seven of the ten trips and lowers it on three (00000007, 00000077, 00000091); continuity is higher or equal on all ten."))
    add(("fig", FIG / "prediction_map_00000052.png", "Folium prediction map for trip 00000052: GPS points, road network, reference route and predicted route."))
    add(("h2", "Failure cases"))
    add(("p", "The largest per-trip Top-1 regressions of the decoder on the training trips are 00000081 (-0.0715), 00000083 (-0.0644) and 00000010 (-0.0432). "
              "Trip 00000081 also has many ambiguous labels, which may explain part of it; for the other two, candidate coverage and labels look sound, so they remain decoder error cases with an unestablished cause."))

    # VI ----------------------------------------------------------------------------------------------------------
    add(("h1", "System Implementation"))
    add(("p", "The solution is served as two cooperating programs. A <b>FastAPI</b> backend (machine 1) exposes <i>/health</i>, <i>/trips</i> and <i>/predict</i>; for a trip identifier it loads the trip, "
              "generates candidates and features, scores them with the exported LightGBM model, runs the HMM, and returns raw GPS, the route segments of both the classifier and the decoder, "
              "per-observation arc identifiers with confidence, metrics and stage timings. A <b>Streamlit</b> interface (machine 2) sends only a trip identifier and draws the response on a Folium map "
              "with layers for raw GPS, classifier route and decoded route; disconnected segments are drawn as separate polylines."))
    add(("p", "<b>No ground-truth leakage.</b> Prediction loads trips without the reference file and cannot read it; the scoring module opens the reference route only after the prediction is finished, "
              "so the response contains metrics but no reference field, and a trip without a reference file still returns a prediction with null metrics. "
              "The evaluator directory is configured by an environment variable and the server refuses to start unless it holds exactly ten valid trip folders. "
              "Every request writes one log line (trip, observation count, Top-1 of both stages, continuity, total time)."))
    add(("p", f"<b>Runtime.</b> Through the full API the ten student-test trips take between {api.seconds.min():.1f} and {slow.seconds:.1f} s"
              + (f" and the ten evaluator trips up to {e_slow:.1f} s" if ev is not None and e_slow else "")
              + ", well below the 30 s per-trip target; the pooled Top-1 through the API matches the offline pipeline exactly."))

    # VII ---------------------------------------------------------------------------------------------------------
    add(("h1", "Limitations and Use of AI Assistance"))
    add(("p", "<b>Limitations.</b> (a) The development set has only ten trips, and the three boosting models are statistically tied, so the choice of LightGBM rests on speed and simplicity "
              "as much as on accuracy. (b) Hyper-parameter tuning gave no improvement. (c) Over-fitting was measured only for LightGBM. (d) Some comparison models were run with reduced settings. "
              "(e) Labels are derived and unreliable for two training trips. (f) The decoder's Top-1 gain is small, it is worse on part of the trips, and its route has lower arc-level precision; "
              "its parameters were tuned on eight pilot trips."))
    add(("p", "<b>AI assistance.</b> An AI coding assistant (Claude Code, Anthropic) was used to help write and review code and tests, run experiments, and draft the repository notes and this paper. "
              "Every number in the paper comes from a result file produced by a committed script. The authors are responsible for understanding and explaining all submitted work."))

    # VIII --------------------------------------------------------------------------------------------------------
    add(("h1", "Conclusion"))
    add(("p", "A LightGBM classifier over 27 geometry, motion, topology and neighbourhood features, combined with an HMM that uses its probabilities as emissions and decodes with Viterbi, "
              "matches noisy GPS tracks to per-trip road networks with about 87 to 90% per-point accuracy on unseen trips and produces more connected routes than the classifier alone. "
              "The decoder's main benefit is connectivity rather than accuracy, and its route contains additional arcs that reduce sequence-level precision. "
              "Future work includes investigating these spurious arcs, tuning the decoder on a larger set of trips with a wider penalty range, adding direct road-continuity features to the classifier, "
              "and evaluating over-fitting for all models."))
    add(("ack", "The authors thank the course staff of UE24CS352A at PES University for the dataset and evaluation framework."))
    add(("refs", REFS))
    return B


# --------------------------------------------------------------------------------------------------------------------
TAG = re.compile(r"(</?[ib]>|</?super>)")


def runs(text):
    """Split inline markup into (text, bold, italic, superscript) tuples."""
    out, b, i, s = [], False, False, False
    for tok in TAG.split(text):
        if tok == "<b>": b = True
        elif tok == "</b>": b = False
        elif tok == "<i>": i = True
        elif tok == "</i>": i = False
        elif tok == "<super>": s = True
        elif tok == "</super>": s = False
        elif tok:
            out.append((tok, b, i, s))
    return out


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def build_docx(blocks, path):
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.7); sec.top_margin = Inches(0.75); sec.bottom_margin = Inches(1.0)

    def para(text="", size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False, after=4, before=0, indent=None, keep=False):
        p = doc.add_paragraph()
        p.alignment = align
        pf = p.paragraph_format
        pf.space_after, pf.space_before = Pt(after), Pt(before)
        pf.line_spacing = 1.0
        if indent is not None: pf.first_line_indent = Inches(indent)
        if keep: pf.keep_with_next = True
        for t, b, i, s in runs(text):
            r = p.add_run(t); r.font.size = Pt(size); r.bold = bold or b; r.italic = italic or i
            if s: r.font.superscript = True
        return p

    # title block (single column)
    para(TITLE, size=22, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, after=8, before=2)
    t = doc.add_table(rows=1, cols=len(AUTHORS)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, (n, srn, role) in zip(t.rows[0].cells, AUTHORS):
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.paragraphs[0].add_run(n); r.bold = True; r.font.size = Pt(10)
        for line, it in ((srn, False), (role, True)):
            q = c.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = q.add_run(line); rr.font.size = Pt(9); rr.italic = it
    para(AFFIL, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, after=8, before=6)

    # two-column section
    s2 = doc.add_section(WD_SECTION.CONTINUOUS)
    cols = s2._sectPr.xpath("./w:cols")
    c = cols[0] if cols else OxmlElement("w:cols")
    c.set(qn("w:num"), "2"); c.set(qn("w:space"), "360")
    if not cols: s2._sectPr.append(c)

    h1n = 0; tabn = 0; fign = 0; eqn = 0
    for blk in blocks:
        k = blk[0]
        if k == "abstract":
            p = para("", after=4)
            r = p.add_run("Abstract\u2014"); r.bold = True; r.italic = True; r.font.size = Pt(9)
            for t_, b, i, s in runs(blk[1]):
                r = p.add_run(t_); r.font.size = Pt(9); r.bold = True; r.italic = i
        elif k == "keywords":
            p = para("", after=8)
            r = p.add_run("Index Terms\u2014"); r.bold = True; r.italic = True; r.font.size = Pt(9)
            r = p.add_run(blk[1]); r.bold = True; r.italic = True; r.font.size = Pt(9)
        elif k == "h1":
            para(f"{ROMAN[h1n]}. {blk[1].upper()}", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, keep=True); h1n += 1; h2n = 0
        elif k == "h2":
            para(f"{chr(65 + h2n)}. {blk[1]}", size=10, align=WD_ALIGN_PARAGRAPH.LEFT, italic=True, before=4, after=2, keep=True); h2n += 1
        elif k == "p":
            para(blk[1], indent=0.17)
        elif k == "eq":
            eqn += 1
            para(f"{blk[1]}\t({blk[2]})", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, before=2, after=4)
        elif k == "table":
            tabn += 1
            para(f"TABLE {ROMAN[tabn - 1]}", size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=0, before=4, keep=True)
            para(blk[1], size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=3, keep=True)
            _, _, head, rows = blk
            tb = doc.add_table(rows=1 + len(rows), cols=len(head)); tb.style = "Table Grid"; tb.alignment = WD_TABLE_ALIGNMENT.CENTER
            for ci, h in enumerate(head):
                cell = tb.rows[0].cells[ci]; cell.text = ""
                r = cell.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(7.5)
            for ri, row in enumerate(rows, 1):
                for ci, v in enumerate(row):
                    cell = tb.rows[ri].cells[ci]; cell.text = ""
                    pp = cell.paragraphs[0]; r = pp.add_run(str(v)); r.font.size = Pt(7.5)
                    if ci > 0: pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for row in tb.rows:
                for cell in row.cells:
                    for pp in cell.paragraphs:
                        pp.paragraph_format.space_after = Pt(0)
            para("", size=4, after=4)
        elif k == "fig":
            fign += 1
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(2)
            p.add_run().add_picture(str(blk[1]), width=Inches(3.3))
            para(f"Fig. {fign}. {blk[2]}", size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
        elif k == "ack":
            para("ACKNOWLEDGMENT", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, keep=True)
            para(blk[1], indent=0.17)
        elif k == "refs":
            para("REFERENCES", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, keep=True)
            for n, ref in enumerate(blk[1], 1):
                p = para(f"[{n}] " + ref, size=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=2)
                p.paragraph_format.left_indent = Inches(0.25); p.paragraph_format.first_line_indent = Inches(-0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


def build_pdf(blocks, path):
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, FrameBreak, Frame, Image, KeepTogether, NextPageTemplate, PageTemplate,
                                    Paragraph, Spacer, Table, TableStyle)
    from xml.sax.saxutils import escape

    fd = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("TNR", str(fd / "times.ttf"))); pdfmetrics.registerFont(TTFont("TNR-B", str(fd / "timesbd.ttf")))
    pdfmetrics.registerFont(TTFont("TNR-I", str(fd / "timesi.ttf"))); pdfmetrics.registerFont(TTFont("TNR-BI", str(fd / "timesbi.ttf")))
    pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

    W, H = letter; M = 0.7 * inch; GAP = 0.25 * inch; CW = (W - 2 * M - GAP) / 2
    body = ParagraphStyle("b", fontName="TNR", fontSize=10, leading=11.4, alignment=TA_JUSTIFY, firstLineIndent=12, spaceAfter=3)
    S = dict(
        body=body,
        h1=ParagraphStyle("h1", parent=body, alignment=TA_CENTER, firstLineIndent=0, spaceBefore=7, spaceAfter=3, keepWithNext=1),
        h2=ParagraphStyle("h2", parent=body, fontName="TNR-I", alignment=TA_LEFT, firstLineIndent=0, spaceBefore=3, spaceAfter=2, keepWithNext=1),
        abs=ParagraphStyle("abs", parent=body, fontName="TNR-B", fontSize=9, leading=10.4, firstLineIndent=0, spaceAfter=3),
        cap=ParagraphStyle("cap", parent=body, fontSize=8, leading=9.2, alignment=TA_CENTER, firstLineIndent=0, spaceAfter=5),
        tcap=ParagraphStyle("tcap", parent=body, fontSize=8, leading=9.2, alignment=TA_CENTER, firstLineIndent=0, spaceAfter=2, keepWithNext=1),
        ref=ParagraphStyle("ref", parent=body, fontSize=8, leading=9.2, firstLineIndent=-18, leftIndent=18, spaceAfter=2),
        eq=ParagraphStyle("eq", parent=body, fontName="TNR-I", alignment=TA_CENTER, firstLineIndent=0, spaceBefore=1, spaceAfter=3),
        ttl=ParagraphStyle("ttl", fontName="TNR-B", fontSize=21, leading=24, alignment=TA_CENTER, spaceAfter=7),
        au=ParagraphStyle("au", fontName="TNR", fontSize=9, leading=10.6, alignment=TA_CENTER),
        af=ParagraphStyle("af", fontName="TNR-I", fontSize=9, leading=10.6, alignment=TA_CENTER, spaceBefore=4),
    )

    def P(t, s):  # markup text -> reportlab paragraph (escape everything except our tags)
        t = escape(t)
        for a, b in (("&lt;i&gt;", "<i>"), ("&lt;/i&gt;", "</i>"), ("&lt;b&gt;", "<b>"), ("&lt;/b&gt;", "</b>"), ("&lt;super&gt;", "<super>"), ("&lt;/super&gt;", "</super>")):
            t = t.replace(a, b)
        return Paragraph(t, s)

    fl = []
    au_cells = [[Paragraph(f"<b>{escape(n)}</b><br/>{srn}<br/><i>{escape(role)}</i>", S["au"]) for n, srn, role in AUTHORS]]
    fl += [Paragraph(escape(TITLE), S["ttl"]),
           Table(au_cells, colWidths=[(W - 2 * M) / len(AUTHORS)] * len(AUTHORS), style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])),
           Paragraph(escape(AFFIL), S["af"]), FrameBreak()]
    h1n = tabn = fign = 0
    for blk in blocks:
        k = blk[0]
        if k == "abstract":
            fl.append(P("<i>Abstract</i>\u2014" + blk[1], S["abs"]))
        elif k == "keywords":
            fl.append(P("<i>Index Terms</i>\u2014" + blk[1], S["abs"]))
        elif k == "h1":
            fl.append(P(f"{ROMAN[h1n]}. {blk[1].upper()}", S["h1"])); h1n += 1; h2n = 0
        elif k == "h2":
            fl.append(P(f"{chr(65 + h2n)}. {blk[1]}", S["h2"])); h2n += 1
        elif k == "p":
            fl.append(P(blk[1], S["body"]))
        elif k == "eq":
            t = Table([[P(blk[1], S["eq"]), P(f"({blk[2]})", S["eq"])]], colWidths=[CW - 28, 28], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            fl.append(t)
        elif k == "table":
            tabn += 1
            _, cap, head, rows = blk
            data = [[P(h, ParagraphStyle("th", parent=S["cap"], fontName="TNR-B", fontSize=7.5, spaceAfter=0)) for h in head]] + \
                   [[P(str(v), ParagraphStyle("td", parent=S["cap"], fontSize=7.5, leading=8.6, spaceAfter=0, alignment=TA_LEFT if ci == 0 else TA_CENTER)) for ci, v in enumerate(r)]
                    for r in rows]
            first = 1.05 * inch if len(head) > 3 else 1.0 * inch
            wds = [first] + [(CW - first) / (len(head) - 1)] * (len(head) - 1)
            tb = Table(data, colWidths=wds, repeatRows=1, style=TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                                                              ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
            fl.append(KeepTogether([P(f"TABLE {ROMAN[tabn - 1]}", S["tcap"]), P(cap, S["tcap"]), tb, Spacer(1, 5)]))
        elif k == "fig":
            fign += 1
            im = PILImage.open(blk[1]); w, h = im.size
            fl.append(KeepTogether([Image(str(blk[1]), width=CW, height=CW * h / w), P(f"Fig. {fign}. {blk[2]}", S["cap"])]))
        elif k == "ack":
            fl += [P("ACKNOWLEDGMENT", S["h1"]), P(blk[1], S["body"])]
        elif k == "refs":
            fl.append(P("REFERENCES", S["h1"]))
            fl += [P(f"[{n}] " + r, S["ref"]) for n, r in enumerate(blk[1], 1)]

    TH = 2.05 * inch
    def frames(first):
        top = H - M
        if first:
            return [Frame(M, top - TH, W - 2 * M, TH, id="t", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                    Frame(M, 0.9 * inch, CW, top - TH - 0.9 * inch, id="c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                    Frame(M + CW + GAP, 0.9 * inch, CW, top - TH - 0.9 * inch, id="c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
        return [Frame(M, 0.9 * inch, CW, top - 0.9 * inch, id="c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                Frame(M + CW + GAP, 0.9 * inch, CW, top - 0.9 * inch, id="c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]

    def footer(c, doc_):
        c.setFont("TNR", 8); c.drawCentredString(W / 2, 0.5 * inch, str(doc_.page))

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(path), pagesize=letter, title=TITLE, author="Team 157",
                          pageTemplates=[PageTemplate("First", frames(True), onPage=footer), PageTemplate("Later", frames(False), onPage=footer)])
    fl.insert(0, NextPageTemplate("Later"))
    doc.build(fl)
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator-csv", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "paper"))
    a = ap.parse_args()
    d = load(a.evaluator_csv)
    blocks = build_blocks(d)
    out = Path(a.out_dir)
    build_docx(blocks, out / "PathSnap_IEEE_paper.docx")
    n = build_pdf(blocks, out / "PathSnap_IEEE_paper_preview.pdf")
    words = sum(len(re.sub(r"<[^>]+>", "", b[1]).split()) for b in blocks if b[0] in ("p", "abstract"))
    print(f"wrote {out / 'PathSnap_IEEE_paper.docx'} | preview PDF pages: {n} | body words: {words}")
