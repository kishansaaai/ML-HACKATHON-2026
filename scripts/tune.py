"""Bayesian tuning of one model, tuned-vs-untuned comparison, and model export.

1. Optuna TPE search (seed 42) on trip-grouped CV of the 80 training trips; every trial saved to
   results/tuning_trials_<model>.csv.
2. Refit with the best parameters (same protocol as the untuned run: train stride 3), evaluate
   with the same CV folds and on student_test -> row '<model>_tuned' in results/experiments.csv.
3. Export the variant (tuned or untuned) with the higher CV Top-1 to artifacts/. The choice is
   made on CV only, never on student_test.

Usage: python scripts/tune.py --model hist_gb [--no-export]
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import optuna
import pandas as pd

from pathsnap.data import load_part, stride_subsample, xy
from pathsnap.features import FEATURE_ORDER
from pathsnap.models.registry import MODEL_SPECS, SEED, make_model
from pathsnap.paths import ARTIFACT_DIR, RESULTS_DIR
from pathsnap.tuning import LOGLOSS_WEIGHT, SEARCH_STRIDE, CVProblem, suggest

from train_heavy import run  # noqa: E402  (scripts dir is on sys.path when run directly)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm", choices=["lightgbm", "hist_gb"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=840, help="seconds budget for the search")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    mname = args.model

    problem = CVProblem()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))

    def objective(trial):
        t0 = time.time()
        params = suggest(trial, mname)
        score, top1, ll = problem.evaluate(params, mname)
        trial.set_user_attr("cv_top1", top1)
        trial.set_user_attr("cv_logloss", ll)
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: top1={top1:.4f} logloss={ll:.4f} obj={score:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return score

    t_search = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    search_s = time.time() - t_search
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    RESULTS_DIR.mkdir(exist_ok=True)
    trials.to_csv(RESULTS_DIR / f"tuning_trials_{mname}.csv", index=False)
    best = study.best_params
    summary = dict(model=mname, n_trials=len(study.trials), search_seconds=round(search_s, 1),
                   search_train_stride=SEARCH_STRIDE, objective=f"mean CV top1 - {LOGLOSS_WEIGHT}*logloss",
                   best_objective=study.best_value, best_trial_cv_top1=study.best_trial.user_attrs["cv_top1"],
                   best_params=best)
    (RESULTS_DIR / f"tuning_summary_{mname}.json").write_text(json.dumps(summary, indent=2))
    print("best:", json.dumps(summary, indent=2), flush=True)

    # refit + compare on the SAME protocol as the untuned model (stride 3, same folds)
    tuned = run(mname, 3, best, "_tuned")
    exp = pd.read_csv(RESULTS_DIR / "experiments.csv")
    untuned = exp[exp.model == mname].iloc[-1]
    use_tuned = tuned["cv_top1_mean"] > untuned["cv_top1_mean"]
    params = best if use_tuned else None
    name = f"{mname}_tuned" if use_tuned else mname
    print(f"{mname}: CV top1 tuned {tuned['cv_top1_mean']:.4f} vs untuned {untuned['cv_top1_mean']:.4f} -> {name}", flush=True)
    if args.no_export:
        return

    train_c, _ = load_part("train")
    X, y = xy(stride_subsample(train_c, 3))
    model = make_model(mname, params)
    model.fit(X, y)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "feature_order.json").write_text(json.dumps(FEATURE_ORDER, indent=1))
    row = tuned if use_tuned else untuned.to_dict()
    (ARTIFACT_DIR / "model_card.json").write_text(json.dumps({
        "model": name, "base_model": mname, "family": MODEL_SPECS[mname]["family"], "seed": SEED,
        "params": {**MODEL_SPECS[mname]["params"], **(params or {})}, "train_stride": 3,
        "trained_on": "80 training trips only (data_cache/candidates/train)",
        "selected_by": "highest grouped-CV Top-1 (student_test not used for selection)",
        "cv_top1_mean": float(row["cv_top1_mean"]), "student_test_top1": float(row["test_top1"]),
        "student_test_logloss": float(row["test_logloss"]), "feature_count": len(FEATURE_ORDER)}, indent=2))
    print("exported artifacts to", ARTIFACT_DIR, flush=True)


if __name__ == "__main__":
    main()

