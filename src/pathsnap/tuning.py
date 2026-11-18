"""Bayesian hyper-parameter optimisation of LightGBM (Optuna TPE) with trip-grouped CV.

Objective (maximised): mean over the 5 grouped folds of
    observation Top-1  -  0.05 * candidate log loss
Top-1 is the route-matching metric we select on (does the best-scored candidate equal the
reference arc?). The small log-loss term only breaks near-ties in favour of better-calibrated
probabilities, which the HMM needs. It is ~0.003 in magnitude, so Top-1 dominates.

Search is run on every 6th GPS point of the training folds (speed); validation folds are always
scored on ALL their points. Only the 80 training trips are used: student_test and evaluator data
play no part in the search space, objective or choice of parameters.
"""
from __future__ import annotations

import numpy as np
import optuna

from .cand_metrics import candidate_metrics, observation_metrics
from .data import load_folds, load_part, stride_subsample, xy
from .models.registry import make_model, proba

SEARCH_STRIDE = 6
LOGLOSS_WEIGHT = 0.05


def suggest(trial: optuna.Trial, model: str = "lightgbm") -> dict:
    if model == "hist_gb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 400, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 30.0, log=True),
        }
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


class CVProblem:
    def __init__(self):
        self.cand, self.obs = load_part("train")
        self.folds = []
        for val_trips in load_folds():
            vt = set(val_trips)
            m = self.cand.trip_id.isin(vt)
            tr = stride_subsample(self.cand[~m], SEARCH_STRIDE)
            va = self.cand[m]
            Xtr, ytr = xy(tr)
            Xva, yva = xy(va)
            self.folds.append((Xtr, ytr, Xva, yva, va[["trip_id", "obs_idx", "arc_id", "label"]],
                               self.obs[self.obs.trip_id.isin(vt)]))

    def evaluate(self, params: dict, model_name: str = "lightgbm") -> tuple[float, float, float]:
        tops, lls = [], []
        for Xtr, ytr, Xva, yva, va_meta, va_obs in self.folds:
            model = make_model(model_name, params)
            model.fit(Xtr, ytr)
            p = proba(model, Xva)
            tops.append(observation_metrics(va_meta, p, va_obs)["top1"])
            lls.append(candidate_metrics(yva, p)["logloss"])
        top1, ll = float(np.mean(tops)), float(np.mean(lls))
        return top1 - LOGLOSS_WEIGHT * ll, top1, ll
