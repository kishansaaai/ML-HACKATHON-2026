"""Candidate-level and observation-level metrics used in model experiments.

Person B's `metrics.py` is the shared, authoritative module (HMM-level, route continuity,
sequence scores). These helpers cover what model selection needs and use the same definitions:

* Observation Top-1: per GPS point with a defined reference arc, the highest-scored candidate
  is the reference arc. Points whose reference arc is not among the candidates count as wrong.
* Candidate miss rate: share of reference-defined points whose reference arc is absent.
* Candidate-level precision / recall / macro-F1 at threshold 0.5, and log loss.
* Route continuity (official, identical to `pathsnap.decoder.metrics.route_continuity`): over ALL
  adjacent predicted pairs, share that are the same arc or connect head-to-tail.
  `distinct_arc_continuity` is the stricter variant used in the first model comparison (repeats
  collapsed first); it is NOT the official metric and must be labelled as such wherever quoted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def top1_predictions(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, obs): columns trip_id, obs_idx, arc_id, label."""
    tmp = df[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    tmp["score"] = score
    idx = tmp.groupby(["trip_id", "obs_idx"], sort=False)["score"].idxmax()
    return tmp.loc[idx, ["trip_id", "obs_idx", "arc_id", "label"]].reset_index(drop=True)


def observation_metrics(df: pd.DataFrame, score: np.ndarray, obs: pd.DataFrame) -> dict:
    ref = obs[obs.ref_arc >= 0]
    n_ref = len(ref)
    pred = top1_predictions(df, score)
    pred = pred.merge(ref[["trip_id", "obs_idx"]], on=["trip_id", "obs_idx"])
    top1 = float(pred["label"].sum() / n_ref)
    has_pos = df.loc[df.label == 1, ["trip_id", "obs_idx"]].drop_duplicates()
    miss = 1.0 - len(has_pos.merge(ref[["trip_id", "obs_idx"]], on=["trip_id", "obs_idx"])) / n_ref
    return {"top1": top1, "candidate_miss_rate": float(miss), "n_ref_points": int(n_ref)}


def candidate_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    yhat = (p >= thr).astype(int)
    return {"macro_f1": float(f1_score(y, yhat, average="macro")),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "logloss": float(log_loss(y, p, labels=[0, 1]))}


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> tuple[float, int]:
    """Official definition: (share of adjacent pairs that are same-arc or head-to-tail, break count)."""
    pred_arcs = np.asarray(pred_arcs)
    if len(pred_arcs) < 2:
        return 1.0, 0
    a, b = pred_arcs[:-1], pred_arcs[1:]
    ok = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(ok.mean()), int((~ok).sum())


def distinct_arc_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> tuple[float, int]:
    """Stricter variant: collapse repeated arcs first, then share of consecutive distinct arcs that connect."""
    pred_arcs = np.asarray(pred_arcs)
    seq = pred_arcs[np.r_[True, pred_arcs[1:] != pred_arcs[:-1]]]
    if len(seq) < 2:
        return 1.0, 0
    ok = arcs[seq[:-1], 1] == arcs[seq[1:], 0]
    return float(ok.mean()), int((~ok).sum())
