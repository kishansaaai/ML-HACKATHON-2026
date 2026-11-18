"""Plot the Optuna search history for the report (reads results/tuning_trials_lightgbm.csv).

    python scripts/plot_tuning.py   ->  results/plots/tuning_trials_lightgbm.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    t = pd.read_csv(ROOT / "results" / "tuning_trials_lightgbm.csv")
    obj = next(c for c in t.columns if c in ("value", "objective", "cv_macro_f1", "cv_top1") or c.startswith("value"))
    t = t.sort_values(t.columns[0]).reset_index(drop=True)
    best = t[obj].cummax()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(range(len(t)), t[obj], s=22, label="trial", color="#1f77b4")
    ax.plot(best, color="#d62728", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel(obj); ax.set_title("LightGBM Bayesian optimisation (Optuna TPE)")
    ax.legend(); ax.grid(alpha=0.3)
    out = ROOT / "results" / "plots" / "tuning_trials_lightgbm.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("saved", out, "| objective column:", obj, "| trials:", len(t), "| best:", round(float(t[obj].max()), 4))
