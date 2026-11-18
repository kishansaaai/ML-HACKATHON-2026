"""Plot Person C's simple baselines from the shared experiment ledger."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
names = ["logistic_regression", "gaussian_nb", "knn"]
data = pd.read_csv(RESULTS / "experiments.csv").drop_duplicates("model", keep="last")
data = data[data.model.isin(names)].set_index("model").reindex(names)
if data.isna().all(axis=None):
    raise SystemExit("No simple model results found; run scripts/train_simple.py first.")

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
axes[0].bar(data.index, data.cv_macro_f1_mean, color="#5277b5")
axes[0].set_title("Trip-grouped CV Macro F1")
axes[1].bar(data.index, data.test_top1, color="#4b9b77")
axes[1].set_title("student_test Top-1")
axes[2].bar(data.index, data.test_macro_f1, color="#d18b45")
axes[2].set_title("student_test Macro F1")
axes[3].bar(data.index, data.test_logloss, color="#a369b5")
axes[3].set_title("student_test Log Loss")
for ax in axes[:3]:
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
axes[3].tick_params(axis="x", rotation=30, labelsize=8)
fig.tight_layout()
out = RESULTS / "plots" / "simple_model_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
plt.close(fig)
print(out)
