"""Model registry. Every model exposes fit(X, y) and predict_proba(X)[:, 1] over FEATURE_ORDER.

Heavy models (Person A): gradient boosting x3, Extra Trees, Random Forest (MLP follows).
Hyper-parameters below are the *initial* (untuned) values reported in the model comparison.
Class weights are not used: positives are ~9 % of candidates, and the HMM needs probabilities
that stay calibrated, which re-weighting would distort.
"""
from __future__ import annotations

SEED = 42

MODEL_SPECS = {
    "lightgbm": dict(family="Boosting (leaf-wise GBDT)",
                     params=dict(n_estimators=300, learning_rate=0.1, num_leaves=63, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)),
    "hist_gb": dict(family="Boosting (histogram GBDT, sklearn)",
                    params=dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=40,
                                l2_regularization=1.0)),
    "xgboost": dict(family="Boosting (XGBoost, depth-wise)",
                    params=dict(n_estimators=300, learning_rate=0.1, max_depth=8, min_child_weight=5,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)),
    "extra_trees": dict(family="Tree ensemble (randomised)",
                        params=dict(n_estimators=100, min_samples_leaf=3, max_features=0.5)),
    "random_forest": dict(family="Tree ensemble (bagging)",
                          params=dict(n_estimators=100, min_samples_leaf=3, max_features="sqrt")),
}


def make_model(name: str, params: dict | None = None, n_jobs: int = 8):
    p = dict(MODEL_SPECS[name]["params"])
    if params:
        p.update(params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, n_jobs=n_jobs, verbose=-1, **p)
    if name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **p)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=SEED, n_jobs=n_jobs, tree_method="hist", **p)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(random_state=SEED, n_jobs=n_jobs, **p)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=SEED, n_jobs=n_jobs, **p)
    raise KeyError(name)


def proba(model, X):
    return model.predict_proba(X)[:, 1]

