"""Model factories, hyperparameter grids, and grouped-CV splitters.

Two regressors, per the benchmark: an elastic-net baseline (expected to win on
average for this task) and a deliberately-shallow XGBoost challenger (given its
best shot only on reduced PCA features, never the raw 56k). Both are wrapped so
tuning happens inside nested CV (see evaluate.py) on splits GROUPED BY CELL LINE.

There is no RepeatedGroupKFold in scikit-learn, so `grouped_kfold_indices`
implements one: for each repeat it seed-permutes the unique cell lines and cuts
them into folds, guaranteeing a cell line never straddles train/test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:  # pragma: no cover - xgboost optional at import time
    _HAS_XGB = False


# ---- estimators ----------------------------------------------------------

def elasticnet_estimator() -> Pipeline:
    """StandardScaler -> ElasticNet. Scaling lives INSIDE the pipeline so it is
    refit per CV fold (no test-fold leakage)."""
    return Pipeline([
        ("scale", StandardScaler()),
        # First-pass tuning: cap iters + looser tol + random coordinate selection so the many
        # nested-CV fits converge fast on small-n data (10000/1e-4 was hitting max_iter and
        # dominating wall-clock). Tighten later for a final run if needed.
        ("enet", ElasticNet(max_iter=3000, tol=1e-3, selection="random", random_state=0)),
    ])


def elasticnet_param_grid() -> dict:
    """Grid for GridSearchCV over the elastic net (inside nested CV)."""
    return {
        # Leaner first-pass grid (21 pts vs 60): drop the near-OLS low-alpha tail that overfits
        # small-n and converges slowly; keep a spread of L1 mixes. Widen for a final run if needed.
        "enet__alpha": np.logspace(-2, 1, 7),
        "enet__l1_ratio": [0.2, 0.5, 0.9],
    }


def xgb_estimator() -> "XGBRegressor":
    """Shallow, regularized XGBoost — constrained hard because n is tiny."""
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed in this environment")
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )


def xgb_param_dist() -> dict:
    """Random-search space for XGBoost (small draw budget, inside nested CV)."""
    return {
        "n_estimators": randint(100, 400),
        "max_depth": randint(2, 4),            # 2-3: shallow to resist overfit
        "learning_rate": uniform(0.01, 0.15),
        "subsample": uniform(0.6, 0.4),        # 0.6-1.0
        "colsample_bytree": uniform(0.5, 0.5),  # 0.5-1.0
        "min_child_weight": randint(3, 10),    # high -> conservative splits
        "reg_lambda": uniform(1.0, 5.0),       # L2 > 0
        "gamma": uniform(0.0, 0.5),
    }


def mean_predictor() -> DummyRegressor:
    """The floor every model must beat: predict the training-fold mean."""
    return DummyRegressor(strategy="mean")


class DrugMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict each drug's average metric, ignoring expression.

    The honest baseline for the POOLED model: the pooled elastic net gets a lot of
    its correlation for free from drug main effects (some drugs are just more potent
    everywhere). This predictor captures exactly that main effect and nothing else,
    so `enet - drugmean` isolates what EXPRESSION actually adds.

    Pooled layout is X = [n_pcs PCs | one-hot(drug) | ...optional omics/tissue blocks].
    The one-hot occupies exactly `n_drugs` columns starting at `n_pcs`; the drug is read
    from the argmax of that slice. Passing `n_drugs` makes the slice exact, so feature
    blocks appended AFTER the one-hot (omics, tissue) don't corrupt drug identification.
    Unseen drugs fall back to the global mean.
    """

    def __init__(self, n_pcs: int, n_drugs: int | None = None):
        self.n_pcs = n_pcs
        self.n_drugs = n_drugs

    def _onehot(self, X):
        if self.n_drugs is None:
            return X[:, self.n_pcs:]
        return X[:, self.n_pcs:self.n_pcs + self.n_drugs]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, float)
        drug_idx = self._onehot(X).argmax(axis=1)
        self.global_mean_ = float(y.mean())
        self.drug_means_ = {}
        for d in np.unique(drug_idx):
            self.drug_means_[int(d)] = float(y[drug_idx == d].mean())
        return self

    def predict(self, X):
        X = np.asarray(X)
        drug_idx = self._onehot(np.asarray(X)).argmax(axis=1)
        return np.array([self.drug_means_.get(int(d), self.global_mean_) for d in drug_idx])


class SignatureScorer(BaseEstimator, TransformerMixin):
    """In-fold sensitivity signature: rank-correlate each feature with sensitivity (-y) on the
    TRAIN fold, keep the top-|rho| features, and project each cell onto that signed signature.

    Used as the first step of a Pipeline: sklearn refits it on every train fold (y is forwarded to
    `fit`), and `transform` is a PER-SAMPLE projection using only stored train statistics, so the
    signature never sees held-out labels -> nested-CV skill stays leakage-free (the honest F2 arm).

    Orientation: sensitivity := -y (y = sens_z, negative = more sensitive), so a higher signature
    score = more sensitive-looking. The downstream linear step sets scale/sign; the reported skill
    is the held-out Spearman of the projection.
    """

    def __init__(self, top_k: int = 100):
        self.top_k = top_k

    def _spearman(self, X, y):
        xr = pd.DataFrame(X).rank().to_numpy()          # avg-rank per feature (ties handled)
        yr = pd.Series(-np.asarray(y, float)).rank().to_numpy()   # rank of sensitivity (-y)
        xr = xr - xr.mean(axis=0)
        yr = yr - yr.mean()
        denom = np.sqrt((xr ** 2).sum(axis=0)) * np.sqrt((yr ** 2).sum())
        return np.divide((xr * yr[:, None]).sum(axis=0), denom,
                         out=np.zeros(X.shape[1]), where=denom > 0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        rho = self._spearman(X, y)
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        k = int(min(self.top_k, X.shape[1]))
        self.idx_ = np.argsort(-np.abs(rho))[:k]
        self.w_ = rho[self.idx_]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = (X[:, self.idx_] - self.mu_[self.idx_]) / self.sd_[self.idx_]
        return (Z @ self.w_).reshape(-1, 1)


def signature_estimator() -> Pipeline:
    """In-fold Spearman signature -> linear fit. The signature score IS the predictor; the linear
    step only sets scale/sign. Tuning is over how many genes/pathways enter the signature."""
    return Pipeline([
        ("sig", SignatureScorer()),
        ("lin", LinearRegression()),
    ])


def signature_param_grid() -> dict:
    return {"sig__top_k": [25, 50, 100, 200, 500]}


MODEL_REGISTRY = {
    "enet": (elasticnet_estimator, elasticnet_param_grid, "grid"),
    "xgb": (xgb_estimator, xgb_param_dist, "random"),
    "mean": (mean_predictor, None, None),
    "signature": (signature_estimator, signature_param_grid, "grid"),
}


def make_estimator(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_REGISTRY)}")
    factory, _, _ = MODEL_REGISTRY[name]
    return factory()


def search_spec(name: str):
    """Return (param_grid_or_dist, 'grid'|'random'|None) for a model name."""
    _, param_factory, kind = MODEL_REGISTRY[name]
    return (param_factory() if param_factory else None), kind


# ---- grouped CV splitting ------------------------------------------------

def grouped_kfold_indices(groups, n_splits=5, n_repeats=1, seed=0):
    """Yield (train_idx, test_idx) for repeated K-fold GROUPED by cell line.

    Each repeat seed-permutes the unique groups and partitions them into
    `n_splits` folds; a group's rows always land together. `n_splits` is
    silently reduced if there are fewer unique groups than requested folds.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    k = int(min(n_splits, len(uniq)))
    if k < 2:
        raise ValueError(f"need >=2 cell-line groups to split, have {len(uniq)}")

    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        order = uniq.copy()
        rng.shuffle(order)
        folds = np.array_split(order, k)
        for f in range(k):
            test_groups = set(folds[f].tolist())
            test_idx = np.concatenate([idx_by_group[g] for g in order if g in test_groups])
            train_idx = np.concatenate([idx_by_group[g] for g in order if g not in test_groups])
            yield np.sort(train_idx), np.sort(test_idx)


def n_effective_splits(groups, n_splits=5):
    return int(min(n_splits, len(np.unique(np.asarray(groups)))))
