"""Honest evaluation: nested grouped-CV, repeated-CV confidence intervals, and
the flat-vs-nested audit that the glioma pilot taught us to always run.

The rules, encoded here so no result can quietly cheat:
  * every split is GROUPED BY CELL LINE (a line never appears in train and test);
  * hyperparameters are tuned INSIDE the outer folds (nested CV), never on the
    fold they're scored on;
  * every headline number is repeated over several CV shuffles and reported with
    a confidence interval;
  * elastic net, XGBoost, and a mean-predictor are compared on identical splits.

The core lesson: a flat search's `best_score_` (tuning and scoring on the same
folds) is optimistic; the nested OOF score is the honest one, and the gap is real.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import models

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# per-line ranking on a cell line whose predictions are constant is undefined and
# skipped explicitly below; silence scipy's constant-input chatter.
warnings.filterwarnings("ignore", message="An input array is constant")

_SCORING = "neg_root_mean_squared_error"

# Inner hyperparameter search parallelism. Default 1 so the DRIVER can parallelize the outer
# task list (disease x cohort x model) across all cores without nested oversubscription.
INNER_NJOBS = 1


# ---- point metrics -------------------------------------------------------

def regression_metrics(y_true, y_pred) -> dict:
    """Spearman/Pearson (pred vs actual), RMSE, R^2. NaN-safe for degenerate folds."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")

    def _corr(fn):
        if len(y_true) < 3 or np.std(y_pred) == 0 or np.std(y_true) == 0:
            return float("nan")
        return float(fn(y_true, y_pred)[0])

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) if len(y_true) else 0.0
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "spearman": _corr(spearmanr),
        "pearson": _corr(pearsonr),
        "rmse": rmse,
        "r2": r2,
    }


# ---- nested CV -----------------------------------------------------------

def _tune_on_fold(name, X_tr, y_tr, groups_tr, n_inner, n_iter, seed, estimator_factory=None):
    """Fit on one outer-train fold, tuning inside grouped inner folds.

    If `estimator_factory` is given, it supplies a fresh (untuned) estimator — used
    for custom baselines like DrugMeanRegressor that need dataset-derived args.
    """
    if estimator_factory is not None:
        est = estimator_factory()
        est.fit(X_tr, y_tr)
        return est, {}
    est = models.make_estimator(name)
    params, kind = models.search_spec(name)
    if kind is None:  # mean-predictor: nothing to tune
        est.fit(X_tr, y_tr)
        return est, {}
    inner_cv = list(models.grouped_kfold_indices(groups_tr, n_inner, 1, seed))
    if kind == "grid":
        search = GridSearchCV(est, params, scoring=_SCORING, cv=inner_cv, n_jobs=INNER_NJOBS)
    else:
        search = RandomizedSearchCV(
            est, params, n_iter=n_iter, scoring=_SCORING, cv=inner_cv,
            random_state=seed, n_jobs=INNER_NJOBS,
        )
    search.fit(X_tr, y_tr)
    return search.best_estimator_, dict(search.best_params_)


def _drug_means(y_tr, drug_tr):
    """Per-drug mean of y over the training rows (+ global mean fallback)."""
    means = {}
    for d in np.unique(drug_tr):
        means[d] = float(y_tr[drug_tr == d].mean())
    return means, float(y_tr.mean())


def _residualize_disease_drug(y, tr, te, drug, disease):
    """In-fold (disease, drug) cell-mean residualization — removes TISSUE before the model sees y.

    `sens_z` is already per-drug z-scored, so subtracting the drug mean (the current arm) is a
    near no-op pan-cancer; subtracting the (disease, drug) CELL mean strips the tissue-of-origin
    component, leaving the honest within-line personalization target. Means are strictly in-fold.

    Fallback hierarchy for test rows whose (disease, drug) cell is absent from the train fold
    (all its lines landed in test): (disease, drug) -> drug -> global. Train rows always sit inside
    their own group, so the train residual needs no fallback. Vectorized (pandas groupby) — the
    per-row Python dict loop would be too slow under the 100-perm null on ~400k rows.
    """
    g = pd.DataFrame({"y": y[tr], "d": drug[tr], "dis": disease[tr]})
    y_tr = y[tr] - g.groupby(["dis", "d"])["y"].transform("mean").to_numpy()
    cell = g.groupby(["dis", "d"])["y"].mean()          # train (disease, drug) cell means
    dmean = g.groupby("d")["y"].mean()                  # train per-drug means (intermediate rung)
    gmean = float(y[tr].mean())
    te_idx = pd.MultiIndex.from_arrays([disease[te], drug[te]])
    m_te = cell.reindex(te_idx).to_numpy()
    m_te = np.where(np.isnan(m_te), pd.Series(drug[te]).map(dmean).to_numpy(), m_te)
    m_te = np.where(np.isnan(m_te), gmean, m_te)
    return y_tr, y[te] - m_te


def nested_cv(ds, name="enet", n_outer=5, n_inner=4, n_iter=40, n_repeats=5, seed=0,
              estimator_factory=None, residualize=False, group_by="line",
              residualize_by="drug"):
    """Repeated nested grouped-CV. Returns per-repeat OOF metrics + aggregate CIs.

    Within each repeat every sample gets exactly one out-of-fold prediction, so the
    correlation metrics are computed on the full n (stable), and the spread across
    repeats gives the confidence interval.

    residualize=True (pooled only): subtract each group's TRAIN-fold mean from y before
    fitting and score on that residual — the mechanism-specific arm. This measures the
    personalization signal that survives *after* removing main effects, with the means
    computed in-fold so there's no leakage. `residualize_by` selects the group:
      * "drug"          -> per-drug mean (near no-op pan-cancer; sens_z is already per-drug z).
      * "disease_drug"  -> per-(disease, drug) CELL mean, which strips TISSUE before the model
                           sees the target — the disease×drug-residualized "tissue-free" arm.

    group_by="drug": leave-DRUG-out splitting (a whole drug held out) — the "new drug"
    blind setting — instead of the default leave-cell-line-out.
    """
    X, y, groups = ds.X, ds.y, ds.groups
    split_groups = np.asarray(ds.drug) if group_by == "drug" else groups
    if residualize and ds.drug is None:
        raise ValueError("residualize=True needs a pooled Dataset (ds.drug)")
    if residualize and residualize_by == "disease_drug" and ds.disease_arr is None:
        raise ValueError("residualize_by='disease_drug' needs ds.disease_arr (per-row disease)")
    drug = np.asarray(ds.drug) if ds.drug is not None else None
    disease = np.asarray(ds.disease_arr) if ds.disease_arr is not None else None
    n = len(y)
    per_repeat = []
    oof_last = ytrue_last = None
    for r in range(n_repeats):
        oof = np.full(n, np.nan)
        ytrue = np.array(y, float)  # equals y unless residualized (then per-fold)
        for tr, te in models.grouped_kfold_indices(split_groups, n_outer, 1, seed + 100 * r):
            y_tr = y[tr]
            if residualize and residualize_by == "disease_drug":
                y_tr, ytrue[te] = _residualize_disease_drug(y, tr, te, drug, disease)
            elif residualize:
                means, gmean = _drug_means(y[tr], drug[tr])
                y_tr = y[tr] - np.array([means.get(d, gmean) for d in drug[tr]])
                ytrue[te] = y[te] - np.array([means.get(d, gmean) for d in drug[te]])
            best, _ = _tune_on_fold(name, X[tr], y_tr, groups[tr], n_inner, n_iter, seed + r,
                                    estimator_factory=estimator_factory)
            oof[te] = best.predict(X[te])
        per_repeat.append(regression_metrics(ytrue, oof))
        oof_last, ytrue_last = oof, ytrue
    agg = _summarize(per_repeat, ["spearman", "pearson", "rmse", "r2"])
    return {
        "model": name,
        "n_rows": int(n),
        "n_groups": int(ds.n_groups),
        "n_repeats": n_repeats,
        "residualized": bool(residualize),
        "residualize_by": residualize_by if residualize else None,
        "group_by": group_by,
        "per_repeat": per_repeat,
        "aggregate": agg,
        "oof_pred": oof_last,      # from the last repeat, for ranking readout
        "y_scored": ytrue_last,    # the (possibly residual) truth the OOF was scored against
    }


def tissue_oracle_cv(ds, n_outer=5, n_repeats=5, seed=0):
    """No-expression TISSUE ORACLE on the drug-residualized target (the interpretive smoking gun).

    Target = drug-mean-residualized sens_z (exactly the current `enet_resid` arm's target).
    Predictor = the in-fold (disease, drug) CELL mean of that residual — pure lineage, no features.
    If this matches or beats the expression arm, the pan-cancer "personalization" the expression
    model finds is recovered tissue-of-origin, not within-line signal. Grouped leave-cell-line-out,
    repeated for CIs; same result shape as nested_cv so `_row` consumes it unchanged.
    """
    y = np.asarray(ds.y, float)
    drug = np.asarray(ds.drug)
    disease = np.asarray(ds.disease_arr)
    groups = ds.groups
    n = len(y)
    per_repeat, oof_last, ytrue_last = [], None, None
    for r in range(n_repeats):
        oof = np.full(n, np.nan)
        ytrue = np.full(n, np.nan)
        for tr, te in models.grouped_kfold_indices(groups, n_outer, 1, seed + 100 * r):
            means, gmean = _drug_means(y[tr], drug[tr])                 # in-fold drug means
            res_tr = y[tr] - np.array([means.get(d, gmean) for d in drug[tr]])
            ytrue[te] = y[te] - np.array([means.get(d, gmean) for d in drug[te]])
            cell = (pd.DataFrame({"r": res_tr, "d": drug[tr], "dis": disease[tr]})
                    .groupby(["dis", "d"])["r"].mean())                 # tissue oracle = cell mean
            te_idx = pd.MultiIndex.from_arrays([disease[te], drug[te]])
            pred = cell.reindex(te_idx).to_numpy()
            oof[te] = np.where(np.isnan(pred), 0.0, pred)               # no cell info -> 0 (null pred)
        per_repeat.append(regression_metrics(ytrue, oof))
        oof_last, ytrue_last = oof, ytrue
    agg = _summarize(per_repeat, ["spearman", "pearson", "rmse", "r2"])
    return {
        "model": "disease_drugmean", "n_rows": int(n), "n_groups": int(ds.n_groups),
        "n_repeats": n_repeats, "residualized": True, "residualize_by": "drug",
        "group_by": "line", "per_repeat": per_repeat, "aggregate": agg,
        "oof_pred": oof_last, "y_scored": ytrue_last,
    }


def flat_vs_nested(ds, name="enet", n_outer=5, n_inner=4, n_iter=40, n_repeats=5, seed=0):
    """Contrast the optimistic flat search score with the honest nested score."""
    X, y, groups = ds.X, ds.y, ds.groups
    est = models.make_estimator(name)
    params, kind = models.search_spec(name)
    flat_rmse = None
    if kind is not None:
        cv = list(models.grouped_kfold_indices(groups, n_outer, 1, seed))
        if kind == "grid":
            s = GridSearchCV(est, params, scoring=_SCORING, cv=cv, n_jobs=-1).fit(X, y)
        else:
            s = RandomizedSearchCV(est, params, n_iter=n_iter, scoring=_SCORING, cv=cv,
                                   random_state=seed, n_jobs=-1).fit(X, y)
        flat_rmse = float(-s.best_score_)  # optimistic: tuned & scored on same folds
    nested = nested_cv(ds, name, n_outer, n_inner, n_iter, n_repeats, seed)
    # keep oof_pred on `nested` so the caller can compute ranking metrics; it is
    # intentionally not persisted to JSON (train.py stores only the aggregate).
    return {"flat_best_rmse": flat_rmse, "nested": nested}


def _summarize(records, keys):
    """mean + approximate 95% CI (normal across repeats) + spread, per metric."""
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in records], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[k] = {"mean": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0}
            continue
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        half = 1.96 * sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[k] = {
            "mean": mean,
            "ci95_lo": mean - half,
            "ci95_hi": mean + half,
            "std": sd,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(len(vals)),
        }
    return out


# ---- ranking metrics (pooled model) -------------------------------------

def ranking_metrics(ds, y_pred, k=3, min_drugs=3) -> dict:
    """Per-cell-line drug-ranking quality from pooled predictions.

    Sensitivity metrics are 'lower = more sensitive', so we rank drugs ascending.
    Averaged over cell lines that have >= `min_drugs` measured drugs.
    """
    if ds.drug is None:
        raise ValueError("ranking_metrics needs a pooled Dataset (ds.drug is None)")
    y_true = np.asarray(ds.y, float)
    y_pred = np.asarray(y_pred, float)
    groups = np.asarray(ds.groups)

    topk_hits, kendalls, spearmans, ndcgs = [], [], [], []
    for g in np.unique(groups):
        m = (groups == g) & np.isfinite(y_pred)
        yt, yp = y_true[m], y_pred[m]
        nd = len(yt)
        if nd < min_drugs:
            continue
        # more sensitive = smaller metric -> relevance = -value (shifted >= 0)
        rel_true = (yt.max() - yt)
        rel_pred = (yp.max() - yp)
        kk = min(k, nd)
        true_order = np.argsort(yt)          # ascending: most sensitive first
        pred_order = np.argsort(yp)
        top_true = set(true_order[:kk].tolist())
        top_pred = set(pred_order[:kk].tolist())
        topk_hits.append(len(top_true & top_pred) / kk)
        if np.std(yp) > 0 and np.std(yt) > 0:
            kendalls.append(float(kendalltau(yt, yp)[0]))
            spearmans.append(float(spearmanr(yt, yp)[0]))
            ndcgs.append(float(ndcg_score(rel_true[None, :], rel_pred[None, :], k=kk)))

    def _m(a):
        a = np.array(a, float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else float("nan")

    return {
        "k": k,
        "n_lines_scored": len(topk_hits),
        f"topk_hit_rate@{k}": _m(topk_hits),
        "kendall_tau": _m(kendalls),
        "spearman_per_line": _m(spearmans),
        f"ndcg@{k}": _m(ndcgs),
    }
