"""Build pooled modeling datasets from the harmonized, DepMap-keyed stores.

Mirrors the legacy pooled() layout so `models.DrugMeanRegressor` and `evaluate.nested_cv`
work unchanged — but reads the ACTIVE ctrp-gdsc-ccle-ml/ stores instead of the discontinued
phase2 parquets:

  X       = [ expression PCs (per cell line) | drug one-hot ]   (n_rows, n_pcs + n_drugs)
  y       = sens_z            (per-(cohort, drug) z of AUC; negative = more sensitive)
  groups  = depmap_id         (a cell line spawns many rows; grouped-CV keeps it in one fold)
  drug    = drug_uid          (InChIKey connectivity block; shared drugs pool across cohorts)

Cohort selection drives the "does GDSC∪CTRP pooling shrink CIs?" ablation:
  cohort="gdsc" / "ctrp"  -> that screen only
  cohort="both"           -> all rows; a drug screened in both cohorts contributes rows from
                             each under one drug_uid (sens_z is per-cohort z, so comparable).

PCA is fit once per cell-line panel (unsupervised, never sees y) — the standard M0 featurization.
Grouping by cell line means every row of a line lands in the same fold, so the PC broadcast and
the (possible) two-cohort rows for one (line, drug) never leak across train/test.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]      # ctrp-gdsc-ccle-ml/
MERGE = ROOT / "2_merge" / "out"
OMICS = ROOT / "3_omics" / "out"
PATHWAY = ROOT / "5_pathway" / "out"             # ssGSEA pathway-activity layers

BINARY_LAYERS = {"mutations", "fusions"}         # base-impute 0; others col-mean


@dataclass
class Dataset:
    """A ready-to-model table with grouped-CV metadata attached (legacy-compatible)."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    disease: str
    metric: str
    drug: np.ndarray | None = None
    disease_arr: np.ndarray | None = None   # per-ROW disease label (for disease×drug residualization)
    meta: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_groups(self) -> int:
        return len(np.unique(self.groups))

    @property
    def n_pcs(self) -> int:
        return int(self.meta.get("n_pcs", self.X.shape[1]))


# ---- expression ----------------------------------------------------------

_EXPR_PATH = MERGE / "expression_all.parquet"


def _expr_columns() -> set[str]:
    return set(pq.read_schema(_EXPR_PATH).names)


def load_expression(depmap_ids: list[str]) -> pd.DataFrame:
    """Return a (cells x genes) log1p(TPM) frame for the requested cells only.

    expression_all is stored features-as-rows (gene_id + one column per depmap_id); we read
    just the wanted columns (cheap) and transpose to cells x genes.
    """
    have = _expr_columns()
    cols = ["gene_id"] + [c for c in depmap_ids if c in have]
    df = pd.read_parquet(_EXPR_PATH, columns=cols).set_index("gene_id")
    return df.T  # cells x genes


def pca_features(expr: pd.DataFrame, n_pcs: int, seed: int = 0) -> pd.DataFrame:
    """StandardScaler -> PCA on (cells x genes); returns (cells x n_pcs).

    Drops zero-variance genes first (unexpressed everywhere in this panel) so scaling is stable.
    n_pcs is capped at n_cells-1. Unsupervised: never sees y.
    """
    X = expr.to_numpy(dtype=np.float64)
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    k = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    Z = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=k, random_state=seed).fit_transform(Z)
    return pd.DataFrame(pcs, index=expr.index, columns=[f"PC{i+1}" for i in range(k)])


# ---- response ------------------------------------------------------------

_RESP_PATH = MERGE / "response_all.parquet"
_RESP_COLS = ["depmap_id", "disease", "cohort", "drug_uid", "drug_name", "sens_z"]


def load_response() -> pd.DataFrame:
    r = pd.read_parquet(_RESP_PATH, columns=_RESP_COLS)
    return r.dropna(subset=["sens_z"])


def disease_sizes(min_lines: int = 1) -> pd.DataFrame:
    """Per disease: #cells (with expression) and #drugs, for scoping the sweep."""
    r = load_response()
    have = _expr_columns()
    r = r[r["depmap_id"].isin(have)]
    g = r.groupby("disease").agg(
        n_cells=("depmap_id", "nunique"),
        n_drugs=("drug_uid", "nunique"),
        n_rows=("sens_z", "size"),
    )
    return g[g["n_cells"] >= min_lines].sort_values("n_cells", ascending=False)


def pooled(
    disease: str,
    cohort: str = "both",
    n_pcs: int = 50,
    min_lines_per_drug: int = 5,
    seed: int = 0,
    response: pd.DataFrame | None = None,
    min_lines_per_cell: int = 1,
) -> Dataset:
    """Long (cell line x drug) table for one disease. cohort in {'gdsc','ctrp','both'}.

    'pancancer' as `disease` pools every disease (the effective-n lever). PCA is fit on the
    disease's cell-line panel; drugs with < `min_lines_per_drug` measured cell lines are dropped
    so the one-hot isn't dominated by near-empty columns.

    `min_lines_per_cell` drops rows whose (disease, drug_uid) cell has fewer than that many
    distinct lines — the disease×drug-residualized arm needs a stable in-fold cell mean. Default 1
    is a no-op; pass 5 for the tissue-free run. Per-disease (non-pancancer) it ~equals
    `min_lines_per_drug` since disease is then constant.
    """
    if cohort not in ("gdsc", "ctrp", "both"):
        raise ValueError("cohort must be 'gdsc', 'ctrp', or 'both'")
    r = load_response() if response is None else response
    if disease != "pancancer":
        r = r[r["disease"] == disease]
    if cohort != "both":
        r = r[r["cohort"].str.lower() == cohort]
    r = r.dropna(subset=["sens_z"])
    if r.empty:
        raise ValueError(f"no rows for {disease}/{cohort}")

    have = _expr_columns()
    cells = sorted(c for c in r["depmap_id"].unique() if c in have)
    if len(cells) < 2:
        raise ValueError(f"{disease}/{cohort}: only {len(cells)} cells with expression")
    r = r[r["depmap_id"].isin(cells)]

    # keep drugs measured in >= min_lines_per_drug distinct cell lines (within this selection)
    per_drug_lines = r.groupby("drug_uid")["depmap_id"].nunique()
    keep_drugs = set(per_drug_lines[per_drug_lines >= min_lines_per_drug].index)
    r = r[r["drug_uid"].isin(keep_drugs)]
    if r.empty:
        raise ValueError(f"{disease}/{cohort}: no drug with >= {min_lines_per_drug} lines")

    # keep (disease, drug_uid) cells with >= min_lines_per_cell distinct lines so the tissue-free
    # arm's in-fold cell mean is estimable. No-op when min_lines_per_cell <= 1.
    if min_lines_per_cell > 1:
        cell_lines = r.groupby(["disease", "drug_uid"])["depmap_id"].transform("nunique")
        r = r[cell_lines >= min_lines_per_cell]
        if r.empty:
            raise ValueError(
                f"{disease}/{cohort}: no (disease,drug) cell with >= {min_lines_per_cell} lines")

    expr = load_expression(cells)
    cells = [c for c in cells if c in expr.index]
    r = r[r["depmap_id"].isin(cells)]
    pcs = pca_features(expr.loc[cells], n_pcs, seed=seed)

    drug_names = sorted(r["drug_uid"].unique())
    drug_index = {d: i for i, d in enumerate(drug_names)}
    n_drugs = len(drug_names)
    name_map = r.drop_duplicates("drug_uid").set_index("drug_uid")["drug_name"].to_dict()

    r = r.sort_values(["drug_uid", "depmap_id"]).reset_index(drop=True)
    pc_block = pcs.loc[r["depmap_id"]].to_numpy(dtype=np.float64)
    onehot = np.zeros((len(r), n_drugs), dtype=np.float64)
    onehot[np.arange(len(r)), [drug_index[d] for d in r["drug_uid"]]] = 1.0

    X = np.hstack([pc_block, onehot])
    y = r["sens_z"].to_numpy(dtype=np.float64)
    groups = r["depmap_id"].to_numpy(dtype=object)
    drug_arr = r["drug_uid"].to_numpy(dtype=object)
    disease_arr = r["disease"].to_numpy(dtype=object)   # per-row tissue label (constant unless pancancer)
    feature_names = list(pcs.columns) + [f"drug={d}" for d in drug_names]
    return Dataset(
        X=X, y=y, groups=groups, feature_names=feature_names,
        disease=disease, metric="sens_z", drug=drug_arr, disease_arr=disease_arr,
        meta={
            "cohort": cohort, "n_pcs": pcs.shape[1], "n_drugs": n_drugs,
            "drugs": drug_names, "drug_display": {d: name_map.get(d, d) for d in drug_names},
            "n_lines": int(r["depmap_id"].nunique()), "min_lines_per_drug": min_lines_per_drug,
            "min_lines_per_cell": min_lines_per_cell, "n_diseases": int(r["disease"].nunique()),
        },
    )


# ---- optional omics layers (join on ds.groups = depmap_id) ----------------

def _layer_dir(name: str) -> Path:
    """Pathway-activity layers live under 5_pathway/out; measured CCLE omics under 3_omics/out."""
    return PATHWAY if name.startswith("pathway_") else OMICS


def load_omics_layer(name: str) -> pd.DataFrame:
    """Return a (cells x features) frame for an omics/pathway layer (stored features-as-rows)."""
    return pd.read_parquet(_layer_dir(name) / f"{name}_all.parquet").set_index("feature_id").T


def pathway_pcs(ds: Dataset, layer: str, n_pcs: int, seed: int = 0) -> Dataset:
    """Replace ds.X with PCA of a pathway-activity layer (per cell, broadcast to rows).

    The fast, dimensionality-matched counterpart to the gene-PC mechanism arm: PCA of the ~1,800
    ssGSEA pathway scores down to `n_pcs`, so `enet_pathwaypcs` costs the same as `enet_resid` (PCs)
    and asks cleanly whether a PATHWAY representation of expression beats a raw-GENE representation.
    Label-blind (PCA never sees y); drug labels/groups are preserved for in-fold residualization.
    """
    L = load_omics_layer(layer)                                  # cells x pathways
    cells = list(dict.fromkeys(np.asarray(ds.groups, dtype=object).tolist()))
    L = L.reindex(cells)
    L = L.fillna(L.mean(numeric_only=True))
    pcs = pca_features(L, n_pcs, seed=seed)                      # cells x k
    block = pcs.loc[ds.groups].to_numpy(dtype=np.float64)        # broadcast to rows
    return replace(ds, X=block, feature_names=list(pcs.columns),
                   meta={**ds.meta, "n_pcs": pcs.shape[1], "pathway_pca": layer})


def attach_omics(ds: Dataset, layers: list[str]) -> Dataset:
    """Append omics blocks joined on each row's depmap_id, with per-layer missingness flag.

    Binary layers (mutations, fusions) impute absent cells with 0; continuous layers impute with
    the column mean over present cells. Response-agnostic join -> no label leakage. Each layer adds
    `<layer>_missing` (1 = cell absent from that layer's profiling).
    """
    if not layers:
        return ds
    row = np.asarray(ds.groups, dtype=object)
    blocks = [ds.X]
    names = list(ds.feature_names)
    attached = []
    for layer in layers:
        L = load_omics_layer(layer)
        aligned = L.reindex(row)
        present = aligned.notna().any(axis=1).to_numpy()
        vals = aligned.to_numpy(dtype=np.float64, copy=True)  # writable: in-place impute below
        if layer in BINARY_LAYERS:
            vals = np.nan_to_num(vals, nan=0.0)
        else:
            col_mean = (np.nanmean(vals[present], axis=0) if present.any()
                        else np.zeros(vals.shape[1]))
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            nan_r, nan_c = np.where(np.isnan(vals))
            vals[nan_r, nan_c] = np.take(col_mean, nan_c)
        blocks.append(vals)
        names.extend(f"{layer}:{c}" for c in L.columns)
        blocks.append((~present).astype(np.float64).reshape(-1, 1))
        names.append(f"{layer}_missing")
        attached.append(f"{layer}({L.shape[1]}; {int(present.sum())}/{len(present)} rows)")
    X = np.hstack(blocks)
    return replace(ds, X=X, feature_names=names,
                   meta={**ds.meta, "omics_layers": list(layers), "omics_attached": attached})
