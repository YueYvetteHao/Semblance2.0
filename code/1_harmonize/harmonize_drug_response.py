"""Harmonize GDSC2 + CTRPv2.2 drug-response into one pooled long table.

Pipeline:
  1. load both cohorts + the cell/drug crosswalks (built by the two sibling modules)
  2. reshape each cohort to long, attach canonical `row_key` (DepMap-first) + `drug_uid`
  3. collapse duplicate measurements per (cohort, cell, drug) by mean AUC
  4. harmonized metric `sens_z` = per-(cohort, drug_uid) z-score of AUC
     (negative = sensitive in BOTH cohorts); `sens_rank` in [0,1] as a robust twin
  5. write union parquet + intersection subset + a QC/counts report

Metric rationale: IC50 exists only in GDSC; AUC exists in both but on different
scales/definitions (GDSC [0,1] vs CTRP integrated ~[0,16]) so raw AUC is not
poolable. Per-drug within-cohort z removes each drug's potency offset and each
cohort's scale/batch offset, computed identically on both sides.

Run:  /opt/miniconda3/envs/ML/bin/python harmonize_drug_response.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import crosswalk_cells as xc
import crosswalk_drugs as xd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / "raw_data"
REF = HERE / "ref"
OUT = HERE / "out"

GDSC_XLSX = RAW / "GDSC2_fitted_dose_response_27Oct23.xlsx"
CTRP_DIR = RAW / "CTRPv2.2_2015_pub_CancerDisc_5_1210"
CTRP_AUC = CTRP_DIR / "v22.data.auc_sensitivities.txt"
CTRP_CELL = CTRP_DIR / "v22.meta.per_cell_line.txt"

MIN_GROUP_N = 5  # min cell lines per (cohort, drug) to trust a z-score

SCHEMA = [
    "depmap_id", "row_key", "cell_line_name", "sanger_model_id", "master_ccl_id",
    "ccle_primary_site", "cohort", "drug_uid", "drug_name", "inchikey",
    "match_channel", "gdsc_drug_id", "broad_cpd_id", "putative_target",
    "auc_raw", "ln_ic50", "gdsc_zscore", "sens_z", "sens_rank",
]


def _load_crosswalks(gdsc_df):
    cell_p, drug_p = REF / "cell_crosswalk.parquet", REF / "drug_crosswalk.parquet"
    cells = pd.read_parquet(cell_p) if cell_p.exists() else xc.build_cell_crosswalk(gdsc_df)
    drugs = pd.read_parquet(drug_p) if drug_p.exists() else xd.build_drug_crosswalk(gdsc_df)
    return cells, drugs


def _gdsc_long(gdsc_df, cells, drugs):
    cmap = cells[cells.cohort == "GDSC"].drop_duplicates("cell_line_name")
    dmap = drugs[drugs.cohort == "GDSC"].drop_duplicates("native_drug_id")
    df = gdsc_df.rename(
        columns={
            "CELL_LINE_NAME": "cell_line_name", "SANGER_MODEL_ID": "sanger_model_id",
            "AUC": "auc_raw", "LN_IC50": "ln_ic50", "Z_SCORE": "gdsc_zscore",
            "PUTATIVE_TARGET": "putative_target",
        }
    )
    df["gdsc_drug_id"] = df["DRUG_ID"].astype(str)
    df = df.merge(
        cmap[["cell_line_name", "depmap_id", "row_key", "master_ccl_id", "ccle_primary_site"]],
        on="cell_line_name", how="left",
    ).merge(
        dmap[["native_drug_id", "drug_uid", "drug_name", "inchikey", "match_channel"]],
        left_on="gdsc_drug_id", right_on="native_drug_id", how="left",
    )
    df["cohort"] = "GDSC"
    df["broad_cpd_id"] = pd.NA
    return df


def _ctrp_long(cells, drugs):
    auc = pd.read_csv(CTRP_AUC, sep="\t")
    meta = pd.read_csv(CTRP_CELL, sep="\t")[["index_ccl", "ccl_name"]]
    auc = auc.merge(meta, on="index_ccl", how="left").rename(
        columns={"ccl_name": "cell_line_name", "area_under_curve": "auc_raw"}
    )
    cmap = cells[cells.cohort == "CTRP"].drop_duplicates("cell_line_name")
    dmap = drugs[drugs.cohort == "CTRP"].drop_duplicates("native_drug_id")
    auc["native_drug_id"] = auc["index_cpd"].astype(str)
    df = auc.merge(
        cmap[["cell_line_name", "depmap_id", "row_key", "sanger_model_id",
              "master_ccl_id", "ccle_primary_site"]],
        on="cell_line_name", how="left",
    ).merge(
        dmap[["native_drug_id", "drug_uid", "drug_name", "inchikey",
              "match_channel", "broad_cpd_id", "target"]],
        on="native_drug_id", how="left",
    )
    df = df.rename(columns={"target": "putative_target"})
    df["cohort"] = "CTRP"
    for c in ("ln_ic50", "gdsc_zscore", "gdsc_drug_id"):
        df[c] = np.nan if c != "gdsc_drug_id" else pd.NA
    return df


def _collapse(df):
    """One row per (cohort, row_key, drug_uid); mean the numeric measurements."""
    # A null key means a crosswalk (possibly stale/cached) failed to cover a raw
    # measurement -> that row would vanish silently. Surface it loudly instead.
    has_auc = df["auc_raw"].notna()
    bad_key = has_auc & (df["row_key"].isna() | df["drug_uid"].isna())
    if bad_key.any():
        miss = df[bad_key]
        print(f"WARNING: {len(miss):,} measured rows dropped for null keys "
              f"(row_key null: {miss.row_key.isna().sum():,}, "
              f"drug_uid null: {miss.drug_uid.isna().sum():,}). "
              f"Crosswalks may be stale vs raw inputs — rebuild ref/*.parquet.")
    df = df.dropna(subset=["row_key", "drug_uid", "auc_raw"])
    num = df.groupby(["cohort", "row_key", "drug_uid"], as_index=False).agg(
        auc_raw=("auc_raw", "mean"),
        ln_ic50=("ln_ic50", "mean"),
        gdsc_zscore=("gdsc_zscore", "mean"),
    )
    meta_cols = ["depmap_id", "cell_line_name", "sanger_model_id", "master_ccl_id",
                 "ccle_primary_site", "drug_name", "inchikey", "match_channel",
                 "gdsc_drug_id", "broad_cpd_id", "putative_target"]
    meta = df.groupby(["cohort", "row_key", "drug_uid"], as_index=False)[meta_cols].first()
    return num.merge(meta, on=["cohort", "row_key", "drug_uid"])


def _add_metric(df):
    grp = df.groupby(["cohort", "drug_uid"])["auc_raw"]
    n = grp.transform("size")
    mu = grp.transform("mean")
    sd = grp.transform("std")
    z = (df["auc_raw"] - mu) / sd
    df["sens_z"] = np.where((n >= MIN_GROUP_N) & (sd > 0), z, np.nan)
    # rank in [0,1], 0 = most sensitive (lowest AUC); NaN for tiny groups
    df["sens_rank"] = np.where(n >= MIN_GROUP_N, grp.rank(pct=True), np.nan)
    return df


def _qc_concordance(shared):
    """Per shared drug, Spearman rho between GDSC and CTRP sens_z across shared lines."""
    g = shared[shared.cohort == "GDSC"][["row_key", "drug_uid", "sens_z"]]
    c = shared[shared.cohort == "CTRP"][["row_key", "drug_uid", "sens_z"]]
    m = g.merge(c, on=["row_key", "drug_uid"], suffixes=("_g", "_c")).dropna()
    rhos = []
    for uid, sub in m.groupby("drug_uid"):
        if len(sub) >= 8:
            rho = spearmanr(sub.sens_z_g, sub.sens_z_c).correlation
            if np.isfinite(rho):
                rhos.append((uid, len(sub), rho))
    return pd.DataFrame(rhos, columns=["drug_uid", "n_lines", "rho"])


def main():
    OUT.mkdir(exist_ok=True)
    print("loading GDSC2 xlsx ...")
    gdsc_df = pd.read_excel(GDSC_XLSX)
    cells, drugs = _load_crosswalks(gdsc_df)

    print("reshaping cohorts ...")
    g = _gdsc_long(gdsc_df, cells, drugs)
    c = _ctrp_long(cells, drugs)
    long = pd.concat([g, c], ignore_index=True, sort=False)

    print("collapsing duplicates + computing metric ...")
    long = _collapse(long)
    long = _add_metric(long)
    long = long.reindex(columns=SCHEMA)

    union_p = OUT / "drug_response_long.parquet"
    long.to_parquet(union_p, index=False)

    # intersection: cell (row_key) AND drug (drug_uid) present in both cohorts
    shared_cells = set(long[long.cohort == "GDSC"].row_key) & set(long[long.cohort == "CTRP"].row_key)
    shared_drugs = set(long[long.cohort == "GDSC"].drug_uid) & set(long[long.cohort == "CTRP"].drug_uid)
    shared = long[long.row_key.isin(shared_cells) & long.drug_uid.isin(shared_drugs)].copy()
    shared_p = OUT / "drug_response_shared.parquet"
    shared.to_parquet(shared_p, index=False)

    qc = _qc_concordance(shared)
    _write_report(long, shared, shared_cells, shared_drugs, qc, drugs)
    print(f"\nwrote {union_p}\nwrote {shared_p}\nwrote {OUT/'harmonize_report.md'}")


def _counts(df):
    return dict(rows=len(df), cells=df.row_key.nunique(), drugs=df.drug_uid.nunique())


def _write_report(long, shared, shared_cells, shared_drugs, qc, drugs):
    g, c = long[long.cohort == "GDSC"], long[long.cohort == "CTRP"]
    zc = long.dropna(subset=["sens_z"])
    lines = []
    A = lines.append
    A("# GDSC2 + CTRPv2.2 Harmonized Drug Response — Report\n")
    A("## Harmonized counts (deliverable)\n")
    A("| set | rows | cell lines | drugs |")
    A("|---|---|---|---|")
    for name, d in [("GDSC", g), ("CTRP", c), ("**Union**", long), ("Intersection", shared)]:
        cc = _counts(d)
        A(f"| {name} | {cc['rows']:,} | {cc['cells']:,} | {cc['drugs']:,} |")
    A(f"\n- Shared cell lines (row_key in both): **{len(shared_cells):,}**")
    A(f"- Shared drugs (drug_uid in both): **{len(shared_drugs):,}**")
    A(f"- Union cell lines: **{long.row_key.nunique():,}** | Union drugs: **{long.drug_uid.nunique():,}**\n")

    A("## Metric sanity (sens_z)\n")
    A(f"- rows with usable sens_z (group n>={MIN_GROUP_N}): {len(zc):,} / {len(long):,}")
    A(f"- sens_z mean={zc.sens_z.mean():.4f} (expect ~0), std={zc.sens_z.std():.4f} (expect ~1)\n")

    A("## Drug crosswalk audit (match_channel)\n")
    A("GDSC drugs by channel:\n```")
    A(drugs[drugs.cohort == "GDSC"].match_channel.value_counts().to_string())
    A("```\n")

    A("## Cross-cohort concordance (QC)\n")
    if len(qc):
        A(f"- shared drugs with >=8 co-measured lines: **{len(qc)}**")
        A(f"- per-drug Spearman rho (GDSC vs CTRP sens_z): "
          f"median **{qc.rho.median():.3f}**, mean {qc.rho.mean():.3f}, "
          f"frac>0 {np.mean(qc.rho > 0):.2f}")
        A("\nTop concordant drugs:\n```")
        A(qc.sort_values('rho', ascending=False).head(10).to_string(index=False))
        A("```")
    else:
        A("- (no shared drugs met the co-measurement threshold)")
    (OUT / "harmonize_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
