"""Merge CCLE TPM expression with the harmonized GDSC+CTRP drug response, then
split by disease into model-ready per-disease datasets.

Cell key = DepMap ID. Universe = cells that have BOTH CCLE TPM expression and at
least one harmonized drug measurement. Expression is log1p(TPM) (TPM is already
length+library normalized). Drug response stays LONG (one row per cell x drug x
cohort) carrying the harmonized `sens_z` plus raw provenance.

Outputs under merge/out/:
    expression_all.parquet   cells x genes (log1p TPM), index depmap_id
    response_all.parquet     long harmonized response for the merged cells + disease
    samples_all.parquet      one row per cell: disease, cohort coverage, counts
    by_disease/<DISEASE>/     expression.parquet | response.parquet | samples.parquet
    _SUMMARY.csv             per-disease cell/drug/row counts
    MERGE_NOTES.md           methodology + headline numbers

Run:  /opt/miniconda3/envs/ML/bin/python build_disease_datasets.py
"""
from __future__ import annotations
import gzip
from pathlib import Path
import numpy as np
import pandas as pd

import disease_labels as dl

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / "raw_data"
HARM = HERE.parent / "harmonize" / "out" / "drug_response_long.parquet"
TPM = RAW / "CCLE2019" / "CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz"
OUT = HERE / "out"
BYDIS = OUT / "by_disease"

RESP_COLS = ["depmap_id", "cohort", "drug_uid", "drug_name", "inchikey",
             "match_channel", "putative_target", "broad_cpd_id", "gdsc_drug_id",
             "auc_raw", "ln_ic50", "gdsc_zscore", "sens_z", "sens_rank"]


def _tpm_header() -> list[str]:
    with gzip.open(TPM, "rt") as f:
        return f.readline().rstrip("\n").split("\t")


def load_expression(keep_depmap: set) -> pd.DataFrame:
    """Read only the needed TPM sample columns; return **genes x cells** log1p(TPM):
    index `gene_id` (Ensembl, version stripped), one column per depmap_id. Genes-as-rows
    is the native TPM orientation and keeps parquet column count = #cells (not 57k), so
    per-disease slices stay small. Transpose (`set_index('gene_id').T`) for samples x genes."""
    maps = dl.ccle_id_to_depmap()
    header = _tpm_header()
    samples = header[2:]
    # sample column -> depmap, keeping only merged cells; first sample wins on collision
    sample_to_dep, seen = {}, set()
    for s in samples:
        d = dl.resolve_sample(s, maps)
        if d in keep_depmap and d not in seen:
            sample_to_dep[s] = d
            seen.add(d)
    needed = list(sample_to_dep)
    print(f"TPM: {len(samples)} samples -> {len(needed)} needed for merged cells")

    expr = pd.read_csv(TPM, sep="\t", usecols=["gene_id"] + needed)
    # strip Ensembl version suffix (ENSG….N -> ENSG….); dedupe defensively
    expr["gene_id"] = expr["gene_id"].str.split(".").str[0]
    if expr["gene_id"].duplicated().any():
        dups = int(expr["gene_id"].duplicated().sum())
        print(f"  {dups} duplicate gene IDs after version strip -> keeping first")
        expr = expr.drop_duplicates("gene_id", keep="first")
    expr = expr.set_index("gene_id")

    expr = expr.rename(columns=sample_to_dep).astype("float32")  # index gene_id, cols depmap
    expr.columns.name = "depmap_id"
    expr = np.log1p(expr).copy()                                  # log1p(TPM); defragment
    print(f"Expression matrix: {expr.shape[1]} cells x {expr.shape[0]} genes "
          f"(log1p TPM, genes-as-rows)")
    return expr


def build_samples(resp: pd.DataFrame, dis: pd.DataFrame, expr_cells: set) -> pd.DataFrame:
    """One row per merged cell: identity, disease, cohort coverage, drug counts."""
    def _first_name(s):
        v = s.dropna()
        return v.iloc[0] if len(v) else None

    agg = resp.groupby("depmap_id").agg(
        cell_line_name=("cell_line_name", _first_name),
        n_response_rows=("sens_z", "size"),
        n_drugs=("drug_uid", "nunique"),
    ).reset_index()
    cohorts = resp.groupby("depmap_id")["cohort"].agg(lambda s: set(s))
    agg["in_gdsc"] = agg["depmap_id"].map(lambda d: "GDSC" in cohorts[d])
    agg["in_ctrp"] = agg["depmap_id"].map(lambda d: "CTRP" in cohorts[d])
    agg["n_drugs_gdsc"] = agg["depmap_id"].map(
        resp[resp.cohort == "GDSC"].groupby("depmap_id")["drug_uid"].nunique()).fillna(0).astype(int)
    agg["n_drugs_ctrp"] = agg["depmap_id"].map(
        resp[resp.cohort == "CTRP"].groupby("depmap_id")["drug_uid"].nunique()).fillna(0).astype(int)
    agg = agg.merge(
        dis[["depmap_id", "disease", "disease_source", "tcga_code", "Site_Primary", "Name"]],
        on="depmap_id", how="left",
    )
    agg["has_expression"] = agg["depmap_id"].isin(expr_cells)
    return agg


def main():
    OUT.mkdir(exist_ok=True)
    BYDIS.mkdir(exist_ok=True)

    print("loading harmonized response + disease labels ...")
    long = pd.read_parquet(HARM)
    dis = dl.disease_table()

    # merged universe: DepMap-keyed harmonized cells that also have TPM expression
    tpm_dep = set()
    maps = dl.ccle_id_to_depmap()
    for s in _tpm_header()[2:]:
        d = dl.resolve_sample(s, maps)
        if d:
            tpm_dep.add(d)
    harm_dep = set(long.loc[long.row_key.str.startswith("ACH"), "row_key"])
    merged = harm_dep & tpm_dep
    print(f"harmonized depmap cells: {len(harm_dep)} | TPM cells: {len(tpm_dep)} "
          f"| MERGED: {len(merged)}")

    # response for merged cells (row_key IS the depmap id here; drop the redundant
    # original depmap_id column before promoting row_key to avoid a dup label)
    resp = long[long.row_key.isin(merged)].copy()
    resp = resp.drop(columns=["depmap_id"]).rename(columns={"row_key": "depmap_id"})
    resp = resp.merge(dis[["depmap_id", "disease", "disease_source"]], on="depmap_id", how="left")

    expr = load_expression(merged)            # genes x cells
    expr_cells = set(expr.columns)
    # a cell could be in `merged` yet drop out if its sole TPM sample collided; keep aligned
    samples = build_samples(resp, dis, expr_cells)

    # ---- combined stores ----
    expr.reset_index().to_parquet(OUT / "expression_all.parquet", index=False)
    resp[["depmap_id", "disease", "disease_source"] +
         [c for c in RESP_COLS if c != "depmap_id"]].to_parquet(OUT / "response_all.parquet", index=False)
    samples.to_parquet(OUT / "samples_all.parquet", index=False)

    # ---- per-disease split ----
    rows = []
    for disease, s_dis in samples.groupby("disease", sort=True):
        cells = list(s_dis.depmap_id)
        d_dir = BYDIS / dl._slug(disease)
        d_dir.mkdir(exist_ok=True)
        present = [c for c in cells if c in expr.columns]      # genes x (disease cells)
        e = expr[present]
        r = resp[resp.depmap_id.isin(cells)]
        e.reset_index().to_parquet(d_dir / "expression.parquet", index=False)
        r.to_parquet(d_dir / "response.parquet", index=False)
        s_dis.to_parquet(d_dir / "samples.parquet", index=False)
        rows.append({
            "disease": disease,
            "disease_source": s_dis.disease_source.iloc[0],
            "n_cells": len(cells),
            "n_cells_with_expr": int(e.shape[1]),
            "n_gdsc": int(s_dis.in_gdsc.sum()),
            "n_ctrp": int(s_dis.in_ctrp.sum()),
            "n_drugs": int(r.drug_uid.nunique()),
            "n_response_rows": int(len(r)),
        })
    summary = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
    summary.to_csv(OUT / "_SUMMARY.csv", index=False)

    _write_notes(long, resp, expr, samples, summary, merged)
    print(f"\nwrote {OUT/'expression_all.parquet'} ({expr.shape[0]} genes x {expr.shape[1]} cells)")
    print(f"wrote {OUT/'response_all.parquet'} ({len(resp):,} rows)")
    print(f"wrote {OUT/'samples_all.parquet'} ({len(samples)} cells)")
    print(f"wrote {len(rows)} per-disease datasets under {BYDIS}")
    print(f"\nTop diseases:\n{summary.head(12).to_string(index=False)}")


def _write_notes(long, resp, expr, samples, summary, merged):
    n_both = int((samples.in_gdsc & samples.in_ctrp).sum())
    lines = [
        "# CCLE × Harmonized Drug Sensitivity — per-disease datasets\n",
        "Merge of CCLE-2019 **TPM expression** (log1p) with the harmonized **GDSC+CTRP** "
        "drug response (`sens_z`), keyed on **DepMap ID**, split by **CCLE `tcga_code`** "
        "(Site_Primary fallback).\n",
        "## Headline",
        f"- Merged cell lines (TPM ∩ harmonized response): **{len(merged)}**",
        f"- Expression matrix: **{expr.shape[1]}** cells × **{expr.shape[0]:,}** genes (log1p TPM, genes-as-rows)",
        f"- In GDSC: {int(samples.in_gdsc.sum())} · in CTRP: {int(samples.in_ctrp.sum())} · in both: {n_both}",
        f"- Response rows (long): **{len(resp):,}** · distinct drugs: {resp.drug_uid.nunique()}",
        f"- Diseases: **{samples.disease.nunique()}** "
        f"({int((samples.disease_source=='tcga_code').sum())} cells labeled by tcga_code, "
        f"{int((samples.disease_source=='site_primary').sum())} by Site_Primary fallback, "
        f"{int((samples.disease_source=='unknown').sum())} unknown)\n",
        "## Layout",
        "```",
        "out/expression_all.parquet   genes × cells (log1p TPM): col 'gene_id' + one column per depmap_id",
        "out/response_all.parquet     long: depmap_id, disease, cohort, drug_uid, sens_z, raw metrics",
        "out/samples_all.parquet      per-cell metadata (disease, cohort coverage, drug counts)",
        "out/by_disease/<DISEASE>/    expression.parquet | response.parquet | samples.parquet",
        "out/_SUMMARY.csv             per-disease counts",
        "```",
        "## Modeling notes",
        "- Expression is stored **genes-as-rows** (efficient parquet). For samples×genes do",
        "  `pd.read_parquet(...).set_index('gene_id').T`. Values are log1p(TPM); recover TPM via expm1.",
        "  Gene IDs are Ensembl with the version suffix stripped.",
        "- `sens_z` is pan-cancer (z-scored per drug×cohort across the whole panel) — the per-disease",
        "  split does NOT recompute it. Negative = sensitive.",
        "- Response is long: a shared drug measured in both cohorts appears as two rows (GDSC + CTRP).",
        "  Pivot to cells×drugs and choose a cohort policy (keep separate, or mean) at modeling time.",
        f"\n## Per-disease summary (top 15 of {len(summary)})\n",
        "```",
        summary.head(15).to_string(index=False),
        "```",
    ]
    (OUT / "MERGE_NOTES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
