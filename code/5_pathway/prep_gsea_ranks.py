"""Atlas step 1 (Python): per-(disease, drug) Spearman-ranked gene lists for pre-ranked GSEA.

The user's harmonized-union alternative to a DESeq2 rerun: for each (disease, drug) panel, rank genes
by Spearman(gene_expr, SENSITIVITY) across pooled GDSC∪CTRP cells, where sensitivity := -sens_z
(negative sens_z = SENSITIVE), so a POSITIVE rho = higher expression in more-sensitive lines — the
same orientation as Phase-1 (positive = sensitivity-associated). Ranks feed clusterProfiler::GSEA in
run_gsea_signatures.R, which only consumes these .rnk files (no R-arrow / no per-gene cor.test in R).

Reads _work/expr_symbol.tsv (symbols x cells, from prep_expression.py) + 2_merge/out/response_all.
Writes _work/ranks/<disease>__<drug_uid>.rnk (symbol<TAB>rho, full ranked list) + _work/ranks_manifest.csv.

Run on the box:  ~/miniconda3/envs/ml/bin/python prep_gsea_ranks.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESP = ROOT / "2_merge" / "out" / "response_all.parquet"
WORK = HERE / "_work"
EXPR_TSV = WORK / "expr_symbol.tsv"
RANKS = WORK / "ranks"


def spearman_vs_sensitivity(expr_cells_x_genes: np.ndarray, sens_z: np.ndarray) -> np.ndarray:
    """Spearman(each gene, -sens_z) across cells, vectorized. Positive = sensitivity-associated."""
    xr = pd.DataFrame(expr_cells_x_genes).rank().to_numpy()        # rank each gene over cells
    yr = pd.Series(-np.asarray(sens_z, float)).rank().to_numpy()    # rank of sensitivity (-sens_z)
    xr = xr - xr.mean(axis=0)
    yr = yr - yr.mean()
    denom = np.sqrt((xr ** 2).sum(axis=0)) * np.sqrt((yr ** 2).sum())
    return np.divide((xr * yr[:, None]).sum(axis=0), denom,
                     out=np.full(expr_cells_x_genes.shape[1], np.nan), where=denom > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-lines", type=int, default=20)
    ap.add_argument("--min-sd", type=float, default=1.0, help="min SD(sens_z) in the panel (Phase-1 used 1.0)")
    args = ap.parse_args()

    RANKS.mkdir(parents=True, exist_ok=True)
    print(f"reading {EXPR_TSV} ...", flush=True)
    expr = pd.read_csv(EXPR_TSV, sep="\t").set_index("symbol")     # genes x cells
    cells_have = list(expr.columns)
    genes = expr.index.to_numpy()
    expr_t = expr.T                                                # cells x genes (align by depmap)

    r = pd.read_parquet(RESP, columns=["depmap_id", "disease", "drug_uid", "drug_name", "sens_z"])
    r = r.dropna(subset=["sens_z"])
    r = r[r["depmap_id"].isin(cells_have)]

    manifest = []
    for (disease, drug_uid), sub in r.groupby(["disease", "drug_uid"]):
        y_by_cell = sub.groupby("depmap_id")["sens_z"].mean()
        cells = [c for c in y_by_cell.index if c in cells_have]
        if len(cells) < args.min_lines:
            continue
        y = y_by_cell.loc[cells].to_numpy(float)
        if np.std(y, ddof=1) < args.min_sd:
            continue
        rho = spearman_vs_sensitivity(expr_t.loc[cells].to_numpy(float), y)
        rnk = pd.DataFrame({"symbol": genes, "rho": rho}).dropna()
        rnk = rnk.sort_values("rho", ascending=False)
        fn = RANKS / f"{str(disease).replace('/', '-')}__{drug_uid}.rnk"
        rnk.to_csv(fn, sep="\t", index=False, header=False)
        manifest.append({"disease": disease, "drug_uid": drug_uid,
                         "drug_name": sub["drug_name"].iloc[0], "n_lines": len(cells),
                         "sd_y": round(float(np.std(y, ddof=1)), 3), "rnk": fn.name})

    mf = pd.DataFrame(manifest).sort_values(["disease", "n_lines"], ascending=[True, False])
    mf.to_csv(WORK / "ranks_manifest.csv", index=False)
    print(f"wrote {len(mf)} ranked lists to {RANKS}/ (+ ranks_manifest.csv); "
          f"{mf['disease'].nunique()} diseases, {mf['drug_uid'].nunique()} drugs", flush=True)


if __name__ == "__main__":
    main()
