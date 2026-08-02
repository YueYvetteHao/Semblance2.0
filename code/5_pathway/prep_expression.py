"""Stage 1 of the pathway-activity build: expression (Ensembl) -> symbol matrix for ssGSEA.

Reads the model-ready log1p(TPM) expression store (2_merge/out/expression_all.parquet, genes-as-rows,
version-stripped Ensembl x 810 DepMap cells), maps Ensembl -> HGNC symbol via ref/ensembl_symbol.tsv
(pre-extracted on the Mac from raw_data/CCLE2019/CCLE_RNAseq_genes_counts_20180929.gct, which is raw
data kept off the box), collapses duplicate symbols keeping the Ensembl with the highest MEAN
log1p-TPM across cells (standard GSEA collapse=max), and writes _work/expr_symbol.tsv (symbols x cells).

Feed log1p-TPM straight through: ssGSEA ranks genes WITHIN each sample, so any monotone transform
(log1p) gives identical ranks and identical scores -> no expm1 needed. This step is label-blind
(never touches sens_z) so the resulting pathway features are leakage-free, exactly like PCA.

Run on the box:  ~/miniconda3/envs/ml/bin/python prep_expression.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # ctrp-gdsc-ccle-ml/
EXPR = ROOT / "2_merge" / "out" / "expression_all.parquet"
MAP = HERE / "ref" / "ensembl_symbol.tsv"
WORK = HERE / "_work"
OUT_TSV = WORK / "expr_symbol.tsv"


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"reading {EXPR} ...", flush=True)
    expr = pd.read_parquet(EXPR).set_index("gene_id")     # genes x cells (log1p TPM)
    print(f"  expression: {expr.shape[0]:,} genes x {expr.shape[1]:,} cells", flush=True)

    gmap = pd.read_csv(MAP, sep="\t", dtype=str)           # ensembl, symbol
    sym = gmap.dropna(subset=["symbol"]).set_index("ensembl")["symbol"]
    sym = sym[sym != ""]

    # collapse duplicate symbols: keep the Ensembl with the highest mean expression
    means = expr.mean(axis=1)
    keep = (
        pd.DataFrame({"symbol": sym.reindex(expr.index).values, "mean": means.values},
                     index=expr.index)
        .dropna(subset=["symbol"])
        .sort_values("mean", ascending=False)
        .drop_duplicates("symbol", keep="first")
    )
    n_mapped = keep.shape[0]
    n_dropped = expr.shape[0] - int(sym.reindex(expr.index).notna().sum())
    expr2 = expr.loc[keep.index].copy()
    expr2.index = keep["symbol"].to_numpy()
    expr2.index.name = "symbol"
    expr2 = expr2.sort_index()

    print(f"  mapped to symbol: {int(sym.reindex(expr.index).notna().sum()):,} genes "
          f"({n_dropped:,} unmapped dropped); {n_mapped:,} unique symbols after collapse=max",
          flush=True)
    expr2.to_csv(OUT_TSV, sep="\t")
    print(f"wrote {OUT_TSV}  ({expr2.shape[0]:,} symbols x {expr2.shape[1]:,} cells)", flush=True)


if __name__ == "__main__":
    main()
