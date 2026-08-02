"""Stage 3 of the pathway-activity build: ssGSEA scores -> model-ready feature layer.

Reads _work/pathway_scores.tsv (pathways x cells, from run_ssgsea.R) and writes, mirroring the
3_omics/out storage pattern so 4_model/data.py can load it as a drop-in layer:

  out/pathway_reactome_all.parquet            features-as-rows (feature_id + one col per depmap_id)
  out/by_disease/<TCGA>/pathway_reactome.parquet   the same, subset to that disease's cells
  out/coverage.csv                            per-disease cell counts

Disease split reuses 2_merge/out/samples_all.parquet (the authoritative 810-cell -> tcga_code map),
so grouping is identical to the expression store. Pathway scores exist only for the 810 cells that
have expression, so coverage is ~100% of the modeling universe by construction.

Run on the box:  ~/miniconda3/envs/ml/bin/python build_pathway_features.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLES = ROOT / "2_merge" / "out" / "samples_all.parquet"
WORK = HERE / "_work"
OUT = HERE / "out"
SCORES = WORK / "pathway_scores.tsv"
LAYER = "pathway_reactome"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "by_disease").mkdir(parents=True, exist_ok=True)

    print(f"reading {SCORES} ...", flush=True)
    sc = pd.read_csv(SCORES, sep="\t")
    sc = sc.rename(columns={sc.columns[0]: "feature_id"})
    n_path = sc.shape[0]
    depmaps = [c for c in sc.columns if c != "feature_id"]
    print(f"  {n_path:,} pathways x {len(depmaps):,} cells", flush=True)

    all_path = OUT / f"{LAYER}_all.parquet"
    sc.to_parquet(all_path, index=False)
    print(f"wrote {all_path}", flush=True)

    # per-disease split via the authoritative sample map
    s = pd.read_parquet(SAMPLES, columns=["depmap_id", "tcga_code"])
    s = s[s["depmap_id"].isin(depmaps)]
    cov = []
    for disease, grp in s.groupby("tcga_code"):
        cells = [c for c in grp["depmap_id"] if c in depmaps]
        if not cells:
            continue
        sub = sc[["feature_id"] + cells]
        ddir = OUT / "by_disease" / str(disease).replace("/", "_")
        ddir.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(ddir / f"{LAYER}.parquet", index=False)
        cov.append({"disease": disease, "n_cells": len(cells), LAYER: len(cells)})

    covdf = pd.DataFrame(cov).sort_values("n_cells", ascending=False)
    covdf.to_csv(OUT / "coverage.csv", index=False)
    n_covered = len(set(depmaps) & set(s["depmap_id"]))
    print(f"wrote by_disease/ for {len(cov)} diseases; coverage.csv "
          f"({n_covered}/{len(depmaps)} cells mapped to a disease)", flush=True)


if __name__ == "__main__":
    main()
