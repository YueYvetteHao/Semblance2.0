"""
Generate the example GSEA table that Tab B's "Load an example" link serves.

A realistic, honest demo signature: the CONSENSUS (mean-NES) Reactome program across melanoma's
BRAF/MEK inhibitors in the harmonized atlas (PLX-4720, PLX-4032, dabrafenib, trametinib,
selumetinib). It is a composite — not any single atlas row — so searching it (scoped SKCM) surfaces
the whole MAPK-inhibitor family rather than trivially self-matching one drug.

Run (from 7_app/precompute/):
  /opt/miniconda3/envs/ML/bin/python make_example_gsea.py
Writes: ../frontend/data/example_gsea.csv  (top-N pathways by |mean NES|, clusterProfiler schema)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # ctrp-gdsc-ccle-ml/

# Melanoma MAPK-pathway inhibitors present in the SKCM atlas panel.
DRUGS = {"plx-4720", "plx-4032", "dabrafenib", "trametinib", "selumetinib"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsea-dir", default=str(ROOT / "5_pathway/out/gsea_signatures"))
    ap.add_argument("--out", default=str(HERE.parent / "frontend/data/example_gsea.csv"))
    ap.add_argument("--disease", default="SKCM")
    ap.add_argument("--top", type=int, default=35, help="top pathways by |mean NES|")
    args = ap.parse_args()

    gd = Path(args.gsea_dir)
    master = pd.read_csv(gd / "MASTER_SIGNATURES.csv")
    sel = master[(master["disease"].str.upper() == args.disease.upper())
                 & (master["drug_name"].str.lower().isin(DRUGS))]
    if sel.empty:
        raise SystemExit(f"no {args.disease} signatures found for {sorted(DRUGS)}")

    frames = []
    for r in sel.itertuples():
        fp = gd / args.disease.replace("/", "-") / f"{r.drug_uid}_GSEA_REACTOME.csv"
        if fp.exists():
            d = pd.read_csv(fp, usecols=["Description", "NES", "p.adjust"])
            frames.append(d.set_index("Description")["NES"].rename(r.drug_uid))

    mean_nes = pd.concat(frames, axis=1).mean(axis=1).dropna()
    top = mean_nes.reindex(mean_nes.abs().sort_values(ascending=False).index).head(args.top)
    out = pd.DataFrame({
        "ID": top.index,
        "Description": top.index,
        "NES": top.values.round(3),
        "pvalue": 0.001,
        "p.adjust": 0.01,
    })
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(out)} pathways: {(out.NES > 0).sum()} up / {(out.NES < 0).sum()} down; "
          f"consensus of {len(frames)} {args.disease} MAPK-inhibitor signatures)")


if __name__ == "__main__":
    main()
