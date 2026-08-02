"""Build DepMap-keyed, per-disease multi-omics datasets — the same treatment the
expression layer got, for the additional modalities (genetics + soft-omics).

For each layer: clean (omics_clean, DepMap-keyed) -> write a full-coverage
`<layer>_all.parquet` (features-as-rows) -> split into per-disease matrices over the
SAME 810-cell / 37-disease universe as the expression merge (reusing
merge/out/samples_all.parquet so the grouping is identical). Coverage + notes emitted.

Orientation matches the expression store: features-as-rows (`feature_id` + one column
per depmap_id). Transpose (`set_index('feature_id').T`) for samples×features at model time.

Run:  /opt/miniconda3/envs/ML/bin/python build_omics_datasets.py
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

import omics_clean as oc

HERE = Path(__file__).resolve().parent
SAMPLES = HERE.parent / "merge" / "out" / "samples_all.parquet"
OUT = HERE / "out"
BYDIS = OUT / "by_disease"
ANNO = OUT / "feature_annotations"

LAYER_ORDER = ["mutations", "fusions", "methylation",           # genetics (primary)
               "metabolomics", "rppa", "mirna", "chromatin",     # soft-omics (secondary)
               "cn_summary"]                                     # bonus


def _slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def clean_all() -> dict:
    maps = oc.ccle_to_depmap()
    print("cleaning layers (mutations streams a 580MB file — ~1-2 min)...")
    layers = {}
    for name, fn in [
        ("mutations", lambda: oc.clean_mutations()),
        ("fusions", lambda: oc.clean_fusions()),
        ("methylation", lambda: oc.clean_methylation(maps)),
        ("metabolomics", lambda: oc.clean_metabolomics()),
        ("rppa", lambda: oc.clean_rppa(maps)),
        ("mirna", lambda: oc.clean_mirna()),
        ("chromatin", lambda: oc.clean_chromatin()),
        ("cn_summary", lambda: oc.clean_cn_summary()),
    ]:
        m = fn()  # cells × features, indexed by depmap_id
        layers[name] = m
        print(f"  {name:14s}: {m.shape[0]:>4} cells × {m.shape[1]:>4} features  ({m.values.dtype})")
    return layers


def _write_features_as_rows(mat: pd.DataFrame, path: Path):
    """mat is cells × features (index depmap_id). Store features-as-rows."""
    wide = mat.T
    wide.index.name = "feature_id"
    wide.reset_index().to_parquet(path, index=False)


def main():
    OUT.mkdir(exist_ok=True)
    BYDIS.mkdir(exist_ok=True)
    ANNO.mkdir(exist_ok=True)

    samples = pd.read_parquet(SAMPLES)[["depmap_id", "disease"]]
    universe = set(samples.depmap_id)                       # the 810 modeling cells
    dep2dis = dict(zip(samples.depmap_id, samples.disease))
    diseases = samples.groupby("disease")["depmap_id"].apply(list).to_dict()
    print(f"modeling universe: {len(universe)} cells, {len(diseases)} diseases\n")

    layers = clean_all()

    # feature annotations (interpretability)
    oc.rppa_annotation().to_csv(ANNO / "rppa_antibody_gene.csv", index=False)

    # combined full-coverage stores + per-disease slices
    cov = {d: {"n_cells": len(cells)} for d, cells in diseases.items()}
    layer_meta = []
    for name in LAYER_ORDER:
        mat = layers[name]
        _write_features_as_rows(mat, OUT / f"{name}_all.parquet")
        in_univ = mat.index.intersection(universe)
        layer_meta.append({
            "layer": name, "n_features": mat.shape[1],
            "n_cells_total": mat.shape[0], "n_cells_in_universe": len(in_univ),
            "dtype": str(mat.values.dtype),
            "kind": "binary" if str(mat.values.dtype) == "int8" else "continuous",
        })
        for disease, cells in diseases.items():
            present = [c for c in cells if c in mat.index]
            cov[disease][name] = len(present)
            if present:
                d_dir = BYDIS / _slug(disease)
                d_dir.mkdir(exist_ok=True)
                _write_features_as_rows(mat.loc[present], d_dir / f"{name}.parquet")

    # coverage table (per disease × layer) + layer summary
    cov_df = (pd.DataFrame.from_dict(cov, orient="index")
              .reset_index(names="disease")
              .sort_values("n_cells", ascending=False))
    cov_df.to_csv(OUT / "coverage.csv", index=False)
    layer_df = pd.DataFrame(layer_meta)
    layer_df.to_csv(OUT / "layer_summary.csv", index=False)

    _write_notes(layer_df, cov_df, len(universe), len(diseases))
    print(f"\nwrote {len(LAYER_ORDER)} layers -> {OUT}")
    print("\nlayer summary:\n", layer_df.to_string(index=False))
    print("\ncoverage (top 8 diseases):\n",
          cov_df.head(8).to_string(index=False))


def _write_notes(layer_df, cov_df, n_cells, n_dis):
    lines = [
        "# Multi-omics per-disease datasets (companion to expression)\n",
        "CCLE-2019 genetics + soft-omics, cleaned and keyed on **DepMap ID**, aligned to the same",
        f"**{n_cells}-cell / {n_dis}-disease** universe as the expression merge "
        "(`merge/out/samples_all.parquet`).\n",
        "Ported from the verified legacy cleaners (`phase2/src/omics.py`): same driver panels,",
        "thresholds, and transforms — re-keyed NORM_ID → depmap_id.\n",
        "## Layers\n",
        "```",
        layer_df.to_string(index=False),
        "```\n",
        "## Layout",
        "```",
        "out/<layer>_all.parquet          features × cells (full CCLE coverage), 'feature_id' + depmap cols",
        "out/by_disease/<DISEASE>/<layer>.parquet   features × (disease's cells having that layer)",
        "out/coverage.csv                 per disease × layer cell counts",
        "out/layer_summary.csv            per-layer feature/coverage/dtype",
        "out/feature_annotations/         rppa_antibody_gene.csv",
        "```",
        "## Modeling notes",
        "- Features-as-rows; transpose (`set_index('feature_id').T`) for samples×features.",
        "- **Binary** layers (mutations, fusions) — int8 presence; a cell ABSENT from a layer's",
        "  columns = no data for that cell (missingness), quantified in coverage.csv. **Continuous**",
        "  layers (methylation, metabolomics, rppa, mirna, chromatin, cn_summary) kept as provided.",
        "- No imputation/standardization here (a modeling step, like expression's raw log1p(TPM)).",
        "  Apply the base+optional impute-and-flag pattern at model time (binary→0, continuous→col mean,",
        "  plus a `<layer>_missing` indicator).",
        "- **cn_summary** is the ABSOLUTE per-sample summary (ploidy/genome-doublings/purity/WGD);",
        "  gene-level copy number is deferred (no gene-coordinate reference in raw_data).",
        f"\n## Coverage (per disease × layer)\n",
        "```",
        cov_df.to_string(index=False),
        "```",
    ]
    (OUT / "OMICS_NOTES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
