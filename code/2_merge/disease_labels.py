"""Disease label per DepMap cell line, from the CCLE-2019 annotation.

Taxonomy = TCGA code (`tcga_code`, e.g. LUAD, SKCM, BRCA). Cells with no tcga_code
fall back to a `SITE_<Site_Primary>` label so they stay grouped by tissue rather than
lumped into one bucket; the `disease_source` column records which was used.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / "raw_data"
ANN = RAW / "CCLE2019" / "Cell_lines_annotations_20181226.txt"


def norm_id(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def disease_table() -> pd.DataFrame:
    """One row per DepMap ID: canonical `disease` label + provenance columns."""
    ann = (
        pd.read_csv(ANN, sep="\t", low_memory=False)
        .dropna(subset=["depMapID"])
        .drop_duplicates("depMapID")
    )
    df = ann[["depMapID", "CCLE_ID", "Name", "tcga_code", "Site_Primary",
              "Histology", "Hist_Subtype1"]].rename(columns={"depMapID": "depmap_id"})

    def _label(r):
        if pd.notna(r.tcga_code) and str(r.tcga_code).strip():
            return str(r.tcga_code).strip(), "tcga_code"
        if pd.notna(r.Site_Primary) and str(r.Site_Primary).strip():
            return "SITE_" + _slug(str(r.Site_Primary).upper()), "site_primary"
        return "UNKNOWN", "unknown"

    labs = [_label(r) for r in df.itertuples(index=False)]
    df["disease"] = [x[0] for x in labs]
    df["disease_source"] = [x[1] for x in labs]
    return df


def ccle_id_to_depmap() -> dict:
    """Map for resolving CCLE sample columns (`22RV1_PROSTATE`) to DepMap IDs,
    direct on CCLE_ID and via the normalized `Name` prefix as a fallback."""
    ann = pd.read_csv(ANN, sep="\t", low_memory=False).dropna(subset=["depMapID"])
    by_ccle = ann.drop_duplicates("CCLE_ID").set_index("CCLE_ID")["depMapID"].to_dict()
    by_name = {norm_id(n): d for n, d in
               ann.drop_duplicates("Name").set_index("Name")["depMapID"].items()}
    return {"ccle": by_ccle, "name": by_name}


def resolve_sample(sample: str, maps: dict) -> str | None:
    if sample in maps["ccle"]:
        return maps["ccle"][sample]
    return maps["name"].get(norm_id(sample.split("_")[0]))
