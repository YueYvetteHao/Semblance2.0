"""Cell-line crosswalk: map GDSC and CTRP cell lines onto a canonical DepMap ID.

Key convention follows the project `NORM_ID` rule (uppercase, alphanumeric-only;
cf. GDSC_CCLE_expression_merged_by_disease/merge_gdsc_ccle.py). The CCLE-2019
annotation (`Name` -> `depMapID`) is the bridge: CTRP is CCLE-native (near-full
coverage); GDSC lines absent from CCLE keep a `SANGER:<id>` fallback key so they
survive in the union (they simply won't have CCLE omics downstream).

Run directly to build/cache `ref/cell_crosswalk.parquet`.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / "raw_data"
REF = HERE / "ref"

GDSC_XLSX = RAW / "GDSC2_fitted_dose_response_27Oct23.xlsx"
CTRP_CELL = RAW / "CTRPv2.2_2015_pub_CancerDisc_5_1210" / "v22.meta.per_cell_line.txt"
CCLE_ANNO = RAW / "CCLE2019" / "Cell_lines_annotations_20181226.txt"


def norm_id(s) -> str:
    """Uppercase, strip everything non-alphanumeric (project NORM_ID convention)."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _ccle_name_to_depmap() -> dict:
    ann = pd.read_csv(CCLE_ANNO, sep="\t", low_memory=False)
    ann = ann.dropna(subset=["depMapID"]).copy()
    ann["nkey"] = ann["Name"].map(norm_id)
    ann = ann[ann["nkey"] != ""].drop_duplicates("nkey")
    return {
        "depmap": ann.set_index("nkey")["depMapID"].to_dict(),
        "site": ann.set_index("nkey")["Site_Primary"].to_dict(),
    }


def build_cell_crosswalk(gdsc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (cohort, native cell line) with canonical depmap_id + row_key."""
    ccle = _ccle_name_to_depmap()
    name2dep, name2site = ccle["depmap"], ccle["site"]

    # --- GDSC cells (from the response table's unique cell lines) ---
    if gdsc_df is None:
        gdsc_df = pd.read_excel(GDSC_XLSX)
    g = (
        gdsc_df[["CELL_LINE_NAME", "SANGER_MODEL_ID"]]
        .drop_duplicates()
        .rename(columns={"CELL_LINE_NAME": "cell_line_name", "SANGER_MODEL_ID": "sanger_model_id"})
    )
    g["cohort"] = "GDSC"
    g["master_ccl_id"] = pd.NA
    g["norm_id"] = g["cell_line_name"].map(norm_id)

    # --- CTRP cells ---
    c_meta = pd.read_csv(CTRP_CELL, sep="\t")
    c = c_meta[["ccl_name", "master_ccl_id", "ccle_primary_site"]].rename(
        columns={"ccl_name": "cell_line_name"}
    )
    c["cohort"] = "CTRP"
    c["sanger_model_id"] = pd.NA
    c["norm_id"] = c["cell_line_name"].map(norm_id)

    cells = pd.concat([g, c], ignore_index=True, sort=False)
    cells["depmap_id"] = cells["norm_id"].map(name2dep)
    # prefer CCLE-annotation site, fall back to CTRP's own site column
    ccle_site = cells["norm_id"].map(name2site)
    cells["ccle_primary_site"] = ccle_site.where(
        ccle_site.notna(), cells.get("ccle_primary_site")
    )

    # Row key: DepMap ID when available, else a cohort-native fallback so the
    # line still survives in the union.
    def _row_key(r):
        # DepMap ID is the canonical key; fall back to a cohort-namespaced native id
        # so a line missing from CCLE still gets a UNIQUE key (never a shared NA slug).
        if pd.notna(r["depmap_id"]):
            return r["depmap_id"]
        if r["cohort"] == "GDSC":
            if pd.notna(r["sanger_model_id"]):
                return f"SANGER:{r['sanger_model_id']}"
            return f"GDSC:{norm_id(r['cell_line_name'])}"
        if pd.notna(r["master_ccl_id"]):
            return f"CTRP:{r['master_ccl_id']}"
        return f"CTRP:{norm_id(r['cell_line_name'])}"

    cells["row_key"] = cells.apply(_row_key, axis=1)
    return cells[
        [
            "cohort", "cell_line_name", "norm_id", "depmap_id", "row_key",
            "sanger_model_id", "master_ccl_id", "ccle_primary_site",
        ]
    ]


if __name__ == "__main__":
    REF.mkdir(exist_ok=True)
    xw = build_cell_crosswalk()
    out = REF / "cell_crosswalk.parquet"
    xw.to_parquet(out, index=False)
    g = xw[xw.cohort == "GDSC"]
    c = xw[xw.cohort == "CTRP"]
    g_dep = set(g.depmap_id.dropna())
    c_dep = set(c.depmap_id.dropna())
    print(f"wrote {out}  ({len(xw)} rows)")
    print(f"GDSC cells: {len(g)}  mapped->DepMap: {g.depmap_id.notna().sum()}")
    print(f"CTRP cells: {len(c)}  mapped->DepMap: {c.depmap_id.notna().sum()}")
    print(f"DepMap overlap: {len(g_dep & c_dep)}  | union row_keys: {xw.row_key.nunique()}")
