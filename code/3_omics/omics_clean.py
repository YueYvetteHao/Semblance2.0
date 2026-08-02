"""Per-layer CCLE-2019 omics cleaners, keyed on DepMap ID.

Ports the verified legacy cleaning logic (phase2/src/omics.py) — identical driver
panels, thresholds, and transforms — but keys every layer on **depmap_id** (native
DepMap columns where the raw file has one; CCLE_ID -> depMapID via the annotation
otherwise) instead of NORM_ID. Each cleaner returns a **cells × features** DataFrame
indexed by `depmap_id`; the builder transposes to features-as-rows for storage.

No imputation / standardization here — that is a modeling step (mirrors the expression
store, which kept raw log1p(TPM)).
"""
from __future__ import annotations
import gzip
import re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / "raw_data" / "CCLE2019"
ANNOT = RAW / "Cell_lines_annotations_20181226.txt"

FILES = {
    "mutations": RAW / "OmicsSomaticMutations.csv",
    "fusions": RAW / "CCLE_Fusions_20181130.txt",
    "methylation": RAW / "CCLE_RRBS_TSS1kb_20181022.txt.gz",
    "metabolomics": RAW / "CCLE_metabolomics_20190502.csv",
    "rppa": RAW / "CCLE_RPPA_20181003.csv",
    "rppa_ab": RAW / "CCLE_RPPA_Ab_info_20181226.csv",
    "mirna": RAW / "CCLE_miRNA_MIMAT.csv",
    "chromatin": RAW / "CCLE_GlobalChromatinProfiling_20181130.csv",
    "cn_absolute": RAW / "CCLE_ABSOLUTE_combined_20181227.xlsx",
}

# --- curated panels / thresholds — verbatim from legacy omics.py -------------
MUTATION_FLAG_COLS = ["Hotspot", "OncogeneHighImpact", "TumorSuppressorHighImpact",
                      "LikelyLoF", "HessDriver"]
DRIVER_FUSION_GENES = {
    "ALK", "ROS1", "RET", "NTRK1", "NTRK2", "NTRK3", "ABL1", "BCR", "EML4",
    "FGFR1", "FGFR2", "FGFR3", "PDGFRA", "PDGFRB", "BRAF", "EGFR", "ERBB2", "KMT2A",
}
METH_PROMOTER_GENES = {
    "MGMT", "MLH1", "BRCA1", "RRM1", "ERCC1", "CDKN2A", "VHL", "WRN", "SLFN11",
}
DRIVER_GENES = {
    "TP53", "KRAS", "NRAS", "HRAS", "BRAF", "EGFR", "PIK3CA", "PTEN", "APC", "RB1",
    "CDKN2A", "CDKN2B", "MYC", "MYCN", "ERBB2", "MET", "ALK", "KIT", "PDGFRA", "FLT3",
    "IDH1", "IDH2", "NPM1", "FGFR1", "FGFR2", "FGFR3", "NF1", "NF2", "VHL", "SMAD4",
    "STK11", "KEAP1", "CTNNB1", "GNAS", "GNAQ", "GNA11", "JAK2", "MPL", "CALR", "ABL1",
    "BCR", "RET", "ROS1", "NTRK1", "MTOR", "AKT1", "ESR1", "AR", "BRCA1", "BRCA2",
    "ATM", "ARID1A", "SMARCA4", "SMARCB1", "KMT2D", "CREBBP", "EP300", "NOTCH1",
    "CCND1", "CDK4", "MDM2", "MDM4", "TERT", "SETD2", "BAP1", "DNMT3A",
    "TET2", "ASXL1", "EZH2", "SF3B1", "U2AF1",
}


def _norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def ccle_to_depmap() -> dict:
    """Resolver for CCLE-style ids -> depMapID: direct on CCLE_ID, else by the
    normalized `Name` prefix (handles `DMS53_LUNG` column headers)."""
    a = pd.read_csv(ANNOT, sep="\t", low_memory=False).dropna(subset=["depMapID"])
    return {
        "ccle": a.drop_duplicates("CCLE_ID").set_index("CCLE_ID")["depMapID"].to_dict(),
        "name": {_norm(n): d for n, d in
                 a.drop_duplicates("Name").set_index("Name")["depMapID"].items()},
    }


def _resolve_ccle(x, maps) -> str | None:
    if x in maps["ccle"]:
        return maps["ccle"][x]
    return maps["name"].get(_norm(str(x).split("_")[0]))


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Index by depmap_id, drop null-id rows, collapse id collisions (keep first)."""
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "depmap_id"
    return df


# ---- soft-omics (continuous, stored as provided) ---------------------------

def clean_metabolomics() -> pd.DataFrame:
    df = pd.read_csv(FILES["metabolomics"])
    feats = df.drop(columns=["CCLE_ID", "DepMap_ID"]).apply(pd.to_numeric, errors="coerce").astype("float32")
    feats.index = pd.Index(df["DepMap_ID"], name="depmap_id")
    return _dedup(feats)


def clean_rppa(maps: dict) -> pd.DataFrame:
    df = pd.read_csv(FILES["rppa"], index_col=0)  # index = CCLE_ID
    feats = df.reset_index(drop=True).apply(pd.to_numeric, errors="coerce").astype("float32")
    feats.index = pd.Index([_resolve_ccle(x, maps) for x in df.index], name="depmap_id")
    return _dedup(feats)


def clean_mirna() -> pd.DataFrame:
    df = pd.read_csv(FILES["mirna"], index_col=0)  # index = DepMap_ID
    feats = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    feats.index = pd.Index(df.index, name="depmap_id")
    return _dedup(feats)


def clean_chromatin() -> pd.DataFrame:
    df = pd.read_csv(FILES["chromatin"])
    feats = df.drop(columns=["CellLineName", "BroadID"]).apply(pd.to_numeric, errors="coerce").astype("float32")
    feats.index = pd.Index(df["BroadID"], name="depmap_id")
    return _dedup(feats)


# ---- genetics --------------------------------------------------------------

def clean_fusions() -> pd.DataFrame:
    """Binary cells × fusion matrix: pass==TRUE, partner ∈ driver set OR cclecnt≥3.

    Reindexed to ALL fusion-profiled cells: a profiled line with no driver/recurrent
    fusion is an explicit all-zero row (not 'missing'). Only truly unprofiled cells
    are absent."""
    fus = pd.read_csv(
        FILES["fusions"], sep="\t",
        usecols=["X.FusionName", "LeftGene", "RightGene", "cclecnt", "pass", "BroadID"],
    )
    profiled = pd.Index(pd.unique(fus["BroadID"].dropna()))   # every fusion-called line
    fus = fus[fus["pass"].astype(str).str.upper().eq("TRUE")].copy()
    is_driver = fus["LeftGene"].isin(DRIVER_FUSION_GENES) | fus["RightGene"].isin(DRIVER_FUSION_GENES)
    is_recurrent = pd.to_numeric(fus["cclecnt"], errors="coerce").fillna(0) >= 3
    fus = fus[is_driver | is_recurrent]
    mat = (
        fus.assign(v=1)
        .pivot_table(index="BroadID", columns="X.FusionName", values="v", aggfunc="max", fill_value=0)
        .astype("int8")
    )
    mat.columns = [f"fus_{c}" for c in mat.columns]
    mat = mat.reindex(profiled, fill_value=0).astype("int8")  # profiled-negative -> zeros
    mat.index.name = "depmap_id"
    return _dedup(mat)


def clean_mutations(genes: set | None = None, chunksize: int = 250_000) -> pd.DataFrame:
    """Binary cells × mut_<gene>: ≥1 FUNCTIONAL variant over the driver panel."""
    genes = genes or DRIVER_GENES
    usecols = ["ModelID", "HugoSymbol"] + MUTATION_FLAG_COLS
    pairs: set[tuple[str, str]] = set()
    profiled: set[str] = set()                                # every sequenced line
    for chunk in pd.read_csv(FILES["mutations"], usecols=usecols, chunksize=chunksize, low_memory=False):
        profiled.update(m for m in chunk["ModelID"].dropna().unique() if isinstance(m, str) and m)
        chunk = chunk[chunk["HugoSymbol"].isin(genes)]
        if chunk.empty:
            continue
        functional = chunk[MUTATION_FLAG_COLS].fillna(False).astype(bool).any(axis=1)
        chunk = chunk[functional]
        for m, g in zip(chunk["ModelID"], chunk["HugoSymbol"]):
            if isinstance(m, str) and m:
                pairs.add((m, f"mut_{g}"))
    prof_idx = pd.Index(sorted(profiled), name="depmap_id")
    if not pairs:
        return pd.DataFrame(0, index=prof_idx, columns=[])
    long = pd.DataFrame(sorted(pairs), columns=["depmap_id", "gene"])
    mat = (long.assign(v=np.int8(1))
               .pivot_table(index="depmap_id", columns="gene", values="v", aggfunc="max", fill_value=0)
               .astype("int8"))
    # profiled-but-no-driver-mutation -> explicit all-zero row (not 'missing')
    mat = mat.reindex(prof_idx, fill_value=0).astype("int8")
    mat.index.name = "depmap_id"
    return _dedup(mat)


def clean_methylation(maps: dict, genes: set | None = None) -> pd.DataFrame:
    """Continuous cells × meth_<gene>: promoter loci averaged per gene, columns are
    CCLE-style RRBS sample headers mapped to depmap_id."""
    genes = genes or METH_PROMOTER_GENES
    with gzip.open(FILES["methylation"], "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    meta_cols = ["locus_id", "CpG_sites_hg19", "avg_coverage"]
    sample_cols = [c for c in header if c not in meta_cols]
    keep = pd.read_csv(FILES["methylation"], sep="\t", compression="gzip",
                       usecols=["locus_id"] + sample_cols, low_memory=False)
    prefixes = tuple(f"{g}_" for g in genes)
    keep = keep[keep["locus_id"].astype(str).str.startswith(prefixes)]
    if keep.empty:
        return pd.DataFrame(index=pd.Index([], name="depmap_id"))
    keep["gene"] = keep["locus_id"].astype(str).str.split("_").str[0]
    keep[sample_cols] = keep[sample_cols].apply(pd.to_numeric, errors="coerce")
    per_gene = keep.groupby("gene")[sample_cols].mean()
    mat = per_gene.T.astype("float32")
    mat.columns = [f"meth_{g}" for g in mat.columns]
    mat.index = pd.Index([_resolve_ccle(s, maps) for s in mat.index], name="depmap_id")
    return _dedup(mat)


def clean_cn_summary() -> pd.DataFrame:
    """ABSOLUTE per-sample summary (bonus): ploidy, genome-doublings, purity, WGD flag.
    Segment-level gene CN is deferred (no gene-coordinate reference in raw_data)."""
    tab = pd.read_excel(FILES["cn_absolute"], sheet_name="ABSOLUTE_combined.table")
    out = pd.DataFrame(index=pd.Index(tab["depMapID"], name="depmap_id"))
    out["cn_ploidy"] = pd.to_numeric(tab["ploidy"].values, errors="coerce")
    out["cn_genome_doublings"] = pd.to_numeric(tab["Genome.doublings"].values, errors="coerce")
    out["cn_purity"] = pd.to_numeric(tab["purity"].values, errors="coerce")
    out["cn_wgd"] = (out["cn_genome_doublings"] >= 1).astype("float32")
    out = out.astype("float32")
    return _dedup(out)


def rppa_annotation() -> pd.DataFrame:
    """Antibody -> target gene(s), for interpretability of the RPPA features."""
    return pd.read_csv(FILES["rppa_ab"])
