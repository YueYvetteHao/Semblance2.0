"""Parse GSEA tables → canonical `GseaResult`.

Mapping-driven, not name-trusting: a column *mapping* (canonical field -> source column)
decides what becomes what. The default mapping is for clusterProfiler GSEA output.

The leading_edge trap: clusterProfiler emits a column literally named `leading_edge` that holds
a stats string ("tags=77%, list=28%, signal=77%") — NOT genes. The genes live in
`core_enrichment` (slash-separated). The default mapping therefore reads genes from
`core_enrichment`, exactly as the canonical schema intends.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import GseaResult, PathwayRow

# Canonical field -> source column (clusterProfiler GSEA). pval/size/leading_edge optional.
CLUSTERPROFILER_MAPPING: dict[str, str] = {
    "pathway": "Description",
    "nes": "NES",
    "padj": "p.adjust",
    "pval": "pvalue",
    "size": "setSize",
    "leading_edge": "core_enrichment",   # genes — NOT the literal `leading_edge` column
}
REQUIRED_FIELDS = ("pathway", "nes", "padj")


def default_mapping() -> dict[str, str]:
    """The clusterProfiler GSEA column mapping (a copy, safe to mutate)."""
    return dict(CLUSTERPROFILER_MAPPING)


def read_table(path: str | Path, sep: Optional[str] = None) -> pd.DataFrame:
    """Read csv/tsv/xlsx into a DataFrame, dispatching on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if sep is None:
        sep = "\t" if suffix in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep)


def _split_genes(value) -> list[str]:
    """Split a clusterProfiler `core_enrichment` cell ('IFIT1/STAT1/...') into symbols."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [g.strip() for g in str(value).split("/") if g.strip()]


def _to_float(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # Reject NaN AND ±inf: R/clusterProfiler can serialize infinities as 'Inf', and an infinite
    # NES would poison the |NES|-weighted centroid (→ NaN vector → scipy.linkage crash).
    return f if math.isfinite(f) else None


def _to_int(value) -> Optional[int]:
    f = _to_float(value)
    return None if f is None else int(round(f))


def _infer_collection(pathways: list[str]) -> Optional[str]:
    """Infer the MSigDB collection from a shared pathway-name prefix (e.g. HALLMARK_)."""
    prefixes = {p.split("_", 1)[0] for p in pathways if "_" in p}
    return prefixes.pop() if len(prefixes) == 1 else None


def apply_mapping(
    df: pd.DataFrame,
    mapping: Optional[dict[str, str]] = None,
    name: str = "result",
    collection: Optional[str] = None,
) -> GseaResult:
    """Transform a DataFrame into a canonical `GseaResult` via a column mapping.

    Deterministic: code applies the mapping; nothing here rewrites data rows. Rows with a
    non-numeric/missing NES or padj are dropped (warned by the engine downstream).
    """
    mapping = mapping or default_mapping()
    for field in REQUIRED_FIELDS:
        col = mapping.get(field)
        if col is None or col not in df.columns:
            raise ValueError(
                f"required field '{field}' maps to column '{col}', which is absent. "
                f"available columns: {list(df.columns)}"
            )

    has_pval = mapping.get("pval") in df.columns
    has_size = mapping.get("size") in df.columns
    has_le = mapping.get("leading_edge") in df.columns

    rows: list[PathwayRow] = []
    for _, r in df.iterrows():
        nes = _to_float(r[mapping["nes"]])
        padj = _to_float(r[mapping["padj"]])
        pathway = r[mapping["pathway"]]
        if nes is None or padj is None or pd.isna(pathway):
            continue  # un-parseable / blank row
        rows.append(PathwayRow(
            pathway=str(pathway).strip(),
            nes=nes,
            padj=padj,
            pval=_to_float(r[mapping["pval"]]) if has_pval else None,
            size=_to_int(r[mapping["size"]]) if has_size else None,
            leading_edge=_split_genes(r[mapping["leading_edge"]]) if has_le else [],
        ))

    if collection is None:
        collection = _infer_collection([row.pathway for row in rows])
    return GseaResult(name=name, collection=collection, rows=rows)


def parse_gsea_file(
    path: str | Path,
    mapping: Optional[dict[str, str]] = None,
    name: Optional[str] = None,
    collection: Optional[str] = None,
    sep: Optional[str] = None,
) -> GseaResult:
    """Read a GSEA file and normalize it to a `GseaResult` (default = clusterProfiler mapping)."""
    path = Path(path)
    name = name or path.stem
    return apply_mapping(read_table(path, sep=sep), mapping=mapping, name=name, collection=collection)


def to_canonical_dict(result: GseaResult) -> dict:
    """JSON-serializable canonical dict, ready to ship over the MCP boundary."""
    return result.model_dump()
