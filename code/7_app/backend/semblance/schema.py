"""Canonical data contracts. Pydantic v2 — validated at the boundaries.

The deterministic core speaks these types; the MCP tools transport their `.model_dump()`
dicts. `ComparisonResult` is the engine's output and the heatmap's input.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Input models tolerate extra keys (clients may send richer canonical dicts).
_LENIENT = ConfigDict(extra="ignore")


class PathwayRow(BaseModel):
    """One enriched pathway from a GSEA result (after column mapping)."""
    model_config = _LENIENT
    pathway: str
    nes: float                                  # signed: >0 higher in test group, <0 lower
    padj: float                                 # adjusted p-value (FDR)
    pval: Optional[float] = None
    size: Optional[int] = None                  # gene-set size
    leading_edge: list[str] = Field(default_factory=list)  # core_enrichment gene symbols


class GseaResult(BaseModel):
    """One uploaded GSEA result, normalized to the canonical schema."""
    model_config = _LENIENT
    name: str
    collection: Optional[str] = None            # e.g. "HALLMARK" (inferred from pathway prefix)
    rows: list[PathwayRow] = Field(default_factory=list)


class CutoffParams(BaseModel):
    """Engine filter + aggregation params."""
    model_config = _LENIENT
    padj_max: float = 0.25
    abs_nes_min: float = 1.0
    top_k_per_direction: int = 50
    aggregation: Literal["centroid", "best_match"] = "centroid"
    weight_by_nes: bool = True


class PairDrillDown(BaseModel):
    """Pairwise comparison detail for two signatures."""
    a: str
    b: str
    semantic: float                             # cosine of NES-weighted centroids
    nes_corr: Optional[float] = None            # Spearman over shared pathway IDs
    shared_pathways: list[str] = Field(default_factory=list)
    top_shared_themes: list[dict] = Field(default_factory=list)  # [{"a","b","sim"}]
    unique_a: list[str] = Field(default_factory=list)
    unique_b: list[str] = Field(default_factory=list)
    leading_edge_jaccard: Optional[float] = None


class ComparisonResult(BaseModel):
    """Full engine output for N results → 2N signatures."""
    signatures: list[str]
    similarity: list[list[float]]               # (2N x 2N) cosine
    order: list[int]                            # dendrogram leaf order
    linkage: list[list[float]]                  # scipy linkage rows [i, j, dist, size]
    clusters: dict[str, int]                    # signature -> flat cluster id
    nes_correlation: list[list[Optional[float]]]  # baseline; null where overlap < 2
    pairs: list[PairDrillDown] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --- Agent-layer models (M6–M7) ---------------------------------------------
class ColumnMapping(BaseModel):
    """LLM output (M6): canonical field -> source column name. Code applies it (golden rule #6)."""
    model_config = _LENIENT
    pathway: str = Field(description="source column holding the pathway/term name")
    nes: str = Field(description="source column holding the signed NES")
    padj: str = Field(description="source column holding the adjusted p-value (FDR)")
    pval: Optional[str] = Field(default=None, description="raw p-value column, if present")
    size: Optional[str] = Field(default=None, description="gene-set size column, if present")
    leading_edge: Optional[str] = Field(
        default=None, description="column with leading-edge / core-enrichment genes, if present")
    collection: Optional[str] = Field(default=None, description="MSigDB collection, e.g. HALLMARK")
    suggested_name: Optional[str] = Field(default=None, description="a short signature name for this file")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_mapping(self) -> dict[str, str]:
        """field -> column dict for `core.parse.apply_mapping` (drops unset optionals)."""
        fields = ("pathway", "nes", "padj", "pval", "size", "leading_edge")
        return {f: getattr(self, f) for f in fields if getattr(self, f)}


class MappingReport(BaseModel):
    """How one file's column mapping was resolved (for the UI + audit trail)."""
    name: str
    mapping: dict[str, str]
    collection: Optional[str] = None
    confidence: float = 1.0
    source: Literal["llm", "default"] = "default"
    needs_confirmation: bool = False
    message: Optional[str] = None


class Claim(BaseModel):
    """One interpreter claim, grounded in a checkable artifact (M7)."""
    model_config = _LENIENT
    statement: str
    kind: Literal["cluster", "similarity", "nes_corr", "leading_edge", "unique"]
    signatures: list[str] = Field(default_factory=list)   # signatures the claim cites
    cited_value: Optional[float] = None                   # the number the claim asserts, if any


class Interpretation(BaseModel):
    """Interpreter output (M7): a summary plus grounded claims."""
    model_config = _LENIENT
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)


class VerifiedClaim(BaseModel):
    """A claim after deterministic verification against the ComparisonResult numbers (M7)."""
    claim: Claim
    verdict: Literal["Supported", "Unsupported", "Contradicted"]
    evidence: str
