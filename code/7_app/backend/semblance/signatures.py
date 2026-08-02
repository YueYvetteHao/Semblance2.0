"""Signatures: filter a GSEA result, split by NES sign, build NES-weighted centroids.

The math: sign is carried by the activated/suppressed split, magnitude by the
weight — never average signed NES into one vector (opposite directions would cancel).
    v_S = normalize( Σ_i |NES_i| · e_i / Σ_i |NES_i| )

This module is pure: it takes an `embed_fn: str -> np.ndarray` and never loads a model or
touches disk. The engine owns the embeddings and injects `embed_fn`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .schema import CutoffParams, GseaResult, PathwayRow

EmbedFn = Callable[[str], np.ndarray]


@dataclass
class Signature:
    label: str                       # "SampleA::activated"
    result_name: str                 # "SampleA"
    direction: str                   # "activated" | "suppressed"
    pathways: list[str]
    nes: np.ndarray                  # signed NES, aligned with `pathways`
    leading_edge: list[list[str]]    # per-pathway gene symbols
    vector: np.ndarray               # (D,) L2-normalized centroid
    embeddings: np.ndarray = field(repr=False, default=None)  # (k, D) per-pathway (for best_match)


def filter_result(result: GseaResult, params: CutoffParams) -> GseaResult:
    """Apply padj/|NES| cutoffs, then cap top-k per direction by |NES|."""
    kept = [r for r in result.rows if r.padj <= params.padj_max and abs(r.nes) >= params.abs_nes_min]
    out: list[PathwayRow] = []
    for sign in (+1, -1):
        side = [r for r in kept if (r.nes > 0) == (sign > 0)]
        side.sort(key=lambda r: abs(r.nes), reverse=True)
        out.extend(side[: params.top_k_per_direction])
    return GseaResult(name=result.name, collection=result.collection, rows=out)


def _build_one(
    result_name: str, direction: str, rows: list[PathwayRow],
    embed_fn: EmbedFn, weight_by_nes: bool,
) -> Optional[Signature]:
    if not rows:
        return None
    pathways = [r.pathway for r in rows]
    nes = np.array([r.nes for r in rows], dtype=float)
    emb = np.stack([np.asarray(embed_fn(p), dtype=float) for p in pathways])  # (k, D)

    if weight_by_nes:
        w = np.abs(nes)
        centroid = (w[:, None] * emb).sum(axis=0) / w.sum()
    else:
        centroid = emb.mean(axis=0)
    norm = np.linalg.norm(centroid)
    vector = centroid / norm if norm > 0 else centroid

    return Signature(
        label=f"{result_name}::{direction}",
        result_name=result_name,
        direction=direction,
        pathways=pathways,
        nes=nes,
        leading_edge=[r.leading_edge for r in rows],
        vector=vector,
        embeddings=emb,
    )


def build_signatures(
    result: GseaResult, params: CutoffParams, embed_fn: EmbedFn,
    warnings: Optional[list[str]] = None,
) -> list[Signature]:
    """Filter → split into activated/suppressed → NES-weighted centroids.

    Returns up to 2 signatures (one per non-empty direction). Empty directions are skipped
    and reported via `warnings` rather than crashing.
    """
    filtered = filter_result(result, params)
    sigs: list[Signature] = []
    for direction, sign in (("activated", +1), ("suppressed", -1)):
        rows = [r for r in filtered.rows if (r.nes > 0) == (sign > 0)]
        sig = _build_one(result.name, direction, rows, embed_fn, params.weight_by_nes)
        if sig is None:
            if warnings is not None:
                warnings.append(f"{result.name}::{direction} had 0 pathways after filtering")
        else:
            if len(rows) < 3 and warnings is not None:
                warnings.append(f"{sig.label} had only {len(rows)} pathway(s) after filtering")
            sigs.append(sig)
    return sigs
