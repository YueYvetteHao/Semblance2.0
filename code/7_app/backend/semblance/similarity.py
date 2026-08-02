"""Similarity matrix, hierarchical clustering, and the NES-correlation baseline.

Operates on `Signature` objects from `core.signatures`. The semantic layer is cosine over
NES-weighted centroids; the baseline is Spearman of NES over *shared* pathway IDs — shipped
alongside to show where semantics diverge from raw identity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from .signatures import Signature


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine between (already L2-normalized) signature vectors → (M x M), clipped to [-1, 1]."""
    V = np.asarray(vectors, dtype=float)
    return np.clip(V @ V.T, -1.0, 1.0)


def signature_similarity(signatures: list[Signature]) -> np.ndarray:
    return cosine_matrix(np.stack([s.vector for s in signatures]))


def best_match_similarity(a: Signature, b: Signature) -> float:
    """GOSemSim clusterSim-style: symmetric mean of max pairwise pathway cosines."""
    M = np.clip(np.asarray(a.embeddings, dtype=float) @ np.asarray(b.embeddings, dtype=float).T, -1.0, 1.0)
    return float((M.max(axis=1).mean() + M.max(axis=0).mean()) / 2.0)


def best_match_matrix(signatures: list[Signature]) -> np.ndarray:
    m = len(signatures)
    S = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            S[i, j] = S[j, i] = best_match_similarity(signatures[i], signatures[j])
    return S


def cluster(similarity: np.ndarray, cut_fraction: float = 0.5) -> tuple[list[int], list[list[float]], list[int]]:
    """Average-linkage clustering on (1 - similarity).

    Returns (leaf order, linkage rows, flat cluster ids). Distance is symmetrized with a zero
    diagonal before condensing — guards against tiny floating-point asymmetries.

    Flat clusters are cut at `cut_fraction` of the dendrogram's own merge-height range (0..1),
    not an absolute distance. This is scale-invariant: BioLORD's compressed cosine baseline would
    defeat a fixed threshold (opposed signatures still sit at ~0.78 cosine), but a fractional cut
    adapts to each run's spread. If there is no spread (all merges equal), everything is one cluster.
    """
    n = similarity.shape[0]
    if n == 0:
        return [], [], []
    if n == 1:
        return [0], [], [1]

    dist = 1.0 - similarity
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z).tolist()

    heights = Z[:, 2]
    hmin, hmax = float(heights.min()), float(heights.max())
    if hmax - hmin < 1e-9:
        flat = [1] * n  # no structure → a single cluster
    else:
        cut = hmin + cut_fraction * (hmax - hmin)
        flat = fcluster(Z, t=cut, criterion="distance").tolist()
    return order, Z.tolist(), flat


def nes_correlation(signatures: list[Signature]) -> list[list[Optional[float]]]:
    """Baseline: Spearman of NES over intersecting pathway IDs; None when overlap < 2.

    Within a single result, activated and suppressed sets are disjoint → None (sign-flip has no
    shared pathways). Across results, shared pathways drive the correlation.
    """
    m = len(signatures)
    nes_maps = [dict(zip(s.pathways, s.nes.tolist())) for s in signatures]
    out: list[list[Optional[float]]] = [[None] * m for _ in range(m)]
    for i in range(m):
        out[i][i] = 1.0
        for j in range(i + 1, m):
            shared = sorted(set(nes_maps[i]) & set(nes_maps[j]))
            val: Optional[float] = None
            if len(shared) >= 2:
                xi = [nes_maps[i][p] for p in shared]
                xj = [nes_maps[j][p] for p in shared]
                res = spearmanr(xi, xj)
                rho = getattr(res, "statistic", getattr(res, "correlation", np.nan))
                if rho is not None and not np.isnan(rho):
                    val = float(rho)
            out[i][j] = out[j][i] = val
    return out
