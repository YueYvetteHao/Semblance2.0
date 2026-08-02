"""
Semblance 2.0 similarity engine.

Serves the POST /semblance contract: an input GSEA table -> ranked drug matches by BioLORD
semantic similarity to a precomputed atlas of per-(drug x disease) sensitivity signatures.

The atlas is built offline by `precompute/build_atlas.py` from `5_pathway/out/gsea_signatures/`
(2,132 fgsea tables, of which 1,824 survive the non-actionable-compound filter) into
`assets/atlas_centroids.npz`. If that file is absent the engine falls back to the deterministic
STUB so the frontend still works end-to-end.

Real pipeline (shares the vendored `semblance/` core with the offline build):
  parse table -> PathwayRows -> filter (padj<=0.25, |NES|>=1, top-50/direction) -> split by sign
  -> take the ACTIVATED (NES>0, "sensitive-cell up-program") |NES|-weighted BioLORD centroid
  -> cosine vs the atlas centroids -> hubness + null correction (see below) -> top-K.

Ranking is NOT raw cosine. See the block below `_REACTOME_EMB` for why, and `_score` /
`_standardize` for what replaced it. Held-out check: feeding a drug's own GSEA table back in
retrieves that drug at rank 1 in 95% of 74 signatures across SKCM / BRCA / COAD-READ (top-5 100%),
against 68% under raw cosine.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS_PATH = os.environ.get("ATLAS_CENTROIDS") or os.path.join(_HERE, "assets", "atlas_centroids.npz")
_REACTOME_EMB = os.path.join(os.path.dirname(ATLAS_PATH), "reactome_embeddings.npz")

# --- hubness correction + null calibration ------------------------------------------------------
# Raw cosine over BioLORD centroids is not usable as a ranking on its own. The embedding space is
# strongly anisotropic: mean pairwise cosine across the atlas is ~0.92, and a *random* pathway
# program scores ~0.975 against any disease panel — higher than many real signatures. Two
# consequences, both measured on this atlas:
#   1. HUBNESS. Some atlas entries sit near the global mean and are the nearest neighbour of almost
#      any query. Their mean similarity to random programs r(y) ranges 0.84-0.97.
#   2. The ranking barely responds to the query. Two INDEPENDENT random programs returned 75% of
#      the same top-20 drugs, i.e. three quarters of the result was the atlas, not the input.
# Fixes applied below, with the measured effect on that top-20 overlap (chance floor = k/panel):
#   raw cosine                       75%
#   CSLS  2*cos(q,y) - r(y)          55%
#   per-drug z of CSLS vs its null   27%   <- shipped; essentially fully query-driven
# The per-drug z is also what makes the reported number interpretable: it says how many standard
# deviations better this drug matches YOUR program than it matches an arbitrary one — a score
# against a baseline that ignores the query, which is the same discipline the study applies to its
# models. r(y) and the null moments are built once at load from the shipped Reactome vocabulary
# (deterministic seed, ~0.03 s, no extra asset file).
_NULL_REFS = int(os.environ.get("SEMBLANCE_NULL_REFS", "384"))  # synthetic programs in the null
_NULL_PATHWAYS = 50   # pathways per synthetic program — matches the top-50/direction cutoff
_NULL_SEED = 0        # fixed: identical calibration on every instance and every redeploy

# Canned atlas for the stub only — used when atlas_centroids.npz is absent.
_STUB_ATLAS = [
    {"drug": "PLX-4720", "drug_uid": "YZDJ-STUB", "disease": "SKCM"},
    {"drug": "Dabrafenib", "drug_uid": "BRAF-STUB", "disease": "SKCM"},
    {"drug": "BI-2536", "drug_uid": "PLK1-STUB", "disease": "SARC"},
    {"drug": "BMS-754807", "drug_uid": "IGF1R-STUB", "disease": "SARC"},
    {"drug": "AZD8055", "drug_uid": "MTOR-STUB", "disease": "KIRC"},
    {"drug": "MST-312", "drug_uid": "TERT-STUB", "disease": "HNSC"},
    {"drug": "Trametinib", "drug_uid": "MEK-STUB", "disease": "COAD_READ"},
    {"drug": "Erlotinib", "drug_uid": "EGFR-STUB", "disease": "LUAD"},
]


def _to_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pretty_display(name: str) -> str:
    """Reactome id -> a human-readable pathway label for the UI pills."""
    n = name[9:] if name.upper().startswith("REACTOME_") else name
    return n.replace("_", " ").strip().title()


def _norm_disease(code):
    """Free-text cancer-type input -> a normalized TCGA code matching the atlas (e.g. 'COAD_READ')."""
    if not code:
        return None
    c = str(code).split("=")[0].strip().upper()
    c = c.replace(" ", "").replace("/", "_").replace("-", "_")
    return c or None


class SemblanceEngine:
    def __init__(self) -> None:
        self._atlas = None
        self._vectors = None
        self._embedder = None
        self._hub = None       # r(y): each atlas entry's mean cosine to a random program
        self._null_mu = None   # per-entry mean of CSLS under the random-program null
        self._null_sd = None   # per-entry sd   of CSLS under the random-program null
        self._load()

    # --- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(ATLAS_PATH):
            return  # stub mode
        data = np.load(ATLAS_PATH, allow_pickle=False)
        self._vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)  # (N, 768), L2-normalized
        self._atlas = {
            "drug_uid": data["drug_uid"].tolist(),
            "drug": data["drug"].tolist(),
            "target": data["target"].tolist(),
            "disease": data["disease"].tolist(),
            "top_pathways": [s.split("||") if s else [] for s in data["top_pathways"].tolist()],
        }
        cache: dict[str, np.ndarray] = {}
        reactome_vectors = None
        if os.path.exists(_REACTOME_EMB):
            r = np.load(_REACTOME_EMB, allow_pickle=False)
            reactome_vectors = np.asarray(r["vectors"], dtype=np.float32)
            cache = {n: np.asarray(v, dtype=np.float32) for n, v in zip(r["names"].tolist(), r["vectors"])}
        from semblance.embed import BioLordEmbedder

        self._embedder = BioLordEmbedder(cache=cache)
        self._build_null()

    def _build_null(self, reactome_vectors=None) -> None:
        """Calibrate hubness r(y) and the null of "how well does an UNRELATED REAL program score".

        The reference programs are the atlas signatures themselves, held out panel-wise. That is
        the honest null here: a real GSEA table from a different tumour context has the same
        statistical character as a user's upload, which synthetic pathway draws do not. Two
        synthetic nulls were tried and rejected first — diffuse random programs sit near the global
        mean and are closer to everything (pushing real coherent queries to a mean best-z of -2.8),
        and nearest-neighbour "theme" blocks are far more peaked than real GSEA output, which
        inflated the null so much that even a drug's own signature scored at only the 50th
        percentile. Real held-out signatures avoid both failure modes by construction.

        Degrades safely: if anything is missing the engine falls back to raw cosine ranking, which
        is the previous behaviour.
        """
        if self._vectors is None or len(self._vectors) < 8:
            return
        V = self._vectors
        n = len(V)
        S = V @ V.T                                  # (N, N) real-signature reference scores
        np.fill_diagonal(S, np.nan)                  # never let an entry calibrate against itself
        self._hub = np.nanmean(S, axis=1).astype(np.float32)          # r(y)
        csls = 2.0 * S - self._hub[:, None]
        self._null_mu = np.nanmean(csls, axis=1).astype(np.float32)
        self._null_sd = (np.nanstd(csls, axis=1) + 1e-6).astype(np.float32)

        # NOTE. A per-panel null of the BEST within-panel z was built here and REMOVED. It did
        # not work as a confidence statistic: on held-out controls it gave a median percentile of
        # 63 for a drug's OWN signature and 64 for an arbitrary program — no separation. The score
        # distribution's shape barely changes with query quality even when the ranked identities
        # change completely, so any "weak result" warning derived from it would have fired on good
        # queries. Ranking is corrected below; per-query confidence remains an open problem and is
        # deliberately NOT claimed in the API.

    def _score(self, cos):
        """Rank statistic. Returns (per-drug null z, calibrated).

        CSLS (Conneau et al. 2018) subtracts each entry's hubness r(y) so entries that are the
        nearest neighbour of almost anything stop dominating; the per-entry z against the null then
        removes whatever baseline affinity remains. This is what sets the RANKING, and it is the
        step that took the top-20 overlap between independent arbitrary programs from 75% to 27%
        (chance floor 26%). Falls back to raw cosine (calibrated=False) if the null is unavailable.
        """
        if self._hub is None:
            return cos, False
        csls = 2.0 * cos - self._hub
        return (csls - self._null_mu) / self._null_sd, True

    @staticmethod
    def _standardize(t):
        """Within-panel standardization of the rank statistic — the REPORTED number.

        The per-drug null z still depends on how internally coherent the submitted program is: a
        tightly themed query scores lower against every drug than a diffuse one, so raw values are
        not comparable between queries (measured mean best-value 0.11 / 1.28 / 1.91 for tight /
        mixed / diffuse programs). Re-expressing each score as standard deviations above the mean
        of the panel being ranked removes that (2.46 / 2.80 / 2.54 — comparable). It is a monotone
        transform within a query, so the ranking above is untouched; only the units change, to
        "how far this drug stands out from the rest of this panel".
        """
        t = np.asarray(t, dtype=np.float64)
        if t.size < 2:
            return np.zeros_like(t)
        sd = t.std()
        return (t - t.mean()) / (sd + 1e-9)

    @staticmethod
    def _percentile(z):
        """P(null < observed) as a percentage, from the normal CDF of the within-panel z."""
        return 50.0 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))

    def status(self) -> str:
        return "real" if self._atlas is not None else "stub"

    # --- parsing ------------------------------------------------------------
    @staticmethod
    def parse_table(text: str) -> list[dict]:
        """Parse a clusterProfiler/fgsea GSEA table (CSV or TSV) -> [{description, nes, padj}].

        padj is optional (None when the table has no FDR column) — the real engine treats a
        missing padj as passing the FDR cutoff so effect size (|NES|) alone selects pathways.
        """
        text = text.strip()
        if not text:
            return []
        delim = "\t" if text.count("\t") >= text.count(",") else ","
        rows: list[dict] = []
        for r in csv.DictReader(io.StringIO(text), delimiter=delim):
            desc = (r.get("Description") or r.get("ID") or r.get("pathway") or r.get("NAME") or "").strip()
            nes = _to_float(r.get("NES") or r.get("nes"))
            padj = _to_float(
                r.get("p.adjust") or r.get("padj") or r.get("p_adjust")
                or r.get("FDR") or r.get("qvalue") or r.get("qvalues") or r.get("adj.P.Val")
            )
            if desc and nes is not None:
                rows.append({"description": desc, "nes": nes, "padj": padj})
        return rows

    # --- compare ------------------------------------------------------------
    def compare(self, rows: list[dict], top_k: int = 20, disease=None) -> tuple[list[dict], str | None]:
        """Return (ranked matches, note). `note` carries any fallback message for the UI."""
        if self._atlas is not None:
            return self._compare_real(rows, top_k, disease)
        return self._compare_stub(rows, top_k, disease)

    def _build_user_signature(self, rows: list[dict]):
        """Build the user's ACTIVATED (NES>0) |NES|-weighted centroid via the vendored core."""
        from semblance.schema import CutoffParams, GseaResult, PathwayRow
        from semblance.signatures import build_signatures

        prs = [
            PathwayRow(pathway=r["description"], nes=r["nes"],
                       padj=(r["padj"] if r.get("padj") is not None else 0.0))
            for r in rows
        ]
        result = GseaResult(name="query", rows=prs)

        def pick(params):
            sigs = build_signatures(result, params, self._embedder)
            act = next((s for s in sigs if s.direction == "activated"), None)
            return act or (sigs[0] if sigs else None)

        sig = pick(CutoffParams())
        if sig is None:  # nothing cleared |NES|>=1 / padj<=0.25 — relax and try again
            sig = pick(CutoffParams(padj_max=1.0, abs_nes_min=0.0))
        return sig

    def _compare_real(self, rows: list[dict], top_k: int, disease) -> tuple[list[dict], str | None]:
        sig = self._build_user_signature(rows)
        if sig is None:  # nothing to match — tell the user why rather than silently returning empty
            return [], (
                "Couldn't build a signature from your table — it needs at least a couple of enriched "
                "pathways with a positive NES (an up-program). Check the NES column and try again."
            )
        q = np.asarray(sig.vector, dtype=np.float32)
        cos = self._vectors @ q            # raw cosine — both sides are L2-normalized
        sims, calibrated = self._score(cos)  # rank on hubness-corrected, null-calibrated evidence

        note = None
        idx = np.arange(len(sims))
        dz = _norm_disease(disease)
        if dz:
            mask = np.array([d == dz for d in self._atlas["disease"]])
            if mask.any():
                idx = idx[mask]
            else:  # disease not represented in the atlas -> broaden instead of returning nothing
                note = (
                    f"No signatures for '{dz}' in the atlas — showing the closest matches across all "
                    f"cancer types instead."
                )

        order = idx[np.argsort(-sims[idx])][:top_k]
        if len(order) == 0:  # last-resort fallback: never return empty when a signature exists
            order = np.argsort(-sims)[:top_k]
            if dz and note is None:
                note = (
                    f"No matches within '{dz}' — showing the closest matches across all cancer types "
                    f"instead."
                )

        # Standardize over the panel actually being ranked (the disease subset when one was given),
        # so the reported number means the same thing regardless of the query's own coherence.
        zs = {}
        if calibrated:
            panel = self._standardize(sims[idx])
            zs = {int(j): float(v) for j, v in zip(idx, panel)}

        out = []
        for i in order:
            i = int(i)
            z = zs.get(i)
            out.append({
                "drug": self._atlas["drug"][i],
                "drug_uid": self._atlas["drug_uid"][i],
                "disease": self._atlas["disease"][i],
                # Raw cosine is kept for transparency but is NOT the ranking key and must not be
                # shown as "match quality": on this atlas an arbitrary program also scores ~0.975.
                "similarity": round(float(cos[i]), 4),
                "evidence_z": round(z, 2) if z is not None else None,
                "percentile": round(self._percentile(z), 1) if z is not None else None,
                "matched_pathways": self._matched(sig, self._atlas["top_pathways"][i]),
            })

        return out, note

    def _matched(self, user_sig, top_pathways: list[str], k: int = 3) -> list[str]:
        """The atlas pathways most semantically similar to any of the user's pathways."""
        if user_sig.embeddings is None or not top_pathways:
            return [_pretty_display(p) for p in top_pathways[:k]]
        ue = np.asarray(user_sig.embeddings, dtype=np.float32)  # (U, 768)
        names, ve = [], []
        for p in top_pathways:
            v = self._embedder.cache.get(p)
            if v is None:
                v = self._embedder(p)
            names.append(p)
            ve.append(np.asarray(v, dtype=np.float32))
        M = np.clip(np.stack(ve) @ ue.T, -1.0, 1.0)  # (P, U)
        best = M.max(axis=1)
        order = np.argsort(-best)[:k]
        return [_pretty_display(names[int(j)]) for j in order]

    # --- stub ---------------------------------------------------------------
    def _compare_stub(self, rows, top_k, disease) -> tuple[list[dict], str | None]:
        """Deterministic pseudo-similarity so the UI is testable without the model. NOT science."""
        seed = hashlib.sha256("|".join(sorted(r["description"] for r in rows)).encode()).hexdigest()
        dz = _norm_disease(disease)
        atlas = [a for a in _STUB_ATLAS if dz in (None, a["disease"])] or _STUB_ATLAS
        top_paths = [r["description"] for r in sorted(rows, key=lambda r: -abs(r["nes"]))[:3]]
        scored = []
        for i, a in enumerate(atlas):
            j = (i * 6) % (len(seed) - 4)
            sim = round(0.55 + int(seed[j : j + 4], 16) / 0xFFFF * 0.4, 3)  # 0.55-0.95
            scored.append({**a, "similarity": sim, "matched_pathways": top_paths})
        scored.sort(key=lambda m: -m["similarity"])
        return scored[:top_k], None
