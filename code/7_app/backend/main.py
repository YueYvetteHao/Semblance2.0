"""
Semblance 2.0 backend — the "Signature Match" (GSEA semantic search) service.

Deploy target: Google Cloud Run (scale-to-zero); see ./README.md and docs/HOSTING_PLAN.md §4.
Deploy with ./deploy.sh.

Endpoints
  GET  /healthz     liveness/readiness for Cloud Run; reports engine "real" or "stub"
  POST /semblance   body: {"table_text": "<GSEA table CSV/TSV>", "top_k": 20, "disease": null}
                    -> ranked atlas drug-signature matches by semantic similarity.

STATUS: LIVE. /semblance runs the real engine: the submitted enrichment table is filtered
(padj <= 0.25, |NES| >= 1, top-50/direction), its ACTIVATED rows are embedded into an
|NES|-weighted BioLORD centroid, and that is cosine-matched against the 1,824-signature atlas in
assets/atlas_centroids.npz (PRODUCT_DESIGN.md §5). SemblanceEngine falls back to a deterministic
canned ranking, and /healthz reports engine "stub", only if that atlas file is missing.
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quota import quota_dependency
from semblance_engine import SemblanceEngine

app = FastAPI(title="Semblance 2.0 — GSEA semantic search", version="0.2.0")

# --- CORS: allow the GitHub Pages origin(s) to call this backend --------------
# ALLOWED_ORIGINS = comma-separated, e.g. "https://<user>.github.io,https://your.domain"
# Keep it explicit (not "*") once auth/quotas are on.
_origins = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,  # set True only when using cookie/JWT auth
)

engine = SemblanceEngine()


class SemblanceRequest(BaseModel):
    table_text: str = Field(..., description="GSEA table as CSV/TSV text (clusterProfiler/fgsea schema).")
    top_k: int = Field(20, ge=1, le=200)
    disease: str | None = Field(None, description="Optional TCGA code to scope atlas matches.")


class Match(BaseModel):
    rank: int
    drug: str
    drug_uid: str
    disease: str
    similarity: float                 # raw cosine; kept for transparency, NOT the ranking key
    evidence_z: float | None = None   # SDs better than this drug matches an arbitrary program
    percentile: float | None = None   # P(null < observed), from the per-drug null
    matched_pathways: list[str]


class SemblanceResponse(BaseModel):
    engine: str
    n_input_pathways: int
    top_k: int
    disclaimer: str
    note: str = ""  # fallback message (e.g. disease not in atlas -> broadened to pan-cancer)
    matches: list[Match]


@app.get("/healthz")
def healthz():
    return {"status": "ok", "engine": engine.status()}


@app.post("/semblance", response_model=SemblanceResponse)
def semblance(req: SemblanceRequest, _user=Depends(quota_dependency)):
    rows = engine.parse_table(req.table_text)
    if len(rows) < 2:
        raise HTTPException(422, "Need at least 2 GSEA rows (pathways) to build a signature.")
    hits, note = engine.compare(rows, top_k=req.top_k, disease=req.disease)
    matches = [Match(rank=i + 1, **h) for i, h in enumerate(hits)]
    return SemblanceResponse(
        engine=engine.status(),
        n_input_pathways=len(rows),
        top_k=req.top_k,
        disclaimer=(
            "Exploratory / hypothesis-generating: semantic similarity of your up-program to "
            "drug-sensitive-cell signatures. Not a validated predictor and NOT clinical advice."
        ),
        note=note or "",
        matches=matches,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
