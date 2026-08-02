"""
Precompute the Semblance drug-sensitivity signature atlas for Tab B (GSEA semantic search).

Reads the 2,132 per-(drug x disease) fgsea signatures in `5_pathway/out/gsea_signatures/`, builds a
|NES|-weighted BioLORD centroid of each one's ACTIVATED (NES>0 = sensitivity-associated) pathways,
and writes into `backend/assets/`:
  atlas_centroids.npz      centroid matrix (N x 768) + drug/disease/pathway metadata
  reactome_embeddings.npz  every unique Reactome pathway-name embedding (serve-time cache, so
                           Reactome queries never load the ~420 MB model)

Reuses the vendored Semblance core (`backend/semblance/`) so the offline build and the live engine
share ONE parse -> filter -> centroid code path.

Sign convention (from 5_pathway): NES>0 = sensitivity-associated. The atlas entry per drug x disease
is the ACTIVATED signature — "the program up in cells sensitive to this drug".

Run (from 7_app/precompute/):
  /opt/miniconda3/envs/ML/bin/python build_atlas.py
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
ROOT = HERE.parent.parent  # ctrp-gdsc-ccle-ml/
sys.path.insert(0, str(BACKEND))  # import the vendored `semblance` package

from semblance.embed import EMBED_DIM, BioLordEmbedder  # noqa: E402
from semblance.schema import CutoffParams, GseaResult, PathwayRow  # noqa: E402
from semblance.signatures import _build_one, filter_result  # noqa: E402


def clean_target(t) -> str | None:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    parts = [p.strip() for p in str(t).replace(";", ",").split(",") if p.strip()]
    return ", ".join(dict.fromkeys(parts[:3])) if parts else None


# --- non-actionable-compound filter (same rule as precompute/build_cellline_json.py) --------------
# Drop uninterpretable screening IDs and a small denylist of non-therapeutic tool compounds so Tab B
# only ever recommends real, nameable drugs. Probes/tool inhibitors are KEPT — they are legitimate
# exploratory hits in a mechanism-match context (mirrors the cell-line lookup's mechanism arm).
_ID_RE = re.compile(r"^(\d+|BRD[-_ ]?[A-Za-z]?\d.*|BD[A-Z]{1,5}\d.*|NSC[-_ ]?\d.*|SID\s*\d+|CIL\d+)$", re.I)
_DENY = (
    "staurosporine",
    "acetalax", "oxyphenisatin",
    "ouabain", "digoxin", "digitoxin",
    "glutathione", "acetyl cysteine", "acetylcysteine",
    "picolin",
)


def drug_ok(name: str) -> bool:
    n = (name or "").strip()
    if not n or _ID_RE.match(n):
        return False
    low = n.lower()
    return not any(d in low for d in _DENY)


def drug_lut(merge_out: Path) -> dict[str, tuple[str, str | None]]:
    """drug_uid -> (display name, target) from the harmonized response store."""
    p = merge_out / "response_all.parquet"
    if not p.exists():
        print(f"WARNING: {p} not found; falling back to MASTER drug_name, no targets")
        return {}
    resp = pd.read_parquet(p, columns=["drug_uid", "drug_name", "putative_target"])
    out: dict[str, tuple[str, str | None]] = {}
    for uid, g in resp.groupby("drug_uid"):
        names = g["drug_name"].dropna()
        name = str(names.mode().iloc[0]) if len(names) else str(uid)
        target = None
        for raw in g["putative_target"].dropna():
            target = clean_target(raw)
            if target:
                break
        out[str(uid)] = (name, target)
    return out


def activated_rows(path: Path, uid: str, params: CutoffParams) -> list[PathwayRow]:
    """Read one _GSEA_REACTOME.csv -> filtered ACTIVATED PathwayRows (top-k by |NES|)."""
    df = pd.read_csv(path, usecols=["Description", "NES", "p.adjust"])
    rows: list[PathwayRow] = []
    for _, r in df.iterrows():
        if pd.isna(r["NES"]) or pd.isna(r["p.adjust"]) or pd.isna(r["Description"]):
            continue
        rows.append(PathwayRow(pathway=str(r["Description"]).strip(),
                               nes=float(r["NES"]), padj=float(r["p.adjust"])))
    filtered = filter_result(GseaResult(name=uid, rows=rows), params)
    return [r for r in filtered.rows if r.nes > 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsea-dir", default=str(ROOT / "5_pathway/out/gsea_signatures"))
    ap.add_argument("--merge-out", default=str(ROOT / "2_merge/out"))
    ap.add_argument("--out", default=str(BACKEND / "assets"))
    ap.add_argument("--top-pathways", type=int, default=10, help="pathway names kept per entry for the UI")
    args = ap.parse_args()

    gsea_dir = Path(args.gsea_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    params = CutoffParams()  # padj<=0.25, |NES|>=1.0, top-50/direction
    lut = drug_lut(Path(args.merge_out))

    master = pd.read_csv(gsea_dir / "MASTER_SIGNATURES.csv")
    entries: list[dict] = []
    all_names: set[str] = set()
    skipped = 0
    filtered = 0
    for _, m in master.iterrows():
        disease_raw = str(m["disease"])
        uid = str(m["drug_uid"])
        fp = gsea_dir / disease_raw.replace("/", "-") / f"{uid}_GSEA_REACTOME.csv"
        if not fp.exists():
            skipped += 1
            continue
        name, target = lut.get(uid, (str(m.get("drug_name") or uid), None))
        if not drug_ok(name):  # drop bare screening IDs / non-therapeutic control compounds
            filtered += 1
            continue
        rows = activated_rows(fp, uid, params)
        if not rows:
            skipped += 1
            continue
        entries.append({
            "uid": uid,
            "disease": disease_raw.replace("/", "_"),  # frontend spelling (e.g. COAD_READ)
            "drug": name,
            "target": target or "",
            "rows": rows,
        })
        all_names.update(r.pathway for r in rows)

    print(f"parsed {len(entries)} signatures ({filtered} non-actionable compounds filtered, "
          f"{skipped} skipped: missing file / no activated pathways)")

    # Embed every unique activated pathway name once, then build centroids from the warm cache.
    embedder = BioLordEmbedder()
    names = sorted(all_names)
    print(f"embedding {len(names)} unique Reactome pathway names with BioLORD (first load ~420 MB)…")
    embedder.warm(names)

    vectors = np.zeros((len(entries), EMBED_DIM), dtype=np.float32)
    uids, drugs, targets, diseases, tops = [], [], [], [], []
    for i, e in enumerate(entries):
        sig = _build_one(e["uid"], "activated", e["rows"], embedder, weight_by_nes=True)
        vectors[i] = np.asarray(sig.vector, dtype=np.float32)
        top = [r.pathway for r in sorted(e["rows"], key=lambda r: abs(r.nes), reverse=True)][: args.top_pathways]
        uids.append(e["uid"]); drugs.append(e["drug"]); targets.append(e["target"])
        diseases.append(e["disease"]); tops.append("||".join(top))

    np.savez_compressed(
        outdir / "atlas_centroids.npz",
        vectors=vectors,
        drug_uid=np.array(uids),
        drug=np.array(drugs),
        target=np.array(targets),
        disease=np.array(diseases),
        top_pathways=np.array(tops),
    )
    cache_names = list(embedder.cache.keys())
    np.savez_compressed(
        outdir / "reactome_embeddings.npz",
        names=np.array(cache_names),
        vectors=np.stack([embedder.cache[n] for n in cache_names]).astype(np.float32),
    )
    print(f"wrote atlas_centroids.npz ({len(entries)} signatures, {vectors.shape[1]}-dim) + "
          f"reactome_embeddings.npz ({len(cache_names)} names) -> {outdir}")


if __name__ == "__main__":
    main()
