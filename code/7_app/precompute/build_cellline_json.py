"""
Precompute Cell-Line-Lookup artifacts: one frontend/data/cellline/<DepMapID>.json per cell line
for the static site (Tab A). No backend, no live model — pure static.

Design (docs/PRODUCT_DESIGN.md §1, §3):
  Arm 1 · Broad-spectrum  = the disease's drugs ranked by WITHIN-DISEASE MEAN sens_z (the "drugmean"
                            model) — "what this tumor type is generically most sensitive to." The same
                            ranking for every line of a disease, annotated with THIS line's own
                            measured value for context.
  Arm 2 · Mechanism (2a)  = the permutation-validated per-drug expression signatures for this line's
                            disease (4_model/out/signature_results_f2_signatures.csv — the latest F2
                            sweep, 7,836 pairs / 20 diseases). Gated at perm_p<=0.01 (the strongest-
                            evidence floor, since n_perm=100 can't resolve p small enough for a 7,836-way
                            FDR) and RANKED by perm_z (SDs above the tissue-matched null), which
                            discriminates within the p-floor. Exploratory; most lines show a short list
                            or none. See DISEASE_DRUG_RESIDUALIZATION_METHOD.md for the estimand.

Sign convention: lower / more-negative sens_z = MORE sensitive.

IMPORTANT — user-facing text only. No internal implementation names (metric column names, model
codenames like "F2"/"drugmean", "perm", etc.) may appear in any field the frontend renders
(`drug`, `target`, `note`). Keys are fine; rendered strings must be plain language.

Run (from 7_app/precompute/):
  /opt/miniconda3/envs/ML/bin/python build_cellline_json.py \
      --merge-out ../../2_merge/out \
      --signatures ../../4_model/out/signature_results_f2_signatures.csv \
      --out ../frontend/data/cellline --top 15 --min-lines 5 --mech-top 20 --max-perm-p 0.01
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

# Readable cancer-type names (kept in sync with the frontend's dz() map).
DZ = {
    "ALL": "Acute lymphoblastic leukemia", "BLCA": "Bladder", "BRCA": "Breast",
    "CLL": "Chronic lymphocytic leukemia", "COAD/READ": "Colorectal",
    "DLBC": "Diffuse large B-cell lymphoma", "ESCA": "Esophageal", "GBM": "Glioblastoma",
    "HNSC": "Head & neck", "KIRC": "Kidney, clear cell", "LAML": "Acute myeloid leukemia",
    "LCML": "Chronic myeloid leukemia", "LGG": "Low-grade glioma", "LIHC": "Liver",
    "LUAD": "Lung adenocarcinoma", "LUSC": "Lung squamous", "MB": "Medulloblastoma",
    "MESO": "Mesothelioma", "MM": "Multiple myeloma", "NB": "Neuroblastoma", "OV": "Ovarian",
    "PAAD": "Pancreatic", "PRAD": "Prostate", "SARC": "Sarcoma", "SCLC": "Small-cell lung",
    "SKCM": "Melanoma", "STAD": "Gastric", "THCA": "Thyroid", "UCEC": "Endometrial",
}


def pretty_disease(code: str) -> str:
    if not code:
        return "unknown"
    if code.startswith("SITE_"):
        return code[5:].replace("_", " ").title()
    if code.upper() == "UNABLE TO CLASSIFY":
        return "Unclassified"
    return DZ.get(code, code)


# --- non-actionable-compound filter (from the 37-disease adversarial biology audit) --------------
# Independent oncology-expert agents repeatedly flagged two classes of "recommendation" that are noise,
# not biology: (a) uninterpretable screening IDs, and (b) a handful of non-therapeutic lab tool
# compounds / assay-control cytotoxins that top lists in weak-driver diseases (staurosporine flagged in
# 8 diseases, acetalax/oxyphenisatin in 8). We drop both from the displayed arms. Conservative by design
# — only bare IDs and unambiguous non-drugs; genuine (even off-lineage) drugs are kept and contextualized
# by target + line count in the UI.
# Uninterpretable screening IDs: bare numbers, Broad experimental-ID families (BRD-…, BDF…/BDILV…/
# BDOCA…), NCI NSC catalog numbers (NSC 74859 / NSC30930), plus SID/CIL. These are catalog codes,
# not drug names.
_ID_RE = re.compile(r"^(\d+|BRD[-_ ]?[A-Za-z]?\d.*|BD[A-Z]{1,5}\d.*|NSC[-_ ]?\d.*|SID\s*\d+|CIL\d+)$", re.I)
_DENY = (
    "staurosporine",   # pan-kinase inhibitor — universal cytotoxic positive control, not a drug
    "acetalax", "oxyphenisatin",  # a laxative repurposed as a broad cytotoxic control
    "ouabain", "digoxin", "digitoxin",  # cardiac glycosides — classic viability-screen artifact
    "glutathione", "acetyl cysteine", "acetylcysteine",  # antioxidants / supplements, not oncology drugs
    "picolin",         # picolinic acid / spelling variants — metal chelator, not a drug
)
# CTRP compound statuses that mark non-therapeutic tool compounds (drop from broad-spectrum only —
# probes are legitimate exploratory hits in the mechanism arm, e.g. GPX4 ferroptosis inducers).
_PROBE_STATUS = {"probe", "GE-active"}


def drug_ok(name: str) -> bool:
    n = (name or "").strip()
    if not n or _ID_RE.match(n):
        return False
    low = n.lower()
    return not any(d in low for d in _DENY)


def clean_target(t) -> str | None:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    parts = [p.strip() for p in str(t).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    return ", ".join(dict.fromkeys(parts[:3]))  # de-dup, cap at 3 for display


def canonical_names(resp: pd.DataFrame) -> dict[str, tuple[str, str | None]]:
    """One display name + target per drug_uid (mode name; first usable target). Plain-Python
    values only — avoids Arrow-string dtype turning None back into NaN on a DataFrame round-trip."""
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


def sanitize(v):
    """Recursively coerce NaN / pandas-NA to None so the JSON is spec-valid (JSON.parse-safe)."""
    if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [sanitize(x) for x in v]
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-out", default="../../2_merge/out")
    ap.add_argument("--signatures", default="../../4_model/out/signature_results_f2_signatures.csv")
    ap.add_argument("--ctrp-meta",
                    default="../../../raw_data/CTRPv2.2_2015_pub_CancerDisc_5_1210/v22.meta.per_compound.txt",
                    help="CTRP per-compound metadata (for cpd_status probe/clinical/FDA gating)")
    ap.add_argument("--out", default="../frontend/data/cellline")
    ap.add_argument("--top", type=int, default=15, help="top-N broad-spectrum drugs per disease")
    ap.add_argument("--min-lines", type=int, default=5, help="min disease lines per drug for a stable mean")
    ap.add_argument("--mech-top", type=int, default=20, help="cap on mechanism rows per disease")
    ap.add_argument("--max-perm-p", type=float, default=0.01,
                    help="mechanism signatures must clear this permutation p (0.01 = the n_perm=100 floor, "
                         "the strongest-evidence set; not FDR-corrected — see DISEASE_DRUG_RESIDUALIZATION_METHOD.md)")
    args = ap.parse_args()

    mo = Path(args.merge_out)
    resp = pd.read_parquet(mo / "response_all.parquet")
    samples = pd.read_parquet(mo / "samples_all.parquet")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- drug_uid -> display name / target ------------------------------------
    lut = canonical_names(resp)  # {uid: (drug, target)}

    # --- CTRP cpd_status → drug_uids that are non-therapeutic probes (broad-spectrum gate only) ---
    probe_uids: set[str] = set()
    meta_path = Path(args.ctrp_meta)
    if meta_path.exists():
        meta = pd.read_csv(meta_path, sep="\t")
        status = dict(zip(meta["broad_cpd_id"], meta["cpd_status"]))
        for uid, g in resp.groupby("drug_uid"):
            bids = [b for b in g["broad_cpd_id"].dropna().unique()]
            if bids and any(status.get(b) in _PROBE_STATUS for b in bids):
                probe_uids.add(str(uid))
    else:
        print(f"WARNING: CTRP meta not found at {meta_path}; skipping cpd_status probe gate")

    # --- Arm 1: disease-mean sens_z per (disease, drug_uid) -------------------
    dm = (resp.groupby(["disease", "drug_uid"])["sens_z"]
              .agg(disease_sens="mean", n_lines="size").reset_index())
    dm = dm[dm["n_lines"] >= args.min_lines]

    # this line's own sens_z + IC50 (µM) per (cell, drug_uid), averaged across cohorts
    line = resp.copy()
    line["ic50_uM"] = line["ln_ic50"].apply(lambda v: math.exp(v) if pd.notna(v) else None)
    line_agg = (line.groupby(["depmap_id", "drug_uid"])
                    .agg(line_sens=("sens_z", "mean"), ic50_uM=("ic50_uM", "mean")).reset_index())

    # --- Arm 2: permutation-validated signatures per (disease, drug_uid) ------
    # Gate on perm_p (permutation test) — the honest per-test significance — at the strongest-evidence
    # floor, and RANK by perm_z (SDs above the null), which discriminates within the p-floor where perm_p
    # alone is constant. Not FDR-corrected (n_perm=100 can't resolve p small enough for 7,836-way BH).
    sig = pd.read_csv(args.signatures)
    sig = sig[(sig["model"] == "signature") & (sig["perm_p"] <= args.max_perm_p)].copy()
    sig = sig.rename(columns={"spearman": "rho", "n_lines": "sig_n_lines"})
    sig = sig[["disease", "drug_uid", "perm_p", "perm_z", "rho", "sig_n_lines"]]

    n_disease_lines = samples.groupby("disease")["depmap_id"].nunique().to_dict()
    sname = samples.set_index("depmap_id")["cell_line_name"].to_dict()

    def disp(uid):
        return lut.get(uid, (str(uid), None))

    index = []
    n_mech_lines = 0
    n_filtered = 0
    for depmap_id, g in resp.groupby("depmap_id"):
        disease = str(g["disease"].iloc[0])
        pdis = pretty_disease(disease)
        name = str(sname.get(depmap_id, depmap_id))
        # SITE_* and UNABLE-TO-CLASSIFY are heterogeneous fallback buckets, not real cancer types —
        # the biology audit could not assess them; flag so the UI can caveat.
        mixed = disease.startswith("SITE_") or disease.upper() == "UNABLE TO CLASSIFY"

        # Arm 1 — disease ranking, ascending disease_sens (most sensitive first). Filter non-actionable
        # compounds (bare IDs / assay artifacts) BEFORE taking top-N so the list stays full of real drugs.
        dis_rank = dm[dm["disease"] == disease].sort_values("disease_sens")
        this_line = line_agg[line_agg["depmap_id"] == depmap_id].set_index("drug_uid")
        bs = []
        for _, r in dis_rank.iterrows():
            uid = r["drug_uid"]
            drug, target = disp(uid)
            if not drug_ok(drug) or str(uid) in probe_uids:  # broad-spectrum = actionable drugs only
                n_filtered += 1
                continue
            ls = this_line.loc[uid] if uid in this_line.index else None
            bs.append({
                "drug": drug, "target": target,
                "disease_sens": round(float(r["disease_sens"]), 3),
                "line_sens": None if ls is None or pd.isna(ls["line_sens"]) else round(float(ls["line_sens"]), 3),
                "ic50_uM": None if ls is None or pd.isna(ls["ic50_uM"]) else round(float(ls["ic50_uM"]), 4),
                "n_lines": int(r["n_lines"]),
            })
            if len(bs) >= args.top:
                break

        # Arm 2 — validated mechanism signatures for this disease, strongest evidence (perm_z) first
        mech = []
        ms = (sig[sig["disease"] == disease]
              .sort_values(["perm_z", "rho"], ascending=[False, False], na_position="last"))
        for _, r in ms.iterrows():
            uid = r["drug_uid"]
            drug, target = disp(uid)
            if not drug_ok(drug):
                continue
            note = f"Validated expression–sensitivity signature in {pdis.lower()}"
            if target:
                note += f" · target {target}"
            mech.append({
                "drug": drug, "target": target,
                "perm_p": round(float(r["perm_p"]), 4),
                "perm_z": None if pd.isna(r["perm_z"]) else round(float(r["perm_z"]), 2),
                "rho": round(float(r["rho"]), 3),
                "n_lines": int(r["sig_n_lines"]),
                "note": note,
            })
            if len(mech) >= args.mech_top:
                break
        if mech:
            n_mech_lines += 1

        doc = {
            "depmap_id": str(depmap_id), "name": name, "disease": disease,
            "mixed_lineage": mixed,
            "n_drugs": int(g["drug_uid"].nunique()),
            "n_disease_lines": int(n_disease_lines.get(disease, 0)),
            "broad_spectrum": bs, "mechanism": mech,
        }
        (outdir / f"{depmap_id}.json").write_text(json.dumps(sanitize(doc), separators=(",", ":"), allow_nan=False))
        index.append({"depmap_id": str(depmap_id), "name": name, "disease": disease})

    index.sort(key=lambda c: (c["disease"], c["name"]))
    (outdir / "_index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"wrote {len(index)} cell-line JSONs -> {outdir}  ({n_mech_lines} have a mechanism arm; "
          f"{n_filtered} non-actionable broad-spectrum entries filtered)")


if __name__ == "__main__":
    main()
