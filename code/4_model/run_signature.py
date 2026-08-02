"""F2 — per-drug, in-fold sensitivity-signature models (the honest mechanism probe).

For one (disease, drug) at a time (one row per cell line, y = sens_z, leave-cell-line-out), compare
on identical splits:

  mean       - training-mean floor
  enet       - ElasticNet on expression PCs           (the per-drug mechanism reference)
  signature  - Pipeline([SignatureScorer, LinearReg]) : an in-fold Spearman(gene,-sens_z) signature,
               refit inside every train fold, projected onto held-out cells -> leakage-free

The question: for a given drug, does a fold-honest mechanism signature predict held-out sensitivity
(Spearman CI_lo > 0), beating the mean floor and the PC ElasticNet? This is where the mechanism
signal is likeliest to surface (per drug, not pooled). --features {pathway,expr} switches the
signature's input between ssGSEA Reactome scores (fast, interpretable) and top-variance genes.

Runs on the c7g.8xlarge box; parallel across (disease, drug) pairs with per-pair checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402
import evaluate as E      # noqa: E402

OUT = HERE / "out"
CKPT = OUT / "sig_checkpoints"


def _ds(X, y, groups, names, disease, drug):
    return D.Dataset(X=np.asarray(X, float), y=np.asarray(y, float),
                     groups=np.asarray(groups, dtype=object), feature_names=list(names),
                     disease=disease, metric="sens_z", drug=None,
                     meta={"cohort": "both", "n_lines": len(np.unique(groups)), "drug": drug})


def build_drug_datasets(disease, drug_uid, resp, feature_kind, n_pcs, max_genes, seed):
    """Return (ds_pcs, ds_sig, info) for one (disease, drug), one row per cell (mean over cohorts)."""
    r = resp[(resp["disease"] == disease) & (resp["drug_uid"] == drug_uid)]
    y_by_cell = r.groupby("depmap_id")["sens_z"].mean()
    cells = sorted(y_by_cell.index)
    expr = D.load_expression(cells)                       # cells x genes
    cells = [c for c in cells if c in expr.index]
    if len(cells) < 3:
        raise ValueError(f"only {len(cells)} cells with expression")
    y = y_by_cell.loc[cells].to_numpy(float)
    groups = np.array(cells, dtype=object)

    pcs = D.pca_features(expr.loc[cells], n_pcs, seed=seed)
    ds_pcs = _ds(pcs.to_numpy(float), y, groups, list(pcs.columns), disease, drug_uid)

    if feature_kind == "pathway":
        P = D.load_omics_layer("pathway_reactome").reindex(cells)   # cells x pathways
        vals = P.to_numpy(float)
        col_mean = np.nan_to_num(np.nanmean(vals, axis=0))
        nan_r, nan_c = np.where(np.isnan(vals))
        vals[nan_r, nan_c] = np.take(col_mean, nan_c)
        ds_sig = _ds(vals, y, groups, list(P.columns), disease, drug_uid)
        n_feat = vals.shape[1]
    else:  # expr: top-variance genes (label-blind)
        Eg = expr.loc[cells]
        var = Eg.var(axis=0)
        keep = list(var.sort_values(ascending=False).index[:max_genes])
        ds_sig = _ds(Eg[keep].to_numpy(float), y, groups, keep, disease, drug_uid)
        n_feat = len(keep)

    info = {"n_lines": len(cells), "sd_y": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            "n_sig_features": n_feat}
    return ds_pcs, ds_sig, info


def _row(disease, drug_uid, drug_name, model, res, info, extra=None):
    a = res["aggregate"]
    row = {
        "disease": disease, "drug_uid": drug_uid, "drug_name": drug_name, "model": model,
        "n_lines": info["n_lines"], "sd_y": round(info["sd_y"], 3),
        "n_sig_features": info["n_sig_features"],
        "spearman": a["spearman"]["mean"], "spearman_lo": a["spearman"]["ci95_lo"],
        "spearman_hi": a["spearman"]["ci95_hi"], "rmse": a["rmse"]["mean"], "r2": a["r2"]["mean"],
    }
    if extra:
        row.update(extra)
    return row


def _permutation_null(ds_sig, cfg):
    """Empirical null for the signature's held-out Spearman: shuffle y n_perm times, 1 repeat each.
    The repeated-CV CI measures split-stability, NOT sampling noise — this calibrates real vs chance."""
    if not cfg["n_perm"]:
        return {}
    rng = np.random.default_rng(cfg["seed"] + 1)
    perms = []
    for _ in range(cfg["n_perm"]):
        yp = ds_sig.y.copy()
        rng.shuffle(yp)
        ps = E.nested_cv(replace(ds_sig, y=yp), "signature", cfg["n_outer"], cfg["n_inner"], 0, 1,
                         cfg["seed"])
        perms.append(ps["aggregate"]["spearman"]["mean"])
    return {"perm": np.array([p for p in perms if np.isfinite(p)], dtype=float)}


def run_one(disease, drug_uid, drug_name, cfg):
    key = f"{disease.replace('/', '-')}__{drug_uid}__{cfg['features']}"
    ckpt = CKPT / f"{key}.json"
    if ckpt.exists() and not cfg["overwrite"]:
        try:
            return json.loads(ckpt.read_text())["rows"]
        except Exception:
            pass
    t0 = time.time()
    try:
        resp = cfg["_resp"]
        ds_pcs, ds_sig, info = build_drug_datasets(
            disease, drug_uid, resp, cfg["features"], cfg["n_pcs"], cfg["max_genes"], cfg["seed"])
        no, ni, nr, seed = cfg["n_outer"], cfg["n_inner"], cfg["n_repeats"], cfg["seed"]
        rows = []
        mean = E.nested_cv(ds_pcs, "mean", no, ni, 0, nr, seed)
        rows.append(_row(disease, drug_uid, drug_name, "mean", mean, info))
        en = E.nested_cv(ds_pcs, "enet", no, ni, cfg["n_iter"], nr, seed)
        rows.append(_row(disease, drug_uid, drug_name, "enet", en, info))
        sg = E.nested_cv(ds_sig, "signature", no, ni, 0, nr, seed)
        extra = {}
        null = _permutation_null(ds_sig, cfg)
        if "perm" in null and len(null["perm"]):
            perm = null["perm"]
            real = sg["aggregate"]["spearman"]["mean"]
            extra = {
                "perm_mean": round(float(perm.mean()), 4), "perm_sd": round(float(perm.std()), 4),
                "perm_p": round((1 + int(np.sum(perm >= real))) / (1 + len(perm)), 4),
                "perm_z": round(float((real - perm.mean()) / (perm.std() + 1e-9)), 3),
                "n_perm": int(len(perm)),
            }
        rows.append(_row(disease, drug_uid, drug_name, "signature", sg, info, extra=extra))
        payload = {"disease": disease, "drug_uid": drug_uid, "ok": True,
                   "secs": round(time.time() - t0, 1), "rows": rows}
    except Exception as e:
        payload = {"disease": disease, "drug_uid": drug_uid, "ok": False,
                   "secs": round(time.time() - t0, 1), "error": str(e),
                   "trace": traceback.format_exc(), "rows": []}
    ckpt.write_text(json.dumps(payload, indent=2))
    status = "ok" if payload["ok"] else f"SKIP/ERR ({payload.get('error', '')[:50]})"
    print(f"[{disease}/{drug_name}] {status} in {payload['secs']}s", flush=True)
    return payload["rows"]


def build_pairs(cfg):
    r = D.load_response()
    r = r[r["depmap_id"].isin(D._expr_columns())]
    if cfg["diseases"] and cfg["diseases"] != ["auto"]:
        r = r[r["disease"].isin(cfg["diseases"])]
    grp = r.groupby(["disease", "drug_uid"]).agg(
        n_lines=("depmap_id", "nunique"), sd_y=("sens_z", "std"),
        drug_name=("drug_name", "first")).reset_index()
    grp = grp[(grp["n_lines"] >= cfg["min_lines"]) & (grp["sd_y"] >= cfg["min_sd"])]
    if cfg["drugs"]:
        pat = "|".join(cfg["drugs"])
        grp = grp[grp["drug_name"].str.contains(pat, case=False, na=False)]
    grp = grp.sort_values("n_lines", ascending=False)
    if cfg["per_disease"]:                       # top-N drugs per disease -> breadth across diseases
        grp = grp.groupby("disease", group_keys=False).head(cfg["per_disease"])
    elif cfg["max_pairs"] and len(grp) > cfg["max_pairs"]:
        grp = grp.head(cfg["max_pairs"])
    return grp, r


def summarize(df: pd.DataFrame) -> str:
    out = ["# Per-drug in-fold signature — summary (F2)\n"]
    out.append(f"- pairs: {df[['disease','drug_uid']].drop_duplicates().shape[0]}  |  "
               f"diseases: {df['disease'].nunique()}  |  rows: {len(df)}")
    sig = df[df["model"] == "signature"].copy()
    enet = df[df["model"] == "enet"].set_index(["disease", "drug_uid"])["spearman"]
    mn = df[df["model"] == "mean"].set_index(["disease", "drug_uid"])["spearman"]
    has_perm = "perm_p" in sig.columns
    out.append(f"- signature mean Spearman: {sig['spearman'].mean():+.3f}  |  "
               f"beats per-drug PC-enet in "
               f"{int((sig.set_index(['disease','drug_uid'])['spearman'] > enet).sum())}/{len(sig)}")
    if has_perm:
        sig_hit = sig[sig["perm_p"] < 0.05]
        out.append(f"- **permutation-significant (perm_p < 0.05): {len(sig_hit)}/{len(sig)} "
                   f"(disease,drug) pairs** — the HONEST count (null = shuffled sens_z)")
        out.append(f"- (naive repeated-CV CI_lo > 0 would claim "
                   f"{int((sig['spearman_lo'] > 0).sum())}/{len(sig)} — inflated; see permutation note)\n")
    else:
        out.append(f"- signature 95% CI_lo > 0: {int((sig['spearman_lo'] > 0).sum())}/{len(sig)} "
                   f"(NOTE: repeated-CV CI understates the null — run with --n-perm for honest counts)\n")

    cols = "| disease | drug | lines | signature | " + ("perm_p | perm_z | " if has_perm else "CI_lo | ") + "enet(PCs) | mean |"
    out.append("## Top pairs by signature Spearman\n")
    out.append(cols)
    out.append("|---|---|--:|--:|--:|" + ("--:|" if has_perm else "") + "--:|--:|")
    for _, row in sig.sort_values("spearman", ascending=False).head(35).iterrows():
        kdx = (row["disease"], row["drug_uid"])
        e = float(enet.get(kdx, float("nan"))); mm = float(mn.get(kdx, float("nan")))
        mid = (f"{row['perm_p']:.3f} | {row.get('perm_z', float('nan')):+.2f} | " if has_perm
               else f"{row['spearman_lo']:+.3f} | ")
        out.append(f"| {row['disease']} | {row['drug_name']} | {int(row['n_lines'])} | "
                   f"{row['spearman']:+.3f} | {mid}{e:+.3f} | {mm:+.3f} |")
    return "\n".join(out) + "\n"


def _summarize_from_checkpoints(tag: str):
    """Reduce step: rebuild signature_results/summary from out/sig_checkpoints/*.json.

    The map (one pod per (disease, drug)) writes a per-pair checkpoint; main() reduces from
    in-process returns, which a distributed run never has. This globs the checkpoint dir instead.
    """
    files = sorted(CKPT.glob("*.json"))
    rows, ok = [], 0
    for f in files:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        if payload.get("ok") and payload.get("rows"):
            rows.extend(payload["rows"]); ok += 1
    if not rows:
        print(f"no rows in {len(files)} checkpoints under {CKPT}", flush=True)
        return
    df = pd.DataFrame(rows)
    suffix = f"_{tag}"
    df.to_csv(OUT / f"signature_results{suffix}.csv", index=False)
    (OUT / f"signature_summary{suffix}.md").write_text(summarize(df))
    print(f"reduced {ok}/{len(files)} checkpoints -> {len(df)} rows -> "
          f"{OUT}/signature_results{suffix}.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["pathway", "expr"], default="pathway")
    ap.add_argument("--diseases", default="auto", help="'auto' or comma list of TCGA codes")
    ap.add_argument("--drugs", default="", help="comma name substrings to restrict to (e.g. dabrafenib,PLX4720)")
    ap.add_argument("--min-lines", type=int, default=20)
    ap.add_argument("--min-sd", type=float, default=0.3)
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--per-disease", type=int, default=0,
                    help="top-N drugs per disease (breadth); overrides --max-pairs when >0")
    ap.add_argument("--n-perm", type=int, default=0,
                    help="permutations per drug for an honest null (e.g. 20); 0 = skip")
    ap.add_argument("--n-pcs", type=int, default=20)
    ap.add_argument("--max-genes", type=int, default=8000)
    ap.add_argument("--n-outer", type=int, default=5)
    ap.add_argument("--n-inner", type=int, default=4)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", default="",
                    help="run ONE unit given as JSON e.g. '{\"disease\":\"SKCM\",\"drug_uid\":\"YZDJ...\"}', "
                         "write its checkpoint, and exit (the Argo fan-out entry point)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip compute; rebuild combined outputs from out/sig_checkpoints/*.json (reduce)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "features": args.features,
        "diseases": [d.strip() for d in args.diseases.split(",")] if args.diseases else ["auto"],
        "drugs": [d.strip() for d in args.drugs.split(",") if d.strip()],
        "min_lines": args.min_lines, "min_sd": args.min_sd, "max_pairs": args.max_pairs,
        "per_disease": args.per_disease, "n_perm": args.n_perm,
        "n_pcs": args.n_pcs, "max_genes": args.max_genes,
        "n_outer": args.n_outer, "n_inner": args.n_inner, "n_repeats": args.n_repeats,
        "n_iter": args.n_iter, "seed": args.seed, "overwrite": args.overwrite,
    }
    if args.summarize_only:                     # reduce: aggregate the fleet's checkpoints
        _summarize_from_checkpoints(args.tag or args.features)
        return
    if args.shard:                              # map: one shard -> its per-drug checkpoint(s)
        unit = json.loads(args.shard)
        resp = D.load_response()
        resp = resp[resp["depmap_id"].isin(D._expr_columns())]
        cfg["_resp"] = resp
        disease = unit["disease"]
        # A shard is either one drug {disease, drug_uid} or a batch {disease, drug_uids:[...]}. A batch
        # loads the response store ONCE, then runs its drugs IN PARALLEL (n_jobs loky workers) — each
        # drug is independent, single-process, and writes its own checkpoint, so --summarize-only
        # aggregates them identically. This is F2's answer to the perm-null bottleneck: drug-level
        # parallelism saturates the pod's cores (batches are ~25 drugs). Keep OMP_NUM_THREADS=1 so
        # n_jobs workers == n_jobs cores (no BLAS oversubscription); run_one's inner perm loop stays
        # serial (1 core/drug), so there is no nested-parallel oversubscription.
        uids = unit.get("drug_uids") or [unit["drug_uid"]]

        def _name(uid):
            sub = resp[(resp["disease"] == disease) & (resp["drug_uid"] == uid)]
            return sub["drug_name"].iloc[0] if len(sub) else uid

        njobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
        Parallel(n_jobs=njobs, backend="loky")(
            delayed(run_one)(disease, uid, _name(uid), cfg) for uid in uids
        )
        return
    pairs, resp = build_pairs(cfg)
    cfg["_resp"] = resp
    njobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    print(f"cores={os.cpu_count()} n_jobs={njobs} pairs={len(pairs)} features={args.features} "
          f"(diseases={pairs['disease'].nunique()})", flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=njobs, backend="loky", verbose=5)(
        delayed(run_one)(row["disease"], row["drug_uid"], row["drug_name"], cfg)
        for _, row in pairs.iterrows()
    )
    rows = [r for group in results for r in group]
    if not rows:
        print("no results produced — check checkpoints", flush=True)
        return
    df = pd.DataFrame(rows)
    suffix = f"_{args.tag}" if args.tag else f"_{args.features}"
    df.to_csv(OUT / f"signature_results{suffix}.csv", index=False)
    (OUT / f"signature_summary{suffix}.md").write_text(summarize(df))
    print(f"\nDONE {len(df)} rows in {time.time()-t0:.0f}s -> {OUT}/signature_results{suffix}.csv",
          flush=True)
    print(summarize(df), flush=True)


if __name__ == "__main__":
    main()
