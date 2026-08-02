"""First modeling pass: the drug-mean-baselined ablation on the harmonized store.

For each disease (and pan-cancer) x cohort {gdsc, ctrp, both}, under honest grouped nested CV
(leave-cell-line-out, repeated for CIs), compare on identical splits:

  mean        - predict the training mean            (the floor)
  drugmean    - predict each drug's train-fold mean  (the HONEST baseline; drug main effects)
  enet        - StandardScaler -> ElasticNet on [expression PCs | drug one-hot], y = sens_z
  enet_resid  - same, but y residualized against in-fold drug means (mechanism-specific arm)

Optional (--omics): repeat enet / enet_resid with genetics blocks appended, to ask whether
mutations/fusions/methylation beat expression-only.

The three questions this answers:
  (1) does expression add anything OVER the drug-mean baseline?          enet   vs drugmean
  (2) is there personalization signal after removing drug main effects?  enet_resid  (> 0?)
  (3) does pooling GDSC u CTRP shrink the CIs / lift skill?              both vs gdsc vs ctrp

Parallelized across (disease, cohort); each writes a JSON checkpoint so a spot interruption
resumes instead of restarting. Run on the c7g.8xlarge box (see docs/AWS_EC2_COMMANDS_WALKTHROUGH §13).
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
import models as M        # noqa: E402

OUT = HERE / "out"
CKPT = OUT / "checkpoints"


def _row(ds, label, res, extra=None):
    a = res["aggregate"]
    row = {
        "disease": ds.disease,
        "cohort": ds.meta["cohort"],
        "model": label,
        "residualized": bool(res.get("residualized", False)),
        "n_rows": res["n_rows"],
        "n_lines": ds.meta["n_lines"],
        "n_drugs": ds.meta["n_drugs"],
        "n_pcs": ds.meta["n_pcs"],
        "spearman": a["spearman"]["mean"],
        "spearman_lo": a["spearman"]["ci95_lo"],
        "spearman_hi": a["spearman"]["ci95_hi"],
        "pearson": a["pearson"]["mean"],
        "rmse": a["rmse"]["mean"],
        "r2": a["r2"]["mean"],
    }
    if extra:
        row.update(extra)
    return row


def _one_perm(ds, arm, no, ni, niter, seed, residualize, perm_seed):
    """One permutation replicate: shuffle y under an independent seed, rerun the arm's held-out CV."""
    yp = ds.y.copy()
    np.random.default_rng(perm_seed).shuffle(yp)
    ps = E.nested_cv(replace(ds, y=yp), arm, no, ni, niter, 1, seed, residualize=residualize)
    return ps["aggregate"]["spearman"]["mean"]


def _permutation_null_arm(ds, arm, cfg, residualize=True):
    """Empirical null for a pathway arm's held-out Spearman: shuffle y n_perm times, 1 repeat each.

    Mirrors run_signature._permutation_null. The repeated-CV CI measures split-stability, NOT
    sampling noise, so it overstates significance (~3x overcount in F2); this calibrates real vs
    chance. Shuffling raw sens_z breaks the feature->y link while the in-fold residualization still
    runs — so the null centers near 0 for a genuinely uninformative arm.

    The n_perm replicates are independent and DOMINATE per-shard cost (~15x the real arms combined),
    so they run in parallel across `cfg['perm_jobs']` loky workers. Each replicate draws an independent
    SeedSequence child, so the null is deterministic regardless of worker count/order. Keep BLAS at 1
    thread/worker (OMP_NUM_THREADS=1 in the env) so perm_jobs workers use perm_jobs cores — not
    perm_jobs x BLAS, the Graviton oversubscription trap that made the serial run take ~a day.
    """
    n_perm = cfg.get("n_perm", 0)
    if not n_perm:
        return None
    no, ni, niter, seed = cfg["n_outer"], cfg["n_inner"], cfg["n_iter"], cfg["seed"]
    seeds = np.random.SeedSequence(seed + 1).spawn(n_perm)
    perms = Parallel(n_jobs=cfg.get("perm_jobs", 1), backend="loky")(
        delayed(_one_perm)(ds, arm, no, ni, niter, seed, residualize, s) for s in seeds
    )
    return np.array([p for p in perms if p is not None and np.isfinite(p)], dtype=float)


def _perm_extra(real, perm):
    """Turn a held-out Spearman + its permutation null into the significance columns (or {})."""
    if perm is None or not len(perm):
        return {}
    return {
        "perm_mean": round(float(perm.mean()), 4), "perm_sd": round(float(perm.std()), 4),
        "perm_p": round((1 + int(np.sum(perm >= real))) / (1 + len(perm)), 4),
        "perm_z": round(float((real - perm.mean()) / (perm.std() + 1e-9)), 3),
        "n_perm": int(len(perm)),
    }


def _permute_pcs_within_disease(ds, perm_seed):
    """Shuffle the line->PC-vector mapping WITHIN each disease.

    The right null for "expression personalizes BEYOND tissue": it preserves tissue, drug, and the
    (disease, drug) cell means (which drive the residualized target), and breaks only the link
    between a line's expression and its own response. A globally-shuffled-y null (the pathway arms'
    `_one_perm`) instead destroys tissue and drug structure together, so it cannot isolate the
    beyond-tissue claim. Rows of one line share a PC vector, so we remap at line granularity.
    """
    X = np.asarray(ds.X)
    groups = np.asarray(ds.groups)
    disease = np.asarray(ds.disease_arr)
    rng = np.random.default_rng(perm_seed)
    lines, first_idx = np.unique(groups, return_index=True)
    first_of = {ln: i for ln, i in zip(lines, first_idx)}
    line_dis = {ln: disease[i] for ln, i in zip(lines, first_idx)}
    mapping = {}
    by_dis = {}
    for ln in lines:
        by_dis.setdefault(line_dis[ln], []).append(ln)
    for lns in by_dis.values():
        perm = list(lns)
        rng.shuffle(perm)
        mapping.update(dict(zip(lns, perm)))
    src = np.fromiter((first_of[mapping[ln]] for ln in groups), dtype=np.int64, count=len(groups))
    return X[src]


def _one_tissue_perm(ds, cfg, perm_seed):
    """One within-disease PC-permutation replicate of the tissue-free arm (1 repeat)."""
    Xp = _permute_pcs_within_disease(ds, perm_seed)
    ps = E.nested_cv(replace(ds, X=Xp), "enet", cfg["n_outer"], cfg["n_inner"],
                     cfg["n_iter"], 1, cfg["seed"], residualize=True, residualize_by="disease_drug")
    return ps["aggregate"]["spearman"]["mean"]


def _tissue_perm_null(ds, cfg):
    """Empirical beyond-tissue null for `enet_resid_disease`: `n_perm` within-disease PC shuffles.

    Deterministic across worker counts (independent SeedSequence children). Dominates per-shard
    cost; runs on `perm_jobs` loky workers with BLAS pinned to 1 thread (OMP_NUM_THREADS=1) to
    avoid the Graviton oversubscription trap. Returns None when --n-perm 0."""
    n_perm = cfg.get("n_perm", 0)
    if not n_perm:
        return None
    seeds = np.random.SeedSequence(cfg["seed"] + 2).spawn(n_perm)
    perms = Parallel(n_jobs=cfg.get("perm_jobs", 1), backend="loky")(
        delayed(_one_tissue_perm)(ds, cfg, s) for s in seeds
    )
    return np.array([p for p in perms if p is not None and np.isfinite(p)], dtype=float)


def _eval_models(ds, cfg):
    """Evaluate the broad-spectrum baseline and the mechanism-specific arms on one dataset.

    Two scientific arms (per the M0 finding + the two-arm recommender goal):
      * BROAD-SPECTRUM  -> `drugmean`: rank drugs by their within-disease mean sensitivity.
                           Fast (a per-drug mean; the wide one-hot only slows ITERATIVE fits).
      * MECHANISM        -> `enet_resid`: ElasticNet on expression PCs predicting the drug-mean-
                           RESIDUALIZED y (in-fold means, no leakage). This is the personalization
                           signal that survives after drug main effects. PCs-only -> fast; the
                           665-col drug one-hot is dropped here (residualize uses ds.drug labels).
      * `enet_expr`      -> ElasticNet on PCs predicting raw sens_z (no drug term) for reference.
      * `enet_resid_<omics>` -> mechanism arm with genetics blocks appended (does genetics add?).
    """
    rows = []
    no, ni, nr, seed, niter, topk = (cfg["n_outer"], cfg["n_inner"], cfg["n_repeats"],
                                     cfg["seed"], cfg["n_iter"], cfg["topk"])
    ds_pcs = replace(ds, X=ds.X[:, :ds.n_pcs].copy(), feature_names=ds.feature_names[:ds.n_pcs])

    mean = E.nested_cv(ds_pcs, "mean", no, ni, 0, nr, seed)
    rows.append(_row(ds, "mean", mean))

    npc, ndr = ds.n_pcs, ds.meta["n_drugs"]
    fac = lambda: M.DrugMeanRegressor(n_pcs=npc, n_drugs=ndr)  # noqa: E731
    dm = E.nested_cv(ds, "drugmean", no, ni, 0, nr, seed, estimator_factory=fac)
    rank = E.ranking_metrics(ds, dm["oof_pred"], k=topk)
    rows.append(_row(ds, "drugmean", dm,
                     extra={f"rank_{k}": v for k, v in rank.items() if isinstance(v, (int, float))}))

    er = E.nested_cv(ds_pcs, "enet", no, ni, niter, nr, seed, residualize=True)
    rows.append(_row(ds, "enet_resid", er))

    ex = E.nested_cv(ds_pcs, "enet", no, ni, niter, nr, seed, residualize=False)
    rows.append(_row(ds, "enet_expr", ex))

    # TISSUE-FREE mechanism arm — only meaningful when >1 disease is pooled (else disease×drug
    # collapses to drug). Residualize on the (disease, drug) cell mean to strip tissue BEFORE the
    # model sees y; score against a within-disease PC-permutation null (beyond-tissue significance).
    if ds.disease_arr is not None and ds.meta.get("n_diseases", 1) > 1:
        erd = E.nested_cv(ds_pcs, "enet", no, ni, niter, nr, seed,
                          residualize=True, residualize_by="disease_drug")
        erd_extra = _perm_extra(erd["aggregate"]["spearman"]["mean"], _tissue_perm_null(ds_pcs, cfg))
        rows.append(_row(ds, "enet_resid_disease", erd, extra=erd_extra))

        # tissue oracle: predict the drug-residualized target from the (disease, drug) cell mean
        # with NO expression — the ceiling of recovered lineage, and the honest within-tissue
        # broad-spectrum baseline (pan-cancer `drugmean` collapses because sens_z is per-drug z).
        dd = E.tissue_oracle_cv(ds_pcs, no, nr, seed)
        rows.append(_row(ds, "disease_drugmean", dd))

    if cfg["omics"] and ds.meta["cohort"] in cfg["omics_cohorts"]:
        ds_gen = D.attach_omics(ds_pcs, cfg["omics"])
        eg = E.nested_cv(ds_gen, "enet", no, ni, niter, nr, seed, residualize=True)
        rows.append(_row(ds, f"enet_resid_{cfg['omics_tag']}", eg,
                         extra={"n_features": ds_gen.X.shape[1]}))

    # pathway-aware mechanism arms (F1): ssGSEA Reactome activity replacing / augmenting the PCs.
    # Leakage-safe (label-blind, like PCA); residualize=True still removes drug means in-fold.
    if cfg["pathway"] and ds.meta["cohort"] in cfg["pathway_cohorts"]:
        layer = cfg["pathway"][0]
        # fast, dimensionality-matched arm: PCA(pathways -> n_pcs) vs the gene-PC arm.
        ds_ppc = D.pathway_pcs(ds_pcs, layer, cfg["n_pcs"], seed)
        eppc = E.nested_cv(ds_ppc, "enet", no, ni, niter, nr, seed, residualize=True)
        rows.append(_row(ds, "enet_pathwaypcs", eppc, extra={"n_features": ds_ppc.X.shape[1]}))

        if not cfg.get("pathway_pcs_only"):
            base0 = replace(ds_pcs, X=np.empty((ds_pcs.n_rows, 0), dtype=np.float64), feature_names=[])
            ds_path = D.attach_omics(base0, cfg["pathway"])      # raw pathways REPLACE the PCs (L1-select)
            ep = E.nested_cv(ds_path, "enet", no, ni, niter, nr, seed, residualize=True)
            ep_extra = {"n_features": ds_path.X.shape[1]}
            ep_extra.update(_perm_extra(ep["aggregate"]["spearman"]["mean"],
                                        _permutation_null_arm(ds_path, "enet", cfg)))
            rows.append(_row(ds, "enet_pathway", ep, extra=ep_extra))

            ds_pcs_path = D.attach_omics(ds_pcs, cfg["pathway"])  # PCs + raw pathways
            epp = E.nested_cv(ds_pcs_path, "enet", no, ni, niter, nr, seed, residualize=True)
            epp_extra = {"n_features": ds_pcs_path.X.shape[1]}
            epp_extra.update(_perm_extra(epp["aggregate"]["spearman"]["mean"],
                                         _permutation_null_arm(ds_pcs_path, "enet", cfg)))
            rows.append(_row(ds, "enet_pcs_pathway", epp, extra=epp_extra))
    return rows


def run_one(disease, cohort, cfg):
    """Build one (disease, cohort) dataset and evaluate; checkpoint + resume-aware."""
    key = f"{disease.replace('/', '-')}__{cohort}"
    ckpt = CKPT / f"{key}.json"
    if ckpt.exists() and not cfg["overwrite"]:
        try:
            return json.loads(ckpt.read_text())["rows"]
        except Exception:
            pass
    t0 = time.time()
    try:
        ds = D.pooled(disease, cohort=cohort, n_pcs=cfg["n_pcs"],
                      min_lines_per_drug=cfg["min_lines_per_drug"], seed=cfg["seed"],
                      min_lines_per_cell=cfg["min_lines_per_cell"])
        if ds.n_groups < cfg["min_lines"] or ds.meta["n_drugs"] < cfg["min_drugs"]:
            raise ValueError(f"too small: {ds.n_groups} lines x {ds.meta['n_drugs']} drugs")
        rows = _eval_models(ds, cfg)
        payload = {"disease": disease, "cohort": cohort, "ok": True,
                   "secs": round(time.time() - t0, 1), "rows": rows}
    except Exception as e:
        payload = {"disease": disease, "cohort": cohort, "ok": False,
                   "secs": round(time.time() - t0, 1), "error": str(e),
                   "trace": traceback.format_exc(), "rows": []}
    ckpt.write_text(json.dumps(payload, indent=2))
    status = "ok" if payload["ok"] else f"SKIP/ERR ({payload.get('error','')[:60]})"
    print(f"[{key}] {status} in {payload['secs']}s", flush=True)
    return payload["rows"]


def build_tasks(cfg):
    sizes = D.disease_sizes(min_lines=cfg["min_lines"])
    diseases = list(sizes.index)
    if cfg["diseases"] and cfg["diseases"] != ["auto"]:
        diseases = cfg["diseases"]
    if cfg["pancancer"]:
        diseases = ["pancancer"] + diseases
    tasks = [(d, c) for d in diseases for c in cfg["cohorts"]]
    return tasks, sizes


def summarize(df: pd.DataFrame, omics_tag: str = "genetics") -> str:
    """Markdown headline: the broad-spectrum baseline vs the mechanism arm, and the pooling effect."""
    gen_model = f"enet_resid_{omics_tag}"
    out = ["# Ablation summary — broad-spectrum baseline vs mechanism-specific arm\n"]
    out.append(f"- rows: {len(df)}  |  diseases: {df['disease'].nunique()}  |  "
               f"cohorts: {sorted(df['cohort'].unique())}")
    out.append("- **drugmean** = broad-spectrum (rank drugs by within-disease mean sensitivity).")
    out.append("- **enet_resid** = mechanism-specific (expression PCs predicting the drug-mean-"
               "residualized sensitivity; >0 ⇒ personalization signal survives).")
    out.append(f"- **{gen_model}** = mechanism arm with genetics (mutations/fusions/methylation) added.\n")

    def pick(sub, model, col="spearman"):
        r = sub[sub["model"] == model]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    out.append("## `both` cohort — leave-cell-line-out nested CV, Spearman (mean over repeats)\n")
    out.append("| disease | lines | drugs | drugmean (broad) | enet_resid (mech) | "
               f"{gen_model} | Δ genetics |")
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    both = df[df["cohort"] == "both"]
    for dis in both["disease"].drop_duplicates():
        sub = both[both["disease"] == dis]
        dm, er, eg = pick(sub, "drugmean"), pick(sub, "enet_resid"), pick(sub, gen_model)
        nl = int(sub["n_lines"].iloc[0]); nd = int(sub["n_drugs"].iloc[0])
        dgen = eg - er
        out.append(f"| {dis} | {nl} | {nd} | {dm:.3f} | {er:+.3f} | "
                   f"{eg:+.3f} | {dgen:+.3f} |")

    out.append("\n## Pooling effect — mechanism arm (`enet_resid`) Spearman + CI width by cohort\n")
    out.append("| cohort | n diseases | mean Spearman | mean 95% CI width |")
    out.append("|---|--:|--:|--:|")
    for c in ["gdsc", "ctrp", "both"]:
        sub = df[(df["cohort"] == c) & (df["model"] == "enet_resid")]
        if not len(sub):
            continue
        width = (sub["spearman_hi"] - sub["spearman_lo"]).mean()
        out.append(f"| {c} | {sub['disease'].nunique()} | {sub['spearman'].mean():+.3f} | {width:.3f} |")

    # count where mechanism arm CI is clearly > 0 (personalization signal present)
    mech = df[(df["cohort"] == "both") & (df["model"] == "enet_resid")]
    pos = int((mech["spearman_lo"] > 0).sum())
    out.append(f"\n**Mechanism signal:** in `both`, {pos}/{len(mech)} diseases have enet_resid "
               f"95% CI lower bound > 0 (expression personalizes beyond drug means).")

    # tissue-free mechanism arm (disease×drug residualization), if run
    if df["model"].isin(["enet_resid_disease"]).any():
        out.append("\n## Tissue-free mechanism arm — disease×drug residualized\n")
        out.append("Removes tissue-of-origin *before* the expression model sees the target. "
                   "`enet_resid` (drug-residualized) is confounded with lineage; `enet_resid_disease` "
                   "is the honest within-line estimate; `disease_drugmean` is the no-expression tissue "
                   "oracle (its ceiling). Report **perm_p** (beyond-tissue null), never CI_lo>0.\n")
        out.append("| cohort | enet_resid (drug) | enet_resid_disease (tissue-free) | "
                   "disease_drugmean (tissue oracle) | perm_p | perm_z |")
        out.append("|---|--:|--:|--:|--:|--:|")
        for c in ["gdsc", "ctrp", "both"]:
            sub = df[df["cohort"] == c]
            prow = sub[sub["model"] == "enet_resid_disease"]
            if not len(prow):
                continue
            pp = prow["perm_p"].iloc[0] if "perm_p" in prow.columns else float("nan")
            pz = prow["perm_z"].iloc[0] if "perm_z" in prow.columns else float("nan")
            pp = f"{pp:.4f}" if pd.notna(pp) else "n/a"
            pz = f"{pz:+.1f}" if pd.notna(pz) else "n/a"
            out.append(f"| {c} | {pick(sub, 'enet_resid'):+.3f} | {pick(sub, 'enet_resid_disease'):+.3f} "
                       f"| {pick(sub, 'disease_drugmean'):+.3f} | {pp} | {pz} |")

    # pathway-aware arms (F1), if run
    pw_models = ["enet_pathwaypcs", "enet_pathway", "enet_pcs_pathway"]
    if df["model"].isin(pw_models).any():
        out.append("\n## Pathway-aware mechanism arm — `both` cohort, Spearman (mean over repeats)\n")
        out.append("| disease | lines | drugs | enet_resid (gene-PCs) | enet_pathwaypcs | "
                   "enet_pathway (raw) | enet_pcs_pathway | Δ (pathwaypcs − gene-PCs) |")
        out.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        both = df[df["cohort"] == "both"]
        for dis in both["disease"].drop_duplicates():
            sub = both[both["disease"] == dis]
            er = pick(sub, "enet_resid"); eppc = pick(sub, "enet_pathwaypcs")
            ep = pick(sub, "enet_pathway"); epp = pick(sub, "enet_pcs_pathway")
            nl = int(sub["n_lines"].iloc[0]); nd = int(sub["n_drugs"].iloc[0])
            out.append(f"| {dis} | {nl} | {nd} | {er:+.3f} | {eppc:+.3f} | {ep:+.3f} | "
                       f"{epp:+.3f} | {eppc - er:+.3f} |")
        for mname in ["enet_pathwaypcs", "enet_pathway"]:
            pw = df[(df["cohort"] == "both") & (df["model"] == mname)]
            if len(pw):
                npos = int((pw["spearman_lo"] > 0).sum())
                out.append(f"\n**{mname} signal:** in `both`, {npos}/{len(pw)} diseases have "
                           f"95% CI lower bound > 0.")
    return "\n".join(out) + "\n"


def _summarize_from_checkpoints(tag: str, omics_tag: str):
    """Reduce step: rebuild the combined outputs from per-unit checkpoints.

    The map (many pods, one `run_one` each) writes `out/checkpoints/<unit>.json`; the monolithic
    main() reduces from in-process returns, which a distributed run never has. This globs the
    checkpoint dir instead, so a fleet's shards aggregate the same way — same summarize(), same files.
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
    suffix = f"_{tag}" if tag else ""
    df.to_csv(OUT / f"ablation_results{suffix}.csv", index=False)
    df.to_json(OUT / f"ablation_results{suffix}.json", orient="records", indent=2)
    (OUT / f"ablation_summary{suffix}.md").write_text(summarize(df, omics_tag))
    print(f"reduced {ok}/{len(files)} checkpoints -> {len(df)} rows -> "
          f"{OUT}/ablation_results{suffix}.csv", flush=True)
    print(summarize(df, omics_tag), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diseases", default="auto", help="'auto' or comma list of TCGA codes")
    ap.add_argument("--cohorts", default="gdsc,ctrp,both")
    ap.add_argument("--pancancer", action="store_true", help="also run the pooled pan-cancer model")
    ap.add_argument("--min-lines", type=int, default=25)
    ap.add_argument("--min-drugs", type=int, default=10)
    ap.add_argument("--min-lines-per-drug", type=int, default=5)
    ap.add_argument("--min-lines-per-cell", type=int, default=1,
                    help="drop (disease,drug) cells with fewer distinct lines (tissue-free arm; "
                         "pass 5 for the pan-cancer disease×drug run). Default 1 = no-op.")
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--n-outer", type=int, default=5)
    ap.add_argument("--n-inner", type=int, default=4)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--omics", default="", help="comma layers to add on omics-cohorts, e.g. mutations,fusions,methylation")
    ap.add_argument("--omics-cohorts", default="both")
    ap.add_argument("--omics-tag", default="genetics")
    ap.add_argument("--pathway", default="", help="comma pathway layers (F1 arms), e.g. pathway_reactome")
    ap.add_argument("--pathway-cohorts", default="both")
    ap.add_argument("--pathway-pcs-only", action="store_true",
                    help="only the fast PCA-of-pathways arm; skip the raw ~1800-feature L1 arms")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--n-perm", type=int, default=0,
                    help="permutations for the raw-pathway arms' honest null (0 = skip); needs --pathway")
    ap.add_argument("--perm-jobs", type=int, default=1,
                    help="parallel workers for the perm-null loop (the per-shard bottleneck). Set to the "
                         "pod's core count with OMP_NUM_THREADS=1 to use all cores without oversubscription.")
    ap.add_argument("--shard", default="",
                    help="run ONE unit given as JSON e.g. '{\"disease\":\"SKCM\",\"cohort\":\"both\"}', "
                         "write its checkpoint, and exit (the Argo fan-out entry point)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip compute; rebuild combined outputs from out/checkpoints/*.json (reduce step)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "diseases": [d.strip() for d in args.diseases.split(",")] if args.diseases else ["auto"],
        "cohorts": [c.strip() for c in args.cohorts.split(",") if c.strip()],
        "pancancer": args.pancancer,
        "min_lines": args.min_lines, "min_drugs": args.min_drugs,
        "min_lines_per_drug": args.min_lines_per_drug,
        "min_lines_per_cell": args.min_lines_per_cell, "n_pcs": args.n_pcs,
        "n_outer": args.n_outer, "n_inner": args.n_inner, "n_repeats": args.n_repeats,
        "n_iter": args.n_iter, "topk": args.topk, "seed": args.seed,
        "omics": [x.strip() for x in args.omics.split(",") if x.strip()],
        "omics_cohorts": [c.strip() for c in args.omics_cohorts.split(",") if c.strip()],
        "omics_tag": args.omics_tag, "overwrite": args.overwrite,
        "pathway": [x.strip() for x in args.pathway.split(",") if x.strip()],
        "pathway_cohorts": [c.strip() for c in args.pathway_cohorts.split(",") if c.strip()],
        "pathway_pcs_only": args.pathway_pcs_only, "n_perm": args.n_perm,
        "perm_jobs": args.perm_jobs,
    }
    if args.summarize_only:                     # reduce: aggregate the fleet's checkpoints
        _summarize_from_checkpoints(args.tag, args.omics_tag)
        return
    if args.shard:                              # map: one (disease, cohort) unit -> its checkpoint
        unit = json.loads(args.shard)
        run_one(unit["disease"], unit["cohort"], cfg)
        return
    tasks, sizes = build_tasks(cfg)
    njobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    print(f"cores={os.cpu_count()}  n_jobs={njobs}  tasks={len(tasks)}  "
          f"(diseases={len({t[0] for t in tasks})} x cohorts={cfg['cohorts']})", flush=True)
    print(f"config: n_pcs={cfg['n_pcs']} outer={cfg['n_outer']} inner={cfg['n_inner']} "
          f"repeats={cfg['n_repeats']} omics={cfg['omics'] or 'none'}", flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=njobs, backend="loky", verbose=5)(
        delayed(run_one)(d, c, cfg) for d, c in tasks
    )
    rows = [r for group in results for r in group]
    if not rows:
        print("no results produced — check checkpoints for errors", flush=True)
        return
    df = pd.DataFrame(rows)
    suffix = f"_{args.tag}" if args.tag else ""
    df.to_csv(OUT / f"ablation_results{suffix}.csv", index=False)
    df.to_json(OUT / f"ablation_results{suffix}.json", orient="records", indent=2)
    (OUT / f"ablation_summary{suffix}.md").write_text(summarize(df, args.omics_tag))
    print(f"\nDONE {len(df)} rows in {time.time()-t0:.0f}s -> {OUT}/ablation_results{suffix}.csv", flush=True)
    print(summarize(df, args.omics_tag), flush=True)


if __name__ == "__main__":
    main()
