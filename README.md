# Semblance 2.0 — a confidence-tiered drug recommender from GDSC × CTRP × CCLE

Static web app, analysis code, and manuscript figures for a two-arm drug-sensitivity
recommender built on a harmonized GDSC2 + CTRP v2.2 drug-response store merged with CCLE
multi-omics, keyed on DepMap ID.

**Live app → https://yueyvettehao.github.io/Semblance2.0/**

The app answers a deliberately narrow question in two tiers of confidence:

1. **Cell Line Lookup (broad-spectrum).** For each of 810 profiled cell lines, the
   within-cancer-type ranking of generically potent drugs, plus any permutation-validated
   mechanism-specific signatures (*p* ≤ 0.01). Fully static — precomputed JSON, no server.
2. **Signature Match (mechanism-specific).** Upload a GSEA results table; it is embedded with
   BioLORD-2023 and matched against an 1,824-signature enrichment atlas. Backed by a
   small CPU-only FastAPI service on Cloud Run.

> Exploratory and hypothesis-generating. **Not a clinical tool** and not a basis for treatment
> decisions.

## Repository layout

```
index.html, config.js, data/      the static frontend (served by GitHub Pages from repo root)
  data/cellline/*.json              810 precomputed per-cell-line artifacts + _index.json
  data/example_gsea.csv             sample input for the Signature Match tab
code/                             the analysis pipeline, in run order
  1_harmonize/                      GDSC + CTRP -> one drug-response store (InChIKey drug identity,
                                    DepMap cell identity, per-cohort per-drug z-scored AUC = sens_z)
  2_merge/                          + CCLE log1p(TPM) expression, split per disease
  3_omics/                          8 companion omics layers (mutations, fusions, methylation,
                                    metabolomics, RPPA, miRNA, chromatin, CN summary)
  4_model/                          grouped nested-CV ablation harness, always vs the drug-mean
                                    baseline; per-drug signature scan with a permutation null
  5_pathway/                        ssGSEA pathway features (R/GSVA) + the fgsea enrichment atlas
  6_eks/                            the AWS EKS + Argo batch fan-out (Dockerfile, workflows, IaC)
  7_app/precompute/                 builds the atlas and the per-cell-line JSON the frontend serves
  7_app/backend/                    the Signature Match service (FastAPI + the ranking engine of
                                    Methods §2.9), its Dockerfile and Cloud Run deploy script
Figures_and_Tables/               every manuscript figure (PDF + PNG) with its make_*.py and inputs
```

## Reproducing

Python 3.11+ with `numpy scipy scikit-learn pandas pyarrow joblib`; the pathway stage additionally
needs R with `GSVA`, `msigdbr`, and `fgsea`. Exact pinned versions are in `code/6_eks/Dockerfile`
(`pyarrow` is held at 18.1.0 on purpose — newer arm64 wheels segfault on these parquet files).

The pipeline reads and writes `out/` directories that are **not** in this repo: the harmonized
stores, the pathway matrix and the enrichment atlas total ~1.2 GB and are archived on Zenodo
(see below). Download the deposit and unpack it so each stage's `out/` sits beside its scripts,
then run the stages in numeric order. Raw inputs (GDSC2, CTRP v2.2, CCLE 2019) come from their
original sources and are not redistributed here.

The heavy stages — the pathway ablation and the 7,836-pair per-drug signature scan — were run as a
322-shard fan-out on EKS (`code/6_eks/`). The contract is that a shard is independently runnable and
idempotent, with S3 as the only data plane: a killed spot pod re-runs its shard, and a finished
shard is skipped because the staged checkpoint makes the driver short-circuit. `emit_shards.py` is
the map step — it reuses the same task enumerators as the monolithic drivers, so the shard list
cannot drift — and `run_ablation.py --summarize-only` is the reduce:

```bash
python code/4_model/emit_shards.py --workload f2_perm > shards.json
argo submit code/6_eks/argo/fanout-f2.yaml -p shards="$(cat shards.json)"
python code/4_model/run_ablation.py --summarize-only     # once the fleet drains
```

Rehearse on `kind` first (`code/6_eks/kind/`, `STAGE_S3=0` mounts the data instead of pulling from
S3) before spending anything on EKS, and run `code/6_eks/scripts/down.sh` when a sweep finishes —
an idle EKS control plane bills continuously.

Nothing in `code/` needs AWS: every stage runs locally against the unpacked `out/` directories, and
the fan-out is an optional accelerator. If you do want to reproduce it, the deployment-specific
values have been replaced with placeholders and must be filled in first:

| placeholder | where | what to put |
|---|---|---|
| `<AWS_ACCOUNT_ID>` | `scripts/build_push.sh`, `argo/fanout*.yaml` | your 12-digit account ID (ECR registry host) |
| `<S3_BUCKET>` | `entrypoint.sh`, `launch_f2.sh`, `infra/irsa-ccle-s3.yaml` | your bucket name — pods use it as the only data plane |
| `<REPO_ROOT>` | `launch_f2.sh`, `kind/kind-config.yaml` | absolute path to your clone |

## Backend service

The Signature Match tab calls a separate container on **Google Cloud Run**, which scales to zero
between requests. Its full source, Dockerfile and deploy script are in `code/7_app/backend/`:
`./deploy.sh` from that directory wraps `gcloud run deploy --source .`, so no git repo or
pre-built image is needed. The settings that matter are baked into that script — **2 GiB** memory
(a pathway name outside the cached Reactome vocabulary lazy-loads BioLORD at ~1.5 GB, which OOMs at
512 MiB), request-time-only CPU allocation (what makes idle cost $0), `--max-instances 3` as the
cost ceiling since Cloud Run has no hard spend cap, and `--concurrency 4` because the service is
memory- rather than CPU-bound. The first build takes ~10–15 min: CPU-only torch and the BioLORD
weights are baked into the image so no request ever waits on a download.

Set the URL once in `config.js`:

```js
var CLOUD_RUN_BACKEND = "https://<service>-<project-number>.<region>.run.app";  // no trailing slash
```

The service must allow this origin — set the `ALLOWED_ORIGINS` environment variable on the Cloud
Run service to `https://yueyvettehao.github.io`. That is an *origin*: scheme and host only, never a
path, so the trailing `/Semblance2.0/` is not part of it. `config.js` picks the backend from where
the page is served (localhost during development, Cloud Run otherwise) and accepts a `?backend=`
query override for testing. Leaving `CLOUD_RUN_BACKEND` empty runs the static tab alone; Signature
Match then shows a configuration note instead of failing.

**How matches are ranked.** Not by raw cosine. BioLORD's embedding space is strongly anisotropic —
mean pairwise similarity across the atlas is ~0.92, so an arbitrary pathway program scores ~0.975
against any panel and cosine barely discriminates. The service therefore applies a hubness
correction (CSLS) and scores each drug against a null built from real held-out signatures, then
reports the result in standard deviations above that null (`evidence_z`). On a held-out check —
feeding a drug's own GSEA table back in — this retrieves that drug at rank 1 for 95% of 74
signatures across SKCM / BRCA / COAD-READ (top-5 100%), against 68% under raw cosine. The raw
cosine is still returned as `similarity` for transparency but is not the ranking key.

## Data and code availability

- **Derived data + full pipeline outputs** — Zenodo: `[DOI to add]`
- **Backend service** — Google Cloud Run; source, Dockerfile and deploy script in
  `code/7_app/backend/`
- **Semblance 1.0**, the pairwise enrichment-comparison tool this reuses —
  https://huggingface.co/spaces/yueyvettehao/Semblance

Primary data are all public: 
GDSC (https://www.cancerrxgene.org),
CTRP and CCLE via the Broad Institute, and DepMap (https://depmap.org).

## Citation

Manuscript in preparation; preprint and citation details to follow.

## Author

Yue Hao
