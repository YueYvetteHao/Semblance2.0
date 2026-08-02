#!/usr/bin/env bash
# Pod entrypoint for the batch fan-out. Stateless map step:
#   1. stage the model-ready store from S3 into the dirs data.py expects (<n>_*/out)
#   2. run the shard command (passed as the container args; reads $SHARD from the env)
#   3. sync results back to S3 (the per-shard checkpoint is the durable artifact)
#
# S3 is the only data plane — a killed spot pod just re-runs its shard; a completed shard is skipped
# (the checkpoint pulled in step 1 makes run_one short-circuit).
#
# Env:
#   BUCKET         s3://…-an                       (default below)
#   RESULT_PREFIX  results/<run tag>               (where this job's checkpoints live)
#   STAGE_S3       1 = sync S3 (EKS) · 0 = data is volume-mounted (kind rehearsal)
#   SHARD          the one shard as JSON            (referenced by the cmd as "$SHARD")
set -euo pipefail

BUCKET="${BUCKET:-s3://<S3_BUCKET>}"
RESULT_PREFIX="${RESULT_PREFIX:-results/ablation_pathway_full}"
STAGE_S3="${STAGE_S3:-1}"
REPO=/app/ctrp-gdsc-ccle-ml

# BLAS thread safety net. The drivers are single-process (evaluate.INNER_NJOBS=1); numpy/scipy read
# these at import. If a template forgot to cap them, a pod's OpenBLAS would spawn one thread per node
# vCPU (it cannot see the cgroup cpu limit) and packed pods would thrash the node. Default to 1; the
# Argo template overrides to == resources.limits.cpu.
: "${OMP_NUM_THREADS:=1}";      export OMP_NUM_THREADS
: "${OPENBLAS_NUM_THREADS:=1}"; export OPENBLAS_NUM_THREADS
: "${MKL_NUM_THREADS:=1}";      export MKL_NUM_THREADS
: "${NUMEXPR_NUM_THREADS:=1}";  export NUMEXPR_NUM_THREADS

if [ "$STAGE_S3" = "1" ]; then
  mkdir -p "$REPO"/2_merge/out "$REPO"/3_omics/out "$REPO"/5_pathway/out "$REPO"/4_model/out
  # S3 layout  ->  the local dirs data.py reads (no data.py change needed)
  aws s3 sync "$BUCKET/data/merge/"               "$REPO/2_merge/out/"  --only-show-errors
  aws s3 sync "$BUCKET/data/omics/"               "$REPO/3_omics/out/"  --only-show-errors
  aws s3 sync "$BUCKET/results/pathway_features/" "$REPO/5_pathway/out/" --only-show-errors
  # pull any prior checkpoints so a retried shard is skipped (idempotent resume); non-fatal if empty
  aws s3 sync "$BUCKET/$RESULT_PREFIX/" "$REPO/4_model/out/" --only-show-errors || true
fi

cd /app
"$@"

if [ "$STAGE_S3" = "1" ]; then
  aws s3 sync "$REPO/4_model/out/" "$BUCKET/$RESULT_PREFIX/" --only-show-errors
fi
