#!/usr/bin/env bash
# F2 per-drug gene-signature fan-out launcher. Emits the 322 batched shards, then submits fanout-f2.yaml.
#
#   !! DO NOT run until the 15 pathway shards (ccle-pathway-c7xj9 + ccle-pathway-oomretry-4gdr8) have all
#      checkpointed to s3://.../results/ablation_pathway_full/checkpoints/ (target: 15 files). F2 reuses
#      the same 8 nodes; launching early would contend for cpu with the pathway perm loops.
#
# Preconditions (verified 2026-07-14, re-check before launch):
#   * image <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/ccle-pipeline:v1 is current (has run_signature.py)
#   * artifact-repositories configMap archiveLogs:false  (MinIO log-archival fix — so pods exit 0, not 64)
#   * 8 spot nodes up (128 vCPU quota)  ->  cpu 5 = 3 pods/node = 24 slots for the 322 shards (~14 waves)
#
# Runtime knob: --n-perm 100 is the honest null (p-resolution 0.01). For a same-day pass use --n-perm 50
#   --n-iter 20 (edit fanout-f2.yaml's cmd). Per-batch cost scales with batch drugs x n-perm; batch-size 25.
set -euo pipefail
export PATH="/opt/miniconda3/envs/ML/bin:/opt/homebrew/bin:$PATH"

REPO="<REPO_ROOT>"
MANIFEST="$REPO/ctrp-gdsc-ccle-ml/6_eks/argo/fanout-f2.yaml"
SHARDS_JSON="$REPO/ctrp-gdsc-ccle-ml/6_eks/f2.json"
PY=/opt/miniconda3/envs/ML/bin/python

cd "$REPO"

echo "== 1/4  guard: pathway run complete? =="
CKPTS=$(aws s3 ls s3://<S3_BUCKET>/results/ablation_pathway_full/checkpoints/ 2>/dev/null | grep -c json || true)
echo "   pathway checkpoints in S3: $CKPTS / 15"
if [ "$CKPTS" -lt 15 ]; then
  echo "   !! pathway run not finished ($CKPTS/15). Re-run this script once it hits 15, or pass FORCE=1 to override."
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

echo "== 2/4  emit F2 shards (batch-size 25) =="
$PY ctrp-gdsc-ccle-ml/4_model/emit_shards.py --workload f2_perm --batch-size 25 > "$SHARDS_JSON"
N=$($PY -c "import json;print(len(json.load(open('$SHARDS_JSON'))))")
echo "   wrote $SHARDS_JSON  ->  $N shards"

echo "== 3/4  lint manifest =="
argo lint -n argo "$MANIFEST"

echo "== 4/4  submit =="
argo submit -n argo "$MANIFEST" -p shards="$(cat "$SHARDS_JSON")"
echo
echo "watch:   kubectl get wf -n argo -w"
echo "results: s3://<S3_BUCKET>/results/f2_signatures/sig_checkpoints/  (target: $N pair-checkpoints)"
echo "reduce:  $PY ctrp-gdsc-ccle-ml/4_model/run_signature.py --summarize-only --tag f2_signatures  (after sync from S3)"
