#!/usr/bin/env bash
# Bring up the per-session batch cluster (~15 min). Tear it down with down.sh at the end.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)   # 6_eks/
eksctl create cluster -f "$HERE/infra/eks-cluster.yaml"
echo "cluster up. next: install Argo, create IRSA (ccle-s3), push the image."
