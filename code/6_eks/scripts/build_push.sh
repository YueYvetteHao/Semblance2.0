#!/usr/bin/env bash
# Build the arm64 batch image and push to ECR. Run from ctrp-gdsc-ccle-ml/ (the build context).
#   6_eks/scripts/build_push.sh [tag]      # default tag: v1
set -euo pipefail
ACCT="<AWS_ACCOUNT_ID>"
REGION=us-east-1
REPO=ccle-pipeline
TAG="${1:-v1}"
IMG="$ACCT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG"

# create the ECR repo if it doesn't exist yet (idempotent)
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCT.dkr.ecr.$REGION.amazonaws.com"

docker buildx build -f 6_eks/Dockerfile --platform linux/arm64 -t "$IMG" --push .
echo "pushed $IMG"
