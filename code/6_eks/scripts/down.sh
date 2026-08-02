#!/usr/bin/env bash
# The cost lever: delete the control plane + nodegroup so billing stops. Run at the end of EVERY
# session. Results are already durable in S3; the ECR image persists (cheap).
set -euo pipefail
eksctl delete cluster --name ccle-batch --region us-east-1
echo "--- verify nothing lingers ---"
eksctl get clusters --region us-east-1 || true
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:alpha.eksctl.io/cluster-name,Values=ccle-batch" \
  --query 'Reservations[].Instances[].{id:InstanceId,state:State.Name}' --output table || true
