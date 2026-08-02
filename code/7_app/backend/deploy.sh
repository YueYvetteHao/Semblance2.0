#!/usr/bin/env bash
# Deploy the Semblance 2.0 backend to Google Cloud Run.
#
#   ./deploy.sh                 # deploy with the defaults below
#   REGION=us-east1 ./deploy.sh # override any variable inline
#
# Settings rationale (see ../../docs/HOSTING_PLAN.md §4):
#   --memory 2Gi     normal use is ~300 MB, but a non-Reactome pathway name lazy-loads
#                    BioLORD (~1.5 GB). At 512Mi that request OOMs and kills the instance.
#   --no-cpu-throttling is NOT set, i.e. CPU is allocated only during request processing.
#                    This is what makes idle cost $0. Do not change it.
#   --max-instances  the real cost ceiling; Cloud Run has no hard spend cap.
#   --concurrency 4  the service is memory-bound, not CPU-bound; the default 80 would OOM.
#   --no-invoker-iam-check  makes the service public. Google's recommended way, and the one that
#                    works under an organization's domain-restricted-sharing policy, which blocks
#                    the older --allow-unauthenticated (that grants allUsers roles/run.invoker).
#                    The two are alternatives, never both: passing both still attempts the blocked
#                    binding. Swap back to --allow-unauthenticated only outside an org.
#
# One-time org setup: builds run as the Compute Engine default service account, which in an
# organization is NOT auto-granted Editor. If the build fails with PERMISSION_DENIED reading the
# source zip, grant it once (PROJECT_NUMBER is in the error message):
#   gcloud projects add-iam-policy-binding "$PROJECT" \
#     --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
#     --role=roles/run.builder
set -euo pipefail

PROJECT="${PROJECT:-semblance2-0}"
SERVICE="${SERVICE:-semblance}"
REGION="${REGION:-us-central1}"
# CORS origins the backend will accept, comma-separated. Add http://localhost:8000 while testing
# the frontend locally against this deployed service.
# NOTE the "^##^" prefix on --set-env-vars below: gcloud splits env-var assignments on commas by
# default, so a comma-separated VALUE is mangled. "^##^" changes the separator to "##", letting the
# commas inside ALLOWED_ORIGINS survive. Without it, multi-origin silently breaks.
ORIGINS="${ORIGINS:-https://yueyvettehao.github.io}"

cd "$(dirname "$0")"

echo "==> project=$PROJECT  service=$SERVICE  region=$REGION"
gcloud config set project "$PROJECT"

echo "==> enabling APIs (no-op if already enabled)"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

echo "==> deploying (first build ~10-15 min: torch + BioLORD are baked into the image)"
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --no-invoker-iam-check \
  --memory 2Gi \
  --cpu 1 \
  --cpu-boost \
  --concurrency 4 \
  --max-instances 3 \
  --min-instances 0 \
  --timeout 120 \
  --set-env-vars "^##^ALLOWED_ORIGINS=$ORIGINS"

# Prefer urls[0] (the modern https://SERVICE-PROJECT_NUMBER.REGION.run.app form). The legacy
# status.url field still reports the old SERVICE-HASH-REGION.a.run.app format, which is NOT
# provisioned for newly created services and 404s.
URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(urls[0])' 2>/dev/null)"
[ -n "$URL" ] || URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo
echo "==> service URL: $URL"
echo "==> health check:"
curl -fsS "$URL/healthz" && echo
echo
echo 'Expect {"status":"ok","engine":"real"}.  "stub" means the assets/*.npz did not make it into the image.'
echo "Next: set BACKEND_URL in Github/Semblance2.0/config.js to  $URL  (no trailing slash)."
