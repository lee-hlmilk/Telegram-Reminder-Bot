#!/usr/bin/env bash
set -euo pipefail

SERVICE="${SERVICE:-reminder-bot}"
REGION="${REGION:-asia-southeast1}"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "No Google Cloud project is selected."
  echo "Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

cd "$(dirname "$0")"

echo "Deploying $SERVICE to $PROJECT_ID in $REGION..."
gcloud run deploy "$SERVICE" \
  --project="$PROJECT_ID" \
  --source=. \
  --region="$REGION" \
  --quiet

SERVICE_URL="$(gcloud run services describe "$SERVICE" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --format='value(status.url)')"

echo "Deployment complete: $SERVICE_URL"
