#!/usr/bin/env bash
# ProxyML basic usage — curl version
#
# Prerequisites: curl, jq
#   export PROXYML_API_KEY="your-api-key"
#
# Build your schema at https://proxyml.github.io/schema-builder/
# then paste it into step 1 below.

set -euo pipefail

BASE_URL="${PROXYML_BASE_URL:-https://api.proxyml.ai/api/v1}"

if [[ -z "${PROXYML_API_KEY:-}" ]]; then
  echo "Error: PROXYML_API_KEY is not set" >&2
  exit 1
fi

AUTH=(-H "X-API-KEY: $PROXYML_API_KEY" -H "Content-Type: application/json")

# ---------------------------------------------------------------------------
# 1. Upload a schema
#    Replace the features array with the JSON from the schema builder.
# ---------------------------------------------------------------------------
echo "==> Uploading schema..."
curl -s -X PUT "$BASE_URL/schema/default" \
  "${AUTH[@]}" \
  -d '{
    "features": [
      { "type": "continuous", "name": "age",          "mean": 45,    "std": 15,    "min": 18,  "max": 90,     "immutable": false },
      { "type": "continuous", "name": "income",        "mean": 60000, "std": 25000, "min": 0,   "max": 200000, "immutable": false },
      { "type": "continuous", "name": "credit_score",  "mean": 680,   "std": 80,    "min": 300, "max": 850,    "immutable": false }
    ]
  }' | jq .

# ---------------------------------------------------------------------------
# 2. Synthesize training data
# ---------------------------------------------------------------------------
echo ""
echo "==> Synthesizing data..."
curl -s -X POST "$BASE_URL/synthesize/neighbors" \
  "${AUTH[@]}" \
  -d '{"n": 500, "schema_name": "default"}' \
  -o synth.json

echo "Synthesized $(jq '.samples | length' synth.json) rows"
echo "Features: $(jq -c '.feature_names' synth.json)"

# ---------------------------------------------------------------------------
# 3. Score with your black-box model
#    Replace the jq expression with your actual model's predictions.
#    Here we use a simple rule as a stand-in: credit_score > 650 → approved.
# ---------------------------------------------------------------------------
echo ""
echo "==> Building training payload (scoring with black-box)..."
TRAIN_PAYLOAD=$(jq '{
  samples:       .samples,
  feature_names: .feature_names,
  predictions:   [.samples[] | if (.[2] > 650) then 1 else 0 end],
  task:          "classification",
  test_size:     0.2,
  schema_name:   "default"
}' synth.json)

# ---------------------------------------------------------------------------
# 4. Train a surrogate
# ---------------------------------------------------------------------------
echo ""
echo "==> Training surrogate..."
TRAIN_RESULT=$(curl -s -X POST "$BASE_URL/surrogate/train" \
  "${AUTH[@]}" \
  -d "$TRAIN_PAYLOAD")

echo "$TRAIN_RESULT" | jq .
VERSION=$(echo "$TRAIN_RESULT" | jq -r '.version')

# ---------------------------------------------------------------------------
# 5. Feature importances
# ---------------------------------------------------------------------------
echo ""
echo "==> Feature importances..."
curl -s -G "$BASE_URL/explain/importance" \
  -H "X-API-KEY: $PROXYML_API_KEY" \
  --data-urlencode "version=$VERSION" | jq .

# ---------------------------------------------------------------------------
# 6. Query the surrogate
# ---------------------------------------------------------------------------
echo ""
echo "==> Predicting for sample [age=50, income=75000, credit_score=720]..."
curl -s -X POST "$BASE_URL/surrogate/predict" \
  "${AUTH[@]}" \
  -d "{\"inputs\": [50, 75000, 720], \"version\": \"$VERSION\"}" | jq .

rm -f synth.json
