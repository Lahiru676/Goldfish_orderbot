#!/bin/bash
# Usage: bash upload_public_key.sh YOUR_WABA_ID
source /opt/goldfish/.env

WABA_ID="$1"
if [ -z "$WABA_ID" ]; then
  echo "ERROR: pass your WABA ID as first argument"
  echo "Usage: bash upload_public_key.sh 123456789012345"
  exit 1
fi

PUBLIC_KEY=$(cat /opt/goldfish/keys/public.pem)

curl -s -X POST \
  "https://graph.facebook.com/v20.0/${WABA_ID}/whatsapp_business_encryption" \
  -H "Authorization: Bearer ${WHATSAPP_ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"business_public_key\": $(echo "$PUBLIC_KEY" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}" \
  | python3 -m json.tool
