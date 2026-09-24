#!/usr/bin/env bash
# Builds the recsys-serving Lambda zip for x86_64 py3.12 (manylinux_2_28 wheels: the python3.12
# runtime is Amazon Linux 2023, glibc 2.34; numpy 2.3+ has no manylinux2014 wheels). boto3 and
# its dependencies come from the Lambda runtime, so they are left out (~19 MB zip, mostly numpy).
# Usage (from serving/): scripts/build_lambda.sh [out.zip]
set -euo pipefail
cd "$(dirname "$0")/.."
out="$(realpath -m "${1:-build/recsys-serving.zip}")"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

uv export --frozen --no-dev --no-hashes --no-emit-project \
  --no-emit-package boto3 --no-emit-package botocore --no-emit-package s3transfer \
  --no-emit-package jmespath --no-emit-package python-dateutil --no-emit-package six \
  --no-emit-package urllib3 \
  -o "$stage/requirements.txt" >/dev/null
uv pip install --quiet -r "$stage/requirements.txt" --target "$stage/pkg" \
  --python-platform x86_64-manylinux_2_28 --python-version 3.12 --only-binary :all:
cp -r src/steam_serving "$stage/pkg/"
find "$stage/pkg" -name '__pycache__' -type d -prune -exec rm -rf {} +

mkdir -p "$(dirname "$out")"
rm -f "$out"
(cd "$stage/pkg" && zip -qr9 "$out" .)
echo "$out"
