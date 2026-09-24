#!/usr/bin/env bash
# Builds the list-partition-game-ids Lambda zip (base deps only, no polars) for x86_64 py3.12.
# Usage (from data_ingestion/): scripts/build_lambda.sh [out.zip]
set -euo pipefail
cd "$(dirname "$0")/.."
out="$(realpath -m "${1:-build/list-partition-game-ids.zip}")"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

uv export --frozen --no-dev --no-hashes --no-emit-project -o "$stage/requirements.txt" >/dev/null
uv pip install --quiet -r "$stage/requirements.txt" --target "$stage/pkg" \
  --python-platform x86_64-manylinux2014 --python-version 3.12 --only-binary :all:
cp -r src/steam_ingestion "$stage/pkg/"
find "$stage/pkg" -name '__pycache__' -type d -prune -exec rm -rf {} +

mkdir -p "$(dirname "$out")"
rm -f "$out"
(cd "$stage/pkg" && zip -qr9 "$out" .)
echo "$out"
