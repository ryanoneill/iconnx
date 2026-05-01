#!/usr/bin/env bash
# Re-vendor ONNX conformance test data from upstream.
#
# Usage:
#   ./regenerate.sh <onnx-tag>     # e.g. ./regenerate.sh v1.19.0
#
# Creates tests/conformance_data/onnx-<tag-without-v>/ alongside the existing
# version directories. Update tests/onnx_conformance.rs to point at the new
# directory once tests are observed to behave reasonably.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <onnx-tag (e.g. v1.19.0)>" >&2
  exit 1
fi

TAG="$1"
SHORT="${TAG#v}"
TMPDIR="$(mktemp -d)"
trap "rm -rf $TMPDIR" EXIT

git clone --depth 1 --branch "$TAG" https://github.com/onnx/onnx.git "$TMPDIR/onnx"

DEST="$(dirname "$0")/onnx-$SHORT"
mkdir -p "$DEST"
cp -r "$TMPDIR/onnx/onnx/backend/test/data/node" "$DEST/"

echo "Vendored ONNX $TAG conformance node tests at $DEST"
echo "Next: update tests/onnx_conformance.rs CONFORMANCE_DATA_DIR to point at the new directory."
