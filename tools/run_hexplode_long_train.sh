#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/tools/outputs"
APP_MODEL_DIR="$ROOT_DIR/adesmegagame/adesmegagame/adesmegagame/HexplodePolicy.mlpackage"
CHECKPOINT="$OUT_DIR/hexplode_policy_value_long_v1.pt"
COREML_PKG="$OUT_DIR/HexplodePolicy_long_v1.mlpackage"
DONE_FILE="$OUT_DIR/hexplode_policy_value_long_v1.done"

mkdir -p "$OUT_DIR"
rm -f "$DONE_FILE"

python3 -u "$ROOT_DIR/tools/hexplode_train_nn.py" \
  --games 15000 \
  --epochs 60 \
  --batch-size 320 \
  --max-turns 200 \
  --sample-stride 2 \
  --max-samples 300000 \
  --progress-every 500 \
  --blocks 8 \
  --channels 96 \
  --out "$CHECKPOINT"

python3 -u "$ROOT_DIR/tools/hexplode_export_coreml.py" \
  --checkpoint "$CHECKPOINT" \
  --out "$COREML_PKG" \
  --ios-target 15

rm -rf "$APP_MODEL_DIR"
cp -R "$COREML_PKG" "$APP_MODEL_DIR"

touch "$DONE_FILE"
echo "Long training complete. Model copied to app bundle path."
