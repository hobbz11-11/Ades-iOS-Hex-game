#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STOP_FILE="tools/outputs/hexplode_counterfactual.stop"
rm -f "$STOP_FILE"

python3 -u tools/hexplode_counterfactual_train.py \
  --hours 8 \
  --channels 96 \
  --blocks 8 \
  --batch-size 384 \
  --lr 6e-4 \
  --weight-decay 1e-4 \
  --value-loss-weight 0.45 \
  --round-samples 15000 \
  --round-epochs 5 \
  --empty-prob 0.46 \
  --max-level 5 \
  --min-turns 2 \
  --max-turns 160 \
  --good-rule after_positive \
  --value-target binary \
  --progress-every 1000 \
  --log-batches-every 25 \
  --bootstrap-checkpoint tools/outputs/hexplode_policy_value_v4_long.pt \
  --stop-file "$STOP_FILE" \
  --out tools/outputs/hexplode_policy_value_counterfactual_overnight.pt
