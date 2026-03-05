#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STOP_FILE="tools/outputs/hexplode_gnn_counterfactual.stop"
rm -f "$STOP_FILE"

python3 -u tools/hexplode_gnn_counterfactual_train.py \
  --hours 12 \
  --hidden-dim 128 \
  --blocks 6 \
  --batch-size 320 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --value-loss-weight 0.45 \
  --round-samples 12000 \
  --round-epochs 5 \
  --start-source selfplay \
  --prefix-random-moves 30 \
  --prefix-policy-min-moves 20 \
  --prefix-policy-max-moves 120 \
  --prefix-policy-epsilon 0.0 \
  --sample-move-epsilon 0.02 \
  --symmetry-rotations 6 \
  --lookahead-ply 5 \
  --max-opponent-checks 0 \
  --good-rule after_positive \
  --value-target binary \
  --progress-every 1000 \
  --log-batches-every 25 \
  --stop-file "$STOP_FILE" \
  --out tools/outputs/hexplode_gnn_counterfactual_overnight.pt
