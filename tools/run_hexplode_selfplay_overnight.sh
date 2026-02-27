#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 -u tools/hexplode_selfplay_train.py \
  --hours 8 \
  --channels 96 \
  --blocks 8 \
  --batch-size 320 \
  --lr 6e-4 \
  --weight-decay 1e-4 \
  --value-loss-weight 0.7 \
  --round-games 1000 \
  --round-epochs 6 \
  --max-turns 200 \
  --sample-stride 2 \
  --top-k 8 \
  --prior-weight 1.0 \
  --value-weight 2.25 \
  --short-term-weight 0.85 \
  --short-term-scale 8.0 \
  --response-top-k 10 \
  --search-temperature 0.65 \
  --move-temperature 0.8 \
  --epsilon 0.06 \
  --dirichlet-alpha 0.25 \
  --dirichlet-frac 0.15 \
  --progress-every 100 \
  --log-batches-every 25 \
  --bootstrap-checkpoint tools/outputs/hexplode_policy_value_v4_long.pt \
  --out tools/outputs/hexplode_policy_value_selfplay_overnight.pt
