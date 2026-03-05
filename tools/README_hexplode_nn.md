# Hexplode Neural Net (Large Board Hard Mode)

This pipeline trains a policy-value neural net for **Hexplode (4-per-side / radius 3)** and exports it to CoreML for iOS.

## Model size

- Input: `8 x 7 x 7`
- Trunk: `64` channels, `6` residual blocks
- Heads:
- Policy head -> logits for `49` board cells (flattened `7x7`)
- Value head -> scalar in `[-1, 1]`

This is intentionally small enough to run quickly on-phone while still stronger than pure heuristics once trained.

## 1) Install Python dependencies

```bash
python3 -m pip install torch coremltools
```

## 2) Train

From repo root:

```bash
python3 tools/hexplode_train_nn.py \
  --games 3000 \
  --epochs 14 \
  --batch-size 256 \
  --max-turns 200 \
  --sample-stride 2 \
  --max-samples 80000 \
  --blocks 6 \
  --channels 64 \
  --out tools/outputs/hexplode_policy_value.pt
```

For a longer run, increase `--games`, `--epochs`, and `--max-samples`.

## 3) Export to CoreML

```bash
python3 tools/hexplode_export_coreml.py \
  --checkpoint tools/outputs/hexplode_policy_value.pt \
  --out tools/outputs/HexplodePolicy.mlpackage \
  --ios-target 15
```

## 4) Add model to Xcode app

1. Drag `tools/outputs/HexplodePolicy.mlpackage` into the `adesmegagame` target.
2. Ensure target membership is enabled.
3. Build and run.

Runtime behavior:
- If `HexplodePolicy` exists in app bundle, the game uses neural priors for **large-board hard Hexplode**.
- If missing, it falls back automatically to the evolved heuristic AI.

## Interactive training UI

You can watch sample generation and online training step-by-step:

```bash
python3 tools/hexplode_training_ui.py \
  --checkpoint tools/outputs/hexplode_policy_value_counterfactual_12h_newlogic_best.pt \
  --lookahead-ply 5 \
  --branch-cap 0
```

What it shows live:
- Initial board state
- Proposed move chosen by current model
- Result after minimax rollout
- Good/Bad label for that proposal
- Input plane view, intermediate activation maps, policy map, value output

Controls:
- `Step (1 sample)` runs one labeled sample + one train update
- `Run` / `Pause` for continuous training
- `Visuals On` toggle to disable heavy drawing for faster throughput
- `Save Checkpoint` writes current weights to the `--out` path

## Graph NN (new)

There is now a Graph Neural Network trainer that uses the true 37-node hex connectivity
instead of a padded 7x7 grid.

Key point:
- `--symmetry-rotations 6` uses all six 60-degree board rotations per labeled base sample.

Quick smoke test:

```bash
python3 tools/hexplode_gnn_counterfactual_train.py \
  --hours 0.1 \
  --device cpu \
  --start-source random \
  --lookahead-ply 3 \
  --symmetry-rotations 6 \
  --round-samples 1200 \
  --round-epochs 2 \
  --out tools/outputs/hexplode_gnn_counterfactual_smoke.pt
```

Long run:

```bash
./tools/run_hexplode_gnn_counterfactual_overnight.sh
```
