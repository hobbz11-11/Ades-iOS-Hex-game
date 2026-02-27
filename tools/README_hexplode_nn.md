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
