#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn

from hexplode_gnn import HexplodeGraphPolicyValueNet
from hexplode_nn import HexplodeEnv


def infer_side_length_from_nodes(num_nodes: int) -> int | None:
    # Hex board nodes: N = 1 + 3r(r+1), where r = sideLength - 1.
    disc = 12 * num_nodes - 3
    if disc <= 0:
        return None
    root = math.sqrt(float(disc))
    side = int(round((3.0 + root) / 6.0))
    if side < 2:
        return None
    r = side - 1
    if 1 + 3 * r * (r + 1) != num_nodes:
        return None
    return side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Hexplode GNN policy-value checkpoint to CoreML.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tools/outputs/hexplode_alphazero_12h_best.pt",
        help="GNN checkpoint (.pt) produced by hexplode_alpha_selfplay.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="adesmegagame/adesmegagame/HexplodePolicy.mlpackage",
        help="Output CoreML package (.mlpackage)",
    )
    parser.add_argument("--ios-target", type=str, default="15", help="Minimum iOS target.")
    return parser.parse_args()


def pick_target(version: str):
    if version == "13":
        return ct.target.iOS13
    if version == "14":
        return ct.target.iOS14
    if version == "15":
        return ct.target.iOS15
    if version == "16":
        return ct.target.iOS16
    if version == "17":
        return ct.target.iOS17
    return ct.target.iOS15


class GraphWrapper(nn.Module):
    def __init__(self, model: HexplodeGraphPolicyValueNet, adj: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("adjacency", adj)

    def forward(self, node_features: torch.Tensor):
        return self.model(node_features, self.adjacency)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_dim = int(ckpt.get("input_dim", 8))
    hidden_dim = int(ckpt.get("hidden_dim", 160))
    blocks = int(ckpt.get("blocks", 8))
    num_nodes = int(ckpt.get("num_nodes", 37))
    side_length_raw = ckpt.get("side_length")
    if side_length_raw is None:
        inferred = infer_side_length_from_nodes(num_nodes)
        if inferred is None:
            raise RuntimeError(
                f"Could not infer side_length for checkpoint with num_nodes={num_nodes}"
            )
        side_length = inferred
    else:
        side_length = int(side_length_raw)

    env = HexplodeEnv(side_length=side_length)
    if len(env.coords) != num_nodes:
        raise RuntimeError(
            f"Checkpoint num_nodes mismatch: ckpt={num_nodes}, env(side_length={side_length})={len(env.coords)}"
        )

    model = HexplodeGraphPolicyValueNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        blocks=blocks,
        num_nodes=num_nodes,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i, neighs in enumerate(env.neighbors):
        adj[i, i] = 1.0
        for j in neighs:
            adj[i, j] = 1.0
    adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1.0)

    wrapped = GraphWrapper(model, adj)
    wrapped.eval()
    sample_features = torch.randn(1, num_nodes, input_dim)
    traced = torch.jit.trace(wrapped, sample_features)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=pick_target(args.ios_target),
        inputs=[ct.TensorType(name="node_features", shape=sample_features.shape)],
        outputs=[
            ct.TensorType(name="policy"),
            ct.TensorType(name="value"),
        ],
    )

    mlmodel.short_description = "Hexplode graph policy-value model (AlphaZero-style self-play)."
    mlmodel.input_description["node_features"] = f"Node feature tensor [1,N,{input_dim}]"
    mlmodel.output_description["policy"] = "Policy logits over N board nodes"
    mlmodel.output_description["value"] = "Estimated state value in [-1,1]"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"Saved CoreML model: {out_path}")


if __name__ == "__main__":
    main()
