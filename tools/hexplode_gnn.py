#!/usr/bin/env python3
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexplode_nn import BLUE, EMPTY, RED, HexplodeEnv, State

GNN_FEATURE_DIM = 12


def build_adj_matrix(env: HexplodeEnv, self_loop: float = 1.0) -> torch.Tensor:
    n = len(env.coords)
    adj = torch.zeros((n, n), dtype=torch.float32)
    for i, neighs in enumerate(env.neighbors):
        if self_loop > 0.0:
            adj[i, i] = self_loop
        for j in neighs:
            adj[i, j] = 1.0
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj = adj / row_sum
    return adj


def encode_node_features(env: HexplodeEnv, state: State, current_player: int) -> torch.Tensor:
    n = len(env.coords)
    x = torch.zeros((n, GNN_FEATURE_DIM), dtype=torch.float32)
    legal = set(env.legal_moves(state, current_player))
    opponent = BLUE if current_player == RED else RED
    turn_progress = min(1.0, float(max(0, state.turns_played)) / 220.0)

    for i in range(n):
        owner = state.owners[i]
        level = state.levels[i]
        level_norm = min(6, max(0, level)) / 6.0

        if owner == current_player:
            x[i, 0] = 1.0
            x[i, 3] = level_norm
            if level >= 5:
                x[i, 5] = 1.0
        elif owner == opponent:
            x[i, 1] = 1.0
            x[i, 4] = level_norm
            if level >= 5:
                x[i, 6] = 1.0
        else:
            x[i, 2] = 1.0

        if i in legal:
            x[i, 7] = 1.0
        if env.is_edge[i]:
            x[i, 8] = 1.0
        if env.is_corner[i]:
            x[i, 9] = 1.0
        # Neighbor degree (corners=3, edges=4, interior=6), normalized to [0, 1].
        x[i, 10] = float(len(env.neighbors[i])) / 6.0
        # Global turn progress helps phase-aware policy/value behavior.
        x[i, 11] = turn_progress

    return x


class GraphResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_proj_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neigh_proj_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_1 = nn.LayerNorm(hidden_dim)

        self.self_proj_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neigh_proj_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C], adj: [N, N]
        neigh = torch.einsum("nm,bmc->bnc", adj, x)
        h = self.self_proj_1(x) + self.neigh_proj_1(neigh)
        h = self.norm_1(h)
        h = F.relu(h, inplace=True)
        h = self.dropout(h)

        neigh2 = torch.einsum("nm,bmc->bnc", adj, h)
        h2 = self.self_proj_2(h) + self.neigh_proj_2(neigh2)
        h2 = self.norm_2(h2)
        h2 = self.dropout(h2)
        return F.relu(x + h2, inplace=True)


class HexplodeGraphPolicyValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int = GNN_FEATURE_DIM,
        hidden_dim: int = 128,
        blocks: int = 6,
        num_nodes: int = 37,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.blocks = blocks
        self.num_nodes = num_nodes

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.trunk = nn.ModuleList([GraphResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, F], adj: [N, N]
        h = F.relu(self.input_proj(x), inplace=True)
        for block in self.trunk:
            h = block(h, adj)

        policy_logits = self.policy_head(h).squeeze(-1)  # [B, N]
        pooled = h.mean(dim=1)  # [B, C]
        value = self.value_head(pooled).squeeze(-1)  # [B]
        return policy_logits, value


def clone_gnn_model(model: HexplodeGraphPolicyValueNet, device: torch.device) -> HexplodeGraphPolicyValueNet:
    cloned = HexplodeGraphPolicyValueNet(
        input_dim=model.input_dim,
        hidden_dim=model.hidden_dim,
        blocks=model.blocks,
        num_nodes=model.num_nodes,
    )
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    cloned.to(device)
    cloned.eval()
    return cloned


@dataclass
class RotationMaps:
    maps: List[List[int]]


def rotate_axial(coord: Tuple[int, int], turns: int) -> Tuple[int, int]:
    x, z = coord
    y = -x - z
    for _ in range(turns % 6):
        x, y, z = -z, -x, -y
    return x, z


def build_rotation_maps(env: HexplodeEnv, num_rotations: int) -> RotationMaps:
    remaps: List[List[int]] = []
    n = len(env.coords)
    for turns in range(max(1, min(6, num_rotations))):
        remap = [0] * n
        for src_idx, coord in enumerate(env.coords):
            rotated = rotate_axial(coord, turns)
            dst_idx = env.index_of.get(rotated)
            if dst_idx is None:
                raise RuntimeError(f"Rotation produced out-of-board coord: {coord} -> {rotated}")
            remap[src_idx] = dst_idx
        remaps.append(remap)
    return RotationMaps(maps=remaps)


def rotate_state(state: State, remap: Sequence[int]) -> State:
    n = len(remap)
    owners = [0] * n
    levels = [0] * n
    for src_idx, dst_idx in enumerate(remap):
        owners[dst_idx] = state.owners[src_idx]
        levels[dst_idx] = state.levels[src_idx]
    return State(owners=owners, levels=levels, turns_played=state.turns_played)
