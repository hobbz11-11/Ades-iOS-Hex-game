#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from hexplode_counterfactual_train import choose_device, format_duration
from hexplode_gnn import (
    GNN_FEATURE_DIM,
    HexplodeGraphPolicyValueNet,
    RotationMaps,
    build_adj_matrix,
    build_rotation_maps,
    encode_node_features,
)
from hexplode_nn import BLUE, RED, HexplodeEnv, State


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero-style self-play trainer for Hexplode (GNN policy-value + MCTS)."
    )
    parser.add_argument("--hours", type=float, default=12.0, help="Wall-clock budget in hours.")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--side-length", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.5)
    parser.add_argument("--round-games", type=int, default=48, help="Self-play games per training round.")
    parser.add_argument("--round-epochs", type=int, default=4, help="Epochs per training round.")
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--mcts-sims", type=int, default=48)
    parser.add_argument("--mcts-cpuct", type=float, default=1.45)
    parser.add_argument("--target-temperature", type=float, default=1.0, help="Temperature for policy targets from MCTS visits.")
    parser.add_argument("--play-temperature", type=float, default=1.0, help="Action sampling temperature in early plies.")
    parser.add_argument("--play-temperature-drop", type=int, default=14, help="After this ply, use near-greedy move selection.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.30)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--replay-size", type=int, default=140000)
    parser.add_argument("--train-samples-cap", type=int, default=50000)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--arena-games", type=int, default=14)
    parser.add_argument("--arena-sims", type=int, default=56)
    parser.add_argument("--promote-threshold", type=float, default=0.53)
    parser.add_argument("--symmetry-rotations", type=int, default=6, help="Use 1..6 rotational symmetries.")
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--log-batches-every", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--bootstrap-checkpoint", type=str, default="", help="Optional model checkpoint to bootstrap from.")
    parser.add_argument("--out", type=str, default="tools/outputs/hexplode_alphazero_latest.pt")
    parser.add_argument("--best-out", type=str, default="tools/outputs/hexplode_alphazero_best.pt")
    parser.add_argument("--stats-out", type=str, default="tools/outputs/hexplode_alphazero_stats.json")
    parser.add_argument("--stop-file", type=str, default="tools/outputs/hexplode_alphazero.stop")
    return parser.parse_args()


@dataclass
class Sample:
    features: torch.Tensor  # [N, F]
    policy: torch.Tensor  # [N]
    value: float


class MCTSNode:
    __slots__ = (
        "to_play",
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "expanded",
        "is_terminal",
        "terminal_winner",
    )

    def __init__(self, to_play: int, prior: float) -> None:
        self.to_play = to_play
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.expanded = False
        self.is_terminal = False
        self.terminal_winner: int | None = None

    @property
    def q_value(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / float(self.visit_count)


def softmax(values: Sequence[float], temperature: float) -> List[float]:
    if not values:
        return []
    t = max(1e-3, float(temperature))
    m = max(values)
    exps = [math.exp(max(-50.0, min(50.0, (v - m) / t))) for v in values]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / s for v in exps]


def sample_index(probs: Sequence[float], rng: random.Random) -> int:
    roll = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if roll <= cumulative:
            return i
    return max(0, len(probs) - 1)


def opponent(player: int) -> int:
    return BLUE if player == RED else RED


def terminal_value_from_perspective(winner: int | None, player: int) -> float:
    if winner is None or winner == 0:
        return 0.0
    return 1.0 if winner == player else -1.0


def model_predict(
    env: HexplodeEnv,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    state: State,
    player: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    model.eval()
    with torch.no_grad():
        xb = encode_node_features(env, state, player).unsqueeze(0).to(device)
        logits, values = model(xb, adj)
    return logits[0].detach().cpu(), float(values[0].detach().cpu().item())


def select_child(node: MCTSNode, cpuct: float, rng: random.Random) -> Tuple[int, MCTSNode]:
    assert node.children, "select_child called on node without children"
    sqrt_parent = math.sqrt(max(1, node.visit_count))
    best_score = -1e30
    best_moves: List[int] = []
    for move, child in node.children.items():
        u = cpuct * child.prior * (sqrt_parent / (1.0 + child.visit_count))
        score = child.q_value + u
        if score > best_score + 1e-12:
            best_score = score
            best_moves = [move]
        elif abs(score - best_score) <= 1e-12:
            best_moves.append(move)
    chosen_move = rng.choice(best_moves)
    return chosen_move, node.children[chosen_move]


def expand_node(
    env: HexplodeEnv,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    node: MCTSNode,
    state: State,
    device: torch.device,
    rng: random.Random,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    add_root_noise: bool,
) -> float:
    winner = env.terminal_winner(state)
    if winner is not None:
        node.expanded = True
        node.is_terminal = True
        node.terminal_winner = winner
        return terminal_value_from_perspective(winner, node.to_play)

    legal = env.legal_moves(state, node.to_play)
    if not legal:
        node.expanded = True
        node.is_terminal = True
        node.terminal_winner = env.terminal_winner(state)
        return terminal_value_from_perspective(node.terminal_winner, node.to_play)

    logits, value = model_predict(env, model, adj, state, node.to_play, device)
    legal_logits = [float(logits[m]) for m in legal]
    priors = softmax(legal_logits, temperature=1.0)

    if add_root_noise and len(legal) > 1 and dirichlet_eps > 0:
        alpha = max(1e-5, dirichlet_alpha)
        noise = [rng.gammavariate(alpha, 1.0) for _ in legal]
        noise_total = sum(noise)
        if noise_total > 0:
            noise = [n / noise_total for n in noise]
            priors = [
                (1.0 - dirichlet_eps) * p + dirichlet_eps * n for p, n in zip(priors, noise)
            ]

    node.children = {
        move: MCTSNode(to_play=opponent(node.to_play), prior=prior)
        for move, prior in zip(legal, priors)
    }
    node.expanded = True
    return value


def run_mcts(
    env: HexplodeEnv,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    state: State,
    to_play: int,
    sims: int,
    cpuct: float,
    device: torch.device,
    rng: random.Random,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    add_root_noise: bool,
) -> MCTSNode:
    root = MCTSNode(to_play=to_play, prior=1.0)
    _ = expand_node(
        env=env,
        model=model,
        adj=adj,
        node=root,
        state=state,
        device=device,
        rng=rng,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        add_root_noise=add_root_noise,
    )

    for _ in range(max(1, sims)):
        node = root
        sim_state = state
        search_path = [node]

        while node.expanded and (not node.is_terminal) and node.children:
            move, child = select_child(node, cpuct=cpuct, rng=rng)
            sim_state = env.apply_move(sim_state, move, node.to_play)
            node = child
            search_path.append(node)

        if not node.expanded:
            leaf_value = expand_node(
                env=env,
                model=model,
                adj=adj,
                node=node,
                state=sim_state,
                device=device,
                rng=rng,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                add_root_noise=False,
            )
        else:
            leaf_value = terminal_value_from_perspective(node.terminal_winner, node.to_play)

        value = leaf_value
        for path_node in reversed(search_path):
            path_node.visit_count += 1
            path_node.value_sum += value
            value = -value

    return root


def policy_from_root(root: MCTSNode, num_nodes: int, temperature: float) -> torch.Tensor:
    target = torch.zeros((num_nodes,), dtype=torch.float32)
    if not root.children:
        return target

    moves = list(root.children.keys())
    visits = [root.children[m].visit_count for m in moves]
    if max(visits) <= 0:
        probs = [1.0 / len(moves)] * len(moves)
    elif temperature <= 1e-3:
        best = max(visits)
        best_moves = [i for i, v in enumerate(visits) if v == best]
        probs = [0.0] * len(visits)
        share = 1.0 / len(best_moves)
        for idx in best_moves:
            probs[idx] = share
    else:
        inv_t = 1.0 / max(1e-3, temperature)
        powered = [float(v) ** inv_t for v in visits]
        total = sum(powered)
        probs = [v / total for v in powered] if total > 0 else [1.0 / len(visits)] * len(visits)

    for move, prob in zip(moves, probs):
        target[move] = float(prob)
    return target


def choose_move_from_policy(policy: torch.Tensor, legal_moves: Sequence[int], rng: random.Random) -> int:
    if not legal_moves:
        return -1
    probs = [max(0.0, float(policy[m])) for m in legal_moves]
    total = sum(probs)
    if total <= 0:
        return rng.choice(list(legal_moves))
    probs = [p / total for p in probs]
    return legal_moves[sample_index(probs, rng)]


def remap_sample(features: torch.Tensor, policy: torch.Tensor, remap: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    out_features = torch.zeros_like(features)
    out_policy = torch.zeros_like(policy)
    for src, dst in enumerate(remap):
        out_features[dst] = features[src]
        out_policy[dst] = policy[src]
    return out_features, out_policy


def finalize_winner(env: HexplodeEnv, state: State) -> int:
    winner = env.terminal_winner(state)
    if winner is not None:
        return winner
    red_score, blue_score = env.score_totals(state)
    if red_score > blue_score:
        return RED
    if blue_score > red_score:
        return BLUE
    return 0


def generate_selfplay_game(
    env: HexplodeEnv,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, int]], int, int]:
    state = env.empty_state()
    player = RED if rng.random() < 0.5 else BLUE
    trajectory: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

    for ply in range(args.max_turns):
        winner = env.terminal_winner(state)
        if winner is not None:
            return trajectory, winner, ply

        legal = env.legal_moves(state, player)
        if not legal:
            winner = finalize_winner(env, state)
            return trajectory, winner, ply

        root = run_mcts(
            env=env,
            model=model,
            adj=adj,
            state=state,
            to_play=player,
            sims=args.mcts_sims,
            cpuct=args.mcts_cpuct,
            device=device,
            rng=rng,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
            add_root_noise=True,
        )

        target_temp = args.target_temperature
        policy_target = policy_from_root(root, len(env.coords), temperature=target_temp)
        features = encode_node_features(env, state, player).float()
        trajectory.append((features, policy_target, player))

        play_temp = args.play_temperature if ply < args.play_temperature_drop else 1e-3
        action_policy = policy_from_root(root, len(env.coords), temperature=play_temp)
        move = choose_move_from_policy(action_policy, legal_moves=legal, rng=rng)
        if move < 0:
            winner = finalize_winner(env, state)
            return trajectory, winner, ply

        state = env.apply_move(state, move, player)
        player = opponent(player)

    winner = finalize_winner(env, state)
    return trajectory, winner, args.max_turns


def arena_play_game(
    env: HexplodeEnv,
    model_red: HexplodeGraphPolicyValueNet,
    model_blue: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    sims: int,
    cpuct: float,
    max_turns: int,
    device: torch.device,
    rng: random.Random,
) -> int:
    state = env.empty_state()
    player = RED
    for _ in range(max_turns):
        winner = env.terminal_winner(state)
        if winner is not None:
            return winner
        legal = env.legal_moves(state, player)
        if not legal:
            return finalize_winner(env, state)

        model = model_red if player == RED else model_blue
        root = run_mcts(
            env=env,
            model=model,
            adj=adj,
            state=state,
            to_play=player,
            sims=sims,
            cpuct=cpuct,
            device=device,
            rng=rng,
            dirichlet_alpha=0.0,
            dirichlet_eps=0.0,
            add_root_noise=False,
        )
        policy = policy_from_root(root, len(env.coords), temperature=1e-3)
        move = choose_move_from_policy(policy, legal_moves=legal, rng=rng)
        if move < 0:
            return finalize_winner(env, state)
        state = env.apply_move(state, move, player)
        player = opponent(player)
    return finalize_winner(env, state)


def evaluate_arena(
    env: HexplodeEnv,
    candidate: HexplodeGraphPolicyValueNet,
    best_model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
) -> Dict[str, float]:
    candidate_wins = 0
    best_wins = 0
    draws = 0

    for game_idx in range(args.arena_games):
        if game_idx % 2 == 0:
            winner = arena_play_game(
                env=env,
                model_red=candidate,
                model_blue=best_model,
                adj=adj,
                sims=args.arena_sims,
                cpuct=args.mcts_cpuct,
                max_turns=args.max_turns,
                device=device,
                rng=rng,
            )
            if winner == RED:
                candidate_wins += 1
            elif winner == BLUE:
                best_wins += 1
            else:
                draws += 1
        else:
            winner = arena_play_game(
                env=env,
                model_red=best_model,
                model_blue=candidate,
                adj=adj,
                sims=args.arena_sims,
                cpuct=args.mcts_cpuct,
                max_turns=args.max_turns,
                device=device,
                rng=rng,
            )
            if winner == BLUE:
                candidate_wins += 1
            elif winner == RED:
                best_wins += 1
            else:
                draws += 1

    total = max(1, args.arena_games)
    win_rate = (candidate_wins + 0.5 * draws) / total
    return {
        "candidate_wins": float(candidate_wins),
        "best_wins": float(best_wins),
        "draws": float(draws),
        "win_rate": float(win_rate),
    }


def save_checkpoint(
    path: Path,
    model: HexplodeGraphPolicyValueNet,
    env: HexplodeEnv,
    args: argparse.Namespace,
    meta: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "blocks": model.blocks,
        "num_nodes": model.num_nodes,
        "side_length": env.side_length,
        "meta": meta,
        "train_args": vars(args),
    }
    torch.save(payload, path)


def load_or_create_model(args: argparse.Namespace, env: HexplodeEnv, device: torch.device) -> Tuple[HexplodeGraphPolicyValueNet, str]:
    target_input_dim = GNN_FEATURE_DIM
    ckpt_raw = (args.bootstrap_checkpoint or "").strip()
    if ckpt_raw:
        ckpt_path = Path(ckpt_raw)
        if ckpt_path.exists() and ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            loaded_input_dim = int(ckpt.get("input_dim", target_input_dim))
            model = HexplodeGraphPolicyValueNet(
                input_dim=target_input_dim,
                hidden_dim=int(ckpt.get("hidden_dim", args.hidden_dim)),
                blocks=int(ckpt.get("blocks", args.blocks)),
                num_nodes=int(ckpt.get("num_nodes", len(env.coords))),
            )
            state_dict = ckpt["state_dict"]
            if loaded_input_dim != target_input_dim and "input_proj.weight" in state_dict:
                old_weight = state_dict["input_proj.weight"]
                new_weight = torch.zeros(
                    (old_weight.shape[0], target_input_dim),
                    dtype=old_weight.dtype,
                )
                copy_dim = min(old_weight.shape[1], target_input_dim)
                new_weight[:, :copy_dim] = old_weight[:, :copy_dim]
                state_dict = dict(state_dict)
                state_dict["input_proj.weight"] = new_weight
                print(
                    f"[Init] Adjusted input_proj weights from {old_weight.shape[1]} to {target_input_dim} features."
                )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[Init] checkpoint load warnings missing={missing} unexpected={unexpected}")
            model.to(device)
            return model, str(ckpt_path)

    model = HexplodeGraphPolicyValueNet(
        input_dim=target_input_dim,
        hidden_dim=args.hidden_dim,
        blocks=args.blocks,
        num_nodes=len(env.coords),
    )
    model.to(device)
    return model, ""


def maybe_subset_samples(replay: Sequence[Sample], cap: int, rng: random.Random) -> List[Sample]:
    if cap <= 0 or len(replay) <= cap:
        return list(replay)
    indices = list(range(len(replay)))
    rng.shuffle(indices)
    return [replay[i] for i in indices[:cap]]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = choose_device(args.device)

    env = HexplodeEnv(side_length=args.side_length)
    adj = build_adj_matrix(env).to(device)
    rotations: RotationMaps = build_rotation_maps(env, num_rotations=max(1, min(6, args.symmetry_rotations)))
    stop_file = Path(args.stop_file) if args.stop_file else None

    model, loaded_from = load_or_create_model(args, env, device)
    best_model = copy.deepcopy(model).to(device)
    best_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    replay: Deque[Sample] = deque(maxlen=max(1000, args.replay_size))
    stats: List[Dict] = []
    start_time = time.time()
    deadline = start_time + max(0.0, args.hours) * 3600.0

    print(
        f"[Init] device={device}, side={args.side_length}, nodes={len(env.coords)}, "
        f"hidden={args.hidden_dim}, blocks={args.blocks}, mcts_sims={args.mcts_sims}, "
        f"loaded_from={loaded_from or 'fresh'}"
    )

    round_idx = 0
    while time.time() < deadline:
        round_idx += 1
        round_start = time.time()
        round_samples: List[Sample] = []
        game_lengths: List[int] = []
        winners = {RED: 0, BLUE: 0, 0: 0}

        for game_idx in range(args.round_games):
            trajectory, winner, turns = generate_selfplay_game(env, model, adj, args, device, rng)
            game_lengths.append(turns)
            winners[winner] = winners.get(winner, 0) + 1

            for features, policy, player in trajectory:
                value = terminal_value_from_perspective(winner, player)
                for remap in rotations.maps:
                    feat_r, pol_r = remap_sample(features, policy, remap)
                    sample = Sample(features=feat_r, policy=pol_r, value=float(value))
                    round_samples.append(sample)
                    replay.append(sample)

            if (game_idx + 1) % max(1, args.progress_every) == 0:
                elapsed = time.time() - round_start
                print(
                    f"[Round {round_idx}] self-play {game_idx + 1}/{args.round_games} "
                    f"games, replay={len(replay)}, elapsed={format_duration(elapsed)}"
                )

        train_pool = maybe_subset_samples(list(replay), args.train_samples_cap, rng)
        if len(train_pool) < 128:
            print(f"[Round {round_idx}] Not enough samples to train yet ({len(train_pool)}).")
            continue

        x = torch.stack([s.features for s in train_pool], dim=0).to(torch.float32)
        y_policy = torch.stack([s.policy for s in train_pool], dim=0).to(torch.float32)
        y_value = torch.tensor([s.value for s in train_pool], dtype=torch.float32)
        dataset = TensorDataset(x, y_policy, y_value)

        val_len = max(1, int(len(dataset) * max(0.0, min(0.5, args.val_split))))
        train_len = len(dataset) - val_len
        if train_len <= 0:
            train_len = len(dataset) - 1
            val_len = 1
        train_set, val_set = random_split(
            dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed + round_idx)
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        round_train_loss = 0.0
        round_train_policy = 0.0
        round_train_value = 0.0
        train_batches = 0

        for epoch in range(1, args.round_epochs + 1):
            model.train()
            for batch_idx, (xb, yb_pol, yb_val) in enumerate(train_loader, start=1):
                xb = xb.to(device)
                yb_pol = yb_pol.to(device)
                yb_val = yb_val.to(device)

                logits, value_pred = model(xb, adj)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(yb_pol * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred, yb_val)
                loss = (args.policy_loss_weight * policy_loss) + (args.value_loss_weight * value_loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                round_train_loss += float(loss.item())
                round_train_policy += float(policy_loss.item())
                round_train_value += float(value_loss.item())
                train_batches += 1

                if batch_idx % max(1, args.log_batches_every) == 0:
                    print(
                        f"[Round {round_idx}][Epoch {epoch}] batch {batch_idx}/{len(train_loader)} "
                        f"loss={loss.item():.4f} pol={policy_loss.item():.4f} val={value_loss.item():.4f}"
                    )

        model.eval()
        val_loss_total = 0.0
        val_policy_total = 0.0
        val_value_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb_pol, yb_val in val_loader:
                xb = xb.to(device)
                yb_pol = yb_pol.to(device)
                yb_val = yb_val.to(device)
                logits, value_pred = model(xb, adj)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(yb_pol * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred, yb_val)
                loss = (args.policy_loss_weight * policy_loss) + (args.value_loss_weight * value_loss)
                val_loss_total += float(loss.item())
                val_policy_total += float(policy_loss.item())
                val_value_total += float(value_loss.item())
                val_batches += 1

        arena = evaluate_arena(env, model, best_model, adj, args, device, rng)
        promoted = arena["win_rate"] >= args.promote_threshold
        if promoted:
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))

        elapsed_total = time.time() - start_time
        elapsed_round = time.time() - round_start
        avg_len = sum(game_lengths) / max(1, len(game_lengths))

        record = {
            "round": round_idx,
            "elapsed_total_sec": elapsed_total,
            "elapsed_round_sec": elapsed_round,
            "selfplay_games": args.round_games,
            "replay_size": len(replay),
            "avg_game_len": avg_len,
            "red_wins": winners.get(RED, 0),
            "blue_wins": winners.get(BLUE, 0),
            "draws": winners.get(0, 0),
            "train_loss": round_train_loss / max(1, train_batches),
            "train_policy_loss": round_train_policy / max(1, train_batches),
            "train_value_loss": round_train_value / max(1, train_batches),
            "val_loss": val_loss_total / max(1, val_batches),
            "val_policy_loss": val_policy_total / max(1, val_batches),
            "val_value_loss": val_value_total / max(1, val_batches),
            "arena_candidate_wins": arena["candidate_wins"],
            "arena_best_wins": arena["best_wins"],
            "arena_draws": arena["draws"],
            "arena_win_rate": arena["win_rate"],
            "promoted": promoted,
        }
        stats.append(record)

        print(
            f"[Round {round_idx}] games={args.round_games} avg_len={avg_len:.1f} "
            f"W/D/L(red={winners.get(RED, 0)},blue={winners.get(BLUE, 0)},draw={winners.get(0, 0)}) | "
            f"train={record['train_loss']:.4f} val={record['val_loss']:.4f} | "
            f"arena_wr={record['arena_win_rate']:.3f} promoted={promoted} | "
            f"round_time={format_duration(elapsed_round)} total={format_duration(elapsed_total)}"
        )

        out_path = Path(args.out)
        best_out_path = Path(args.best_out)
        stats_path = Path(args.stats_out)

        save_checkpoint(out_path, model, env, args, meta={"round": round_idx, "record": record, "type": "candidate"})
        if promoted or (not best_out_path.exists()):
            save_checkpoint(
                best_out_path,
                best_model,
                env,
                args,
                meta={
                    "round": round_idx,
                    "record": record,
                    "type": "best" if promoted else "best-initial",
                },
            )

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps({"args": vars(args), "stats": stats}, indent=2))

        if stop_file and stop_file.exists():
            print(f"[Stop] Stop file detected: {stop_file}")
            break

    if not Path(args.best_out).exists():
        save_checkpoint(
            Path(args.best_out),
            best_model,
            env,
            args,
            meta={"round": round_idx, "type": "best-final-fallback"},
        )

    print(
        f"[Done] rounds={round_idx}, total_time={format_duration(time.time() - start_time)}, "
        f"latest='{args.out}', best='{args.best_out}', stats='{args.stats_out}'"
    )


if __name__ == "__main__":
    main()
