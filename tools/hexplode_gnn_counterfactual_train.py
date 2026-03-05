#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from hexplode_counterfactual_train import (
    build_random_state,
    choose_device,
    choose_player_with_legal,
    format_duration,
    label_from_outcome,
    minimax_score_diff,
    opponent_of,
    score_diff_for_player,
    value_target_from_outcome,
)
from hexplode_gnn import (
    HexplodeGraphPolicyValueNet,
    build_adj_matrix,
    build_rotation_maps,
    clone_gnn_model,
    encode_node_features,
    rotate_state,
)
from hexplode_nn import HexplodeEnv, State


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Hexplode Graph NN (policy-value) with counterfactual rollout labels "
            "and optional 6x rotational augmentation."
        )
    )
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--side-length", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.45)
    parser.add_argument("--round-samples", type=int, default=12000)
    parser.add_argument("--round-epochs", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.12)
    parser.add_argument("--log-batches-every", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--start-source", type=str, default="selfplay", choices=["selfplay", "random"])
    parser.add_argument("--prefix-random-moves", type=int, default=30)
    parser.add_argument("--prefix-policy-min-moves", type=int, default=20)
    parser.add_argument("--prefix-policy-max-moves", type=int, default=120)
    parser.add_argument("--prefix-policy-epsilon", type=float, default=0.0)
    parser.add_argument("--sample-move-epsilon", type=float, default=0.02)
    parser.add_argument("--symmetry-rotations", type=int, default=6)
    parser.add_argument("--empty-prob", type=float, default=0.46)
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=140)
    parser.add_argument("--good-rule", type=str, default="after_positive", choices=["after_positive", "delta_nonnegative", "both"])
    parser.add_argument("--good-margin", type=float, default=0.0)
    parser.add_argument("--value-target", type=str, default="binary", choices=["binary", "scaled_after", "scaled_delta"])
    parser.add_argument("--value-scale", type=float, default=12.0)
    parser.add_argument("--max-opponent-checks", type=int, default=0)
    parser.add_argument("--lookahead-ply", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--eval-device",
        type=str,
        default="cpu",
        choices=["same", "cpu"],
        help="Device used for validation pass. Use cpu to avoid accelerator numeric instability in val metrics.",
    )
    parser.add_argument(
        "--max-valid-loss",
        type=float,
        default=1e6,
        help="Reject validation batches with loss magnitude above this threshold.",
    )
    parser.add_argument("--bootstrap-checkpoint", type=str, default="")
    parser.add_argument("--prefix-selfplay-checkpoint", type=str, default="")
    parser.add_argument("--out", type=str, default="tools/outputs/hexplode_gnn_counterfactual.pt")
    parser.add_argument("--best-out", type=str, default="")
    parser.add_argument("--stop-file", type=str, default="tools/outputs/hexplode_gnn_counterfactual.stop")
    parser.add_argument("--min-round-seconds", type=int, default=300)
    return parser.parse_args()


def load_model(args: argparse.Namespace, num_nodes: int, device: torch.device) -> Tuple[HexplodeGraphPolicyValueNet, str]:
    ckpt_raw = (args.bootstrap_checkpoint or "").strip()
    ckpt_path = Path(ckpt_raw) if ckpt_raw else None
    model = HexplodeGraphPolicyValueNet(
        input_dim=8,
        hidden_dim=args.hidden_dim,
        blocks=args.blocks,
        num_nodes=num_nodes,
    )
    loaded_from = ""
    if ckpt_path is not None and ckpt_path.exists() and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = HexplodeGraphPolicyValueNet(
            input_dim=int(ckpt.get("input_dim", 8)),
            hidden_dim=int(ckpt.get("hidden_dim", args.hidden_dim)),
            blocks=int(ckpt.get("blocks", args.blocks)),
            num_nodes=int(ckpt.get("num_nodes", num_nodes)),
        )
        model.load_state_dict(ckpt["state_dict"])
        loaded_from = str(ckpt_path)
    model.to(device)
    return model, loaded_from


def load_fixed_model(checkpoint_path: str, fallback: HexplodeGraphPolicyValueNet, device: torch.device) -> Tuple[HexplodeGraphPolicyValueNet, str]:
    ckpt_raw = (checkpoint_path or "").strip()
    if not ckpt_raw:
        return fallback, ""
    path = Path(ckpt_raw)
    if not path.exists() or not path.is_file():
        print(f"Warning: prefix-selfplay-checkpoint not found: {path}; using current model.")
        return fallback, ""
    ckpt = torch.load(path, map_location="cpu")
    model = HexplodeGraphPolicyValueNet(
        input_dim=int(ckpt.get("input_dim", 8)),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        blocks=int(ckpt.get("blocks", 6)),
        num_nodes=int(ckpt.get("num_nodes", 37)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, str(path)


def choose_model_move_gnn(
    env: HexplodeEnv,
    state: State,
    player: int,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    device: torch.device,
    rng: random.Random,
    epsilon: float,
) -> int:
    legal = env.legal_moves(state, player)
    if not legal:
        return -1
    if epsilon > 0.0 and rng.random() < epsilon:
        return rng.choice(legal)

    model.eval()
    with torch.no_grad():
        xb = encode_node_features(env, state, player).unsqueeze(0).to(device)
        logits, _ = model(xb, adj)
    row = logits[0].detach().cpu()
    best_score = -1e30
    best_moves: List[int] = []
    for mv in legal:
        score = float(row[mv])
        if score > best_score + 1e-9:
            best_score = score
            best_moves = [mv]
        elif abs(score - best_score) <= 1e-9:
            best_moves.append(mv)
    return rng.choice(best_moves) if best_moves else legal[0]


def choose_random_move(env: HexplodeEnv, state: State, player: int, rng: random.Random) -> int:
    legal = env.legal_moves(state, player)
    if not legal:
        return -1
    return rng.choice(legal)


@dataclass
class StartPosition:
    state: State
    to_move: int
    source: str
    random_prefix_moves: int
    policy_prefix_moves: int


def build_selfplay_prefix_state_gnn(
    env: HexplodeEnv,
    rng: random.Random,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    device: torch.device,
    random_warmup_moves: int,
    policy_min_moves: int,
    policy_max_moves: int,
    epsilon: float,
) -> StartPosition | None:
    random_warmup_moves = max(0, random_warmup_moves)
    policy_min_moves = max(0, policy_min_moves)
    policy_max_moves = max(policy_min_moves, policy_max_moves)
    target_policy_moves = rng.randint(policy_min_moves, policy_max_moves)
    state = env.empty_state()
    player = 1 if rng.random() < 0.5 else 2
    random_done = 0
    policy_done = 0

    for _ in range(random_warmup_moves):
        if env.terminal_winner(state) is not None:
            return None
        mv = choose_random_move(env, state, player, rng)
        if mv < 0:
            other = opponent_of(player)
            mv = choose_random_move(env, state, other, rng)
            if mv < 0:
                return None
            player = other
        state = env.apply_move(state, mv, player)
        random_done += 1
        player = opponent_of(player)

    for _ in range(target_policy_moves):
        if env.terminal_winner(state) is not None:
            return None
        mv = choose_model_move_gnn(env, state, player, model, adj, device, rng, epsilon)
        if mv < 0:
            other = opponent_of(player)
            mv = choose_model_move_gnn(env, state, other, model, adj, device, rng, epsilon)
            if mv < 0:
                return None
            player = other
        state = env.apply_move(state, mv, player)
        policy_done += 1
        player = opponent_of(player)

    if env.terminal_winner(state) is not None:
        return None
    if not env.legal_moves(state, player):
        other = opponent_of(player)
        if not env.legal_moves(state, other):
            return None
        player = other
    return StartPosition(
        state=state,
        to_move=player,
        source="selfplay",
        random_prefix_moves=random_done,
        policy_prefix_moves=policy_done,
    )


def evaluate_model_move_with_rollout_gnn(
    env: HexplodeEnv,
    state: State,
    player: int,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    device: torch.device,
    rng: random.Random,
    move_epsilon: float,
    good_rule: str,
    margin: float,
    value_target_mode: str,
    value_scale: float,
    lookahead_ply: int,
    max_branch_checks: int,
) -> Tuple[int, float, float, int, int, int]:
    legal = env.legal_moves(state, player)
    if not legal:
        return -1, 0.0, -1.0, 0, 0, 0
    move = choose_model_move_gnn(
        env=env,
        state=state,
        player=player,
        model=model,
        adj=adj,
        device=device,
        rng=rng,
        epsilon=move_epsilon,
    )
    before_diff = score_diff_for_player(env, state, player)
    next_state = env.apply_move(state, move, player)
    opponent = opponent_of(player)
    opponent_branches = len(env.legal_moves(next_state, opponent))

    if lookahead_ply <= 0:
        after_rollout_diff = score_diff_for_player(env, next_state, player)
    else:
        after_rollout_diff = minimax_score_diff(
            env=env,
            state=next_state,
            to_move=opponent,
            root_player=player,
            plies_left=lookahead_ply,
            max_branch_checks=max_branch_checks,
            cache={},
        )

    policy_label = label_from_outcome(before_diff, after_rollout_diff, good_rule, margin)
    value_target = value_target_from_outcome(
        before_diff=before_diff,
        after_reply_diff=after_rollout_diff,
        policy_label=policy_label,
        value_target_mode=value_target_mode,
        value_scale=value_scale,
    )
    return move, policy_label, value_target, before_diff, after_rollout_diff, opponent_branches


def generate_round_dataset(
    env: HexplodeEnv,
    adj: torch.Tensor,
    model: HexplodeGraphPolicyValueNet,
    prefix_model: HexplodeGraphPolicyValueNet,
    device: torch.device,
    samples: int,
    seed: int,
    start_source: str,
    prefix_random_moves: int,
    prefix_policy_min_moves: int,
    prefix_policy_max_moves: int,
    prefix_policy_epsilon: float,
    sample_move_epsilon: float,
    symmetry_rotations: int,
    empty_prob: float,
    max_level: int,
    min_turns: int,
    max_turns: int,
    good_rule: str,
    margin: float,
    value_target_mode: str,
    value_scale: float,
    lookahead_ply: int,
    max_branch_checks: int,
    progress_every: int,
    stop_file: Path | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, bool]:
    rng = random.Random(seed)
    model.eval()
    features: List[torch.Tensor] = []
    move_indices: List[int] = []
    policy_labels: List[float] = []
    value_targets: List[float] = []

    attempts = 0
    max_attempts = max(samples * 120, 40_000)
    good_count = 0
    bad_count = 0
    base_samples = 0
    source_random = 0
    source_selfplay = 0
    total_random_prefix_moves = 0
    total_policy_prefix_moves = 0
    total_before_diff = 0.0
    total_after_diff = 0.0
    total_responses = 0
    started = time.monotonic()
    stop_requested = False
    rotation_maps = build_rotation_maps(env, symmetry_rotations).maps

    while len(features) < samples and attempts < max_attempts:
        attempts += 1
        if stop_file is not None and stop_file.exists() and len(features) > 0:
            stop_requested = True
            print(f"  stop-file detected during generation ({stop_file})")
            break

        start: StartPosition | None = None
        if start_source == "selfplay":
            start = build_selfplay_prefix_state_gnn(
                env=env,
                rng=rng,
                model=prefix_model,
                adj=adj,
                device=device,
                random_warmup_moves=prefix_random_moves,
                policy_min_moves=prefix_policy_min_moves,
                policy_max_moves=prefix_policy_max_moves,
                epsilon=prefix_policy_epsilon,
            )
        else:
            state = build_random_state(
                env=env,
                rng=rng,
                empty_prob=empty_prob,
                max_level=max_level,
                min_turns=min_turns,
                max_turns=max_turns,
            )
            if env.terminal_winner(state) is None:
                player = choose_player_with_legal(env, state, rng)
                if player is not None and env.legal_moves(state, player):
                    start = StartPosition(
                        state=state,
                        to_move=player,
                        source="random",
                        random_prefix_moves=0,
                        policy_prefix_moves=0,
                    )

        if start is None:
            continue

        move, policy_label, value_target, before_diff, after_reply_diff, response_count = evaluate_model_move_with_rollout_gnn(
            env=env,
            state=start.state,
            player=start.to_move,
            model=model,
            adj=adj,
            device=device,
            rng=rng,
            move_epsilon=sample_move_epsilon,
            good_rule=good_rule,
            margin=margin,
            value_target_mode=value_target_mode,
            value_scale=value_scale,
            lookahead_ply=lookahead_ply,
            max_branch_checks=max_branch_checks,
        )
        if move < 0:
            continue

        base_samples += 1
        if start.source == "selfplay":
            source_selfplay += 1
        else:
            source_random += 1
        total_random_prefix_moves += start.random_prefix_moves
        total_policy_prefix_moves += start.policy_prefix_moves
        if policy_label >= 0.5:
            good_count += 1
        else:
            bad_count += 1
        total_before_diff += before_diff
        total_after_diff += after_reply_diff
        total_responses += response_count

        for remap in rotation_maps:
            if len(features) >= samples:
                break
            rotated_state = rotate_state(start.state, remap)
            rotated_move = remap[move]
            features.append(encode_node_features(env, rotated_state, start.to_move))
            move_indices.append(rotated_move)
            policy_labels.append(policy_label)
            value_targets.append(value_target)

        if stop_file is not None and stop_file.exists() and len(features) > 0:
            stop_requested = True
            print(f"  stop-file detected during generation ({stop_file})")
            break

        sample_count = len(features)
        if progress_every > 0 and sample_count % progress_every == 0:
            elapsed = time.monotonic() - started
            rate = sample_count / max(1e-6, elapsed)
            print(
                f"  gen {sample_count}/{samples} ({100.0 * sample_count / samples:5.1f}%) "
                f"base={base_samples} rot={len(rotation_maps)} "
                f"good={good_count} bad={bad_count} "
                f"goodRate={good_count / max(1, base_samples):.3f} "
                f"avgBefore={total_before_diff / max(1, base_samples):+.2f} "
                f"avgAfter={total_after_diff / max(1, base_samples):+.2f} "
                f"avgOppReplies={total_responses / max(1, base_samples):.2f} "
                f"avgRndPrefix={total_random_prefix_moves / max(1, base_samples):.1f} "
                f"avgPolPrefix={total_policy_prefix_moves / max(1, base_samples):.1f} "
                f"rate={rate:.1f}/s"
            )

    if not features:
        raise RuntimeError("Round dataset generation produced zero samples.")

    x = torch.stack(features, dim=0)  # [S, N, F]
    y_move_idx = torch.tensor(move_indices, dtype=torch.long)
    y_policy = torch.tensor(policy_labels, dtype=torch.float32)
    y_value = torch.tensor(value_targets, dtype=torch.float32)
    stats = {
        "samples": len(features),
        "base_samples": base_samples,
        "symmetry_rotations": len(rotation_maps),
        "start_source": start_source,
        "source_selfplay": source_selfplay,
        "source_random": source_random,
        "attempts": attempts,
        "accept_rate": len(features) / max(1, attempts),
        "good_count": good_count,
        "bad_count": bad_count,
        "good_rate": good_count / max(1, base_samples),
        "avg_before_diff": total_before_diff / max(1, base_samples),
        "avg_after_diff": total_after_diff / max(1, base_samples),
        "avg_opponent_responses": total_responses / max(1, base_samples),
        "avg_random_prefix_moves": total_random_prefix_moves / max(1, base_samples),
        "avg_policy_prefix_moves": total_policy_prefix_moves / max(1, base_samples),
        "lookahead_ply": lookahead_ply,
        "stopped_early": stop_requested,
    }
    return x, y_move_idx, y_policy, y_value, stats, stop_requested


def evaluate(
    model: HexplodeGraphPolicyValueNet,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    value_loss_weight: float,
    max_valid_loss: float,
) -> dict:
    model.eval()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    invalid_batches = 0
    with torch.no_grad():
        for xb, move_idx, y_policy, y_value in loader:
            xb = xb.to(device, non_blocking=True)
            move_idx = move_idx.to(device, non_blocking=True)
            y_policy = y_policy.to(device, non_blocking=True)
            y_value = y_value.to(device, non_blocking=True)

            logits, value = model(xb, adj)
            if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                invalid_batches += 1
                continue
            selected_logits = logits.gather(1, move_idx.unsqueeze(1)).squeeze(1)
            policy_loss = F.binary_cross_entropy_with_logits(selected_logits, y_policy)
            value_loss = F.mse_loss(value, y_value)
            loss = policy_loss + value_loss_weight * value_loss
            if not torch.isfinite(policy_loss) or not torch.isfinite(value_loss) or not torch.isfinite(loss):
                invalid_batches += 1
                continue
            loss_item = float(loss.item())
            if not math.isfinite(loss_item) or abs(loss_item) > max_valid_loss:
                invalid_batches += 1
                continue
            pred = (torch.sigmoid(selected_logits) >= 0.5).float()
            acc = (pred == (y_policy >= 0.5).float()).float().mean()

            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_loss += loss_item
            total_acc += acc.item()
            batches += 1

    if batches == 0:
        return {
            "loss": float("inf"),
            "policy_loss": float("inf"),
            "value_loss": float("inf"),
            "policy_acc": 0.0,
            "valid": False,
            "invalid_batches": invalid_batches,
        }

    denom = max(1, batches)
    return {
        "loss": total_loss / denom,
        "policy_loss": total_policy / denom,
        "value_loss": total_value / denom,
        "policy_acc": total_acc / denom,
        "valid": invalid_batches == 0,
        "invalid_batches": invalid_batches,
    }


def train_round(
    model: HexplodeGraphPolicyValueNet,
    device: torch.device,
    adj: torch.Tensor,
    eval_device: torch.device,
    adj_eval: torch.Tensor,
    x: torch.Tensor,
    y_move_idx: torch.Tensor,
    y_policy: torch.Tensor,
    y_value: torch.Tensor,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    value_loss_weight: float,
    val_split: float,
    log_batches_every: int,
    max_valid_loss: float,
) -> Tuple[List[dict], float]:
    dataset = TensorDataset(x, y_move_idx, y_policy, y_value)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(len(dataset)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_size > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[dict] = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_policy = 0.0
        running_value = 0.0
        running_total = 0.0
        running_acc = 0.0
        batches = 0
        total_batches = max(1, len(train_loader))
        epoch_start = time.monotonic()

        for batch_idx, (xb, move_idx, pb, vb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            move_idx = move_idx.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)

            logits, value = model(xb, adj)
            selected_logits = logits.gather(1, move_idx.unsqueeze(1)).squeeze(1)
            policy_loss = F.binary_cross_entropy_with_logits(selected_logits, pb)
            value_loss = F.mse_loss(value, vb)
            loss = policy_loss + value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pred = (torch.sigmoid(selected_logits) >= 0.5).float()
            acc = (pred == (pb >= 0.5).float()).float().mean()

            running_policy += policy_loss.item()
            running_value += value_loss.item()
            running_total += loss.item()
            running_acc += acc.item()
            batches += 1

            if log_batches_every > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % log_batches_every == 0):
                elapsed = time.monotonic() - epoch_start
                eta = (elapsed / max(1, batch_idx)) * max(0, total_batches - batch_idx)
                print(
                    f"    epoch {epoch:02d} batch {batch_idx:04d}/{total_batches:04d} "
                    f"({100.0 * batch_idx / total_batches:5.1f}%) "
                    f"loss={running_total / batches:.4f} "
                    f"p={running_policy / batches:.4f} "
                    f"v={running_value / batches:.4f} "
                    f"acc={running_acc / batches:.4f} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                )

        train_metrics = {
            "loss": running_total / max(1, batches),
            "policy_loss": running_policy / max(1, batches),
            "value_loss": running_value / max(1, batches),
            "policy_acc": running_acc / max(1, batches),
        }
        if val_loader is not None:
            if eval_device == device:
                val_model = model
            else:
                val_model = copy.deepcopy(model).to(eval_device)
                val_model.eval()

            val_metrics = evaluate(
                model=val_model,
                loader=val_loader,
                adj=adj_eval,
                device=eval_device,
                value_loss_weight=value_loss_weight,
                max_valid_loss=max_valid_loss,
            )
            if eval_device != device:
                del val_model

            if val_metrics["valid"] and math.isfinite(val_metrics["loss"]):
                best_val = min(best_val, float(val_metrics["loss"]))
            else:
                print(
                    f"    warning: invalid validation metrics at epoch {epoch:02d} "
                    f"(invalid_batches={val_metrics.get('invalid_batches', 0)}); ignoring for best selection"
                )
        else:
            val_metrics = {
                "loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "policy_acc": float("nan"),
                "valid": True,
                "invalid_batches": 0,
            }

        summary = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(summary)
        print(
            f"  epoch {epoch:02d} "
            f"train(loss={train_metrics['loss']:.4f}, p={train_metrics['policy_loss']:.4f}, "
            f"v={train_metrics['value_loss']:.4f}, acc={train_metrics['policy_acc']:.4f}) "
            f"val(loss={val_metrics['loss']:.4f}, p={val_metrics['policy_loss']:.4f}, "
            f"v={val_metrics['value_loss']:.4f}, acc={val_metrics['policy_acc']:.4f})"
        )

    return history, best_val


def save_checkpoint(
    path: Path,
    model: HexplodeGraphPolicyValueNet,
    history: List[dict],
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "blocks": model.blocks,
        "num_nodes": model.num_nodes,
        "meta": meta,
        "history": history,
    }
    torch.save(checkpoint, path)
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    env = HexplodeEnv(side_length=max(2, args.side_length))
    num_nodes = len(env.coords)
    adj = build_adj_matrix(env).to(device)
    eval_device = device if args.eval_device == "same" else torch.device("cpu")
    adj_eval = adj if eval_device == device else build_adj_matrix(env).to(eval_device)

    model, loaded_from = load_model(args, num_nodes=num_nodes, device=device)
    prefix_seed_model, prefix_loaded_from = load_fixed_model(args.prefix_selfplay_checkpoint, model, device)
    best_prefix_model = clone_gnn_model(prefix_seed_model, device)

    out_path = Path(args.out)
    best_out_path = Path(args.best_out) if args.best_out else out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")
    stop_file = Path(args.stop_file) if args.stop_file else None

    started_at = time.time()
    deadline = started_at + args.hours * 3600.0
    round_index = 0
    all_history: List[dict] = []
    best_val = float("inf")
    best_round = 0

    print(
        f"Starting GNN counterfactual training on {device}. "
        f"budget={args.hours:.2f}h lookahead_ply={args.lookahead_ply} "
        f"start_source={args.start_source} rotations={max(1, min(6, args.symmetry_rotations))} "
        f"bootstrap={'yes' if loaded_from else 'no'} eval_device={eval_device}"
    )
    if loaded_from:
        print(f"Loaded bootstrap checkpoint: {loaded_from}")
    if prefix_loaded_from:
        print(f"Loaded initial prefix self-play checkpoint: {prefix_loaded_from}")
    print(f"Best checkpoint path: {best_out_path}")
    if stop_file is not None:
        print(f"Stop file path: {stop_file}")

    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        if remaining < max(30.0, float(args.min_round_seconds)):
            print("Remaining budget too small for another full round; stopping.")
            break
        if stop_file is not None and stop_file.exists():
            print("Stop file already exists; stopping before starting next round.")
            break

        round_index += 1
        elapsed = time.time() - started_at
        print(
            f"\nRound {round_index:02d} | elapsed={format_duration(elapsed)} "
            f"remaining={format_duration(remaining)}"
        )

        x, y_move_idx, y_policy, y_value, stats, generation_stopped = generate_round_dataset(
            env=env,
            adj=adj,
            model=model,
            prefix_model=best_prefix_model,
            device=device,
            samples=args.round_samples,
            seed=args.seed + (round_index * 101),
            start_source=args.start_source,
            prefix_random_moves=max(0, args.prefix_random_moves),
            prefix_policy_min_moves=max(0, args.prefix_policy_min_moves),
            prefix_policy_max_moves=max(0, args.prefix_policy_max_moves),
            prefix_policy_epsilon=max(0.0, min(1.0, args.prefix_policy_epsilon)),
            sample_move_epsilon=max(0.0, min(1.0, args.sample_move_epsilon)),
            symmetry_rotations=max(1, min(6, args.symmetry_rotations)),
            empty_prob=args.empty_prob,
            max_level=args.max_level,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            good_rule=args.good_rule,
            margin=args.good_margin,
            value_target_mode=args.value_target,
            value_scale=args.value_scale,
            lookahead_ply=max(0, args.lookahead_ply),
            max_branch_checks=args.max_opponent_checks,
            progress_every=max(0, args.progress_every),
            stop_file=stop_file,
        )
        print(
            f"  round dataset: samples={stats['samples']} attempts={stats['attempts']} "
            f"base={stats['base_samples']} rot={stats['symmetry_rotations']} "
            f"acceptRate={stats['accept_rate']:.3f} goodRate={stats['good_rate']:.3f} "
            f"avgBefore={stats['avg_before_diff']:+.2f} avgAfter={stats['avg_after_diff']:+.2f} "
            f"avgOppReplies={stats['avg_opponent_responses']:.2f} "
            f"avgRndPrefix={stats['avg_random_prefix_moves']:.1f} "
            f"avgPolPrefix={stats['avg_policy_prefix_moves']:.1f}"
        )

        model.to(device)
        round_history, round_best_val = train_round(
            model=model,
            device=device,
            adj=adj,
            eval_device=eval_device,
            adj_eval=adj_eval,
            x=x,
            y_move_idx=y_move_idx,
            y_policy=y_policy,
            y_value=y_value,
            batch_size=args.batch_size,
            epochs=args.round_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            value_loss_weight=args.value_loss_weight,
            val_split=args.val_split,
            log_batches_every=max(0, args.log_batches_every),
            max_valid_loss=max(1.0, float(args.max_valid_loss)),
        )
        improved = math.isfinite(round_best_val) and (round_best_val < best_val)
        if improved:
            best_val = round_best_val
            best_round = round_index
            best_prefix_model.load_state_dict(copy.deepcopy(model.state_dict()))
            best_prefix_model.to(device)
            best_prefix_model.eval()
            print(f"  new best validation loss: {best_val:.4f} at round {best_round}")
        else:
            print(f"  best validation loss remains {best_val:.4f} from round {best_round}")
        all_history.append({"round": round_index, "stats": stats, "train_history": round_history})

        meta = {
            "mode": "gnn_counterfactual_rollout",
            "bootstrap_checkpoint": loaded_from,
            "prefix_selfplay_checkpoint": prefix_loaded_from,
            "hours_budget": args.hours,
            "round_samples": args.round_samples,
            "round_epochs": args.round_epochs,
            "lookahead_ply": max(0, args.lookahead_ply),
            "start_source": args.start_source,
            "prefix_random_moves": args.prefix_random_moves,
            "prefix_policy_min_moves": args.prefix_policy_min_moves,
            "prefix_policy_max_moves": args.prefix_policy_max_moves,
            "prefix_policy_epsilon": args.prefix_policy_epsilon,
            "sample_move_epsilon": args.sample_move_epsilon,
            "symmetry_rotations": max(1, min(6, args.symmetry_rotations)),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "value_loss_weight": args.value_loss_weight,
            "empty_prob": args.empty_prob,
            "max_level": args.max_level,
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "good_rule": args.good_rule,
            "good_margin": args.good_margin,
            "value_target": args.value_target,
            "value_scale": args.value_scale,
            "max_opponent_checks": args.max_opponent_checks,
            "device": str(device),
            "eval_device": str(eval_device),
            "max_valid_loss": float(args.max_valid_loss),
            "num_nodes": num_nodes,
            "hidden_dim": int(model.hidden_dim),
            "blocks": int(model.blocks),
            "rounds_completed": round_index,
            "best_val_loss": best_val,
            "best_round": best_round,
            "best_checkpoint_path": str(best_out_path),
            "elapsed_seconds": time.time() - started_at,
            "last_round_stats": stats,
        }
        save_checkpoint(out_path, model, all_history, meta)
        model.to(device)
        print(f"Saved checkpoint: {out_path}")
        if improved:
            save_checkpoint(best_out_path, model, all_history, meta)
            model.to(device)
            print(f"Saved best checkpoint: {best_out_path}")

        if generation_stopped:
            print("Generation stopped via stop-file; ending after checkpoint save.")
            break
        if stop_file is not None and stop_file.exists():
            print("Stop file detected; ending after checkpoint save.")
            break

    print("\nGNN counterfactual rollout training complete.")
    print(f"Final checkpoint: {out_path}")


if __name__ == "__main__":
    main()
