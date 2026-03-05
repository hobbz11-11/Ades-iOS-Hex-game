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
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from hexplode_nn import BLUE, EMPTY, RED, HexplodeEnv, HexplodePolicyValueNet, State


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Hexplode policy-value net with counterfactual rollout labels. "
            "Supports random or self-play-derived start states plus symmetry augmentation."
        )
    )
    parser.add_argument("--hours", type=float, default=4.0, help="Maximum wall-clock budget in hours.")
    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument("--side-length", type=int, default=4, help="Board side length (4 for large board).")
    parser.add_argument("--channels", type=int, default=96, help="Residual trunk channels.")
    parser.add_argument("--blocks", type=int, default=8, help="Residual blocks.")
    parser.add_argument("--batch-size", type=int, default=384, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--value-loss-weight", type=float, default=0.45, help="Scale factor for value MSE.")
    parser.add_argument("--round-samples", type=int, default=12000, help="Generated samples per round.")
    parser.add_argument("--round-epochs", type=int, default=5, help="Training epochs per round.")
    parser.add_argument("--val-split", type=float, default=0.12, help="Validation split per round.")
    parser.add_argument("--log-batches-every", type=int, default=30, help="Log every N train batches.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Generation progress interval.")
    parser.add_argument(
        "--start-source",
        type=str,
        default="selfplay",
        choices=["selfplay", "random"],
        help="How to build initial board states for each labeled sample.",
    )
    parser.add_argument(
        "--prefix-random-moves",
        type=int,
        default=30,
        help="Number of fully random plies before policy-driven prefix when start-source=selfplay.",
    )
    parser.add_argument(
        "--prefix-policy-min-moves",
        type=int,
        default=20,
        help="Minimum best-policy plies after random warmup when start-source=selfplay.",
    )
    parser.add_argument(
        "--prefix-policy-max-moves",
        type=int,
        default=120,
        help="Maximum best-policy plies after random warmup when start-source=selfplay.",
    )
    parser.add_argument(
        "--prefix-policy-epsilon",
        type=float,
        default=0.0,
        help="Exploration epsilon for best-policy prefix plies (0 = pure policy).",
    )
    parser.add_argument(
        "--sample-move-epsilon",
        type=float,
        default=0.02,
        help="Exploration epsilon for picking the candidate move to be labeled.",
    )
    parser.add_argument(
        "--symmetry-rotations",
        type=int,
        default=6,
        help="How many 60-degree rotations to include per base sample (1..6).",
    )
    parser.add_argument("--empty-prob", type=float, default=0.46, help="Probability a random tile starts empty.")
    parser.add_argument("--max-level", type=int, default=5, help="Maximum random level for occupied tiles.")
    parser.add_argument("--min-turns", type=int, default=2, help="Minimum random turns_played for generated states.")
    parser.add_argument("--max-turns", type=int, default=140, help="Maximum random turns_played for generated states.")
    parser.add_argument(
        "--good-rule",
        type=str,
        default="after_positive",
        choices=["after_positive", "delta_nonnegative", "both"],
        help=(
            "How to mark random move as good: "
            "after_positive => scoreDiff(afterReply) > margin; "
            "delta_nonnegative => afterReply-before >= margin; "
            "both => satisfy both."
        ),
    )
    parser.add_argument("--good-margin", type=float, default=0.0, help="Margin used by good-rule.")
    parser.add_argument(
        "--value-target",
        type=str,
        default="binary",
        choices=["binary", "scaled_after", "scaled_delta"],
        help=(
            "Value target style: binary (+1/-1), scaled_after (tanh(after/value-scale)), "
            "scaled_delta (tanh((after-before)/value-scale))."
        ),
    )
    parser.add_argument("--value-scale", type=float, default=12.0, help="Scale used by non-binary value targets.")
    parser.add_argument(
        "--max-opponent-checks",
        type=int,
        default=0,
        help=(
            "If >0, cap lookahead branch expansion to first N ordered legal moves at each ply. "
            "0 means all."
        ),
    )
    parser.add_argument(
        "--lookahead-ply",
        type=int,
        default=5,
        help=(
            "Number of minimax plies to roll forward after the proposed move "
            "to score that move (0 means no rollout)."
        ),
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--bootstrap-checkpoint",
        type=str,
        default="tools/outputs/hexplode_policy_value_v4_long.pt",
        help="Optional checkpoint to initialize model weights from.",
    )
    parser.add_argument(
        "--prefix-selfplay-checkpoint",
        type=str,
        default="",
        help=(
            "Optional fixed checkpoint used only for generating self-play prefix start states. "
            "If empty, uses the currently trained model each round."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tools/outputs/hexplode_policy_value_counterfactual.pt",
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--best-out",
        type=str,
        default="",
        help="Optional path for continuously-updated best checkpoint. Defaults to <out>_best.pt",
    )
    parser.add_argument(
        "--stop-file",
        type=str,
        default="tools/outputs/hexplode_counterfactual.stop",
        help="Stop cleanly after current round checkpoint when this file exists.",
    )
    parser.add_argument(
        "--min-round-seconds",
        type=int,
        default=300,
        help="Do not start a new round when remaining budget is below this many seconds.",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:d}h {mins:02d}m {sec:02d}s"
    return f"{mins:02d}m {sec:02d}s"


def choose_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available.")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[HexplodePolicyValueNet, str]:
    checkpoint_path = Path(args.bootstrap_checkpoint) if args.bootstrap_checkpoint else Path("")
    model = HexplodePolicyValueNet(channels=args.channels, blocks=args.blocks)
    loaded_from = ""

    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = HexplodePolicyValueNet(
            channels=int(checkpoint.get("channels", args.channels)),
            blocks=int(checkpoint.get("blocks", args.blocks)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        loaded_from = str(checkpoint_path)

    model.to(device)
    return model, loaded_from


def load_fixed_model(checkpoint_path: str, fallback: HexplodePolicyValueNet, device: torch.device) -> Tuple[HexplodePolicyValueNet, str]:
    if not checkpoint_path:
        return fallback, ""
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Warning: prefix-selfplay-checkpoint not found: {path}; using current training model instead.")
        return fallback, ""
    checkpoint = torch.load(path, map_location="cpu")
    model = HexplodePolicyValueNet(
        channels=int(checkpoint.get("channels", 96)),
        blocks=int(checkpoint.get("blocks", 8)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, str(path)


def clone_model(model: HexplodePolicyValueNet, device: torch.device) -> HexplodePolicyValueNet:
    cloned = HexplodePolicyValueNet(
        input_channels=int(model.stem[0].in_channels),
        channels=int(model.stem[0].out_channels),
        blocks=int(len(model.trunk)),
        board_size=int(model.board_size),
    )
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    cloned.to(device)
    cloned.eval()
    return cloned


def score_diff_for_player(env: HexplodeEnv, state: State, player: int) -> int:
    red_score, blue_score = env.score_totals(state)
    return red_score - blue_score if player == RED else blue_score - red_score


def opponent_of(player: int) -> int:
    return BLUE if player == RED else RED


def build_random_state(
    env: HexplodeEnv,
    rng: random.Random,
    empty_prob: float,
    max_level: int,
    min_turns: int,
    max_turns: int,
) -> State:
    n = len(env.coords)
    owners = [EMPTY] * n
    levels = [0] * n
    p_empty = min(0.95, max(0.05, empty_prob))
    max_level = max(1, min(5, max_level))

    for i in range(n):
        roll = rng.random()
        if roll < p_empty:
            owners[i] = EMPTY
            levels[i] = 0
            continue
        owners[i] = RED if rng.random() < 0.5 else BLUE
        levels[i] = rng.randint(1, max_level)

    red_count = sum(1 for o in owners if o == RED)
    blue_count = sum(1 for o in owners if o == BLUE)
    if red_count == 0:
        idx = rng.randrange(n)
        owners[idx] = RED
        levels[idx] = rng.randint(1, max_level)
    if blue_count == 0:
        idx = rng.randrange(n)
        if n > 1 and owners[idx] == RED:
            idx = (idx + 1) % n
        owners[idx] = BLUE
        levels[idx] = rng.randint(1, max_level)

    min_turns = max(0, min_turns)
    max_turns = max(min_turns, max_turns)
    turns_played = rng.randint(min_turns, max_turns)
    return State(owners=owners, levels=levels, turns_played=turns_played)


def choose_player_with_legal(env: HexplodeEnv, state: State, rng: random.Random) -> int | None:
    candidates = []
    if env.legal_moves(state, RED):
        candidates.append(RED)
    if env.legal_moves(state, BLUE):
        candidates.append(BLUE)
    if not candidates:
        return None
    return rng.choice(candidates)


@dataclass
class StartPosition:
    state: State
    to_move: int
    source: str
    random_prefix_moves: int
    policy_prefix_moves: int


def choose_model_move(
    env: HexplodeEnv,
    state: State,
    player: int,
    model: HexplodePolicyValueNet,
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
        xb = env.encode_features(state, player).unsqueeze(0).to(device)
        logits, _ = model(xb)
    row = logits[0].detach().cpu()
    best_score = -1e30
    best_moves: List[int] = []
    for mv in legal:
        score = float(row[env.flat_index[mv]])
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


def build_selfplay_prefix_state(
    env: HexplodeEnv,
    rng: random.Random,
    model: HexplodePolicyValueNet,
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
    player = RED if rng.random() < 0.5 else BLUE
    random_done = 0
    policy_done = 0

    for _ in range(random_warmup_moves):
        if env.terminal_winner(state) is not None:
            return None
        mv = choose_random_move(
            env=env,
            state=state,
            player=player,
            rng=rng,
        )
        if mv < 0:
            other = opponent_of(player)
            mv = choose_random_move(env=env, state=state, player=other, rng=rng)
            if mv < 0:
                return None
            player = other
        state = env.apply_move(state, mv, player)
        random_done += 1
        player = opponent_of(player)

    for _ in range(target_policy_moves):
        if env.terminal_winner(state) is not None:
            return None
        mv = choose_model_move(
            env=env,
            state=state,
            player=player,
            model=model,
            device=device,
            rng=rng,
            epsilon=epsilon,
        )
        if mv < 0:
            other = opponent_of(player)
            mv = choose_model_move(
                env=env,
                state=state,
                player=other,
                model=model,
                device=device,
                rng=rng,
                epsilon=epsilon,
            )
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


def rotate_axial(coord: Tuple[int, int], turns: int) -> Tuple[int, int]:
    x, z = coord
    y = -x - z
    for _ in range(turns % 6):
        x, y, z = -z, -x, -y
    return x, z


def build_rotation_maps(env: HexplodeEnv, num_rotations: int) -> List[List[int]]:
    maps: List[List[int]] = []
    n = len(env.coords)
    for turns in range(max(1, min(6, num_rotations))):
        remap = [0] * n
        for src_idx, coord in enumerate(env.coords):
            rotated = rotate_axial(coord, turns)
            dst_idx = env.index_of.get(rotated)
            if dst_idx is None:
                raise RuntimeError(f"Rotation produced out-of-board coord: {coord} -> {rotated}")
            remap[src_idx] = dst_idx
        maps.append(remap)
    return maps


def rotate_state(state: State, remap: Sequence[int]) -> State:
    n = len(remap)
    owners = [0] * n
    levels = [0] * n
    for src_idx, dst_idx in enumerate(remap):
        owners[dst_idx] = state.owners[src_idx]
        levels[dst_idx] = state.levels[src_idx]
    return State(owners=owners, levels=levels, turns_played=state.turns_played)


def label_from_outcome(
    before_diff: int,
    after_reply_diff: int,
    good_rule: str,
    margin: float,
) -> float:
    if good_rule == "after_positive":
        good = after_reply_diff > margin
    elif good_rule == "delta_nonnegative":
        good = (after_reply_diff - before_diff) >= margin
    else:
        good = after_reply_diff > margin and (after_reply_diff - before_diff) >= margin
    return 1.0 if good else 0.0


def value_target_from_outcome(
    before_diff: int,
    after_reply_diff: int,
    policy_label: float,
    value_target_mode: str,
    value_scale: float,
) -> float:
    if value_target_mode == "binary":
        return 1.0 if policy_label >= 0.5 else -1.0
    scale = max(1.0, value_scale)
    if value_target_mode == "scaled_after":
        return math.tanh(after_reply_diff / scale)
    return math.tanh((after_reply_diff - before_diff) / scale)


def ordered_rollout_moves(
    env: HexplodeEnv,
    state: State,
    to_move: int,
    root_player: int,
    max_branch_checks: int,
) -> List[int]:
    legal = env.legal_moves(state, to_move)
    if not legal:
        return legal

    scored: List[Tuple[int, int]] = []
    for mv in legal:
        nxt = env.apply_move(state, mv, to_move)
        diff = score_diff_for_player(env, nxt, root_player)
        scored.append((mv, diff))

    maximizing = to_move == root_player
    scored.sort(key=lambda pair: pair[1], reverse=maximizing)
    if max_branch_checks > 0:
        scored = scored[:max_branch_checks]
    return [mv for mv, _ in scored]


def minimax_score_diff(
    env: HexplodeEnv,
    state: State,
    to_move: int,
    root_player: int,
    plies_left: int,
    max_branch_checks: int,
    cache: dict[Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int], int],
) -> int:
    winner = env.terminal_winner(state)
    if winner is not None:
        if winner == root_player:
            return 10_000 + plies_left
        if winner == 0:
            return 0
        return -10_000 - plies_left

    if plies_left <= 0:
        return score_diff_for_player(env, state, root_player)

    key = (tuple(state.owners), tuple(state.levels), state.turns_played, to_move, plies_left)
    cached = cache.get(key)
    if cached is not None:
        return cached

    legal = ordered_rollout_moves(
        env=env,
        state=state,
        to_move=to_move,
        root_player=root_player,
        max_branch_checks=max_branch_checks,
    )
    if not legal:
        result = score_diff_for_player(env, state, root_player)
        cache[key] = result
        return result

    next_player = BLUE if to_move == RED else RED
    if to_move == root_player:
        best = -1_000_000_000
        for mv in legal:
            nxt = env.apply_move(state, mv, to_move)
            score = minimax_score_diff(
                env=env,
                state=nxt,
                to_move=next_player,
                root_player=root_player,
                plies_left=plies_left - 1,
                max_branch_checks=max_branch_checks,
                cache=cache,
            )
            if score > best:
                best = score
        cache[key] = best
        return best

    best = 1_000_000_000
    for mv in legal:
        nxt = env.apply_move(state, mv, to_move)
        score = minimax_score_diff(
            env=env,
            state=nxt,
            to_move=next_player,
            root_player=root_player,
            plies_left=plies_left - 1,
            max_branch_checks=max_branch_checks,
            cache=cache,
        )
        if score < best:
            best = score
    cache[key] = best
    return best


def evaluate_model_move_with_rollout(
    env: HexplodeEnv,
    state: State,
    player: int,
    model: HexplodePolicyValueNet,
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
    move = choose_model_move(
        env=env,
        state=state,
        player=player,
        model=model,
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
    model: HexplodePolicyValueNet,
    prefix_model: HexplodePolicyValueNet,
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
    rotation_maps = build_rotation_maps(env, symmetry_rotations)

    while len(features) < samples and attempts < max_attempts:
        attempts += 1
        if stop_file is not None and stop_file.exists() and len(features) > 0:
            stop_requested = True
            print(f"  stop-file detected during generation ({stop_file})")
            break

        start: StartPosition | None = None
        if start_source == "selfplay":
            start = build_selfplay_prefix_state(
                env=env,
                rng=rng,
                model=prefix_model,
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

        move, policy_label, value_target, before_diff, after_reply_diff, response_count = evaluate_model_move_with_rollout(
            env=env,
            state=start.state,
            player=start.to_move,
            model=model,
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
            features.append(env.encode_features(rotated_state, start.to_move))
            move_indices.append(env.flat_index[rotated_move])
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
                f"  gen {sample_count}/{samples} "
                f"({100.0 * sample_count / samples:5.1f}%) "
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
            if stop_file is not None and stop_file.exists():
                stop_requested = True
                print(f"  stop-file detected during generation ({stop_file})")
                break

    if not features:
        raise RuntimeError("Round dataset generation produced zero samples.")

    x = torch.stack(features, dim=0)
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
    model: HexplodePolicyValueNet,
    loader: DataLoader,
    device: torch.device,
    value_loss_weight: float,
) -> dict:
    model.eval()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    with torch.no_grad():
        for xb, move_idx, y_policy, y_value in loader:
            xb = xb.to(device, non_blocking=True)
            move_idx = move_idx.to(device, non_blocking=True)
            y_policy = y_policy.to(device, non_blocking=True)
            y_value = y_value.to(device, non_blocking=True)

            logits, value = model(xb)
            selected_logits = logits.gather(1, move_idx.unsqueeze(1)).squeeze(1)
            policy_loss = F.binary_cross_entropy_with_logits(selected_logits, y_policy)
            value_loss = F.mse_loss(value, y_value)
            loss = policy_loss + value_loss_weight * value_loss
            pred = (torch.sigmoid(selected_logits) >= 0.5).float()
            acc = (pred == (y_policy >= 0.5).float()).float().mean()

            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_loss += loss.item()
            total_acc += acc.item()
            batches += 1

    denom = max(1, batches)
    return {
        "loss": total_loss / denom,
        "policy_loss": total_policy / denom,
        "value_loss": total_value / denom,
        "policy_acc": total_acc / denom,
    }


def train_round(
    model: HexplodePolicyValueNet,
    device: torch.device,
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

            logits, value = model(xb)
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
            val_metrics = evaluate(model, val_loader, device, value_loss_weight)
            best_val = min(best_val, val_metrics["loss"])
        else:
            val_metrics = {"loss": float("nan"), "policy_loss": float("nan"), "value_loss": float("nan"), "policy_acc": float("nan")}

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
    model: HexplodePolicyValueNet,
    args: argparse.Namespace,
    history: List[dict],
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = int(model.stem[0].out_channels)
    blocks = int(len(model.trunk))
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "channels": channels,
        "blocks": blocks,
        "board_size": 7,
        "input_channels": 8,
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
    model, loaded_from = load_model(args, device)
    prefix_seed_model, prefix_loaded_from = load_fixed_model(args.prefix_selfplay_checkpoint, model, device)
    stop_file = Path(args.stop_file) if args.stop_file else None
    out_path = Path(args.out)
    best_out_path = Path(args.best_out) if args.best_out else out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")
    best_prefix_model = clone_model(prefix_seed_model, device)

    started_at = time.time()
    deadline = started_at + args.hours * 3600.0
    round_index = 0
    all_history: List[dict] = []
    best_val = float("inf")
    best_round = 0

    print(
        f"Starting counterfactual rollout training on {device}. "
        f"budget={args.hours:.2f}h lookahead_ply={args.lookahead_ply} "
        f"start_source={args.start_source} rotations={max(1, min(6, args.symmetry_rotations))} "
        f"bootstrap={'yes' if loaded_from else 'no'}"
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
        )
        improved = round_best_val < best_val
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
            "mode": "counterfactual_rollout",
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
            "rounds_completed": round_index,
            "best_val_loss": best_val,
            "best_round": best_round,
            "best_checkpoint_path": str(best_out_path),
            "elapsed_seconds": time.time() - started_at,
            "last_round_stats": stats,
        }
        save_checkpoint(out_path, model, args, all_history, meta)
        model.to(device)
        print(f"Saved checkpoint: {out_path}")
        if improved:
            save_checkpoint(best_out_path, model, args, all_history, meta)
            model.to(device)
            print(f"Saved best checkpoint: {best_out_path}")

        if generation_stopped:
            print("Generation stopped via stop-file; ending after checkpoint save.")
            break
        if stop_file is not None and stop_file.exists():
            print("Stop file detected; ending after checkpoint save.")
            break

    print("\nCounterfactual rollout training complete.")
    print(f"Final checkpoint: {out_path}")


if __name__ == "__main__":
    main()
