#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from hexplode_counterfactual_train import (
    choose_device,
    format_duration,
    minimax_score_diff,
    opponent_of,
)
from hexplode_gnn import (
    GNN_FEATURE_DIM,
    HexplodeGraphPolicyValueNet,
    build_adj_matrix,
    build_rotation_maps,
    clone_gnn_model,
    encode_node_features,
    rotate_state,
)
from hexplode_nn import BLUE, RED, HexplodeEnv, State


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Hexplode GNN by imitating a minimax teacher. "
            "Each sampled self-play state labels all legal moves using minimax lookahead."
        )
    )
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--side-length", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.6)
    parser.add_argument("--round-games", type=int, default=6)
    parser.add_argument("--round-epochs", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--sample-state-prob", type=float, default=1.0)
    parser.add_argument("--min-sample-turn", type=int, default=0)
    parser.add_argument("--symmetry-rotations", type=int, default=6)
    parser.add_argument("--teacher-ply", type=int, default=5)
    parser.add_argument("--teacher-branch-cap", type=int, default=0)
    parser.add_argument("--policy-target-temp", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=16.0)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--progress-every-games", type=int, default=1)
    parser.add_argument("--log-batches-every", type=int, default=25)
    parser.add_argument("--state-random-min-plies", type=int, default=30)
    parser.add_argument("--state-random-max-plies", type=int, default=100)
    parser.add_argument("--state-model-min-plies", type=int, default=20)
    parser.add_argument("--state-model-max-plies", type=int, default=120)
    parser.add_argument("--state-minimax-min-plies", type=int, default=6)
    parser.add_argument("--state-minimax-max-plies", type=int, default=18)
    parser.add_argument("--state-model-epsilon", type=float, default=0.10)
    parser.add_argument("--state-model-temperature", type=float, default=0.85)
    parser.add_argument("--state-minimax-ply", type=int, default=2)
    parser.add_argument("--state-minimax-branch-cap", type=int, default=8)
    parser.add_argument("--replay-max-samples", type=int, default=18000)
    parser.add_argument("--replay-sample-ratio", type=float, default=0.75)
    parser.add_argument("--holdout-base-states", type=int, default=500)
    parser.add_argument("--holdout-symmetry-rotations", type=int, default=6)
    parser.add_argument("--holdout-seed", type=int, default=31337)
    parser.add_argument("--holdout-cache-file", type=str, default="")
    parser.add_argument("--holdout-progress-every", type=int, default=10)
    parser.add_argument("--holdout-progress-file", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--eval-device",
        type=str,
        default="cpu",
        choices=["same", "cpu"],
        help="Validation device. CPU is safer for stable metrics.",
    )
    parser.add_argument("--bootstrap-checkpoint", type=str, default="")
    parser.add_argument("--out", type=str, default="tools/outputs/hexplode_small_teacher_latest.pt")
    parser.add_argument("--best-out", type=str, default="tools/outputs/hexplode_small_teacher_best.pt")
    parser.add_argument("--stats-out", type=str, default="tools/outputs/hexplode_small_teacher_stats.json")
    parser.add_argument("--stop-file", type=str, default="tools/outputs/hexplode_small_teacher.stop")
    return parser.parse_args()


def load_model(args: argparse.Namespace, num_nodes: int, device: torch.device) -> Tuple[HexplodeGraphPolicyValueNet, str]:
    model = HexplodeGraphPolicyValueNet(
        input_dim=GNN_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        blocks=args.blocks,
        num_nodes=num_nodes,
    )
    loaded_from = ""
    ckpt_raw = (args.bootstrap_checkpoint or "").strip()
    ckpt_path = Path(ckpt_raw) if ckpt_raw else None
    if ckpt_path is not None and ckpt_path.exists() and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = HexplodeGraphPolicyValueNet(
            input_dim=int(ckpt.get("input_dim", GNN_FEATURE_DIM)),
            hidden_dim=int(ckpt.get("hidden_dim", args.hidden_dim)),
            blocks=int(ckpt.get("blocks", args.blocks)),
            num_nodes=int(ckpt.get("num_nodes", num_nodes)),
        )
        model.load_state_dict(ckpt["state_dict"])
        loaded_from = str(ckpt_path)
    model.to(device)
    return model, loaded_from


def choose_model_move(
    env: HexplodeEnv,
    state: State,
    player: int,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    device: torch.device,
    rng: random.Random,
    epsilon: float,
    temperature: float,
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

    if temperature <= 1e-6:
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

    vals = [float(row[mv]) / temperature for mv in legal]
    vmax = max(vals)
    exps = [math.exp(v - vmax) for v in vals]
    total = sum(exps)
    if total <= 0:
        return rng.choice(legal)
    pick = rng.random() * total
    run = 0.0
    for mv, e in zip(legal, exps):
        run += e
        if run >= pick:
            return mv
    return legal[-1]


def winner_for_stats(env: HexplodeEnv, state: State) -> int:
    winner = env.terminal_winner(state)
    if winner is not None:
        return winner
    red_score, blue_score = env.score_totals(state)
    if red_score > blue_score:
        return RED
    if blue_score > red_score:
        return BLUE
    return 0


def score_to_value(score: float, scale: float) -> float:
    if score >= 9000:
        return 1.0
    if score <= -9000:
        return -1.0
    return math.tanh(score / max(1e-6, scale))


def legal_policy_from_scores(
    legal_moves: Sequence[int],
    move_scores: Sequence[float],
    num_nodes: int,
    temperature: float,
) -> torch.Tensor:
    target = torch.zeros((num_nodes,), dtype=torch.float32)
    if not legal_moves:
        return target

    if temperature <= 1e-6:
        best = max(move_scores)
        best_moves = [mv for mv, sc in zip(legal_moves, move_scores) if abs(sc - best) <= 1e-9]
        p = 1.0 / max(1, len(best_moves))
        for mv in best_moves:
            target[mv] = p
        return target

    vmax = max(move_scores)
    exps = [math.exp((sc - vmax) / temperature) for sc in move_scores]
    total = sum(exps)
    if total <= 0:
        p = 1.0 / len(legal_moves)
        for mv in legal_moves:
            target[mv] = p
        return target
    for mv, e in zip(legal_moves, exps):
        target[mv] = e / total
    return target


def rotate_policy_target(target: torch.Tensor, remap: Sequence[int]) -> torch.Tensor:
    out = torch.zeros_like(target)
    for src_idx, dst_idx in enumerate(remap):
        out[dst_idx] = target[src_idx]
    return out


def evaluate_all_legal_moves(
    env: HexplodeEnv,
    state: State,
    player: int,
    teacher_ply: int,
    teacher_branch_cap: int,
) -> Tuple[List[int], List[float]]:
    legal = env.legal_moves(state, player)
    if not legal:
        return [], []
    cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int], int] = {}
    scores: List[float] = []
    opponent = opponent_of(player)
    for mv in legal:
        nxt = env.apply_move(state, mv, player)
        if teacher_ply <= 0:
            red_score, blue_score = env.score_totals(nxt)
            score = red_score - blue_score if player == RED else blue_score - red_score
        else:
            score = minimax_score_diff(
                env=env,
                state=nxt,
                to_move=opponent,
                root_player=player,
                plies_left=teacher_ply,
                max_branch_checks=teacher_branch_cap,
                cache=cache,
            )
        scores.append(float(score))
    return legal, scores


def choose_minimax_move(
    env: HexplodeEnv,
    state: State,
    player: int,
    ply: int,
    branch_cap: int,
    rng: random.Random,
) -> int:
    legal, scores = evaluate_all_legal_moves(
        env=env,
        state=state,
        player=player,
        teacher_ply=max(0, ply),
        teacher_branch_cap=max(0, branch_cap),
    )
    if not legal:
        return -1
    best = max(scores)
    best_moves = [mv for mv, sc in zip(legal, scores) if abs(sc - best) <= 1e-9]
    return rng.choice(best_moves) if best_moves else legal[0]


def choose_rollout_move(
    env: HexplodeEnv,
    state: State,
    player: int,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    device: torch.device,
    rng: random.Random,
    ply_index: int,
    random_prefix: int,
    model_prefix: int,
    minimax_prefix: int,
    model_epsilon: float,
    model_temperature: float,
    minimax_ply: int,
    minimax_branch_cap: int,
) -> int:
    legal = env.legal_moves(state, player)
    if not legal:
        return -1

    if ply_index < random_prefix:
        return rng.choice(legal)
    if ply_index < random_prefix + model_prefix:
        return choose_model_move(
            env=env,
            state=state,
            player=player,
            model=model,
            adj=adj,
            device=device,
            rng=rng,
            epsilon=max(0.0, min(1.0, model_epsilon)),
            temperature=max(1e-6, model_temperature),
        )
    if ply_index < random_prefix + model_prefix + minimax_prefix:
        return choose_minimax_move(
            env=env,
            state=state,
            player=player,
            ply=max(0, minimax_ply),
            branch_cap=max(0, minimax_branch_cap),
            rng=rng,
        )

    return choose_model_move(
        env=env,
        state=state,
        player=player,
        model=model,
        adj=adj,
        device=device,
        rng=rng,
        epsilon=max(0.0, min(1.0, model_epsilon)),
        temperature=max(1e-6, model_temperature),
    )


def build_state_symmetry_key(state: State, player: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    return (tuple(state.owners), tuple(state.levels), player)


def append_labeled_state_with_symmetry(
    env: HexplodeEnv,
    state: State,
    player: int,
    policy_target: torch.Tensor,
    value: float,
    rotation_maps: Sequence[Sequence[int]],
    features: List[torch.Tensor],
    policy_targets: List[torch.Tensor],
    value_targets: List[float],
) -> int:
    seen: set[Tuple[Tuple[int, ...], Tuple[int, ...], int]] = set()
    added = 0
    for remap in rotation_maps:
        rotated = rotate_state(state, remap)
        key = build_state_symmetry_key(rotated, player)
        if key in seen:
            continue
        seen.add(key)
        features.append(encode_node_features(env, rotated, player))
        policy_targets.append(rotate_policy_target(policy_target, remap))
        value_targets.append(value)
        added += 1
    return added


def generate_round_dataset(
    env: HexplodeEnv,
    model: HexplodeGraphPolicyValueNet,
    adj: torch.Tensor,
    device: torch.device,
    round_games: int,
    max_turns: int,
    sample_state_prob: float,
    min_sample_turn: int,
    symmetry_rotations: int,
    teacher_ply: int,
    teacher_branch_cap: int,
    policy_target_temp: float,
    value_scale: float,
    state_random_min_plies: int,
    state_random_max_plies: int,
    state_model_min_plies: int,
    state_model_max_plies: int,
    state_minimax_min_plies: int,
    state_minimax_max_plies: int,
    state_model_epsilon: float,
    state_model_temperature: float,
    state_minimax_ply: int,
    state_minimax_branch_cap: int,
    seed: int,
    progress_every_games: int,
    stop_file: Path | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, bool]:
    rng = random.Random(seed)
    model.eval()

    features: List[torch.Tensor] = []
    policy_targets: List[torch.Tensor] = []
    value_targets: List[float] = []

    red_wins = 0
    blue_wins = 0
    draws = 0
    game_lengths: List[int] = []
    base_states = 0
    teacher_evals = 0
    sum_legal = 0
    sum_best_score = 0.0
    sum_expected_score = 0.0
    symmetry_unique_samples = 0
    random_prefix_total = 0
    model_prefix_total = 0
    minimax_prefix_total = 0
    stop_requested = False
    started = time.monotonic()

    rotation_maps = build_rotation_maps(env, symmetry_rotations).maps
    num_nodes = len(env.coords)

    for game_idx in range(1, max(1, round_games) + 1):
        if stop_file is not None and stop_file.exists() and len(features) > 0:
            stop_requested = True
            break

        state = env.empty_state()
        player = RED if rng.random() < 0.5 else BLUE
        turns = 0
        random_prefix = rng.randint(max(0, state_random_min_plies), max(max(0, state_random_min_plies), max(0, state_random_max_plies)))
        model_prefix = rng.randint(max(0, state_model_min_plies), max(max(0, state_model_min_plies), max(0, state_model_max_plies)))
        minimax_prefix = rng.randint(max(0, state_minimax_min_plies), max(max(0, state_minimax_min_plies), max(0, state_minimax_max_plies)))
        random_prefix_total += random_prefix
        model_prefix_total += model_prefix
        minimax_prefix_total += minimax_prefix

        for ply_idx in range(max_turns):
            legal = env.legal_moves(state, player)
            if not legal:
                other = opponent_of(player)
                legal_other = env.legal_moves(state, other)
                if not legal_other:
                    break
                player = other
                legal = legal_other

            if state.turns_played >= min_sample_turn and rng.random() <= sample_state_prob:
                legal_moves, scores = evaluate_all_legal_moves(
                    env=env,
                    state=state,
                    player=player,
                    teacher_ply=max(0, teacher_ply),
                    teacher_branch_cap=max(0, teacher_branch_cap),
                )
                if legal_moves:
                    teacher_evals += 1
                    base_states += 1
                    sum_legal += len(legal_moves)
                    best_score = max(scores)
                    policy_target = legal_policy_from_scores(
                        legal_moves=legal_moves,
                        move_scores=scores,
                        num_nodes=num_nodes,
                        temperature=max(1e-6, policy_target_temp),
                    )
                    expected_score = sum(float(policy_target[mv]) * float(sc) for mv, sc in zip(legal_moves, scores))
                    sum_best_score += best_score
                    sum_expected_score += expected_score
                    value = score_to_value(best_score, value_scale)
                    symmetry_unique_samples += append_labeled_state_with_symmetry(
                        env=env,
                        state=state,
                        player=player,
                        policy_target=policy_target,
                        value=value,
                        rotation_maps=rotation_maps,
                        features=features,
                        policy_targets=policy_targets,
                        value_targets=value_targets,
                    )

            move = choose_rollout_move(
                env=env,
                state=state,
                player=player,
                model=model,
                adj=adj,
                device=device,
                rng=rng,
                ply_index=ply_idx,
                random_prefix=random_prefix,
                model_prefix=model_prefix,
                minimax_prefix=minimax_prefix,
                model_epsilon=max(0.0, min(1.0, state_model_epsilon)),
                model_temperature=max(1e-6, state_model_temperature),
                minimax_ply=max(0, state_minimax_ply),
                minimax_branch_cap=max(0, state_minimax_branch_cap),
            )
            if move < 0:
                break
            state = env.apply_move(state, move, player)
            turns += 1
            winner = env.terminal_winner(state)
            if winner is not None:
                break
            player = opponent_of(player)

        game_lengths.append(turns)
        result = winner_for_stats(env, state)
        if result == RED:
            red_wins += 1
        elif result == BLUE:
            blue_wins += 1
        else:
            draws += 1

        if progress_every_games > 0 and game_idx % progress_every_games == 0:
            elapsed = time.monotonic() - started
            print(
                f"  self-play game {game_idx}/{round_games} "
                f"samples={len(features)} baseStates={base_states} "
                f"uniqueSym={symmetry_unique_samples} "
                f"avgLen={sum(game_lengths)/max(1, len(game_lengths)):.1f} "
                f"avgLegal={sum_legal/max(1, teacher_evals):.2f} "
                f"avgBestScore={sum_best_score/max(1, teacher_evals):+.2f} "
                f"elapsed={format_duration(elapsed)}"
            )

    if not features:
        raise RuntimeError("Generated zero samples. Try more games or higher sample-state-prob.")

    x = torch.stack(features, dim=0)
    y_policy = torch.stack(policy_targets, dim=0)
    y_value = torch.tensor(value_targets, dtype=torch.float32)
    stats = {
        "selfplay_games": round_games,
        "sampled_states": base_states,
        "samples": int(x.shape[0]),
        "avg_game_len": (sum(game_lengths) / max(1, len(game_lengths))),
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "teacher_evals": teacher_evals,
        "avg_legal_moves": (sum_legal / max(1, teacher_evals)),
        "avg_best_score": (sum_best_score / max(1, teacher_evals)),
        "avg_expected_score": (sum_expected_score / max(1, teacher_evals)),
        "unique_symmetry_samples": symmetry_unique_samples,
        "avg_random_prefix": (random_prefix_total / max(1, round_games)),
        "avg_model_prefix": (model_prefix_total / max(1, round_games)),
        "avg_minimax_prefix": (minimax_prefix_total / max(1, round_games)),
        "symmetry_rotations": len(rotation_maps),
        "stopped_early": stop_requested,
    }
    return x, y_policy, y_value, stats, stop_requested


def merge_with_replay(
    x_new: torch.Tensor,
    y_policy_new: torch.Tensor,
    y_value_new: torch.Tensor,
    replay_x: torch.Tensor | None,
    replay_policy: torch.Tensor | None,
    replay_value: torch.Tensor | None,
    replay_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if replay_x is None or replay_policy is None or replay_value is None:
        return x_new, y_policy_new, y_value_new, 0
    if replay_ratio <= 0.0 or len(replay_x) == 0:
        return x_new, y_policy_new, y_value_new, 0

    target = int(max(0, round(len(x_new) * replay_ratio)))
    if target <= 0:
        return x_new, y_policy_new, y_value_new, 0
    target = min(target, len(replay_x))

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(replay_x), generator=g)[:target]
    x_mix = torch.cat([x_new, replay_x[idx]], dim=0)
    y_policy_mix = torch.cat([y_policy_new, replay_policy[idx]], dim=0)
    y_value_mix = torch.cat([y_value_new, replay_value[idx]], dim=0)
    return x_mix, y_policy_mix, y_value_mix, target


def update_replay_buffer(
    replay_x: torch.Tensor | None,
    replay_policy: torch.Tensor | None,
    replay_value: torch.Tensor | None,
    x_new: torch.Tensor,
    y_policy_new: torch.Tensor,
    y_value_new: torch.Tensor,
    max_samples: int,
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    cap = max(0, max_samples)
    if cap <= 0:
        return None, None, None

    if replay_x is None:
        x_all = x_new.detach().cpu()
        y_policy_all = y_policy_new.detach().cpu()
        y_value_all = y_value_new.detach().cpu()
    else:
        x_all = torch.cat([replay_x, x_new.detach().cpu()], dim=0)
        y_policy_all = torch.cat([replay_policy, y_policy_new.detach().cpu()], dim=0)
        y_value_all = torch.cat([replay_value, y_value_new.detach().cpu()], dim=0)

    if len(x_all) > cap:
        x_all = x_all[-cap:]
        y_policy_all = y_policy_all[-cap:]
        y_value_all = y_value_all[-cap:]
    return x_all, y_policy_all, y_value_all


def build_fixed_holdout_dataset(
    env: HexplodeEnv,
    base_states: int,
    symmetry_rotations: int,
    teacher_ply: int,
    teacher_branch_cap: int,
    value_scale: float,
    state_random_min_plies: int,
    state_random_max_plies: int,
    state_minimax_min_plies: int,
    state_minimax_max_plies: int,
    state_minimax_ply: int,
    state_minimax_branch_cap: int,
    seed: int,
    cache_file: Path | None = None,
    progress_every: int = 0,
    progress_file: Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def save_cache(records: List[dict]) -> None:
        if cache_file is None:
            return
        payload = {
            "version": 1,
            "side_length": int(env.side_length),
            "num_nodes": int(len(env.coords)),
            "states": records,
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(cache_file)

    def load_cache() -> List[dict]:
        if cache_file is None or not cache_file.exists():
            return []
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[Init] holdout cache unreadable ({cache_file}): {exc}. Starting fresh.")
            return []

        if isinstance(raw, dict):
            side = int(raw.get("side_length", env.side_length))
            if side != int(env.side_length):
                print(
                    f"[Init] holdout cache side mismatch (cache={side}, run={env.side_length}); ignoring cache."
                )
                return []
            records = raw.get("states", [])
        elif isinstance(raw, list):
            records = raw
        else:
            records = []

        valid: List[dict] = []
        node_count = len(env.coords)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            owners = rec.get("owners")
            levels = rec.get("levels")
            player = rec.get("player")
            turns = rec.get("turns_played", 0)
            if not isinstance(owners, list) or not isinstance(levels, list):
                continue
            if len(owners) != node_count or len(levels) != node_count:
                continue
            if player not in (RED, BLUE):
                continue
            try:
                valid.append(
                    {
                        "owners": [int(v) for v in owners],
                        "levels": [int(v) for v in levels],
                        "turns_played": int(turns),
                        "player": int(player),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return valid

    def emit_progress(
        *,
        phase: str,
        status: str,
        base_done: int,
        base_target: int,
        attempts_done: int,
        attempts_target: int,
        elapsed: float,
        labeled_done: int = 0,
        labeled_target: int = 0,
        sample_count: int = 0,
    ) -> None:
        base_pct = (100.0 * base_done / max(1, base_target))
        msg = (
            f"[Init] {phase} {status}: base_states={base_done}/{base_target} ({base_pct:.1f}%) "
            f"attempts={attempts_done}/{attempts_target} elapsed={format_duration(elapsed)}"
        )
        if labeled_target > 0:
            label_pct = (100.0 * labeled_done / max(1, labeled_target))
            msg += f" labels={labeled_done}/{labeled_target} ({label_pct:.1f}%)"
        if sample_count > 0:
            msg += f" samples={sample_count}"
        print(msg, flush=True)

        if progress_file is None:
            return
        payload = {
            "phase": phase,
            "status": status,
            "base_states_completed": int(base_done),
            "base_states_target": int(base_target),
            "attempts": int(attempts_done),
            "max_attempts": int(attempts_target),
            "elapsed_sec": float(elapsed),
            "labeled_completed": int(labeled_done),
            "labeled_target": int(labeled_target),
            "samples_generated": int(sample_count),
        }
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_file.with_suffix(progress_file.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(progress_file)

    rotation_maps = build_rotation_maps(env, symmetry_rotations).maps
    num_nodes = len(env.coords)

    target_states = max(32, base_states)
    cache_records = load_cache()
    if len(cache_records) > target_states:
        cache_records = cache_records[:target_states]
    print(
        f"[Init] holdout cache: {len(cache_records)}/{target_states} base states "
        f"from {cache_file if cache_file is not None else '(in-memory only)'}",
        flush=True,
    )

    gen_started = time.monotonic()
    needed = max(0, target_states - len(cache_records))
    attempts = 0
    max_attempts = max(needed, 1) * 60
    rng = random.Random(seed + (len(cache_records) * 9973))
    emit_progress(
        phase="holdout_cache",
        status="progress",
        base_done=len(cache_records),
        base_target=target_states,
        attempts_done=attempts,
        attempts_target=max_attempts,
        elapsed=time.monotonic() - gen_started,
    )

    while attempts < max_attempts and len(cache_records) < target_states:
        attempts += 1
        if progress_every > 0 and (attempts == 1 or attempts % progress_every == 0):
            emit_progress(
                phase="holdout_cache",
                status="attempt",
                base_done=len(cache_records),
                base_target=target_states,
                attempts_done=attempts,
                attempts_target=max_attempts,
                elapsed=time.monotonic() - gen_started,
            )
        state = env.empty_state()
        player = RED if rng.random() < 0.5 else BLUE
        random_prefix = rng.randint(max(0, state_random_min_plies), max(max(0, state_random_min_plies), max(0, state_random_max_plies)))
        minimax_prefix = rng.randint(max(0, state_minimax_min_plies), max(max(0, state_minimax_min_plies), max(0, state_minimax_max_plies)))
        total_prefix = random_prefix + minimax_prefix
        if total_prefix <= 0:
            total_prefix = 1

        for ply_idx in range(total_prefix):
            legal = env.legal_moves(state, player)
            if not legal:
                other = opponent_of(player)
                legal_other = env.legal_moves(state, other)
                if not legal_other:
                    break
                player = other
                legal = legal_other

            if ply_idx < random_prefix:
                move = rng.choice(legal)
            else:
                move = choose_minimax_move(
                    env=env,
                    state=state,
                    player=player,
                    ply=max(0, state_minimax_ply),
                    branch_cap=max(0, state_minimax_branch_cap),
                    rng=rng,
                )
            if move < 0:
                break
            state = env.apply_move(state, move, player)
            winner = env.terminal_winner(state)
            if winner is not None:
                break
            player = opponent_of(player)

        if env.terminal_winner(state) is not None:
            continue
        legal_now = env.legal_moves(state, player)
        if not legal_now:
            other = opponent_of(player)
            legal_other = env.legal_moves(state, other)
            if not legal_other:
                continue
            player = other

        legal_moves, scores = evaluate_all_legal_moves(
            env=env,
            state=state,
            player=player,
            teacher_ply=max(0, teacher_ply),
            teacher_branch_cap=max(0, teacher_branch_cap),
        )
        if not legal_moves:
            continue
        policy_target = legal_policy_from_scores(
            legal_moves=legal_moves,
            move_scores=scores,
            num_nodes=num_nodes,
            temperature=1e-6,
        )
        _ = policy_target  # used to ensure state is labelable; relabeled below.

        cache_records.append(
            {
                "owners": state.owners[:],
                "levels": state.levels[:],
                "turns_played": int(state.turns_played),
                "player": int(player),
            }
        )
        if progress_every > 0 and (
            len(cache_records) % progress_every == 0 or len(cache_records) == target_states
        ):
            save_cache(cache_records)
            emit_progress(
                phase="holdout_cache",
                status="progress",
                base_done=len(cache_records),
                base_target=target_states,
                attempts_done=attempts,
                attempts_target=max_attempts,
                elapsed=time.monotonic() - gen_started,
            )

    if len(cache_records) < target_states:
        raise RuntimeError(
            f"Failed to build fixed holdout base states ({len(cache_records)}/{target_states})."
        )

    save_cache(cache_records)
    emit_progress(
        phase="holdout_cache",
        status="done",
        base_done=len(cache_records),
        base_target=target_states,
        attempts_done=attempts,
        attempts_target=max_attempts,
        elapsed=time.monotonic() - gen_started,
    )

    features: List[torch.Tensor] = []
    policy_targets: List[torch.Tensor] = []
    value_targets: List[float] = []
    label_started = time.monotonic()
    base_records = cache_records[:target_states]
    emit_progress(
        phase="holdout_label",
        status="progress",
        base_done=len(base_records),
        base_target=target_states,
        attempts_done=attempts,
        attempts_target=max_attempts,
        elapsed=time.monotonic() - label_started,
        labeled_done=0,
        labeled_target=len(base_records),
        sample_count=0,
    )

    for idx, rec in enumerate(base_records, start=1):
        state = State(
            owners=[int(v) for v in rec["owners"]],
            levels=[int(v) for v in rec["levels"]],
            turns_played=int(rec.get("turns_played", 0)),
        )
        player = int(rec["player"])
        legal_moves, scores = evaluate_all_legal_moves(
            env=env,
            state=state,
            player=player,
            teacher_ply=max(0, teacher_ply),
            teacher_branch_cap=max(0, teacher_branch_cap),
        )
        if not legal_moves:
            continue
        policy_target = legal_policy_from_scores(
            legal_moves=legal_moves,
            move_scores=scores,
            num_nodes=num_nodes,
            temperature=1e-6,
        )
        value = score_to_value(max(scores), value_scale)
        append_labeled_state_with_symmetry(
            env=env,
            state=state,
            player=player,
            policy_target=policy_target,
            value=value,
            rotation_maps=rotation_maps,
            features=features,
            policy_targets=policy_targets,
            value_targets=value_targets,
        )
        if progress_every > 0 and (idx % progress_every == 0 or idx == len(base_records)):
            emit_progress(
                phase="holdout_label",
                status="progress",
                base_done=len(base_records),
                base_target=target_states,
                attempts_done=attempts,
                attempts_target=max_attempts,
                elapsed=time.monotonic() - label_started,
                labeled_done=idx,
                labeled_target=len(base_records),
                sample_count=len(features),
            )

    if not features:
        raise RuntimeError("Failed to label fixed holdout dataset.")

    x = torch.stack(features, dim=0)
    y_policy = torch.stack(policy_targets, dim=0)
    y_value = torch.tensor(value_targets, dtype=torch.float32)
    emit_progress(
        phase="holdout_label",
        status="done",
        base_done=len(base_records),
        base_target=target_states,
        attempts_done=attempts,
        attempts_target=max_attempts,
        elapsed=time.monotonic() - label_started,
        labeled_done=len(base_records),
        labeled_target=len(base_records),
        sample_count=len(features),
    )
    return x, y_policy, y_value


def evaluate(
    model: HexplodeGraphPolicyValueNet,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    value_loss_weight: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_acc = 0.0
    batches = 0

    with torch.no_grad():
        for xb, ypb, yvb in loader:
            xb = xb.to(device, non_blocking=True)
            ypb = ypb.to(device, non_blocking=True)
            yvb = yvb.to(device, non_blocking=True)

            logits, value = model(xb, adj)
            legal_mask = xb[:, :, 7] > 0.5
            masked_logits = logits.masked_fill(~legal_mask, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=1)
            policy_loss = -(ypb * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, yvb)
            loss = policy_loss + value_loss_weight * value_loss

            pred = masked_logits.argmax(dim=1)
            tgt = ypb.argmax(dim=1)
            acc = (pred == tgt).float().mean()

            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_acc += float(acc.item())
            batches += 1

    denom = max(1, batches)
    return {
        "loss": total_loss / denom,
        "policy_loss": total_policy / denom,
        "value_loss": total_value / denom,
        "policy_acc": total_acc / denom,
    }


def train_round(
    model: HexplodeGraphPolicyValueNet,
    device: torch.device,
    adj: torch.Tensor,
    eval_device: torch.device,
    adj_eval: torch.Tensor,
    x: torch.Tensor,
    y_policy: torch.Tensor,
    y_value: torch.Tensor,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    value_loss_weight: float,
    val_split: float,
    log_batches_every: int,
) -> Tuple[List[dict], dict]:
    dataset = TensorDataset(x, y_policy, y_value)
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

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_policy = 0.0
        running_value = 0.0
        running_acc = 0.0
        batches = 0
        total_batches = max(1, len(train_loader))
        epoch_start = time.monotonic()

        for batch_idx, (xb, ypb, yvb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            ypb = ypb.to(device, non_blocking=True)
            yvb = yvb.to(device, non_blocking=True)

            logits, value = model(xb, adj)
            legal_mask = xb[:, :, 7] > 0.5
            masked_logits = logits.masked_fill(~legal_mask, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=1)
            policy_loss = -(ypb * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, yvb)
            loss = policy_loss + value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pred = masked_logits.argmax(dim=1)
            tgt = ypb.argmax(dim=1)
            acc = (pred == tgt).float().mean()

            running_loss += float(loss.item())
            running_policy += float(policy_loss.item())
            running_value += float(value_loss.item())
            running_acc += float(acc.item())
            batches += 1

            if log_batches_every > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % log_batches_every == 0):
                elapsed = time.monotonic() - epoch_start
                eta = (elapsed / max(1, batch_idx)) * max(0, total_batches - batch_idx)
                print(
                    f"    epoch {epoch:02d} batch {batch_idx:04d}/{total_batches:04d} "
                    f"loss={running_loss / batches:.4f} p={running_policy / batches:.4f} "
                    f"v={running_value / batches:.4f} acc={running_acc / batches:.4f} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                )

        train_metrics = {
            "loss": running_loss / max(1, batches),
            "policy_loss": running_policy / max(1, batches),
            "value_loss": running_value / max(1, batches),
            "policy_acc": running_acc / max(1, batches),
        }
        if val_loader is None:
            val_metrics = {
                "loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "policy_acc": float("nan"),
            }
        else:
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
            )
            if eval_device != device:
                del val_model

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"  epoch {epoch:02d} "
            f"train(loss={train_metrics['loss']:.4f}, p={train_metrics['policy_loss']:.4f}, "
            f"v={train_metrics['value_loss']:.4f}, acc={train_metrics['policy_acc']:.4f}) "
            f"val(loss={val_metrics['loss']:.4f}, p={val_metrics['policy_loss']:.4f}, "
            f"v={val_metrics['value_loss']:.4f}, acc={val_metrics['policy_acc']:.4f})"
        )

    best_epoch = min(history, key=lambda r: r["val"]["loss"] if math.isfinite(r["val"]["loss"]) else float("inf"))
    return history, best_epoch


def save_checkpoint(
    path: Path,
    model: HexplodeGraphPolicyValueNet,
    history: List[dict],
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.cpu().state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "blocks": model.blocks,
        "num_nodes": model.num_nodes,
        "meta": meta,
        "history": history,
    }
    torch.save(payload, path)
    model.to(torch.device(meta.get("device", "cpu")))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = HexplodeEnv(side_length=max(2, args.side_length))
    num_nodes = len(env.coords)
    device = choose_device(args.device)
    eval_device = device if args.eval_device == "same" else torch.device("cpu")
    adj = build_adj_matrix(env).to(device)
    adj_eval = adj if eval_device == device else build_adj_matrix(env).to(eval_device)

    model, loaded_from = load_model(args, num_nodes=num_nodes, device=device)
    best_model = clone_gnn_model(model, device)
    best_val = float("inf")
    best_round = 0

    out_path = Path(args.out)
    best_out_path = Path(args.best_out) if args.best_out else out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")
    stats_path = Path(args.stats_out)
    holdout_progress_path = (
        Path(args.holdout_progress_file)
        if (args.holdout_progress_file or "").strip()
        else stats_path.with_name(f"{stats_path.stem}_holdout_progress.json")
    )
    holdout_cache_path = (
        Path(args.holdout_cache_file)
        if (args.holdout_cache_file or "").strip()
        else stats_path.with_name(f"{stats_path.stem}_holdout_states.json")
    )
    stop_file = Path(args.stop_file) if args.stop_file else None

    print("[Init] building fixed holdout dataset...")
    holdout_x, holdout_policy, holdout_value = build_fixed_holdout_dataset(
        env=env,
        base_states=max(32, args.holdout_base_states),
        symmetry_rotations=max(1, min(6, args.holdout_symmetry_rotations)),
        teacher_ply=max(0, args.teacher_ply),
        teacher_branch_cap=max(0, args.teacher_branch_cap),
        value_scale=max(1e-6, args.value_scale),
        state_random_min_plies=max(0, args.state_random_min_plies),
        state_random_max_plies=max(0, args.state_random_max_plies),
        state_minimax_min_plies=max(0, args.state_minimax_min_plies),
        state_minimax_max_plies=max(0, args.state_minimax_max_plies),
        state_minimax_ply=max(0, args.state_minimax_ply),
        state_minimax_branch_cap=max(0, args.state_minimax_branch_cap),
        seed=args.holdout_seed,
        cache_file=holdout_cache_path,
        progress_every=max(0, args.holdout_progress_every),
        progress_file=holdout_progress_path,
    )
    holdout_ds = TensorDataset(holdout_x, holdout_policy, holdout_value)
    holdout_loader = DataLoader(holdout_ds, batch_size=max(1, args.batch_size), shuffle=False)
    print(f"[Init] holdout samples={len(holdout_ds)}")

    replay_x: torch.Tensor | None = None
    replay_policy: torch.Tensor | None = None
    replay_value: torch.Tensor | None = None

    started = time.time()
    deadline = started + max(0.1, args.hours) * 3600.0
    round_idx = 0
    stats_payload = {"args": vars(args), "stats": []}

    print(
        f"[Init] device={device}, side={args.side_length}, nodes={num_nodes}, "
        f"hidden={args.hidden_dim}, blocks={args.blocks}, teacher_ply={args.teacher_ply}, "
        f"branch_cap={args.teacher_branch_cap}, replay_cap={args.replay_max_samples}, "
        f"loaded_from={loaded_from or 'fresh'}, holdout_cache={holdout_cache_path}"
    )

    while time.time() < deadline:
        if stop_file is not None and stop_file.exists():
            print(f"[Stop] stop-file found before round start: {stop_file}")
            break

        round_idx += 1
        round_start = time.time()
        print(f"\n[Round {round_idx}] generating minimax-teacher labels from mixed playout states...")

        x_new, y_policy_new, y_value_new, gen_stats, stopped_early = generate_round_dataset(
            env=env,
            model=model,
            adj=adj,
            device=device,
            round_games=max(1, args.round_games),
            max_turns=max(2, args.max_turns),
            sample_state_prob=max(0.0, min(1.0, args.sample_state_prob)),
            min_sample_turn=max(0, args.min_sample_turn),
            symmetry_rotations=max(1, min(6, args.symmetry_rotations)),
            teacher_ply=max(0, args.teacher_ply),
            teacher_branch_cap=max(0, args.teacher_branch_cap),
            policy_target_temp=max(1e-6, args.policy_target_temp),
            value_scale=max(1e-6, args.value_scale),
            state_random_min_plies=max(0, args.state_random_min_plies),
            state_random_max_plies=max(0, args.state_random_max_plies),
            state_model_min_plies=max(0, args.state_model_min_plies),
            state_model_max_plies=max(0, args.state_model_max_plies),
            state_minimax_min_plies=max(0, args.state_minimax_min_plies),
            state_minimax_max_plies=max(0, args.state_minimax_max_plies),
            state_model_epsilon=max(0.0, min(1.0, args.state_model_epsilon)),
            state_model_temperature=max(1e-6, args.state_model_temperature),
            state_minimax_ply=max(0, args.state_minimax_ply),
            state_minimax_branch_cap=max(0, args.state_minimax_branch_cap),
            seed=args.seed + (round_idx * 977),
            progress_every_games=max(0, args.progress_every_games),
            stop_file=stop_file,
        )

        x, y_policy, y_value, replay_used = merge_with_replay(
            x_new=x_new,
            y_policy_new=y_policy_new,
            y_value_new=y_value_new,
            replay_x=replay_x,
            replay_policy=replay_policy,
            replay_value=replay_value,
            replay_ratio=max(0.0, args.replay_sample_ratio),
            seed=args.seed + round_idx * 1237,
        )

        print(
            f"[Round {round_idx}] dataset samples(new={gen_stats['samples']}, replay={replay_used}, train={len(x)}) "
            f"base_states={gen_stats['sampled_states']} avg_len={gen_stats['avg_game_len']:.1f} "
            f"W/D/L(red={gen_stats['red_wins']},blue={gen_stats['blue_wins']},draw={gen_stats['draws']})"
        )

        history, best_epoch = train_round(
            model=model,
            device=device,
            adj=adj,
            eval_device=eval_device,
            adj_eval=adj_eval,
            x=x,
            y_policy=y_policy,
            y_value=y_value,
            batch_size=max(1, args.batch_size),
            epochs=max(1, args.round_epochs),
            lr=args.lr,
            weight_decay=args.weight_decay,
            value_loss_weight=args.value_loss_weight,
            val_split=max(0.0, min(0.5, args.val_split)),
            log_batches_every=max(0, args.log_batches_every),
        )

        replay_x, replay_policy, replay_value = update_replay_buffer(
            replay_x=replay_x,
            replay_policy=replay_policy,
            replay_value=replay_value,
            x_new=x_new,
            y_policy_new=y_policy_new,
            y_value_new=y_value_new,
            max_samples=max(0, args.replay_max_samples),
        )

        if eval_device == device:
            holdout_model = model
        else:
            holdout_model = copy.deepcopy(model).to(eval_device)
            holdout_model.eval()
        holdout_metrics = evaluate(
            model=holdout_model,
            loader=holdout_loader,
            adj=adj_eval,
            device=eval_device,
            value_loss_weight=args.value_loss_weight,
        )
        if eval_device != device:
            del holdout_model

        val_loss = float(best_epoch["val"]["loss"]) if math.isfinite(float(best_epoch["val"]["loss"])) else float(best_epoch["train"]["loss"])
        promo_metric = float(holdout_metrics["loss"]) if math.isfinite(float(holdout_metrics["loss"])) else val_loss
        promoted = promo_metric < best_val
        if promoted:
            best_val = promo_metric
            best_round = round_idx
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
            best_model.to(device)
            best_model.eval()

        round_elapsed = time.time() - round_start
        total_elapsed = time.time() - started
        last_epoch = history[-1]
        round_stat = {
            "round": round_idx,
            "elapsed_total_sec": total_elapsed,
            "elapsed_round_sec": round_elapsed,
            "selfplay_games": gen_stats["selfplay_games"],
            "samples": gen_stats["samples"],
            "sampled_states": gen_stats["sampled_states"],
            "avg_game_len": gen_stats["avg_game_len"],
            "red_wins": gen_stats["red_wins"],
            "blue_wins": gen_stats["blue_wins"],
            "draws": gen_stats["draws"],
            "avg_legal_moves": gen_stats["avg_legal_moves"],
            "avg_best_score": gen_stats["avg_best_score"],
            "avg_expected_score": gen_stats["avg_expected_score"],
            "unique_symmetry_samples": gen_stats["unique_symmetry_samples"],
            "avg_random_prefix": gen_stats["avg_random_prefix"],
            "avg_model_prefix": gen_stats["avg_model_prefix"],
            "avg_minimax_prefix": gen_stats["avg_minimax_prefix"],
            "replay_used_samples": replay_used,
            "replay_size": 0 if replay_x is None else int(len(replay_x)),
            "train_loss": float(last_epoch["train"]["loss"]),
            "train_policy_loss": float(last_epoch["train"]["policy_loss"]),
            "train_value_loss": float(last_epoch["train"]["value_loss"]),
            "train_policy_acc": float(last_epoch["train"]["policy_acc"]),
            "val_loss": float(last_epoch["val"]["loss"]),
            "val_policy_loss": float(last_epoch["val"]["policy_loss"]),
            "val_value_loss": float(last_epoch["val"]["value_loss"]),
            "val_policy_acc": float(last_epoch["val"]["policy_acc"]),
            "holdout_loss": float(holdout_metrics["loss"]),
            "holdout_policy_loss": float(holdout_metrics["policy_loss"]),
            "holdout_value_loss": float(holdout_metrics["value_loss"]),
            "holdout_policy_acc": float(holdout_metrics["policy_acc"]),
            "best_epoch_val_loss": val_loss,
            "promotion_metric": promo_metric,
            "promoted": promoted,
            "best_round_so_far": best_round,
            "best_val_so_far": best_val,
        }
        stats_payload["stats"].append(round_stat)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats_payload, f, indent=2)

        meta = {
            "mode": "gnn_minimax_teacher",
            "round": round_idx,
            "device": str(device),
            "side_length": args.side_length,
            "teacher_ply": args.teacher_ply,
            "teacher_branch_cap": args.teacher_branch_cap,
            "round_games": args.round_games,
            "round_epochs": args.round_epochs,
            "symmetry_rotations": args.symmetry_rotations,
            "sample_state_prob": args.sample_state_prob,
            "bootstrap_checkpoint": loaded_from,
            "best_round": best_round,
            "best_holdout_loss": best_val,
            "holdout_samples": len(holdout_ds),
            "last_round": round_stat,
        }
        save_checkpoint(out_path, model, history, meta)
        if promoted:
            save_checkpoint(best_out_path, model, history, meta)

        print(
            f"[Round {round_idx}] "
            f"train={round_stat['train_loss']:.4f} val={round_stat['val_loss']:.4f} "
            f"holdout={round_stat['holdout_loss']:.4f} acc={round_stat['holdout_policy_acc']:.3f} "
            f"promoted={promoted} "
            f"round_time={format_duration(round_elapsed)} total={format_duration(total_elapsed)}"
        )

        if stopped_early:
            print("[Stop] generation stopped early due to stop-file; ending after checkpoint save.")
            break
        if stop_file is not None and stop_file.exists():
            print(f"[Stop] stop-file detected after round: {stop_file}")
            break

    print("\nTraining complete.")
    print(f"Latest checkpoint: {out_path}")
    print(f"Best checkpoint:   {best_out_path}")
    print(f"Stats:             {stats_path}")


if __name__ == "__main__":
    main()
