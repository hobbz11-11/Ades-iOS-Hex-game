#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from hexplode_nn import BLUE, RED, HexplodeEnv, HexplodePolicyValueNet, soft_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure self-play trainer for Hexplode large-board hard AI.")
    parser.add_argument("--hours", type=float, default=8.0, help="Maximum wall-clock training budget in hours.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("--channels", type=int, default=96, help="Residual trunk channels.")
    parser.add_argument("--blocks", type=int, default=8, help="Residual blocks.")
    parser.add_argument("--batch-size", type=int, default=320, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--value-loss-weight", type=float, default=0.7, help="Value loss multiplier.")
    parser.add_argument("--round-games", type=int, default=800, help="Self-play games generated each round.")
    parser.add_argument("--round-epochs", type=int, default=6, help="Training epochs per round.")
    parser.add_argument("--max-turns", type=int, default=200, help="Max turns per self-play game.")
    parser.add_argument("--sample-stride", type=int, default=2, help="Keep one sample every N turns.")
    parser.add_argument("--top-k", type=int, default=8, help="Candidate moves evaluated per position.")
    parser.add_argument("--prior-weight", type=float, default=1.0, help="Weight for root policy prior in search.")
    parser.add_argument("--value-weight", type=float, default=2.25, help="Weight for next-state value in search.")
    parser.add_argument(
        "--short-term-weight",
        type=float,
        default=0.85,
        help="Weight for 2-ply tactical score swing in move scoring.",
    )
    parser.add_argument(
        "--short-term-scale",
        type=float,
        default=8.0,
        help="Scale for tanh-normalized 2-ply tactical swing.",
    )
    parser.add_argument(
        "--response-top-k",
        type=int,
        default=10,
        help="Maximum opponent responses evaluated for short-term tactical scoring.",
    )
    parser.add_argument("--search-temperature", type=float, default=0.65, help="Softmax temperature for searched move targets.")
    parser.add_argument("--move-temperature", type=float, default=0.8, help="Sampling temperature for self-play move selection.")
    parser.add_argument("--epsilon", type=float, default=0.06, help="Random move exploration rate.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.25, help="Dirichlet alpha for root exploration.")
    parser.add_argument("--dirichlet-frac", type=float, default=0.15, help="Dirichlet mixing factor for root exploration.")
    parser.add_argument("--val-split", type=float, default=0.12, help="Validation split per round.")
    parser.add_argument("--progress-every", type=int, default=100, help="Log self-play progress every N games.")
    parser.add_argument("--log-batches-every", type=int, default=40, help="Log every N train batches.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--bootstrap-checkpoint",
        type=str,
        default="tools/outputs/hexplode_policy_value_v4_long.pt",
        help="Optional checkpoint to bootstrap from before pure self-play.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tools/outputs/hexplode_policy_value_selfplay.pt",
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--stop-file",
        type=str,
        default="",
        help="Optional path; if this file exists, stop cleanly after current round checkpoint.",
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


def softmax(values: Sequence[float], temperature: float) -> List[float]:
    if not values:
        return []
    t = max(0.05, temperature)
    max_value = max(values)
    exps = [math.exp(max(-40.0, min(40.0, (v - max_value) / t))) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def sample_index(weights: Sequence[float], rng: random.Random) -> int:
    roll = rng.random()
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if roll <= cumulative:
            return idx
    return max(0, len(weights) - 1)


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[HexplodePolicyValueNet, dict]:
    checkpoint_path = Path(args.bootstrap_checkpoint)
    model = HexplodePolicyValueNet(channels=args.channels, blocks=args.blocks)
    metadata: dict = {"bootstrapped": False}

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = HexplodePolicyValueNet(
            channels=int(checkpoint.get("channels", args.channels)),
            blocks=int(checkpoint.get("blocks", args.blocks)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        metadata = dict(checkpoint.get("meta", {}))
        metadata["bootstrapped"] = True

    model.to(device)
    return model, metadata


def infer_policy_value(
    model: HexplodePolicyValueNet,
    env: HexplodeEnv,
    states: Sequence,
    player: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack([env.encode_features(state, player) for state in states], dim=0).to(device)
    model.eval()
    with torch.no_grad():
        logits, values = model(features)
    return logits.cpu(), values.cpu()


def score_diff_for_player(env: HexplodeEnv, state, player: int) -> float:
    red_score, blue_score = env.score_totals(state)
    return float(red_score - blue_score if player == RED else blue_score - red_score)


def two_ply_swing_signal(
    env: HexplodeEnv,
    state,
    next_state,
    player: int,
    response_top_k: int,
    short_term_scale: float,
) -> float:
    before_diff = score_diff_for_player(env, state, player)
    after_own_move = score_diff_for_player(env, next_state, player)

    opponent = BLUE if player == RED else RED
    opponent_moves = env.legal_moves(next_state, opponent)
    if response_top_k > 0 and len(opponent_moves) > response_top_k:
        opponent_moves = opponent_moves[:response_top_k]

    # Assume opponent chooses the response that hurts current player most.
    worst_after_reply = after_own_move
    for response in opponent_moves:
        reply_state = env.apply_move(next_state, response, opponent)
        reply_diff = score_diff_for_player(env, reply_state, player)
        if reply_diff < worst_after_reply:
            worst_after_reply = reply_diff
        if env.terminal_winner(reply_state) == opponent:
            worst_after_reply = reply_diff
            break

    raw_swing = worst_after_reply - before_diff
    return math.tanh(raw_swing / max(1.0, short_term_scale))


def choose_selfplay_move(
    env: HexplodeEnv,
    model: HexplodePolicyValueNet,
    state,
    player: int,
    device: torch.device,
    rng: random.Random,
    top_k: int,
    prior_weight: float,
    value_weight: float,
    short_term_weight: float,
    short_term_scale: float,
    response_top_k: int,
    search_temperature: float,
    move_temperature: float,
    epsilon: float,
    dirichlet_alpha: float,
    dirichlet_frac: float,
) -> Tuple[int, torch.Tensor]:
    legal_moves = env.legal_moves(state, player)
    if not legal_moves:
        empty = torch.zeros((env.grid_size * env.grid_size,), dtype=torch.float32)
        return -1, empty

    root_logits, _ = infer_policy_value(model, env, [state], player, device)
    logits = root_logits[0]
    root_scores = [float(logits[env.flat_index[idx]]) for idx in legal_moves]
    priors = softmax(root_scores, temperature=1.0)

    if dirichlet_frac > 0 and len(legal_moves) > 1:
        noise = list(rng.gammavariate(dirichlet_alpha, 1.0) for _ in legal_moves)
        noise_total = sum(noise)
        noise = [n / noise_total for n in noise]
        priors = [
            (1.0 - dirichlet_frac) * prior + dirichlet_frac * n
            for prior, n in zip(priors, noise)
        ]

    ranked = sorted(zip(legal_moves, priors), key=lambda item: item[1], reverse=True)
    candidate_moves = ranked[: max(1, min(top_k, len(ranked)))]
    next_states = [env.apply_move(state, move, player) for move, _ in candidate_moves]

    winning_moves = [
        (move, prior)
        for (move, prior), next_state in zip(candidate_moves, next_states)
        if env.terminal_winner(next_state) == player
    ]
    if winning_moves:
        weight_sum = sum(prior for _, prior in winning_moves)
        target = env.encode_policy_target(
            (move, prior / max(1e-8, weight_sum)) for move, prior in winning_moves
        )
        best_idx = sample_index([prior / max(1e-8, weight_sum) for _, prior in winning_moves], rng)
        return winning_moves[best_idx][0], target

    opponent = BLUE if player == RED else RED
    _, opponent_values = infer_policy_value(model, env, next_states, opponent, device)

    candidate_scores: List[float] = []
    candidate_moves_only: List[int] = []
    for ((move, prior), next_state), opponent_value in zip(zip(candidate_moves, next_states), opponent_values.tolist()):
        candidate_moves_only.append(move)
        short_term_signal = 0.0
        if short_term_weight > 1e-8:
            short_term_signal = two_ply_swing_signal(
                env=env,
                state=state,
                next_state=next_state,
                player=player,
                response_top_k=response_top_k,
                short_term_scale=short_term_scale,
            )
        candidate_scores.append(
            prior_weight * math.log(max(prior, 1e-8))
            + value_weight * (-opponent_value)
            + short_term_weight * short_term_signal
        )

    target_probs = softmax(candidate_scores, temperature=search_temperature)
    target = env.encode_policy_target(zip(candidate_moves_only, target_probs))

    if rng.random() < epsilon:
        chosen_move = rng.choice(candidate_moves_only)
        return chosen_move, target

    move_probs = softmax(candidate_scores, temperature=move_temperature)
    chosen_index = sample_index(move_probs, rng)
    return candidate_moves_only[chosen_index], target


def generate_selfplay_dataset(
    env: HexplodeEnv,
    model: HexplodePolicyValueNet,
    device: torch.device,
    games: int,
    seed: int,
    max_turns: int,
    sample_stride: int,
    top_k: int,
    prior_weight: float,
    value_weight: float,
    short_term_weight: float,
    short_term_scale: float,
    response_top_k: int,
    search_temperature: float,
    move_temperature: float,
    epsilon: float,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    progress_every: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    rng = random.Random(seed)
    features: List[torch.Tensor] = []
    policies: List[torch.Tensor] = []
    values: List[float] = []

    red_wins = 0
    blue_wins = 0
    draws = 0
    total_turns = 0

    for game_idx in range(games):
        state = env.empty_state()
        player = RED if rng.random() < 0.5 else BLUE
        records: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

        winner = None
        turns = 0
        while turns < max_turns:
            winner = env.terminal_winner(state)
            if winner is not None:
                break

            move, policy_target = choose_selfplay_move(
                env=env,
                model=model,
                state=state,
                player=player,
                device=device,
                rng=rng,
                top_k=top_k,
                prior_weight=prior_weight,
                value_weight=value_weight,
                short_term_weight=short_term_weight,
                short_term_scale=short_term_scale,
                response_top_k=response_top_k,
                search_temperature=search_temperature,
                move_temperature=move_temperature,
                epsilon=epsilon,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_frac=dirichlet_frac,
            )
            if move < 0:
                break

            if sample_stride <= 1 or (turns % sample_stride == 0):
                records.append((env.encode_features(state, player), policy_target, player))

            state = env.apply_move(state, move, player)
            turns += 1
            winner = env.terminal_winner(state)
            if winner is not None:
                break
            player = BLUE if player == RED else RED

        if winner is None:
            winner = env.terminal_winner(state)
        if winner is None:
            red_score, blue_score = env.score_totals(state)
            if red_score > blue_score:
                winner = RED
            elif blue_score > red_score:
                winner = BLUE
            else:
                winner = 0

        if winner == RED:
            red_wins += 1
        elif winner == BLUE:
            blue_wins += 1
        else:
            draws += 1
        total_turns += turns

        for feature, policy_target, sample_player in records:
            outcome = 0.0 if winner == 0 else (1.0 if sample_player == winner else -1.0)
            features.append(feature)
            policies.append(policy_target)
            values.append(outcome)

        if progress_every > 0 and (game_idx + 1) % progress_every == 0:
            avg_turns = total_turns / max(1, game_idx + 1)
            print(
                f"  self-play: {game_idx + 1} games, {len(features)} samples, "
                f"R/B/D={red_wins}/{blue_wins}/{draws}, avgTurns={avg_turns:.1f}"
            )

    x = torch.stack(features, dim=0)
    y_policy = torch.stack(policies, dim=0)
    y_value = torch.tensor(values, dtype=torch.float32)
    stats = {
        "games": games,
        "samples": len(features),
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "avg_turns": total_turns / max(1, games),
    }
    return x, y_policy, y_value, stats


def evaluate(model: HexplodePolicyValueNet, loader: DataLoader, device: torch.device, value_w: float) -> dict:
    model.eval()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for x, y_policy, y_value in loader:
            x = x.to(device, non_blocking=True)
            y_policy = y_policy.to(device, non_blocking=True)
            y_value = y_value.to(device, non_blocking=True)
            logits, value = model(x)
            policy_loss = soft_cross_entropy(logits, y_policy)
            value_loss = torch.nn.functional.mse_loss(value, y_value)
            loss = policy_loss + value_w * value_loss
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_loss += loss.item()
            total_batches += 1
    denom = max(1, total_batches)
    return {
        "loss": total_loss / denom,
        "policy_loss": total_policy / denom,
        "value_loss": total_value / denom,
    }


def train_round(
    model: HexplodePolicyValueNet,
    device: torch.device,
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
) -> Tuple[List[dict], float]:
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
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_policy = 0.0
        running_value = 0.0
        running_total = 0.0
        batches = 0
        total_batches = max(1, len(train_loader))
        epoch_start = time.monotonic()

        for batch_idx, (xb, pb, vb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)

            logits, value = model(xb)
            policy_loss = soft_cross_entropy(logits, pb)
            value_loss = torch.nn.functional.mse_loss(value, vb)
            loss = policy_loss + value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_policy += policy_loss.item()
            running_value += value_loss.item()
            running_total += loss.item()
            batches += 1

            if log_batches_every > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % log_batches_every == 0):
                elapsed = time.monotonic() - epoch_start
                eta = (elapsed / max(1, batch_idx)) * max(0, total_batches - batch_idx)
                print(
                    f"    epoch {epoch:02d} batch {batch_idx:04d}/{total_batches:04d} "
                    f"({(100.0 * batch_idx / total_batches):5.1f}%) "
                    f"loss={running_total / batches:.4f} "
                    f"p={running_policy / batches:.4f} "
                    f"v={running_value / batches:.4f} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                )

        train_metrics = {
            "loss": running_total / max(1, batches),
            "policy_loss": running_policy / max(1, batches),
            "value_loss": running_value / max(1, batches),
        }
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, value_loss_weight)
            best_val = min(best_val, val_metrics["loss"])
        else:
            val_metrics = {"loss": float("nan"), "policy_loss": float("nan"), "value_loss": float("nan")}

        summary = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(summary)
        print(
            f"  epoch {epoch:02d} "
            f"train(loss={train_metrics['loss']:.4f}, p={train_metrics['policy_loss']:.4f}, v={train_metrics['value_loss']:.4f}) "
            f"val(loss={val_metrics['loss']:.4f}, p={val_metrics['policy_loss']:.4f}, v={val_metrics['value_loss']:.4f})"
        )

    return history, best_val


def save_checkpoint(path: Path, model: HexplodePolicyValueNet, args: argparse.Namespace, history: List[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "channels": args.channels,
        "blocks": args.blocks,
        "board_size": 7,
        "input_channels": 8,
        "meta": meta,
        "history": history,
    }
    torch.save(checkpoint, path)
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    model.to(choose_device(args.device))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    env = HexplodeEnv(side_length=4)
    model, bootstrap_meta = load_model(args, device)

    out_path = Path(args.out)
    stop_file = Path(args.stop_file) if args.stop_file else None
    started_at = time.time()
    deadline = started_at + args.hours * 3600.0
    round_index = 0
    all_history: List[dict] = []
    best_val = float("inf")

    print(
        f"Starting pure self-play training on {device}. "
        f"Budget={args.hours:.1f}h bootstrap={'yes' if bootstrap_meta.get('bootstrapped') else 'no'}"
    )

    while time.time() < deadline:
        round_index += 1
        elapsed = time.time() - started_at
        remaining = max(0.0, deadline - time.time())
        print(
            f"\nRound {round_index:02d} | elapsed={format_duration(elapsed)} "
            f"remaining={format_duration(remaining)}"
        )

        x, y_policy, y_value, stats = generate_selfplay_dataset(
            env=env,
            model=model,
            device=device,
            games=args.round_games,
            seed=args.seed + round_index * 101,
            max_turns=args.max_turns,
            sample_stride=max(1, args.sample_stride),
            top_k=max(1, args.top_k),
            prior_weight=args.prior_weight,
            value_weight=args.value_weight,
            short_term_weight=args.short_term_weight,
            short_term_scale=args.short_term_scale,
            response_top_k=max(1, args.response_top_k),
            search_temperature=args.search_temperature,
            move_temperature=args.move_temperature,
            epsilon=args.epsilon,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_frac=args.dirichlet_frac,
            progress_every=max(0, args.progress_every),
        )
        print(
            f"  round dataset: samples={stats['samples']} "
            f"R/B/D={stats['red_wins']}/{stats['blue_wins']}/{stats['draws']} "
            f"avgTurns={stats['avg_turns']:.1f}"
        )

        round_history, round_best_val = train_round(
            model=model,
            device=device,
            x=x,
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
        best_val = min(best_val, round_best_val)
        all_history.append(
            {
                "round": round_index,
                "stats": stats,
                "train_history": round_history,
            }
        )

        meta = {
            "mode": "pure_self_play",
            "bootstrapped_from": args.bootstrap_checkpoint if bootstrap_meta.get("bootstrapped") else "",
            "hours_budget": args.hours,
            "round_games": args.round_games,
            "round_epochs": args.round_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "value_loss_weight": args.value_loss_weight,
            "max_turns": args.max_turns,
            "sample_stride": args.sample_stride,
            "top_k": args.top_k,
            "prior_weight": args.prior_weight,
            "value_weight": args.value_weight,
            "short_term_weight": args.short_term_weight,
            "short_term_scale": args.short_term_scale,
            "response_top_k": args.response_top_k,
            "search_temperature": args.search_temperature,
            "move_temperature": args.move_temperature,
            "epsilon": args.epsilon,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_frac": args.dirichlet_frac,
            "device": str(device),
            "rounds_completed": round_index,
            "best_val_loss": best_val,
            "elapsed_seconds": time.time() - started_at,
            "last_round_stats": stats,
        }
        save_checkpoint(out_path, model, args, all_history, meta)
        print(f"Saved checkpoint: {out_path}")

        if stop_file is not None and stop_file.exists():
            print(f"Stop file detected ({stop_file}). Stopping cleanly at round boundary.")
            break

        if time.time() + 600 > deadline:
            print("Stopping before next round because remaining budget is below 10 minutes.")
            break

    print("\nPure self-play training complete.")
    print(f"Final checkpoint: {out_path}")


if __name__ == "__main__":
    main()
