#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from hexplode_nn import HexplodePolicyValueNet, generate_self_play_dataset, soft_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a policy-value net for Hexplode (large board).")
    parser.add_argument("--games", type=int, default=2400, help="Number of self-play games for dataset generation.")
    parser.add_argument("--epochs", type=int, default=14, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--channels", type=int, default=64, help="Residual trunk channels.")
    parser.add_argument("--blocks", type=int, default=6, help="Number of residual blocks.")
    parser.add_argument("--val-split", type=float, default=0.12, help="Validation split [0, 0.5).")
    parser.add_argument("--value-loss-weight", type=float, default=0.5, help="Scale factor for value MSE.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=1.4, help="Teacher softmax temperature.")
    parser.add_argument("--epsilon", type=float, default=0.12, help="Action exploration during self-play.")
    parser.add_argument("--max-turns", type=int, default=200, help="Max turns per self-play game.")
    parser.add_argument("--sample-stride", type=int, default=2, help="Keep one sample every N turns.")
    parser.add_argument("--max-samples", type=int, default=80000, help="Cap total training samples.")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N games.")
    parser.add_argument("--log-batches-every", type=int, default=100, help="Log every N train batches per epoch.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--out", type=str, default="tools/outputs/hexplode_policy_value.pt", help="Checkpoint path.")
    parser.add_argument(
        "--dataset-out",
        type=str,
        default="",
        help="Optional torch file for saved generated dataset (features/policy/value).",
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Generating self-play dataset...")
    def progress(game_count: int, sample_count: int) -> None:
        print(f"  self-play: {game_count} games, {sample_count} samples")

    x, y_policy, y_value = generate_self_play_dataset(
        games=args.games,
        seed=args.seed,
        epsilon=args.epsilon,
        temperature=args.temperature,
        max_turns=args.max_turns,
        sample_stride=max(1, args.sample_stride),
        max_samples=max(0, args.max_samples),
        progress_every=max(0, args.progress_every),
        progress_fn=progress if args.progress_every > 0 else None,
    )
    print(f"Samples: {x.shape[0]}")

    dataset = TensorDataset(x, y_policy, y_value)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_size > 0 else None

    device = choose_device(args.device)
    print(f"Device: {device}")

    model = HexplodePolicyValueNet(channels=args.channels, blocks=args.blocks)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
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
            loss = policy_loss + args.value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_policy += policy_loss.item()
            running_value += value_loss.item()
            running_total += loss.item()
            batches += 1

            should_log = (
                args.log_batches_every > 0
                and (batch_idx == 1 or batch_idx == total_batches or batch_idx % args.log_batches_every == 0)
            )
            if should_log:
                elapsed = time.monotonic() - epoch_start
                eta = (elapsed / max(1, batch_idx)) * max(0, total_batches - batch_idx)
                print(
                    f"  epoch {epoch:02d} batch {batch_idx:04d}/{total_batches:04d} "
                    f"({(100.0 * batch_idx / total_batches):5.1f}%) "
                    f"loss={running_total / batches:.4f} "
                    f"p={running_policy / batches:.4f} "
                    f"v={running_value / batches:.4f} "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(eta)}"
                )

        train_metrics = {
            "loss": running_total / max(1, batches),
            "policy_loss": running_policy / max(1, batches),
            "value_loss": running_value / max(1, batches),
        }

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args.value_loss_weight)
            best_val = min(best_val, val_metrics["loss"])
        else:
            val_metrics = {"loss": float("nan"), "policy_loss": float("nan"), "value_loss": float("nan")}

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:02d} "
            f"train(loss={train_metrics['loss']:.4f}, p={train_metrics['policy_loss']:.4f}, v={train_metrics['value_loss']:.4f}) "
            f"val(loss={val_metrics['loss']:.4f}, p={val_metrics['policy_loss']:.4f}, v={val_metrics['value_loss']:.4f})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "channels": args.channels,
        "blocks": args.blocks,
        "board_size": 7,
        "input_channels": 8,
        "meta": {
            "games": args.games,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "value_loss_weight": args.value_loss_weight,
            "seed": args.seed,
            "temperature": args.temperature,
            "epsilon": args.epsilon,
            "max_turns": args.max_turns,
            "sample_stride": args.sample_stride,
            "max_samples": args.max_samples,
            "best_val_loss": best_val,
        },
        "history": history,
    }
    torch.save(checkpoint, out_path)
    print(f"Saved checkpoint: {out_path}")

    if args.dataset_out:
        dataset_out = Path(args.dataset_out)
        dataset_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": x, "policy": y_policy, "value": y_value}, dataset_out)
        print(f"Saved dataset: {dataset_out}")

    metrics_path = out_path.with_suffix(".json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint["meta"], f, indent=2)
    print(f"Saved metadata: {metrics_path}")


if __name__ == "__main__":
    main()
