#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

import torch
import torch.nn.functional as F

from hexplode_counterfactual_train import (
    build_random_state,
    build_selfplay_prefix_state,
    choose_device,
    choose_model_move,
    label_from_outcome,
    opponent_of,
    ordered_rollout_moves,
    score_diff_for_player,
    value_target_from_outcome,
)
from hexplode_nn import BLUE, EMPTY, RED, HexplodeEnv, HexplodePolicyValueNet, State


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive UI for Hexplode counterfactual training.")
    parser.add_argument("--checkpoint", type=str, default="tools/outputs/hexplode_policy_value_counterfactual_12h_newlogic_best.pt")
    parser.add_argument("--side-length", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--lookahead-ply", type=int, default=5)
    parser.add_argument("--branch-cap", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.45)
    parser.add_argument("--sample-move-epsilon", type=float, default=0.02)
    parser.add_argument("--prefix-policy-epsilon", type=float, default=0.0)
    parser.add_argument("--good-rule", type=str, default="after_positive", choices=["after_positive", "delta_nonnegative", "both"])
    parser.add_argument("--good-margin", type=float, default=0.0)
    parser.add_argument("--value-target", type=str, default="binary", choices=["binary", "scaled_after", "scaled_delta"])
    parser.add_argument("--value-scale", type=float, default=12.0)
    parser.add_argument("--out", type=str, default="tools/outputs/hexplode_policy_value_ui.pt")
    return parser.parse_args()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def color_interp(hex_a: str, hex_b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a = tuple(int(hex_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i : i + 2], 16) for i in (1, 3, 5))
    out = tuple(int(lerp(float(a[i]), float(b[i]), t)) for i in range(3))
    return f"#{out[0]:02x}{out[1]:02x}{out[2]:02x}"


def heat_color(value: float) -> str:
    v = max(-1.0, min(1.0, value))
    if v >= 0.0:
        return color_interp("#1a2030", "#ff7043", v)
    return color_interp("#1a2030", "#4a90ff", -v)


def load_checkpoint_model(path: Path, device: torch.device) -> Tuple[HexplodePolicyValueNet, str]:
    if path.exists():
        ckpt = torch.load(path, map_location="cpu")
        model = HexplodePolicyValueNet(
            input_channels=int(ckpt.get("input_channels", 8)),
            channels=int(ckpt.get("channels", 96)),
            blocks=int(ckpt.get("blocks", 8)),
            board_size=int(ckpt.get("board_size", 7)),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model, str(path)
    model = HexplodePolicyValueNet(channels=96, blocks=8)
    model.to(device)
    return model, ""


def minimax_with_line(
    env: HexplodeEnv,
    state: State,
    to_move: int,
    root_player: int,
    plies_left: int,
    max_branch_checks: int,
    cache: dict[Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int], Tuple[int, List[int]]],
) -> Tuple[int, List[int]]:
    winner = env.terminal_winner(state)
    if winner is not None:
        if winner == root_player:
            return 10_000 + plies_left, []
        if winner == 0:
            return 0, []
        return -10_000 - plies_left, []
    if plies_left <= 0:
        return score_diff_for_player(env, state, root_player), []

    key = (tuple(state.owners), tuple(state.levels), state.turns_played, to_move, plies_left)
    cached = cache.get(key)
    if cached is not None:
        return cached[0], list(cached[1])

    legal = ordered_rollout_moves(
        env=env,
        state=state,
        to_move=to_move,
        root_player=root_player,
        max_branch_checks=max_branch_checks,
    )
    if not legal:
        score = score_diff_for_player(env, state, root_player)
        cache[key] = (score, [])
        return score, []

    next_player = opponent_of(to_move)
    maximizing = to_move == root_player
    best_score = -10**12 if maximizing else 10**12
    best_line: List[int] = []

    for mv in legal:
        nxt = env.apply_move(state, mv, to_move)
        child_score, child_line = minimax_with_line(
            env=env,
            state=nxt,
            to_move=next_player,
            root_player=root_player,
            plies_left=plies_left - 1,
            max_branch_checks=max_branch_checks,
            cache=cache,
        )
        better = child_score > best_score if maximizing else child_score < best_score
        if better:
            best_score = child_score
            best_line = [mv] + child_line

    cache[key] = (best_score, list(best_line))
    return best_score, best_line


def apply_move_line(env: HexplodeEnv, state: State, to_move: int, line: Sequence[int]) -> State:
    cur = state
    player = to_move
    for mv in line:
        cur = env.apply_move(cur, mv, player)
        player = opponent_of(player)
    return cur


@dataclass
class ReplaySample:
    x: torch.Tensor
    move_idx: int
    policy_label: float
    value_target: float


class GridHeatmap(ttk.Frame):
    def __init__(self, parent: tk.Widget, title: str, grid_size: int, valid_mask: List[List[bool]]):
        super().__init__(parent)
        self.grid_size = grid_size
        self.valid_mask = valid_mask
        ttk.Label(self, text=title).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=220, height=220, bg="#111319", highlightthickness=1, highlightbackground="#2f3441")
        self.canvas.pack(fill="both", expand=True)
        self.value_label = ttk.Label(self, text="", font=("Menlo", 10))
        self.value_label.pack(anchor="w")

    def draw(self, matrix: torch.Tensor, note: str = "") -> None:
        self.canvas.delete("all")
        n = self.grid_size
        pad = 8.0
        w = float(self.canvas.winfo_width() or 220)
        h = float(self.canvas.winfo_height() or 220)
        cell = min((w - 2 * pad) / n, (h - 2 * pad) / n)

        max_abs = max(1e-6, float(matrix.abs().max().item()))
        for r in range(n):
            for c in range(n):
                x0 = pad + c * cell
                y0 = pad + r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                if not self.valid_mask[r][c]:
                    color = "#1a1d25"
                else:
                    val = float(matrix[r, c].item()) / max_abs
                    color = heat_color(val)
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#1b1f28")
        self.value_label.config(text=note)


class HexBoardCanvas(ttk.Frame):
    def __init__(self, parent: tk.Widget, env: HexplodeEnv, title: str):
        super().__init__(parent)
        self.env = env
        self.title = ttk.Label(self, text=title)
        self.title.pack(anchor="w")
        self.canvas = tk.Canvas(self, width=340, height=320, bg="#111319", highlightthickness=1, highlightbackground="#2f3441")
        self.canvas.pack(fill="both", expand=True)
        self.info = ttk.Label(self, text="", font=("Menlo", 10))
        self.info.pack(anchor="w")

    def draw_state(self, state: State, info: str = "", highlight_idx: int = -1) -> None:
        self.canvas.delete("all")
        w = float(self.canvas.winfo_width() or 340)
        h = float(self.canvas.winfo_height() or 320)
        size = min(w, h) / (2.8 * (self.env.radius + 1))
        sqrt3 = math.sqrt(3.0)

        positions: List[Tuple[float, float]] = []
        for q, r in self.env.coords:
            px = size * sqrt3 * (q + r / 2.0)
            py = size * 1.5 * r
            positions.append((px, py))
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        cx = (w - (max_x - min_x)) / 2.0 - min_x
        cy = (h - (max_y - min_y)) / 2.0 - min_y

        for idx, ((q, r), owner, lvl) in enumerate(zip(self.env.coords, state.owners, state.levels)):
            px = size * sqrt3 * (q + r / 2.0) + cx
            py = size * 1.5 * r + cy
            pts: List[float] = []
            for k in range(6):
                ang = math.radians(60 * k - 30)
                pts.extend([px + size * math.cos(ang), py + size * math.sin(ang)])

            fill = "#4c5260"
            if owner == RED:
                fill = "#5a2020"
            elif owner == BLUE:
                fill = "#1f355e"

            outline = "#292f3a"
            width = 2
            if idx == highlight_idx:
                outline = "#f2c84b"
                width = 3
            self.canvas.create_polygon(pts, fill=fill, outline=outline, width=width)

            if owner != EMPTY and lvl > 0:
                ball_color = "#ff5e5e" if owner == RED else "#68a2ff"
                rad = size * 0.44
                self.canvas.create_oval(px - rad, py - rad, px + rad, py + rad, fill=ball_color, outline="#111")
                self.canvas.create_text(px, py, text=str(lvl), fill="#ffffff", font=("Menlo", max(8, int(size * 0.55)), "bold"))

        self.info.config(text=info)


class TrainingUI(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.title("Hexplode Training Visualizer")
        self.geometry("1640x980")
        self.minsize(1400, 900)

        self.args = args
        self.device = choose_device(args.device)
        self.env = HexplodeEnv(side_length=max(2, args.side_length))
        self.rng = random.Random(args.seed)
        self.model, loaded = load_checkpoint_model(Path(args.checkpoint), self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.replay: Deque[ReplaySample] = deque(maxlen=max(128, args.replay_size))
        self.running = False

        self.step_count = 0
        self.good_count = 0
        self.bad_count = 0
        self.loss_count = 0
        self.loss_sum = 0.0
        self.last_loss = float("nan")
        self.last_value = 0.0
        self.last_policy_conf = 0.0
        self.last_elapsed = 0.0

        self.lookahead_var = tk.IntVar(value=max(0, args.lookahead_ply))
        self.branch_cap_var = tk.IntVar(value=max(0, args.branch_cap))
        self.batch_var = tk.IntVar(value=max(8, args.batch_size))
        self.delay_var = tk.IntVar(value=1)
        self.visuals_var = tk.BooleanVar(value=True)
        self.source_var = tk.StringVar(value="selfplay")
        self.good_rule_var = tk.StringVar(value=args.good_rule)
        self.input_plane_var = tk.IntVar(value=0)
        self.out_path_var = tk.StringVar(value=args.out)
        self.status_var = tk.StringVar(value=f"Device: {self.device} | bootstrap: {loaded if loaded else 'new model'}")

        self._build_ui()
        self._refresh_stats()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        ctrl = ttk.LabelFrame(root, text="Controls", padding=8)
        ctrl.pack(fill="x")

        ttk.Button(ctrl, text="Step (1 sample)", command=self.step_once).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(ctrl, text="Run", command=self.start_run).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(ctrl, text="Pause", command=self.pause_run).grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(ctrl, text="Save Checkpoint", command=self.save_checkpoint).grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        ttk.Label(ctrl, text="Lookahead").grid(row=0, column=4, padx=(12, 4), sticky="e")
        ttk.Entry(ctrl, textvariable=self.lookahead_var, width=5).grid(row=0, column=5, padx=4, sticky="w")
        ttk.Label(ctrl, text="Branch Cap").grid(row=0, column=6, padx=(12, 4), sticky="e")
        ttk.Entry(ctrl, textvariable=self.branch_cap_var, width=5).grid(row=0, column=7, padx=4, sticky="w")
        ttk.Label(ctrl, text="Batch").grid(row=0, column=8, padx=(12, 4), sticky="e")
        ttk.Entry(ctrl, textvariable=self.batch_var, width=5).grid(row=0, column=9, padx=4, sticky="w")

        ttk.Label(ctrl, text="Tick ms").grid(row=1, column=0, padx=4, sticky="e")
        ttk.Entry(ctrl, textvariable=self.delay_var, width=6).grid(row=1, column=1, padx=4, sticky="w")
        ttk.Checkbutton(ctrl, text="Visuals On (slower)", variable=self.visuals_var).grid(row=1, column=2, columnspan=2, padx=4, sticky="w")
        ttk.Label(ctrl, text="Start Source").grid(row=1, column=4, padx=(12, 4), sticky="e")
        ttk.Combobox(ctrl, textvariable=self.source_var, values=["selfplay", "random"], width=10, state="readonly").grid(row=1, column=5, padx=4, sticky="w")
        ttk.Label(ctrl, text="Good Rule").grid(row=1, column=6, padx=(12, 4), sticky="e")
        ttk.Combobox(ctrl, textvariable=self.good_rule_var, values=["after_positive", "delta_nonnegative", "both"], width=14, state="readonly").grid(row=1, column=7, columnspan=2, padx=4, sticky="w")
        ttk.Label(ctrl, text="Input Plane").grid(row=1, column=9, padx=(12, 4), sticky="e")
        ttk.Combobox(ctrl, textvariable=self.input_plane_var, values=list(range(8)), width=4, state="readonly").grid(row=1, column=10, padx=4, sticky="w")

        ttk.Label(ctrl, text="Checkpoint Out").grid(row=2, column=0, padx=4, sticky="e")
        ttk.Entry(ctrl, textvariable=self.out_path_var, width=88).grid(row=2, column=1, columnspan=10, padx=4, pady=(4, 0), sticky="ew")
        for col in range(11):
            ctrl.grid_columnconfigure(col, weight=1 if col in (10,) else 0)
        ctrl.grid_columnconfigure(3, weight=1)

        status = ttk.Label(root, textvariable=self.status_var, font=("Menlo", 11))
        status.pack(fill="x", pady=(6, 8))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        boards = ttk.LabelFrame(body, text="Sample Visuals", padding=8)
        boards.pack(side="left", fill="both", expand=True)
        right = ttk.LabelFrame(body, text="Neural Net Visuals", padding=8)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.start_board = HexBoardCanvas(boards, self.env, "Initial Board")
        self.start_board.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
        self.proposed_board = HexBoardCanvas(boards, self.env, "After Proposed Move")
        self.proposed_board.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")
        self.rollout_board = HexBoardCanvas(boards, self.env, "After Minimax Rollout")
        self.rollout_board.grid(row=0, column=2, padx=4, pady=4, sticky="nsew")
        for col in range(3):
            boards.grid_columnconfigure(col, weight=1)
        boards.grid_rowconfigure(0, weight=1)

        valid_mask = [[False for _ in range(self.env.grid_size)] for _ in range(self.env.grid_size)]
        for q, r in self.env.coords:
            rr = r + self.env.radius
            cc = q + self.env.radius
            valid_mask[rr][cc] = True

        self.input_map = GridHeatmap(right, "Input Plane", self.env.grid_size, valid_mask)
        self.input_map.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
        self.stem_map = GridHeatmap(right, "Stem Mean Activation", self.env.grid_size, valid_mask)
        self.stem_map.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")
        self.trunk_map = GridHeatmap(right, "Trunk Mean Activation", self.env.grid_size, valid_mask)
        self.trunk_map.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        self.policy_map = GridHeatmap(right, "Policy Probability Map", self.env.grid_size, valid_mask)
        self.policy_map.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        for col in range(2):
            right.grid_columnconfigure(col, weight=1)
        for row in range(2):
            right.grid_rowconfigure(row, weight=1)

    def _refresh_stats(self) -> None:
        total_labeled = max(1, self.good_count + self.bad_count)
        avg_loss = self.loss_sum / max(1, self.loss_count)
        msg = (
            f"steps={self.step_count} replay={len(self.replay)} "
            f"good={self.good_count} bad={self.bad_count} goodRate={self.good_count / total_labeled:.3f} "
            f"lastLoss={self.last_loss:.4f} avgLoss={avg_loss:.4f} "
            f"lastPolicyConf={self.last_policy_conf:.3f} lastValue={self.last_value:+.3f} "
            f"stepTime={self.last_elapsed:.3f}s"
        )
        self.status_var.set(msg)

    def _sample_start(self) -> Tuple[State, int]:
        source = self.source_var.get()
        if source == "selfplay":
            start = build_selfplay_prefix_state(
                env=self.env,
                rng=self.rng,
                model=self.model,
                device=self.device,
                random_warmup_moves=30,
                policy_min_moves=20,
                policy_max_moves=120,
                epsilon=max(0.0, min(1.0, self.args.prefix_policy_epsilon)),
            )
            if start is not None:
                return start.state, start.to_move

        while True:
            state = build_random_state(
                env=self.env,
                rng=self.rng,
                empty_prob=0.46,
                max_level=5,
                min_turns=2,
                max_turns=140,
            )
            if self.env.terminal_winner(state) is not None:
                continue
            player_candidates = [RED, BLUE]
            self.rng.shuffle(player_candidates)
            for player in player_candidates:
                if self.env.legal_moves(state, player):
                    return state, player

    def _train_one_batch(self) -> None:
        batch_size = max(8, int(self.batch_var.get()))
        if len(self.replay) < batch_size:
            self.last_loss = float("nan")
            return

        batch = self.rng.sample(list(self.replay), batch_size)
        xb = torch.stack([s.x for s in batch], dim=0).to(self.device, non_blocking=True)
        move_idx = torch.tensor([s.move_idx for s in batch], dtype=torch.long, device=self.device)
        y_policy = torch.tensor([s.policy_label for s in batch], dtype=torch.float32, device=self.device)
        y_value = torch.tensor([s.value_target for s in batch], dtype=torch.float32, device=self.device)

        self.model.train()
        logits, value = self.model(xb)
        chosen_logits = logits.gather(1, move_idx.unsqueeze(1)).squeeze(1)
        policy_loss = F.binary_cross_entropy_with_logits(chosen_logits, y_policy)
        value_loss = F.mse_loss(value, y_value)
        loss = policy_loss + float(self.args.value_loss_weight) * value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.loss_sum += self.last_loss
        self.loss_count += 1

    def _update_nn_visuals(self, features: torch.Tensor, player: int, move_idx: int) -> None:
        with torch.no_grad():
            x = features.unsqueeze(0).to(self.device)
            stem = self.model.stem(x)
            trunk = self.model.trunk(stem)
            pol_feat = self.model.policy_head(trunk)
            logits = self.model.policy_fc(torch.flatten(pol_feat, 1))
            val_feat = self.model.value_head(trunk)
            value = torch.tanh(self.model.value_fc2(F.relu(self.model.value_fc1(torch.flatten(val_feat, 1)), inplace=False))).squeeze(1)

            policy_probs = F.softmax(logits, dim=1)[0].detach().cpu().reshape(self.env.grid_size, self.env.grid_size)
            selected_input = int(self.input_plane_var.get())
            selected_input = max(0, min(7, selected_input))
            input_plane = features[selected_input].detach().cpu()
            stem_map = stem[0].mean(dim=0).detach().cpu()
            trunk_map = trunk[0].mean(dim=0).detach().cpu()

            conf = torch.sigmoid(logits[0, move_idx]).item()
            self.last_policy_conf = float(conf)
            self.last_value = float(value.item())

        self.input_map.draw(input_plane, note=f"plane={selected_input} player={player}")
        self.stem_map.draw(stem_map, note=f"mean={float(stem_map.mean()):+.4f}")
        self.trunk_map.draw(trunk_map, note=f"mean={float(trunk_map.mean()):+.4f}")
        self.policy_map.draw(policy_probs, note=f"sigmoid(chosen)={self.last_policy_conf:.3f} value={self.last_value:+.3f}")

    def _perform_step(self, render: bool) -> None:
        started = time.monotonic()
        self.model.eval()
        state, player = self._sample_start()
        legal = self.env.legal_moves(state, player)
        if not legal:
            return

        move = choose_model_move(
            env=self.env,
            state=state,
            player=player,
            model=self.model,
            device=self.device,
            rng=self.rng,
            epsilon=max(0.0, min(1.0, self.args.sample_move_epsilon)),
        )
        before_diff = score_diff_for_player(self.env, state, player)
        proposed_state = self.env.apply_move(state, move, player)

        lookahead = max(0, int(self.lookahead_var.get()))
        branch_cap = max(0, int(self.branch_cap_var.get()))
        rollout_line: List[int] = []
        if lookahead > 0:
            after_diff, rollout_line = minimax_with_line(
                env=self.env,
                state=proposed_state,
                to_move=opponent_of(player),
                root_player=player,
                plies_left=lookahead,
                max_branch_checks=branch_cap,
                cache={},
            )
            rollout_state = apply_move_line(self.env, proposed_state, opponent_of(player), rollout_line)
        else:
            after_diff = score_diff_for_player(self.env, proposed_state, player)
            rollout_state = proposed_state

        policy_label = label_from_outcome(before_diff, after_diff, self.good_rule_var.get(), float(self.args.good_margin))
        value_target = value_target_from_outcome(
            before_diff=before_diff,
            after_reply_diff=after_diff,
            policy_label=policy_label,
            value_target_mode=self.args.value_target,
            value_scale=float(self.args.value_scale),
        )

        sample = ReplaySample(
            x=self.env.encode_features(state, player),
            move_idx=self.env.flat_index[move],
            policy_label=float(policy_label),
            value_target=float(value_target),
        )
        self.replay.append(sample)
        if policy_label >= 0.5:
            self.good_count += 1
        else:
            self.bad_count += 1

        self._train_one_batch()
        self.step_count += 1
        self.last_elapsed = time.monotonic() - started

        if render:
            rollout_text = " ".join(str(m) for m in rollout_line[:8])
            if len(rollout_line) > 8:
                rollout_text += " ..."
            self.start_board.draw_state(
                state,
                info=f"to_move={'RED' if player == RED else 'BLUE'} beforeDiff={before_diff:+d}",
            )
            self.proposed_board.draw_state(
                proposed_state,
                info=f"proposed_move={move} (flat={self.env.flat_index[move]})",
                highlight_idx=move,
            )
            self.rollout_board.draw_state(
                rollout_state,
                info=f"afterDiff={after_diff:+d} label={'GOOD' if policy_label >= 0.5 else 'BAD'} pv={rollout_text}",
            )
            self._update_nn_visuals(sample.x, player, sample.move_idx)

        self._refresh_stats()

    def _run_tick(self) -> None:
        if not self.running:
            return
        try:
            self._perform_step(render=bool(self.visuals_var.get()))
        except Exception as exc:  # pylint: disable=broad-except
            self.running = False
            messagebox.showerror("Training error", str(exc))
            return
        delay = max(1, int(self.delay_var.get()))
        self.after(delay, self._run_tick)

    def start_run(self) -> None:
        if self.running:
            return
        self.running = True
        self._run_tick()

    def pause_run(self) -> None:
        self.running = False

    def step_once(self) -> None:
        if self.running:
            return
        try:
            self._perform_step(render=True)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Step error", str(exc))

    def save_checkpoint(self) -> None:
        out_path = Path(self.out_path_var.get().strip())
        if not out_path:
            messagebox.showwarning("Missing path", "Please provide an output path.")
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "mode": "ui_counterfactual",
            "steps": self.step_count,
            "good_count": self.good_count,
            "bad_count": self.bad_count,
            "avg_loss": self.loss_sum / max(1, self.loss_count),
            "device": str(self.device),
            "saved_at": time.time(),
        }
        checkpoint = {
            "state_dict": self.model.cpu().state_dict(),
            "channels": int(self.model.stem[0].out_channels),
            "blocks": int(len(self.model.trunk)),
            "board_size": int(self.model.board_size),
            "input_channels": int(self.model.stem[0].in_channels),
            "meta": meta,
        }
        torch.save(checkpoint, out_path)
        self.model.to(self.device)
        self.status_var.set(f"Saved checkpoint to {out_path}")


def main() -> None:
    args = parse_args()
    app = TrainingUI(args)
    app.mainloop()


if __name__ == "__main__":
    main()
