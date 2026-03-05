#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

RED = 1
BLUE = 2
EMPTY = 0
DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


@dataclass(frozen=True)
class HeuristicWeights:
    bias: float = 0.383594
    score_diff: float = 1.306195
    piece_diff: float = 4.303645
    critical_diff: float = 4.725590
    edge_diff: float = -2.498602
    center_diff: float = 0.425609
    swing: float = 1.614324
    immediate_win: float = 11935.626926
    opponent_immediate_win: float = 2558.127751


DEFAULT_HEURISTIC_WEIGHTS = HeuristicWeights()


@dataclass
class PolicySummary:
    score: int = 0
    pieces: int = 0
    critical: int = 0
    edge_load: int = 0
    center_control: int = 0


@dataclass
class State:
    owners: List[int]
    levels: List[int]
    turns_played: int


class HexplodeEnv:
    def __init__(self, side_length: int = 4) -> None:
        self.side_length = side_length
        self.radius = side_length - 1
        self.grid_size = self.radius * 2 + 1

        self.coords = self._generate_hexagon(self.radius)
        self.index_of = {coord: i for i, coord in enumerate(self.coords)}
        self.neighbors: List[List[int]] = [[] for _ in self.coords]
        self.is_edge: List[bool] = [False] * len(self.coords)
        self.is_corner: List[bool] = [False] * len(self.coords)
        self.center_weight: List[int] = [0] * len(self.coords)
        self.flat_index: List[int] = [0] * len(self.coords)
        self._build_geometry()

    def _generate_hexagon(self, radius: int) -> List[Tuple[int, int]]:
        coords: List[Tuple[int, int]] = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                coords.append((q, r))
        return coords

    def _build_geometry(self) -> None:
        corner_set = {
            (self.radius, 0),
            (0, self.radius),
            (-self.radius, self.radius),
            (-self.radius, 0),
            (0, -self.radius),
            (self.radius, -self.radius),
        }
        for i, (q, r) in enumerate(self.coords):
            for dq, dr in DIRECTIONS:
                n = (q + dq, r + dr)
                if n in self.index_of:
                    self.neighbors[i].append(self.index_of[n])

            x = q
            z = r
            y = -x - z
            dist = max(abs(x), abs(y), abs(z))
            self.is_edge[i] = dist == self.radius
            self.is_corner[i] = (q, r) in corner_set
            self.center_weight[i] = max(0, self.radius - dist)

            row = r + self.radius
            col = q + self.radius
            self.flat_index[i] = row * self.grid_size + col

    def empty_state(self) -> State:
        n = len(self.coords)
        return State(owners=[EMPTY] * n, levels=[0] * n, turns_played=0)

    def legal_moves(self, state: State, player: int) -> List[int]:
        moves = [i for i, owner in enumerate(state.owners) if owner == EMPTY or owner == player]
        moves.sort(key=lambda i: (state.levels[i] if state.owners[i] == player else 0), reverse=True)
        return moves

    def apply_move(self, state: State, move_idx: int, player: int) -> State:
        owners = state.owners[:]
        levels = state.levels[:]
        turns_played = state.turns_played + 1

        owners[move_idx] = player
        levels[move_idx] += 1

        if levels[move_idx] > 5:
            queue = [move_idx]
            q_head = 0
            queued = {move_idx}
            while q_head < len(queue):
                cur = queue[q_head]
                q_head += 1
                queued.discard(cur)

                cur_owner = owners[cur]
                cur_level = levels[cur]
                if cur_owner == EMPTY or cur_level <= 5:
                    continue

                exploding_player = cur_owner
                burst_count = cur_level // 6
                remainder = cur_level % 6
                if burst_count <= 0:
                    continue

                if remainder > 0:
                    owners[cur] = exploding_player
                    levels[cur] = remainder
                else:
                    owners[cur] = EMPTY
                    levels[cur] = 0

                for n in self.neighbors[cur]:
                    owners[n] = exploding_player
                    levels[n] += burst_count
                    if levels[n] > 5 and n not in queued:
                        queue.append(n)
                        queued.add(n)

        return State(owners=owners, levels=levels, turns_played=turns_played)

    def summarize(self, state: State, player: int) -> PolicySummary:
        summary = PolicySummary()
        for i, owner in enumerate(state.owners):
            if owner != player:
                continue
            level = state.levels[i]
            summary.score += level
            summary.pieces += 1
            if level >= 5:
                summary.critical += 1
            if self.is_corner[i]:
                summary.edge_load += level * 2
            elif self.is_edge[i]:
                summary.edge_load += level
            summary.center_control += self.center_weight[i] * level
        return summary

    def score_totals(self, state: State) -> Tuple[int, int]:
        red = 0
        blue = 0
        for i, owner in enumerate(state.owners):
            if owner == RED:
                red += state.levels[i]
            elif owner == BLUE:
                blue += state.levels[i]
        return red, blue

    def piece_totals(self, state: State) -> Tuple[int, int]:
        red = 0
        blue = 0
        for owner in state.owners:
            if owner == RED:
                red += 1
            elif owner == BLUE:
                blue += 1
        return red, blue

    def terminal_winner(self, state: State) -> int | None:
        # Match in-game Hexplode hard rules: no score target terminal.
        # A game ends only by wipeout (after opening turns), otherwise callers
        # decide by max-turn fallback using total ball count.
        if state.turns_played >= 2:
            red_pieces, blue_pieces = self.piece_totals(state)
            if red_pieces > 0 and blue_pieces == 0:
                return RED
            if blue_pieces > 0 and red_pieces == 0:
                return BLUE
            if red_pieces == 0 and blue_pieces == 0:
                return 0
        return None

    def is_win(self, state: State, player: int) -> bool:
        winner = self.terminal_winner(state)
        return winner == player

    def count_immediate_wins(self, state: State, player: int, max_checks: int = 14) -> int:
        wins = 0
        for i, move in enumerate(self.legal_moves(state, player)):
            if i >= max_checks:
                break
            nxt = self.apply_move(state, move, player)
            if self.is_win(nxt, player):
                wins += 1
                if wins >= 2:
                    return wins
        return wins

    def encode_features(self, state: State, current_player: int) -> torch.Tensor:
        x = torch.zeros((8, self.grid_size, self.grid_size), dtype=torch.float32)
        legal = set(self.legal_moves(state, current_player))
        opponent = BLUE if current_player == RED else RED

        for i, (q, r) in enumerate(self.coords):
            row = r + self.radius
            col = q + self.radius
            owner = state.owners[i]
            level = state.levels[i]
            level_norm = min(6, max(0, level)) / 6.0

            if owner == current_player:
                x[0, row, col] = 1.0
                x[3, row, col] = level_norm
                if level >= 5:
                    x[5, row, col] = 1.0
            elif owner == opponent:
                x[1, row, col] = 1.0
                x[4, row, col] = level_norm
                if level >= 5:
                    x[6, row, col] = 1.0
            else:
                x[2, row, col] = 1.0

            if i in legal:
                x[7, row, col] = 1.0

        turn_scalar = min(1.0, state.turns_played / 40.0)
        x[7] = torch.clamp(x[7] * (0.8 + 0.2 * turn_scalar), 0.0, 1.0)
        return x

    def encode_policy_target(self, move_probs: Sequence[Tuple[int, float]]) -> torch.Tensor:
        y = torch.zeros((self.grid_size * self.grid_size,), dtype=torch.float32)
        for move_idx, prob in move_probs:
            y[self.flat_index[move_idx]] = float(prob)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        return F.relu(x, inplace=True)


class HexplodePolicyValueNet(nn.Module):
    def __init__(self, input_channels: int = 8, channels: int = 64, blocks: int = 6, board_size: int = 7) -> None:
        super().__init__()
        self.board_size = board_size

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.trunk(x)

        policy = self.policy_head(x)
        policy = torch.flatten(policy, 1)
        policy_logits = self.policy_fc(policy)

        value = self.value_head(x)
        value = torch.flatten(value, 1)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value)).squeeze(1)
        return policy_logits, value


def score_move(
    env: HexplodeEnv,
    state: State,
    player: int,
    move: int,
    weights: HeuristicWeights = DEFAULT_HEURISTIC_WEIGHTS,
) -> float:
    opponent = BLUE if player == RED else RED

    before_player = env.summarize(state, player)
    before_opp = env.summarize(state, opponent)
    nxt = env.apply_move(state, move, player)

    if env.is_win(nxt, player):
        return weights.immediate_win

    after_player = env.summarize(nxt, player)
    after_opp = env.summarize(nxt, opponent)
    opp_immediate = env.count_immediate_wins(nxt, opponent, max_checks=14)

    score_diff = after_player.score - after_opp.score
    piece_diff = after_player.pieces - after_opp.pieces
    critical_diff = after_player.critical - after_opp.critical
    edge_diff = after_player.edge_load - after_opp.edge_load
    center_diff = after_player.center_control - after_opp.center_control
    swing = (after_player.score - before_player.score) - (after_opp.score - before_opp.score)

    return (
        weights.bias
        + weights.score_diff * score_diff
        + weights.piece_diff * piece_diff
        + weights.critical_diff * critical_diff
        + weights.edge_diff * edge_diff
        + weights.center_diff * center_diff
        + weights.swing * swing
        - weights.opponent_immediate_win * opp_immediate
    )


def teacher_policy(
    env: HexplodeEnv,
    state: State,
    player: int,
    weights: HeuristicWeights = DEFAULT_HEURISTIC_WEIGHTS,
    temperature: float = 1.4,
) -> Tuple[List[int], List[float], torch.Tensor]:
    legal = env.legal_moves(state, player)
    if not legal:
        empty = torch.zeros((env.grid_size * env.grid_size,), dtype=torch.float32)
        return [], [], empty

    scores = [score_move(env, state, player, m, weights) for m in legal]
    max_score = max(scores)
    scaled = [(s - max_score) / max(0.25, temperature) for s in scores]
    exp_scores = [math.exp(max(-40.0, min(40.0, s))) for s in scaled]
    total = sum(exp_scores)
    probs = [s / total for s in exp_scores]
    target = env.encode_policy_target(zip(legal, probs))
    return legal, probs, target


def sample_from_probs(indices: Sequence[int], probs: Sequence[float], rng: random.Random) -> int:
    roll = rng.random()
    cumulative = 0.0
    for idx, p in zip(indices, probs):
        cumulative += p
        if roll <= cumulative:
            return idx
    return indices[-1]


def game_winner_by_score(env: HexplodeEnv, state: State) -> int:
    red, blue = env.score_totals(state)
    if red > blue:
        return RED
    if blue > red:
        return BLUE
    return 0


def generate_self_play_dataset(
    games: int,
    seed: int = 7,
    max_turns: int = 280,
    epsilon: float = 0.12,
    temperature: float = 1.4,
    weights: HeuristicWeights = DEFAULT_HEURISTIC_WEIGHTS,
    side_length: int = 4,
    sample_stride: int = 2,
    max_samples: int = 80_000,
    progress_every: int = 0,
    progress_fn: Callable[[int, int], None] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    env = HexplodeEnv(side_length=side_length)
    rng = random.Random(seed)

    features: List[torch.Tensor] = []
    policies: List[torch.Tensor] = []
    values: List[float] = []

    stop_early = False
    for game_idx in range(games):
        state = env.empty_state()
        player = RED if rng.random() < 0.5 else BLUE
        records: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

        winner: int | None = None
        for turn in range(max_turns):
            winner = env.terminal_winner(state)
            if winner is not None:
                break

            legal, probs, target = teacher_policy(
                env=env,
                state=state,
                player=player,
                weights=weights,
                temperature=temperature,
            )
            if not legal:
                break

            if sample_stride <= 1 or (turn % sample_stride == 0):
                records.append((env.encode_features(state, player), target, player))

            if rng.random() < epsilon:
                move = rng.choice(legal)
            else:
                move = sample_from_probs(legal, probs, rng)

            state = env.apply_move(state, move, player)
            winner = env.terminal_winner(state)
            if winner is not None:
                break
            player = BLUE if player == RED else RED

        if winner is None:
            winner = game_winner_by_score(env, state)

        for x, policy, sample_player in records:
            v = 0.0 if winner == 0 else (1.0 if sample_player == winner else -1.0)
            features.append(x)
            policies.append(policy)
            values.append(v)
            if max_samples > 0 and len(features) >= max_samples:
                stop_early = True
                break

        if progress_every > 0 and (game_idx + 1) % progress_every == 0 and progress_fn is not None:
            progress_fn(game_idx + 1, len(features))
        if stop_early:
            break

    if not features:
        raise RuntimeError("Dataset generation produced no samples.")

    x = torch.stack(features, dim=0)
    y_policy = torch.stack(policies, dim=0)
    y_value = torch.tensor(values, dtype=torch.float32)
    return x, y_policy, y_value


def soft_cross_entropy(policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(policy_logits, dim=1)
    return -(policy_target * log_probs).sum(dim=1).mean()
