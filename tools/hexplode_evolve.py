#!/usr/bin/env python3
"""
Evolutionary self-play trainer for Hexplode large-board hard AI.

Usage:
  python3 tools/hexplode_evolve.py --generations 20 --population 20 --duels 12
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple


RED = 1
BLUE = 2
EMPTY = 0

DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


@dataclass
class Weights:
    bias: float
    score_diff: float
    piece_diff: float
    critical_diff: float
    edge_diff: float
    center_diff: float
    swing: float
    immediate_win: float
    opponent_immediate_win: float


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
        self.coords: List[Tuple[int, int]] = self._generate_hexagon(self.radius)
        self.index_of = {coord: i for i, coord in enumerate(self.coords)}
        self.neighbors: List[List[int]] = [[] for _ in self.coords]
        self.is_edge: List[bool] = [False] * len(self.coords)
        self.is_corner: List[bool] = [False] * len(self.coords)
        self.center_weight: List[int] = [0] * len(self.coords)
        self._build_geometry_caches()

    def _generate_hexagon(self, radius: int) -> List[Tuple[int, int]]:
        coords: List[Tuple[int, int]] = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                coords.append((q, r))
        return coords

    def _build_geometry_caches(self) -> None:
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
            queue_head = 0
            queued = {move_idx}
            while queue_head < len(queue):
                cur = queue[queue_head]
                queue_head += 1
                queued.discard(cur)
                cur_owner = owners[cur]
                cur_level = levels[cur]
                if cur_owner == EMPTY or cur_level <= 5:
                    continue

                exploding_player = cur_owner
                owners[cur] = EMPTY
                levels[cur] = 0

                for n in self.neighbors[cur]:
                    owners[n] = exploding_player
                    levels[n] += 1
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

    def terminal_score(self, state: State, player: int, depth: int = 0) -> int | None:
        opponent = BLUE if player == RED else RED
        player_sum = self.summarize(state, player)
        opp_sum = self.summarize(state, opponent)

        if player_sum.score >= 85 or opp_sum.score >= 85:
            if player_sum.score > opp_sum.score:
                return 10000 + depth
            if opp_sum.score > player_sum.score:
                return -10000 - depth
            return 0

        if state.turns_played >= 2:
            if player_sum.pieces > 0 and opp_sum.pieces == 0:
                return 10000 + depth
            if opp_sum.pieces > 0 and player_sum.pieces == 0:
                return -10000 - depth
            if player_sum.pieces == 0 and opp_sum.pieces == 0:
                return 0

        return None

    def is_win(self, state: State, player: int) -> bool:
        term = self.terminal_score(state, player, depth=0)
        return term is not None and term > 0

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


def score_move(env: HexplodeEnv, state: State, player: int, move: int, w: Weights) -> float:
    opp = BLUE if player == RED else RED
    before_p = env.summarize(state, player)
    before_o = env.summarize(state, opp)
    nxt = env.apply_move(state, move, player)

    if env.is_win(nxt, player):
        return w.immediate_win

    after_p = env.summarize(nxt, player)
    after_o = env.summarize(nxt, opp)
    opp_immediate = env.count_immediate_wins(nxt, opp, max_checks=14)

    score_diff = after_p.score - after_o.score
    piece_diff = after_p.pieces - after_o.pieces
    critical_diff = after_p.critical - after_o.critical
    edge_diff = after_p.edge_load - after_o.edge_load
    center_diff = after_p.center_control - after_o.center_control
    swing = (after_p.score - before_p.score) - (after_o.score - before_o.score)

    return (
        w.bias
        + w.score_diff * score_diff
        + w.piece_diff * piece_diff
        + w.critical_diff * critical_diff
        + w.edge_diff * edge_diff
        + w.center_diff * center_diff
        + w.swing * swing
        - w.opponent_immediate_win * opp_immediate
    )


def choose_move(env: HexplodeEnv, state: State, player: int, w: Weights, rng: random.Random) -> int:
    moves = env.legal_moves(state, player)
    if not moves:
        return -1
    best_score = -math.inf
    best_moves: List[int] = []
    for move in moves:
        s = score_move(env, state, player, move, w)
        if s > best_score:
            best_score = s
            best_moves = [move]
        elif s == best_score:
            best_moves.append(move)
    return rng.choice(best_moves)


def play_game(env: HexplodeEnv, w_red: Weights, w_blue: Weights, seed: int, max_turns: int = 280) -> int:
    rng = random.Random(seed)
    state = env.empty_state()
    current = RED if rng.random() < 0.5 else BLUE

    for _ in range(max_turns):
        w = w_red if current == RED else w_blue
        move = choose_move(env, state, current, w, rng)
        if move < 0:
            break
        state = env.apply_move(state, move, current)

        if env.is_win(state, current):
            return current

        current = BLUE if current == RED else RED

    red_sum = env.summarize(state, RED)
    blue_sum = env.summarize(state, BLUE)
    if red_sum.score == blue_sum.score:
        return 0
    return RED if red_sum.score > blue_sum.score else BLUE


def mutate(w: Weights, rng: random.Random, scale: float) -> Weights:
    return Weights(
        bias=w.bias + rng.gauss(0, 0.4 * scale),
        score_diff=max(0.1, w.score_diff + rng.gauss(0, 0.22 * scale)),
        piece_diff=max(0.1, w.piece_diff + rng.gauss(0, 0.36 * scale)),
        critical_diff=max(0.1, w.critical_diff + rng.gauss(0, 0.6 * scale)),
        edge_diff=min(-0.05, w.edge_diff + rng.gauss(0, 0.28 * scale)),
        center_diff=max(0.05, w.center_diff + rng.gauss(0, 0.18 * scale)),
        swing=max(0.05, w.swing + rng.gauss(0, 0.2 * scale)),
        immediate_win=max(1000.0, w.immediate_win + rng.gauss(0, 280 * scale)),
        opponent_immediate_win=max(50.0, w.opponent_immediate_win + rng.gauss(0, 120 * scale)),
    )


def crossover(a: Weights, b: Weights, rng: random.Random) -> Weights:
    mix = lambda x, y: (x + y) * 0.5 + rng.gauss(0, abs(x - y) * 0.08 + 1e-9)
    return Weights(
        bias=mix(a.bias, b.bias),
        score_diff=max(0.1, mix(a.score_diff, b.score_diff)),
        piece_diff=max(0.1, mix(a.piece_diff, b.piece_diff)),
        critical_diff=max(0.1, mix(a.critical_diff, b.critical_diff)),
        edge_diff=min(-0.05, mix(a.edge_diff, b.edge_diff)),
        center_diff=max(0.05, mix(a.center_diff, b.center_diff)),
        swing=max(0.05, mix(a.swing, b.swing)),
        immediate_win=max(1000.0, mix(a.immediate_win, b.immediate_win)),
        opponent_immediate_win=max(50.0, mix(a.opponent_immediate_win, b.opponent_immediate_win)),
    )


def evaluate_candidate(
    env: HexplodeEnv, candidate: Weights, population: List[Weights], duels: int, seed_base: int
) -> float:
    score = 0.0
    for duel_idx in range(duels):
        opp = population[(seed_base + duel_idx * 7) % len(population)]
        seed = seed_base * 1000 + duel_idx

        winner = play_game(env, candidate, opp, seed=seed)
        if winner == RED:
            score += 1.0
        elif winner == 0:
            score += 0.5

        winner = play_game(env, opp, candidate, seed=seed + 131)
        if winner == BLUE:
            score += 1.0
        elif winner == 0:
            score += 0.5
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-length", type=int, default=4)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--duels", type=int, default=12)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = HexplodeEnv(side_length=args.side_length)

    seed = Weights(
        bias=0.0,
        score_diff=1.45,
        piece_diff=3.10,
        critical_diff=6.85,
        edge_diff=-1.55,
        center_diff=0.95,
        swing=1.35,
        immediate_win=12000.0,
        opponent_immediate_win=2600.0,
    )

    population: List[Weights] = [seed]
    while len(population) < args.population:
        population.append(mutate(seed, rng, scale=1.8))

    for gen in range(args.generations):
        ranked: List[Tuple[float, Weights]] = []
        for i, cand in enumerate(population):
            fit = evaluate_candidate(env, cand, population, args.duels, seed_base=(gen + 1) * 100 + i)
            ranked.append((fit, cand))
        ranked.sort(key=lambda x: x[0], reverse=True)
        best_fit, best = ranked[0]
        avg_fit = sum(s for s, _ in ranked) / max(1, len(ranked))
        print(f"gen {gen + 1:02d} | best={best_fit:.2f} avg={avg_fit:.2f} | {best}")

        elite = [w for _, w in ranked[: max(1, min(args.elite, len(ranked)))]]
        new_population = elite[:]
        while len(new_population) < args.population:
            parent_a = rng.choice(elite)
            parent_b = rng.choice(ranked[: max(2, len(ranked) // 2)])[1]
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng, scale=max(0.35, 1.4 - gen / max(1, args.generations)))
            new_population.append(child)
        population = new_population

    final_ranked: List[Tuple[float, Weights]] = []
    for i, cand in enumerate(population):
        fit = evaluate_candidate(env, cand, population, args.duels * 2, seed_base=9000 + i)
        final_ranked.append((fit, cand))
    final_ranked.sort(key=lambda x: x[0], reverse=True)
    winner = final_ranked[0][1]

    print("\nBest evolved weights:")
    print(json.dumps(asdict(winner), indent=2))
    print("\nSwift literal:")
    print(
        "HexpandEvolvedWeights(\n"
        f"    bias: {winner.bias:.6f},\n"
        f"    scoreDiff: {winner.score_diff:.6f},\n"
        f"    pieceDiff: {winner.piece_diff:.6f},\n"
        f"    criticalDiff: {winner.critical_diff:.6f},\n"
        f"    edgeDiff: {winner.edge_diff:.6f},\n"
        f"    centerDiff: {winner.center_diff:.6f},\n"
        f"    swing: {winner.swing:.6f},\n"
        f"    immediateWin: {winner.immediate_win:.6f},\n"
        f"    opponentImmediateWin: {winner.opponent_immediate_win:.6f}\n"
        ")"
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(asdict(winner), f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
