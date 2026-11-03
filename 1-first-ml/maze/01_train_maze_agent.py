"""Project 2 — Train a tiny maze agent with tabular Q-learning.

Run from the repository root:
    python 1-first-ml/maze/01_train_maze_agent.py
    python 1-first-ml/maze/01_train_maze_agent.py --episodes 10000

Only Python's standard library is used.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass

Position = tuple[int, int]
Action = str

ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")
DELTAS: dict[Action, Position] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID = (
    "#########",
    "#S#.....#",
    "#.#.###.#",
    "#.#...#.#",
    "#...#...#",
    "###.#.#.#",
    "#...#...#",
    "#.......#",
    "#########",
)


@dataclass
class Maze:
    """Small deterministic grid-world environment."""

    grid: tuple[str, ...] = GRID
    step_reward: float = -0.04
    wall_reward: float = -0.15
    goal_reward: float = 1.0

    def __post_init__(self) -> None:
        self.start = self._find("S")
        self.goal = self._find("G")
        self.position = self.start

    def _find(self, marker: str) -> Position:
        for row, line in enumerate(self.grid):
            for column, cell in enumerate(line):
                if cell == marker:
                    return (row, column)
        raise ValueError(f"Maze has no {marker!r} marker.")

    def reset(self) -> Position:
        self.position = self.start
        return self.position

    def step(self, action: Action) -> tuple[Position, float, bool]:
        if action not in DELTAS:
            raise ValueError(f"Unknown action: {action}")

        dr, dc = DELTAS[action]
        row, column = self.position
        next_position = (row + dr, column + dc)
        nr, nc = next_position

        if self.grid[nr][nc] == "#":
            return self.position, self.wall_reward, False

        self.position = next_position
        if self.position == self.goal:
            return self.position, self.goal_reward, True
        return self.position, self.step_reward, False

    def render(self, path: set[Position] | None = None) -> str:
        """Show the maze; optional path cells are marked with dots."""
        path = path or set()
        lines = []
        for row, line in enumerate(self.grid):
            rendered = []
            for column, cell in enumerate(line):
                position = (row, column)
                if position == self.start:
                    rendered.append("S")
                elif position == self.goal:
                    rendered.append("G")
                elif cell == "#":
                    rendered.append("#")
                elif position in path:
                    rendered.append("*")
                else:
                    rendered.append(" ")
            lines.append("".join(rendered))
        return "\n".join(lines)


def choose_action(
    q_values: defaultdict[tuple[Position, Action], float],
    state: Position,
    epsilon: float,
    rng: random.Random,
) -> Action:
    """With probability epsilon explore; otherwise choose a best-known action."""
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)

    values = [q_values[(state, action)] for action in ACTIONS]
    best_value = max(values)
    best_actions = [
        action
        for action, value in zip(ACTIONS, values)
        if value == best_value
    ]
    return rng.choice(best_actions)


def train(
    maze: Maze,
    episodes: int,
    seed: int,
    learning_rate: float = 0.15,
    discount: float = 0.95,
) -> defaultdict[tuple[Position, Action], float]:
    rng = random.Random(seed)
    q_values: defaultdict[tuple[Position, Action], float] = defaultdict(float)

    for episode in range(episodes):
        state = maze.reset()
        epsilon = max(0.03, 1.0 - episode / max(1, episodes * 0.8))

        for _ in range(200):
            action = choose_action(q_values, state, epsilon, rng)
            next_state, reward, done = maze.step(action)

            if done:
                target = reward
            else:
                future_value = max(q_values[(next_state, a)] for a in ACTIONS)
                target = reward + discount * future_value

            old_value = q_values[(state, action)]
            q_values[(state, action)] = old_value + learning_rate * (
                target - old_value
            )
            state = next_state

            if done:
                break

        if (episode + 1) % max(1, episodes // 10) == 0:
            print(f"Training progress: {episode + 1}/{episodes} episodes")

    return q_values


def follow_greedy_policy(
    maze: Maze,
    q_values: defaultdict[tuple[Position, Action], float],
    max_steps: int = 100,
) -> tuple[list[Position], bool]:
    """Follow the learned best action without exploration."""
    state = maze.reset()
    path = [state]
    visited = {state}

    for _ in range(max_steps):
        values = [q_values[(state, action)] for action in ACTIONS]
        best_value = max(values)
        best_actions = [
            action
            for action, value in zip(ACTIONS, values)
            if value == best_value
        ]
        # Prefer a move that has not already been visited if values are tied.
        candidates = []
        for action in best_actions:
            dr, dc = DELTAS[action]
            candidate = (state[0] + dr, state[1] + dc)
            if maze.grid[candidate[0]][candidate[1]] != "#":
                candidates.append((action, candidate))
        unvisited = [item for item in candidates if item[1] not in visited]
        choices = unvisited or candidates
        if not choices:
            return path, False

        action, _ = choices[0]
        state, _, done = maze.step(action)
        path.append(state)
        visited.add(state)
        if done:
            return path, True

    return path, state == maze.goal


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a maze-solving Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=3000, help="training episodes (default: 3000)")
    parser.add_argument("--seed", type=int, default=7, help="random seed for repeatable training")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    maze = Maze()
    print("MAZE: S = start, G = goal, # = wall")
    print(maze.render())
    print(f"\nTraining the agent for {args.episodes:,} episodes...")

    q_values = train(maze, args.episodes, args.seed)
    path, reached_goal = follow_greedy_policy(maze, q_values)

    print("\nLEARNED ROUTE: * = route chosen by the agent")
    print(maze.render(set(path)))
    if reached_goal:
        print(f"\nThe agent reached the goal in {len(path) - 1} steps.")
    else:
        print("\nThe agent did not reach the goal this time. Try more episodes.")
    print("\nTry: python 1-first-ml/maze/01_train_maze_agent.py --episodes 10000")


if __name__ == "__main__":
    main()
